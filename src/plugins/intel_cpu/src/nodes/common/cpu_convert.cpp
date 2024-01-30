// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"

#include "cpu_memcpy.h"
#include "utils/bfloat16.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#include "nodes/kernels/x64/jit_kernel.hpp"
#else
#include "cpu_memory.h"
#include "openvino/core/type/element_type_traits.hpp"
#include "selective_build.h"
#include "utils/general_utils.h"
#endif

namespace ov {
namespace intel_cpu {
namespace {

#if defined(OPENVINO_ARCH_X86_64)

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

template <typename src_t, typename dst_t, bool clamp = false>
void convert_vec(jit_generator & gen,
                 const RegExp & src,
                 const RegExp & dst);

template <typename src_t, typename dst_t, bool clamp = false>
void convert_vec_prepare(jit_generator& gen) {}

template <>
void convert_vec<ov::float16, float>(jit_generator & gen,
                                     const RegExp & src,
                                     const RegExp & dst) {
    auto const & f16vec = gen.xmm3;
    auto const & f32vec = gen.ymm4;

    gen.movdqu(f16vec, gen.xword[src]);
    gen.vcvtph2ps(f32vec, f16vec);
    gen.vmovups(gen.yword[dst], f32vec);
}

template <>
void convert_vec<float, ov::float16>(jit_generator & gen,
                                     const RegExp & src,
                                     const RegExp & dst) {
    auto const & f16vec = gen.xmm3;
    auto const & f32vec = gen.ymm4;

    gen.vmovups(f32vec, gen.yword[src]);
    gen.vcvtps2ph(f16vec, f32vec, 0);
    gen.movdqu(gen.xword[dst], f16vec);
}

template <>
void convert_vec_prepare<float, ov::float16, true>(jit_generator& gen) {
    auto upper_bound = gen.ymm5;
    auto lower_bound = gen.ymm6;
    auto addr = gen.r15;

    static const float f16_max = std::numeric_limits<ov::float16>::max();
    static const float f16_min = std::numeric_limits<ov::float16>::lowest();
    static const float upper_bounds[8] = {f16_max, f16_max, f16_max, f16_max, f16_max, f16_max, f16_max, f16_max};
    static const float lower_bounds[8] = {f16_min, f16_min, f16_min, f16_min, f16_min, f16_min, f16_min, f16_min};

    gen.mov(addr, (size_t)upper_bounds);
    gen.vmovdqu(upper_bound, gen.yword[addr]);
    gen.mov(addr, (size_t)lower_bounds);
    gen.vmovdqu(lower_bound, gen.yword[addr]);
}

template <>
void convert_vec<float, ov::float16, true>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto f16vec = gen.xmm3;
    auto f32vec = gen.ymm4;
    auto upper_bound = gen.ymm5;
    auto lower_bound = gen.ymm6;

    gen.vmovups(f32vec, gen.yword[src]);
    gen.vminps(f32vec, f32vec, upper_bound);
    gen.vmaxps(f32vec, f32vec, lower_bound);
    gen.vcvtps2ph(f16vec, f32vec, 0);
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void convert_vec<uint8_t, ov::float16>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto u8vec = gen.xmm1;
    auto i32vec = gen.ymm2;
    auto f16vec = gen.xmm3;
    auto fvec = gen.ymm4;

    gen.movq(u8vec, gen.qword[src]);
    gen.vpmovzxbd(i32vec, u8vec);
    gen.vcvtdq2ps(fvec, i32vec);
    gen.vcvtps2ph(f16vec, fvec, 0);
    gen.vzeroupper();
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void convert_vec_prepare<float, int8_t>(jit_generator& gen) {
    auto order = gen.ymm1;
    auto addr = gen.r15;

    static const int8_t offsets[32] = {0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                       -1, -1, -1, -1, 0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1};

    gen.mov(addr, (size_t)offsets);       // get offsets[] address
    gen.vmovdqu(order, gen.yword[addr]);  // save offsets[] to ymm register
}

template <>
void convert_vec<float, int8_t>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto order = gen.ymm1;
    auto p32vec = gen.ymm2;
    auto p32vec_lo = gen.xmm2;
    auto p32vec_hi = gen.xmm3;

    gen.vcvttps2dq(p32vec, gen.yword[src]);     // convert 8 floats to 8 ints
    gen.vpshufb(p32vec, p32vec, order);         // Shuffle the bytes according to the order
    gen.vextracti128(p32vec_hi, p32vec, 1);     // extract upper part of p32vec
    gen.vpor(p32vec_lo, p32vec_lo, p32vec_hi);  // p32vec_lo = p32vec_lo | p32vec_hi
    gen.movq(gen.qword[dst], p32vec_lo);        // save the result
}

template <>
void convert_vec_prepare<ov::float16, int8_t>(jit_generator& gen) {
    convert_vec_prepare<float, int8_t>(gen);
}

template <>
void convert_vec<ov::float16, int8_t>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto order = gen.ymm1;
    auto p32vec = gen.ymm2;
    auto p32vec_lo = gen.xmm2;
    auto p32vec_hi = gen.xmm3;

    gen.vcvtph2ps(p32vec, gen.xword[src]);      // convert 8 fp16's to 8 floats
    gen.vcvttps2dq(p32vec, p32vec);             // convert 8 floats to 8 ints
    gen.vpshufb(p32vec, p32vec, order);         // Shuffle the bytes according to the order
    gen.vextracti128(p32vec_hi, p32vec, 1);     // extract upper part of p32vec
    gen.vpor(p32vec_lo, p32vec_lo, p32vec_hi);  // p32vec_lo = p32vec_lo | p32vec_hi
    gen.movq(gen.qword[dst], p32vec_lo);        // save the result
}

class jit_convert_array : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_array)

    void generate() override {
        constexpr size_t vlen = 8u;
        constexpr size_t vlen_log2 = 3;

        preamble();

        _convert_vec_prepare(*this);

        // Get arguments addresses
        auto src = arg(&args_t::src);
        auto dst = arg(&args_t::out);
        auto size = arg(&args_t::count);

        size >>= vlen_log2;

        foreach(0, size, [&, this](const Xbyak::Reg64& idx) {
            _convert_vec(*this, src, dst);
            src += _src_size * vlen;
            dst += _dst_size * vlen;
        });

        mov(size, argPtr(&args_t::count));
        size &= vlen - 1;

        // Tail conversion
        _if(size != 0)
        ._then([&] {
            auto tmp = stack(vlen * sizeof(float));
            tmp.clear();

            auto tail_size = var<size_t>();

            tail_size = size * _src_size;
            copy<uint8_t>(tmp.pointer(), src, tail_size);

            _convert_vec(*this, tmp.pointer(), tmp.pointer());

            tail_size = size * _dst_size;
            copy<uint8_t>(dst, tmp.pointer(), tail_size);
        });

        postamble();
    }

public:
    typedef struct {
        const void* src;
        void* out;
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    typedef void (*convert_vec_t)(jit_generator &,
                                  const RegExp &,
                                  const RegExp &);
    typedef void (*convert_vec_prepare_t)(jit_generator&);

    jit_convert_array(convert_vec_prepare_t convert_vec_prepare,
                      convert_vec_t convert_vec,
                      size_t src_size,
                      size_t dst_size)
        : jit_kernel(jit_name()),
          _convert_vec_prepare(convert_vec_prepare),
          _convert_vec(convert_vec),
          _src_size(src_size),
          _dst_size(dst_size) {}

    template<typename src_t, typename dst_t, bool clamp = false>
    static fn_t get() {
        if (mayiuse(cpu_isa_t::avx2)
            && dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tF16C)) {
            static jit_convert_array converter(convert_vec_prepare<src_t, dst_t, clamp>, convert_vec<src_t, dst_t, clamp>, sizeof(src_t), sizeof(dst_t));
            auto & generator = static_cast<jit_generator&>(converter);
            generator.create_kernel();
            return (fn_t)generator.jit_ker();
        }
        return nullptr;
    }

private:
    convert_vec_prepare_t _convert_vec_prepare;
    convert_vec_t _convert_vec;
    size_t _src_size;
    size_t _dst_size;
};

template <typename TI, typename TO, bool clamp = false>
void jit_convert(const TI* arg, TO* out, size_t count) {
    using jit_impl = jit_convert_array;
    static auto converter = jit_impl::get<TI, TO, clamp>();

    if (converter) {
        typename jit_impl::args_t args = { arg, out, count };
        converter(&args);
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<TO>(arg[i]);
        }
    }
}

template <>
void jit_convert<float, ov::float16, true>(const float* arg, ov::float16* out, size_t count) {
    using jit_impl = jit_convert_array;
    static auto converter = jit_impl::get<float, ov::float16, true>();

    if (converter) {
        typename jit_impl::args_t args = {arg, out, count};
        converter(&args);
    } else {
        for (size_t i = 0; i < count; ++i) {
            if (arg[i] > std::numeric_limits<ov::float16>::max()) {
                out[i] = std::numeric_limits<ov::float16>::max();
            } else if (arg[i] < std::numeric_limits<ov::float16>::lowest()) {
                out[i] = std::numeric_limits<ov::float16>::lowest();
            } else {
                out[i] = static_cast<ov::float16>(arg[i]);
            }
        }
    }
}

#endif

template <ov::element::Type_t p>
struct PrecisionInfo {
    using value_type = typename element_type_traits<p>::value_type;
};

template <>
struct PrecisionInfo<ov::element::bf16> {
    using value_type = ov::intel_cpu::bfloat16_t;
};

template <>
struct PrecisionInfo<ov::element::f16> {
    using value_type = ov::float16;
};

template <>
struct PrecisionInfo<ov::element::boolean> {
    using value_type = uint8_t;
};

template<typename T,
         typename U = typename std::conditional<
                        std::is_same<ov::float16, T>::value
                        || std::is_same<ov::intel_cpu::bfloat16_t, T>::value,
                        float, T>::type>
struct Range {
    const std::tuple<U, U> & fit(const ov::element::Type & prec);

private:
    std::tuple<U, U> _range {
        std::numeric_limits<T>::lowest(),
        std::numeric_limits<T>::max()
    };
};

template<typename T, typename U>
const std::tuple<U, U> & Range<T, U>::fit(const ov::element::Type & prec) {
    if (prec.is_real()) {
        double lbound, ubound;
        switch (prec) {
            case ov::element::bf16:
                lbound = static_cast<double>(std::numeric_limits<ov::intel_cpu::bfloat16_t>::lowest());
                ubound = static_cast<double>(std::numeric_limits<ov::intel_cpu::bfloat16_t>::max());
                break;
            case ov::element::f16:
                lbound = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
                ubound = static_cast<double>(std::numeric_limits<ov::float16>::max());
                break;
            case ov::element::f32:
                lbound = static_cast<double>(std::numeric_limits<float>::lowest());
                ubound = static_cast<double>(std::numeric_limits<float>::max());
                break;
            case ov::element::f64:
                lbound = std::numeric_limits<double>::lowest();
                ubound = std::numeric_limits<double>::max();
                break;
            default:
                OPENVINO_THROW("Unsupported precision");
        }
        // If U is integral, its range always less than float, so not need update _range
        // Else it will be overflow, for example static_cast double to int64_t:
        //         int64_t ubound = 9223372036854775807
        //         double  dd_ubound = static_cast<double>(ubbound)
        //         static_cast<int64_t>(dd_ubound) will return -9223372036854775808
        if (!std::is_integral<U>::value) {
                std::get<0>(_range) = static_cast<U>(std::max(static_cast<double>(std::get<0>(_range)), lbound));
                std::get<1>(_range) = static_cast<U>(std::min(static_cast<double>(std::get<1>(_range)), ubound));
        }
    } else {
        int64_t lbound;
        uint64_t ubound;
        switch (prec) {
            case ov::element::boolean:
            case ov::element::u8:
                lbound = static_cast<int64_t>(std::numeric_limits<uint8_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint8_t>::max());
                break;
            case ov::element::i8:
                lbound = static_cast<int64_t>(std::numeric_limits<int8_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int8_t>::max());
                break;
            case ov::element::u16:
                lbound = static_cast<int64_t>(std::numeric_limits<uint16_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
                break;
            case ov::element::i16:
                lbound = static_cast<int64_t>(std::numeric_limits<int16_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int16_t>::max());
                break;
            case ov::element::u32:
                lbound = static_cast<int64_t>(std::numeric_limits<uint32_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
                break;
            case ov::element::i32:
                lbound = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
                break;
            case ov::element::u64:
                lbound = static_cast<int64_t>(std::numeric_limits<uint64_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint64_t>::max());
                break;
            case ov::element::i64:
                lbound = static_cast<int64_t>(std::numeric_limits<int64_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
                break;
            default:
                OPENVINO_THROW("Unsupported precision");
        }
        using ltype = typename std::conditional<
                            std::is_floating_point<U>::value,
                            double, int64_t>::type;
        using utype = typename std::conditional<
                            std::is_floating_point<U>::value,
                            double, uint64_t>::type;
        std::get<0>(_range) = static_cast<U>(std::max(static_cast<ltype>(std::get<0>(_range)), static_cast<ltype>(lbound)));
        std::get<1>(_range) = static_cast<U>(std::min(static_cast<utype>(std::get<1>(_range)), static_cast<utype>(ubound)));
    }
    return _range;
}

struct ConvertContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    ov::element::Type interimPrc;
    ov::element::Type dstPrc;
    bool converted;
    bool clamp;

    template<typename T>
    std::tuple<T, T> range() const {
        Range<T> r;
        r.fit(interimPrc);
        return r.fit(dstPrc);
    }
};

template<typename T>
struct ConvertPrecision;

template<typename src_t, typename dst_t>
struct ConvertPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);
        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t>();
        auto clamp = ctx.clamp;

        if (std::is_integral<src_t>::value
            || ctx.interimPrc.is_real()
            || std::is_integral<dst_t>::value) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(clamp ? std::max(std::min(src[i], ubound), lbound) : src[i]);
            });
        } else {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(std::trunc(clamp ? std::max(std::min(src[i], ubound), lbound) : src[i]));
            });
        }

        ctx.converted = true;
    }
};

template<>
struct ConvertPrecision<std::tuple<float, ov::intel_cpu::bfloat16_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const float *>(ctx.srcPtr);
        auto dst = static_cast<ov::intel_cpu::bfloat16_t *>(ctx.dstPtr);
        auto clamp = ctx.clamp;

        if (ctx.interimPrc.is_real() || !clamp) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<ov::intel_cpu::bfloat16_t>(src[i]);
            });
        } else {
            float lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<float>();
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<ov::intel_cpu::bfloat16_t>(std::trunc(std::max(std::min(src[i], ubound), lbound)));
            });
        }

        ctx.converted = true;
    }
};

template<>
struct ConvertPrecision<std::tuple<ov::intel_cpu::bfloat16_t, float>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::intel_cpu::bfloat16_t *>(ctx.srcPtr);
        auto dst = static_cast<float *>(ctx.dstPtr);
        auto clamp = ctx.clamp;

        if (ctx.interimPrc.is_real()) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<float>(src[i]);
            });
        } else {
            float lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<ov::intel_cpu::bfloat16_t>();
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = clamp ? std::trunc(std::max(std::min(static_cast<float>(src[i]), ubound), lbound))
                               : static_cast<float>(src[i]);
            });
        }

        ctx.converted = true;
    }
};

#if defined(OPENVINO_ARCH_X86_64)
template<typename src_t>
struct ConvertPrecision<std::tuple<src_t, ov::float16>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<ov::float16 *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];
        auto clamp = ctx.clamp;

        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t>();

        if (std::is_integral<src_t>::value
            || ctx.interimPrc.is_real()) {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j)         // src_t -> fp32
                    tmp[j] = static_cast<float>(clamp ? std::max(std::min(src[offset + j], ubound), lbound)
                                                      : src[offset + j]);
                jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j)         // src_t -> fp32
                    tmp[j] = static_cast<float>(
                        std::trunc(clamp ? std::max(std::min(src[offset + j], ubound), lbound) : src[offset + j]));
                jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};

template<typename dst_t>
struct ConvertPrecision<std::tuple<ov::float16, dst_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::float16 *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];
        auto clamp = ctx.clamp;

        float lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<ov::float16>();

        if (ctx.interimPrc.is_real()
            || std::is_integral<dst_t>::value) {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)         // fp32 -> dst_t
                    dst[offset + j] = static_cast<dst_t>(clamp ? std::max(std::min(tmp[j], ubound), lbound) : tmp[j]);
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)         // fp32 -> dst_t
                    dst[offset + j] =
                        static_cast<dst_t>(clamp ? std::trunc(std::max(std::min(tmp[j], ubound), lbound)) : tmp[j]);
            });
        }

        ctx.converted = true;
    }
};

template<>
struct ConvertPrecision<std::tuple<ov::float16, ov::float16>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::float16 *>(ctx.srcPtr);
        auto dst = static_cast<ov::float16 *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];
        auto clamp = ctx.clamp;

        float lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<ov::float16>();

        if (ctx.interimPrc.is_real()) {
            cpu_memcpy(dst, src, ctx.size * sizeof(ov::float16));
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)         // truncate fp32
                    tmp[j] = clamp ? std::trunc(std::max(std::min(tmp[j], ubound), lbound)) : tmp[j];
                jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};

template <>
struct ConvertPrecision<std::tuple<float, ov::float16>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const float*>(ctx.srcPtr);
        auto dst = static_cast<ov::float16*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        auto clamp = ctx.clamp;

        if (!clamp) {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, dst + offset, current_batch_size);  // fp32 -> fp16
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert<float, ov::float16, true>(src + offset, dst + offset, current_batch_size);  // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};

template <>
struct ConvertPrecision<std::tuple<uint8_t, ov::float16>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const uint8_t*>(ctx.srcPtr);
        auto dst = static_cast<ov::float16*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        auto clamp = ctx.clamp;

        if (!clamp) {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, dst + offset, current_batch_size);  // uint8 -> fp16
            });

        } else {
            typedef float batch_type[batch];
            uint8_t lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<uint8_t>();
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j)  // uint8 -> fp32
                    tmp[j] = static_cast<float>(std::max(std::min(src[offset + j], ubound), lbound));
                jit_convert(tmp, dst + offset, current_batch_size);  // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};

template <>
struct ConvertPrecision<std::tuple<float, int8_t>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const float*>(ctx.srcPtr);
        auto dst = static_cast<int8_t*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        auto clamp = ctx.clamp;

        if (!clamp) {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, dst + offset, current_batch_size);  // fp32 -> int8
            });
        } else {
            float lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<float>();
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<int8_t>(std::max(std::min(src[i], ubound), lbound));
            });
        }

        ctx.converted = true;
    }
};

template <>
struct ConvertPrecision<std::tuple<ov::float16, int8_t>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const ov::float16*>(ctx.srcPtr);
        auto dst = static_cast<int8_t*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);

        auto clamp = ctx.clamp;

        if (!clamp) {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, dst + offset, current_batch_size);  // float16 -> int8_t
            });
        } else {
            typedef float batch_type[batch];
            float lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<ov::float16>();
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);  // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)      // fp32 -> int8_t
                    dst[offset + j] = static_cast<int8_t>(std::max(std::min(tmp[j], ubound), lbound));
            });
        }

        ctx.converted = true;
    }
};

#endif

}   // namespace

#define INTEL_CPU_CVT(ST, DT)                            \
    OV_CASE2(ov::element::ST,                            \
             ov::element::DT,                            \
             PrecisionInfo<ov::element::ST>::value_type, \
             PrecisionInfo<ov::element::DT>::value_type)

#define INTEL_CPU_CVT_LIST                                                                                             \
    INTEL_CPU_CVT(u8, i8), INTEL_CPU_CVT(u8, u16), INTEL_CPU_CVT(u8, i16), INTEL_CPU_CVT(u8, u32),                     \
        INTEL_CPU_CVT(u8, i32), INTEL_CPU_CVT(u8, u64), INTEL_CPU_CVT(u8, i64), INTEL_CPU_CVT(u8, f32),                \
        INTEL_CPU_CVT(u8, f16), INTEL_CPU_CVT(u8, bf16), INTEL_CPU_CVT(u8, f64), INTEL_CPU_CVT(u8, boolean),           \
        INTEL_CPU_CVT(i8, u8), INTEL_CPU_CVT(i8, u16), INTEL_CPU_CVT(i8, i16), INTEL_CPU_CVT(i8, u32),                 \
        INTEL_CPU_CVT(i8, i32), INTEL_CPU_CVT(i8, u64), INTEL_CPU_CVT(i8, i64), INTEL_CPU_CVT(i8, f32),                \
        INTEL_CPU_CVT(i8, f16), INTEL_CPU_CVT(i8, bf16), INTEL_CPU_CVT(i8, f64), INTEL_CPU_CVT(i8, boolean),           \
        INTEL_CPU_CVT(u16, u8), INTEL_CPU_CVT(u16, i8), INTEL_CPU_CVT(u16, i16), INTEL_CPU_CVT(u16, u32),              \
        INTEL_CPU_CVT(u16, i32), INTEL_CPU_CVT(u16, u64), INTEL_CPU_CVT(u16, i64), INTEL_CPU_CVT(u16, f32),            \
        INTEL_CPU_CVT(u16, f16), INTEL_CPU_CVT(u16, bf16), INTEL_CPU_CVT(u16, f64), INTEL_CPU_CVT(u16, boolean),       \
        INTEL_CPU_CVT(i16, u8), INTEL_CPU_CVT(i16, i8), INTEL_CPU_CVT(i16, u16), INTEL_CPU_CVT(i16, u32),              \
        INTEL_CPU_CVT(i16, i32), INTEL_CPU_CVT(i16, u64), INTEL_CPU_CVT(i16, i64), INTEL_CPU_CVT(i16, f32),            \
        INTEL_CPU_CVT(i16, f16), INTEL_CPU_CVT(i16, bf16), INTEL_CPU_CVT(i16, f64), INTEL_CPU_CVT(i16, boolean),       \
        INTEL_CPU_CVT(u32, u8), INTEL_CPU_CVT(u32, i8), INTEL_CPU_CVT(u32, u16), INTEL_CPU_CVT(u32, i16),              \
        INTEL_CPU_CVT(u32, i32), INTEL_CPU_CVT(u32, u64), INTEL_CPU_CVT(u32, i64), INTEL_CPU_CVT(u32, f32),            \
        INTEL_CPU_CVT(u32, f16), INTEL_CPU_CVT(u32, bf16), INTEL_CPU_CVT(u32, f64), INTEL_CPU_CVT(u32, boolean),       \
        INTEL_CPU_CVT(i32, u8), INTEL_CPU_CVT(i32, i8), INTEL_CPU_CVT(i32, u16), INTEL_CPU_CVT(i32, i16),              \
        INTEL_CPU_CVT(i32, u32), INTEL_CPU_CVT(i32, u64), INTEL_CPU_CVT(i32, i64), INTEL_CPU_CVT(i32, f32),            \
        INTEL_CPU_CVT(i32, f16), INTEL_CPU_CVT(i32, bf16), INTEL_CPU_CVT(i32, f64), INTEL_CPU_CVT(i32, boolean),       \
        INTEL_CPU_CVT(u64, u8), INTEL_CPU_CVT(u64, i8), INTEL_CPU_CVT(u64, u16), INTEL_CPU_CVT(u64, i16),              \
        INTEL_CPU_CVT(u64, u32), INTEL_CPU_CVT(u64, i32), INTEL_CPU_CVT(u64, i64), INTEL_CPU_CVT(u64, f32),            \
        INTEL_CPU_CVT(u64, f16), INTEL_CPU_CVT(u64, bf16), INTEL_CPU_CVT(u64, f64), INTEL_CPU_CVT(u64, boolean),       \
        INTEL_CPU_CVT(i64, u8), INTEL_CPU_CVT(i64, i8), INTEL_CPU_CVT(i64, u16), INTEL_CPU_CVT(i64, i16),              \
        INTEL_CPU_CVT(i64, u32), INTEL_CPU_CVT(i64, i32), INTEL_CPU_CVT(i64, u64), INTEL_CPU_CVT(i64, f32),            \
        INTEL_CPU_CVT(i64, f16), INTEL_CPU_CVT(i64, bf16), INTEL_CPU_CVT(i64, f64), INTEL_CPU_CVT(i64, boolean),       \
        INTEL_CPU_CVT(f32, u8), INTEL_CPU_CVT(f32, i8), INTEL_CPU_CVT(f32, u16), INTEL_CPU_CVT(f32, i16),              \
        INTEL_CPU_CVT(f32, u32), INTEL_CPU_CVT(f32, i32), INTEL_CPU_CVT(f32, u64), INTEL_CPU_CVT(f32, i64),            \
        INTEL_CPU_CVT(f32, f16), INTEL_CPU_CVT(f32, bf16), INTEL_CPU_CVT(f32, f64), INTEL_CPU_CVT(f32, boolean),       \
        INTEL_CPU_CVT(f16, u8), INTEL_CPU_CVT(f16, i8), INTEL_CPU_CVT(f16, u16), INTEL_CPU_CVT(f16, i16),              \
        INTEL_CPU_CVT(f16, u32), INTEL_CPU_CVT(f16, i32), INTEL_CPU_CVT(f16, u64), INTEL_CPU_CVT(f16, i64),            \
        INTEL_CPU_CVT(f16, f32), INTEL_CPU_CVT(f16, bf16), INTEL_CPU_CVT(f16, f64), INTEL_CPU_CVT(f16, boolean),       \
        INTEL_CPU_CVT(bf16, u8), INTEL_CPU_CVT(bf16, i8), INTEL_CPU_CVT(bf16, u16), INTEL_CPU_CVT(bf16, i16),          \
        INTEL_CPU_CVT(bf16, u32), INTEL_CPU_CVT(bf16, i32), INTEL_CPU_CVT(bf16, u64), INTEL_CPU_CVT(bf16, i64),        \
        INTEL_CPU_CVT(bf16, f32), INTEL_CPU_CVT(bf16, f16), INTEL_CPU_CVT(bf16, f64), INTEL_CPU_CVT(bf16, boolean),    \
        INTEL_CPU_CVT(f64, u8), INTEL_CPU_CVT(f64, i8), INTEL_CPU_CVT(f64, u16), INTEL_CPU_CVT(f64, i16),              \
        INTEL_CPU_CVT(f64, u32), INTEL_CPU_CVT(f64, i32), INTEL_CPU_CVT(f64, u64), INTEL_CPU_CVT(f64, i64),            \
        INTEL_CPU_CVT(f64, f32), INTEL_CPU_CVT(f64, f16), INTEL_CPU_CVT(f64, bf16), INTEL_CPU_CVT(f64, boolean),       \
        INTEL_CPU_CVT(boolean, u8), INTEL_CPU_CVT(boolean, i8), INTEL_CPU_CVT(boolean, u16),                           \
        INTEL_CPU_CVT(boolean, i16), INTEL_CPU_CVT(boolean, u32), INTEL_CPU_CVT(boolean, i32),                         \
        INTEL_CPU_CVT(boolean, u64), INTEL_CPU_CVT(boolean, i64), INTEL_CPU_CVT(boolean, f32),                         \
        INTEL_CPU_CVT(boolean, f16), INTEL_CPU_CVT(boolean, bf16), INTEL_CPU_CVT(boolean, f64), INTEL_CPU_CVT(u8, u8), \
        INTEL_CPU_CVT(i8, i8), INTEL_CPU_CVT(u16, u16), INTEL_CPU_CVT(i16, i16), INTEL_CPU_CVT(u32, u32),              \
        INTEL_CPU_CVT(i32, i32), INTEL_CPU_CVT(u64, u64), INTEL_CPU_CVT(i64, i64), INTEL_CPU_CVT(f32, f32),            \
        INTEL_CPU_CVT(f16, f16), INTEL_CPU_CVT(bf16, bf16), INTEL_CPU_CVT(f64, f64), INTEL_CPU_CVT(boolean, boolean)

#define INTEL_CPU_CVT_FROM_BIN(DT) OV_CASE(ov::element::DT, PrecisionInfo<ov::element::DT>::value_type)

#define INTEL_CPU_CVT_FROM_BIN_LIST                                                            \
    INTEL_CPU_CVT_FROM_BIN(f32), INTEL_CPU_CVT_FROM_BIN(f16), INTEL_CPU_CVT_FROM_BIN(bf16),    \
        INTEL_CPU_CVT_FROM_BIN(f64), INTEL_CPU_CVT_FROM_BIN(i16), INTEL_CPU_CVT_FROM_BIN(u8),  \
        INTEL_CPU_CVT_FROM_BIN(i8), INTEL_CPU_CVT_FROM_BIN(u16), INTEL_CPU_CVT_FROM_BIN(i32),  \
        INTEL_CPU_CVT_FROM_BIN(u32), INTEL_CPU_CVT_FROM_BIN(i64), INTEL_CPU_CVT_FROM_BIN(u64), \
        INTEL_CPU_CVT_FROM_BIN(boolean)

struct ConvertFromBinContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    bool converted;
};

template<typename T>
struct ConvertFromBinPrecision {
    void operator()(ConvertFromBinContext &ctx) {
        auto src = static_cast<const uint8_t *>(ctx.srcPtr);
        auto dst = static_cast<T *>(ctx.dstPtr);
        const size_t nBits = 8;
        const size_t nBytes = rnd_up(ctx.size, nBits);
        parallel_for(nBytes, [&](size_t byteIndex) {
            auto currentBitNum = std::min(nBits, ctx.size - byteIndex * nBits);
            for (size_t bitIndex = 0; bitIndex < currentBitNum; ++bitIndex) {
                dst[byteIndex * nBits + bitIndex] = static_cast<T>((src[byteIndex] & (1 << bitIndex)) >> bitIndex);
            }
        });
        ctx.converted = true;
    }
};


void cpu_convert(const void *srcPtr, void *dstPtr, ov::element::Type srcPrc, ov::element::Type dstPrc, const size_t size) {
    cpu_convert(srcPtr, dstPtr, srcPrc, dstPrc, dstPrc, size);
}

void cpu_convert(const void *srcPtr,
                 void *dstPtr,
                 ov::element::Type srcPrc,
                 ov::element::Type interimPrc,
                 ov::element::Type dstPrc,
                 const size_t size) {
    if (size == 0) {
        return;
    }
    if (srcPtr == nullptr || dstPtr == nullptr)
        OPENVINO_THROW("cpu_convert has null data pointer");

    if (srcPrc == dstPrc && srcPrc == interimPrc) {
        const size_t L2_cache_size = dnnl::utils::get_cache_size(2, true);
        const size_t totalSize = size * dstPrc.size();
        if (srcPrc == element::string) {
            auto str_src = reinterpret_cast<const StringMemory::OvString *>(srcPtr);
            auto str_dst = reinterpret_cast<StringMemory::OvString *>(dstPtr);
            std::copy(str_src, str_src + size, str_dst);
        } else if (totalSize >= L2_cache_size) {
            auto src = static_cast<const uint8_t *>(srcPtr);
            auto dst = static_cast<uint8_t *>(dstPtr);
            parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
                size_t start = 0, end = 0;
                splitter(totalSize, nthr, ithr, start, end);
                cpu_memcpy(dst + start, src + start, end - start);
            });
        } else {
            cpu_memcpy(dstPtr, srcPtr, size * dstPrc.size());
        }
    } else if (srcPrc == ov::element::u1) {
        if (srcPrc.bitwidth() != 1)
            OPENVINO_THROW("cpu_convert can't convert from: ",
                           srcPrc,
                           " <bitsSize == ",
                           srcPrc.bitwidth(),
                           "> precision to: ",
                           dstPrc,
                           ". Not implemented.");
        ConvertFromBinContext ctx {
                srcPtr,
                dstPtr,
                size,
                false
        };
        OV_SWITCH(intel_cpu, ConvertFromBinPrecision, ctx, dstPrc, INTEL_CPU_CVT_FROM_BIN_LIST);
        if (!ctx.converted)
            OPENVINO_THROW("cpu_convert can't convert from: ",
                           srcPrc,
                           " <bitsSize == ",
                           srcPrc.bitwidth(),
                           "> precision to: ",
                           dstPrc);
    } else {
        ConvertContext ctx {
            srcPtr,
            dstPtr,
            size,
            interimPrc,
            dstPrc,
            false,
            false
        };
        OV_SWITCH(intel_cpu, ConvertPrecision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_LIST);
        if (!ctx.converted)
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
    }
}

#undef INTEL_CPU_CVT
#undef INTEL_CPU_CVT_LIST

}   // namespace intel_cpu
}   // namespace ov
