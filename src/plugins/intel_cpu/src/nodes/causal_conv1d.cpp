// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/reference.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "ov_ops/causal_conv1d.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu/x64/jit_generator.hpp"
#endif

namespace ov::intel_cpu::node {

CausalConv1D::CausalConv1D(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool CausalConv1D::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = ov::as_type_ptr<const ov::op::internal::CausalConv1D>(op);
        if (!node) {
            errorMessage = "Only CausalConv1D operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void CausalConv1D::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    const auto& inShapes = inputShapes;
    const auto& outShapes = outputShapes;

    for (size_t i = 0; i < inShapes.size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, getOriginalInputPrecisionAtPort(i), inShapes[i], false, -1);
    }

    for (size_t i = 0; i < outShapes.size(); ++i) {
        outPortConfigs.emplace_back(LayoutType::ncsp, getOriginalOutputPrecisionAtPort(i), outShapes[i], false, -1);
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

static size_t get_weight_k(const ov::intel_cpu::PlainTensor& t_weight) {
    const auto rank = t_weight.m_rank;
    if (rank >= 1) {
        return t_weight.size(rank - 1);
    }
    return 0;
}

static const float* get_weight_ptr(const PlainTensor& t_weight, size_t c) {
    if (t_weight.m_rank == 4) {
        return t_weight.ptr<float>(c, 0, 0, 0);
    }
    return t_weight.ptr<float>(c, 0, 0);
}

#if defined(OPENVINO_ARCH_X86_64)
namespace {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace Xbyak::util;

struct jit_dot_call_args {
    const float* src;
    const float* weights;
    size_t len;
    float* dst;
};

#    define GET_OFF(field) offsetof(jit_dot_call_args, field)

template <cpu_isa_t isa>
struct jit_dot_kernel : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_dot_kernel)

    static constexpr size_t vec_size = cpu_isa_traits_t<isa>::vlen / sizeof(float);

    jit_dot_kernel() : jit_generator_t(jit_name()) {}

    void create_ker() {
        jit_generator_t::create_kernel();
        ker_ = reinterpret_cast<ker_t>(jit_ker());
    }

    void operator()(const jit_dot_call_args* args) const {
        ker_(args);
    }

private:
    using Vmm = typename dnnl::impl::utils::
        conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[abi_param1 + GET_OFF(src)]);
        mov(reg_wei, ptr[abi_param1 + GET_OFF(weights)]);
        mov(reg_len, ptr[abi_param1 + GET_OFF(len)]);
        mov(reg_dst, ptr[abi_param1 + GET_OFF(dst)]);

        mov(reg_iter, reg_len);
        if (vec_size == 16) {
            shr(reg_iter, 4);
        } else if (vec_size == 8) {
            shr(reg_iter, 3);
        }

        uni_vpxor(vmm_acc, vmm_acc, vmm_acc);

        Xbyak::Label loop, exit;
        L(loop);
        cmp(reg_iter, 0);
        je(exit, T_NEAR);
        uni_vmovups(vmm_src, ptr[reg_src]);
        uni_vmovups(vmm_wei, ptr[reg_wei]);
        vfmadd231ps(vmm_acc, vmm_src, vmm_wei);
        add(reg_src, vec_size * sizeof(float));
        add(reg_wei, vec_size * sizeof(float));
        dec(reg_iter);
        jmp(loop);
        L(exit);

        uni_vmovups(ptr[reg_dst], vmm_acc);

        this->postamble();
    }

    using ker_t = void (*)(const jit_dot_call_args*);
    ker_t ker_ = nullptr;

    const Xbyak::Reg64 reg_src = r8;
    const Xbyak::Reg64 reg_wei = r9;
    const Xbyak::Reg64 reg_len = r10;
    const Xbyak::Reg64 reg_dst = r11;
    const Xbyak::Reg64 reg_iter = r12;

    const Vmm vmm_acc = Vmm(0);
    const Vmm vmm_src = Vmm(1);
    const Vmm vmm_wei = Vmm(2);
};

#    undef GET_OFF

template <cpu_isa_t isa>
static std::shared_ptr<jit_dot_kernel<isa>> get_dot_kernel() {
    static std::shared_ptr<jit_dot_kernel<isa>> ker;
    if (!ker) {
        ker = std::make_shared<jit_dot_kernel<isa>>();
        ker->create_ker();
    }
    return ker;
}

}  // namespace
#endif

using ActivationType = ov::op::internal::CausalConv1D::ActivationType;

static size_t tensor_size(const PlainTensor& t) {
    size_t sz = 1;
    for (size_t i = 0; i < t.m_rank; ++i) {
        sz *= t.m_dims[i];
    }
    return sz;
}

static ActivationType read_activation_type(const PlainTensor& t_act) {
    const auto act_size = tensor_size(t_act);
    if (act_size == 0) {
        return ActivationType::None;
    }
    if (t_act.get_precision() == ov::element::i32) {
        const int32_t v = t_act.ptr<int32_t>()[0];
        return static_cast<ActivationType>(static_cast<int64_t>(v));
    }
    if (t_act.get_precision() == ov::element::i64) {
        const int64_t v = t_act.ptr<int64_t>()[0];
        return static_cast<ActivationType>(v);
    }
    return ActivationType::None;
}

static float apply_activation(float x, ActivationType activation) {
    if (activation == ActivationType::SiLU) {
        return x / (1.0f + std::exp(-x));
    }
    return x;
}

static void causal_conv1d_reference_impl(const PlainTensor& t_cache,
                                         const PlainTensor& t_hidden,
                                         const PlainTensor& t_weight,
                                         const PlainTensor* t_pos,
                                         bool has_cache_position,
                                         ActivationType activation,
                                         PlainTensor& t_out,
                                         PlainTensor& t_state) {
    const size_t B = t_hidden.size(0);
    const size_t C = t_hidden.size(1);
    const size_t seq_len = t_hidden.size(2);
    const size_t L = t_cache.size(2);

    if (seq_len == 1) {
        // decoding path
        auto read_pos0 = [&]() -> int64_t {
            if (!has_cache_position || !t_pos) {
                return -1;
            }
            if (t_pos->get_precision() == ov::element::i64) {
                return static_cast<int64_t>(t_pos->ptr<int64_t>()[0]);
            }
            return static_cast<int64_t>(t_pos->ptr<int32_t>()[0]);
        };
        auto clamp_pos = [&](int64_t p) -> size_t {
            if (!has_cache_position || p < 0) {
                return (L > 0) ? (L - 1) : 0;
            }
            if (p < 0) {
                p = 0;
            }
            const int64_t max_p = static_cast<int64_t>(L) - 1;
            if (p > max_p) {
                p = max_p;
            }
            return static_cast<size_t>(p);
        };
        const size_t pos = clamp_pos(read_pos0());
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                // roll cache by -1 and write at cache_position
                for (size_t k = 0; k + 1 < L; ++k) {
                    t_state.at<float>({b, c, k}) = t_cache.at<float>({b, c, k + 1});
                }
                t_state.at<float>({b, c, pos}) = t_hidden.at<float>({b, c, 0});

                float sum = 0.0f;
                const auto* weight = get_weight_ptr(t_weight, c);
                for (size_t k = 0; k < L; ++k) {
                    sum += t_state.at<float>({b, c, k}) * weight[k];
                }
                t_out.at<float>({b, c, 0}) = apply_activation(sum, activation);
            }
        }
        return;
    }

    // prefill path
    const size_t copy_len = std::min(seq_len, L);
    const size_t pad = L - copy_len;
    const size_t start = (seq_len > L) ? (seq_len - L) : 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t k = 0; k < pad; ++k) {
                t_state.at<float>({b, c, k}) = 0.0f;
            }
            for (size_t t = 0; t < copy_len; ++t) {
                t_state.at<float>({b, c, pad + t}) = t_hidden.at<float>({b, c, start + t});
            }

            for (size_t t = 0; t < seq_len; ++t) {
                float sum = 0.0f;
                const auto* weight = get_weight_ptr(t_weight, c);
                for (size_t k = 0; k < L; ++k) {
                    // Conv1d uses cross-correlation: idx = t + k - (L - 1)，represent left padding conv
                    const auto idx = static_cast<int64_t>(t) - static_cast<int64_t>(L) + 1 + static_cast<int64_t>(k);
                    if (idx >= 0) {
                        sum += t_hidden.at<float>({b, c, static_cast<size_t>(idx)}) * weight[k];
                    }
                }
                t_out.at<float>({b, c, t}) = apply_activation(sum, activation);
            }
        }
    }
}

[[maybe_unused]] static void causal_conv1d_optimized_impl(const PlainTensor& t_cache,
                                                          const PlainTensor& t_hidden,
                                                          const PlainTensor& t_weight,
                                                          const PlainTensor* t_pos,
                                                          bool has_cache_position,
                                                          ActivationType activation,
                                                          PlainTensor& t_out,
                                                          PlainTensor& t_state) {
#if !defined(OPENVINO_ARCH_X86_64)
    causal_conv1d_reference_impl(t_cache, t_hidden, t_weight, t_pos, has_cache_position, activation, t_out, t_state);
    return;
#else
    using namespace dnnl::impl::cpu::x64;

    std::shared_ptr<jit_dot_kernel<cpu::x64::avx512_core>> ker_avx512;
    std::shared_ptr<jit_dot_kernel<cpu::x64::avx2>> ker_avx2;
    size_t vec_size = 0;

    if (mayiuse(cpu::x64::avx512_core)) {
        ker_avx512 = get_dot_kernel<cpu::x64::avx512_core>();
        vec_size = jit_dot_kernel<cpu::x64::avx512_core>::vec_size;
    } else if (mayiuse(cpu::x64::avx2)) {
        ker_avx2 = get_dot_kernel<cpu::x64::avx2>();
        vec_size = jit_dot_kernel<cpu::x64::avx2>::vec_size;
    } else {
        causal_conv1d_reference_impl(t_cache,
                                     t_hidden,
                                     t_weight,
                                     t_pos,
                                     has_cache_position,
                                     activation,
                                     t_out,
                                     t_state);
        return;
    }

    auto dot_product = [&](const float* src, const float* weights, size_t len) -> float {
        if (len == 0) {
            return 0.0f;
        }
        if (vec_size == 0 || len < vec_size) {
            float sum = 0.0f;
            for (size_t i = 0; i < len; ++i) {
                sum += src[i] * weights[i];
            }
            return sum;
        }

        const size_t len_vec = (len / vec_size) * vec_size;
        alignas(64) float acc_buf[16] = {0.0f};
        jit_dot_call_args args{src, weights, len_vec, acc_buf};
        if (ker_avx512) {
            (*ker_avx512)(&args);
        } else {
            (*ker_avx2)(&args);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < vec_size; ++i) {
            sum += acc_buf[i];
        }
        for (size_t i = len_vec; i < len; ++i) {
            sum += src[i] * weights[i];
        }
        return sum;
    };

    const size_t B = t_hidden.size(0);
    const size_t C = t_hidden.size(1);
    const size_t seq_len = t_hidden.size(2);
    const size_t L = t_cache.size(2);

    if (seq_len == 1) {
        auto read_pos0 = [&]() -> int64_t {
            if (!has_cache_position || !t_pos) {
                return -1;
            }
            if (t_pos->get_precision() == ov::element::i64) {
                return static_cast<int64_t>(t_pos->ptr<int64_t>()[0]);
            }
            return static_cast<int64_t>(t_pos->ptr<int32_t>()[0]);
        };
        auto clamp_pos = [&](int64_t p) -> size_t {
            if (!has_cache_position || p < 0) {
                return (L > 0) ? (L - 1) : 0;
            }
            if (p < 0) {
                p = 0;
            }
            const int64_t max_p = static_cast<int64_t>(L) - 1;
            if (p > max_p) {
                p = max_p;
            }
            return static_cast<size_t>(p);
        };
        const size_t pos = clamp_pos(read_pos0());

        if (pos != L - 1) {
            causal_conv1d_reference_impl(t_cache,
                                         t_hidden,
                                         t_weight,
                                         t_pos,
                                         has_cache_position,
                                         activation,
                                         t_out,
                                         t_state);
            return;
        }

        parallel_for(C, [&](size_t c) {
            for (size_t b = 0; b < B; ++b) {
                auto* state = t_state.ptr<float>(b, c, 0);
                const auto* cache = t_cache.ptr<float>(b, c, 0);
                if (L <= 4) {
                    if (L > 1) {
                        state[0] = cache[1];
                    }
                    if (L > 2) {
                        state[1] = cache[2];
                    }
                    if (L > 3) {
                        state[2] = cache[3];
                    }
                } else {
                    std::memcpy(state, cache + 1, (L - 1) * sizeof(float));
                }
                state[L - 1] = *t_hidden.ptr<float>(b, c, 0);
            }
        });

        parallel_for(C, [&](size_t c) {
            const auto* weight = get_weight_ptr(t_weight, c);
            for (size_t b = 0; b < B; ++b) {
                const auto* state = t_state.ptr<float>(b, c, 0);
                *t_out.ptr<float>(b, c, 0) = apply_activation(dot_product(state, weight, L), activation);
            }
        });
        return;
    }

    if (seq_len < L) {
        causal_conv1d_reference_impl(t_cache,
                                     t_hidden,
                                     t_weight,
                                     t_pos,
                                     has_cache_position,
                                     activation,
                                     t_out,
                                     t_state);
        return;
    }

    const size_t copy_len = std::min(seq_len, L);
    const size_t pad = L - copy_len;
    const size_t start = (seq_len > L) ? (seq_len - L) : 0;

    parallel_for(C, [&](size_t c) {
        for (size_t b = 0; b < B; ++b) {
            auto* state = t_state.ptr<float>(b, c, 0);
            if (pad > 0) {
                std::memset(state, 0, pad * sizeof(float));
            }
            const auto* hidden = t_hidden.ptr<float>(b, c, start);
            if (copy_len <= 4) {
                if (copy_len > 0) {
                    state[pad] = hidden[0];
                }
                if (copy_len > 1) {
                    state[pad + 1] = hidden[1];
                }
                if (copy_len > 2) {
                    state[pad + 2] = hidden[2];
                }
                if (copy_len > 3) {
                    state[pad + 3] = hidden[3];
                }
            } else {
                std::memcpy(state + pad, hidden, copy_len * sizeof(float));
            }
        }
    });

    parallel_for(C, [&](size_t c) {
        const auto* weight = get_weight_ptr(t_weight, c);
        for (size_t b = 0; b < B; ++b) {
            const auto* hidden_base = t_hidden.ptr<float>(b, c, 0);
            auto* out = t_out.ptr<float>(b, c, 0);

            for (size_t t = 0; t + 1 < L; ++t) {
                float sum = 0.0f;
                for (size_t k = 0; k < L; ++k) {
                    const auto idx = static_cast<int64_t>(t) - static_cast<int64_t>(L) + 1 + static_cast<int64_t>(k);
                    if (idx >= 0) {
                        sum += hidden_base[static_cast<size_t>(idx)] * weight[k];
                    }
                }
                out[t] = apply_activation(sum, activation);
            }

            for (size_t t = L - 1; t < seq_len; ++t) {
                const auto* src = hidden_base + (t - (L - 1));
                out[t] = apply_activation(dot_product(src, weight, L), activation);
            }
        }
    });
#endif
}

void CausalConv1D::execute(const dnnl::stream& strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size());
    std::vector<MemoryPtr> outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getDstMemoryAtPort(i);
    }

    const auto precision = getOriginalInputPrecisionAtPort(1);
    OPENVINO_ASSERT(precision == ov::element::f32,
                    "CausalConv1D: only f32 is supported in this reference implementation");

    size_t bias_idx = std::numeric_limits<size_t>::max();
    size_t activation_idx = std::numeric_limits<size_t>::max();
    if (inputs.size() == 5) {
        PlainTensor t_last(inputs[4]);
        const auto act_size = tensor_size(t_last);
        if ((t_last.get_precision() == ov::element::i32 || t_last.get_precision() == ov::element::i64) &&
            act_size <= 1) {
            activation_idx = 4;
        } else {
            bias_idx = 4;
        }
    } else if (inputs.size() >= 6) {
        bias_idx = 4;
        activation_idx = 5;
    }

    if (bias_idx != std::numeric_limits<size_t>::max()) {
        OPENVINO_ASSERT(false, "CausalConv1D: bias is not supported in this reference implementation");
    }

    PlainTensor t_cache(inputs[0]);
    PlainTensor t_hidden(inputs[1]);
    PlainTensor t_weight(inputs[2]);
    const bool has_cache_position_input = inputs.size() >= 4;
    PlainTensor t_pos = has_cache_position_input ? PlainTensor(inputs[3]) : PlainTensor();

    PlainTensor t_out(outputs[0]);
    PlainTensor t_state(outputs[1]);

    const size_t L = t_cache.size(2);
    const size_t K = get_weight_k(t_weight);

    OPENVINO_ASSERT(K == L || K == 0, "CausalConv1D: weight K mismatch with cache length");

    const auto weight_prec = t_weight.get_precision();
    OPENVINO_ASSERT(weight_prec == ov::element::f32,
                    "CausalConv1D: only f32 weight is supported in this reference implementation");
    OPENVINO_ASSERT(t_weight.m_rank == 4 || t_weight.m_rank == 3,
                    "CausalConv1D: only 3D/4D weight is supported in this reference implementation");

    OPENVINO_ASSERT(t_cache.get_precision() == ov::element::f32,
                    "CausalConv1D: cache must be f32 in this reference implementation");
    OPENVINO_ASSERT(t_hidden.get_precision() == ov::element::f32,
                    "CausalConv1D: hidden_states must be f32 in this reference implementation");
    OPENVINO_ASSERT(t_out.get_precision() == ov::element::f32,
                    "CausalConv1D: output must be f32 in this reference implementation");
    OPENVINO_ASSERT(t_state.get_precision() == ov::element::f32,
                    "CausalConv1D: state must be f32 in this reference implementation");

    ActivationType activation = ActivationType::None;
    if (activation_idx != std::numeric_limits<size_t>::max()) {
        PlainTensor t_act(inputs[activation_idx]);
        activation = read_activation_type(t_act);
    }

    bool has_cache_position = false;
    if (has_cache_position_input) {
        const auto pos_prec = t_pos.get_precision();
        OPENVINO_ASSERT(pos_prec == ov::element::i32 || pos_prec == ov::element::i64,
                        "CausalConv1D: cache_position must be i32/i64 in this reference implementation");
        has_cache_position = tensor_size(t_pos) > 0;
        if (has_cache_position) {
            int64_t pos_val = (pos_prec == ov::element::i64) ? t_pos.ptr<int64_t>()[0]
                                                             : static_cast<int64_t>(t_pos.ptr<int32_t>()[0]);
            if (pos_val < 0) {
                has_cache_position = false;
            }
        }
    }

    // causal_conv1d_reference_impl(t_cache,
    //                              t_hidden,
    //                              t_weight,
    //                              has_cache_position_input ? &t_pos : nullptr,
    //                              has_cache_position,
    //                              activation,
    //                              t_out,
    //                              t_state);
    causal_conv1d_optimized_impl(t_cache,
                                 t_hidden,
                                 t_weight,
                                 has_cache_position_input ? &t_pos : nullptr,
                                 has_cache_position,
                                 activation,
                                 t_out,
                                 t_state);
}

}  // namespace ov::intel_cpu::node
