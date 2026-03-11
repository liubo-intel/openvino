// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d.h"

#include <algorithm>
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

static void causal_conv1d_reference_impl(const PlainTensor& t_hidden,
                                         const PlainTensor& t_cache,
                                         const PlainTensor& t_weight,
                                         const PlainTensor* t_bias,
                                         PlainTensor& t_out,
                                         PlainTensor& t_state) {
    const size_t B = t_hidden.size(0);
    const size_t C = t_hidden.size(1);
    const size_t T = t_hidden.size(2);
    const size_t S = t_cache.size(2);
    const size_t K = get_weight_k(t_weight);

    const size_t concat_len = S + T;
    OPENVINO_ASSERT(K > 0 && concat_len >= K, "CausalConv1D: invalid kernel/state/seq lengths");

    const size_t full_out_len = concat_len - K + 1;
    OPENVINO_ASSERT(full_out_len >= T, "CausalConv1D: insufficient history for taking last seq_len outputs");
    const size_t out_start = full_out_len - T;

    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            const float* cache = t_cache.ptr<float>(b, c, 0);
            const float* hidden = t_hidden.ptr<float>(b, c, 0);
            float* out = t_out.ptr<float>(b, c, 0);
            float* state = t_state.ptr<float>(b, c, 0);
            const float* weight = get_weight_ptr(t_weight, c);
            const float bias = t_bias ? t_bias->ptr<float>(c)[0] : 0.0f;

            for (size_t t = 0; t < T; ++t) {
                const size_t win_start = out_start + t;
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    const size_t idx = win_start + k;
                    const float v = idx < S ? cache[idx] : hidden[idx - S];
                    sum += v * weight[k];
                }
                out[t] = sum + bias;
            }

            const size_t tail_start = concat_len - S;
            for (size_t s = 0; s < S; ++s) {
                const size_t idx = tail_start + s;
                state[s] = idx < S ? cache[idx] : hidden[idx - S];
            }
        }
    }
}

static void causal_conv1d_optimized_impl(const PlainTensor& t_hidden,
                                         const PlainTensor& t_cache,
                                         const PlainTensor& t_weight,
                                         const PlainTensor* t_bias,
                                         PlainTensor& t_out,
                                         PlainTensor& t_state) {
#if !defined(OPENVINO_ARCH_X86_64)
    causal_conv1d_reference_impl(t_hidden, t_cache, t_weight, t_bias, t_out, t_state);
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
        causal_conv1d_reference_impl(t_hidden, t_cache, t_weight, t_bias, t_out, t_state);
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
    const size_t T = t_hidden.size(2);
    const size_t S = t_cache.size(2);
    const size_t K = get_weight_k(t_weight);

    const size_t concat_len = S + T;
    OPENVINO_ASSERT(K > 0 && concat_len >= K, "CausalConv1D: invalid kernel/state/seq lengths");

    const size_t full_out_len = concat_len - K + 1;
    OPENVINO_ASSERT(full_out_len >= T, "CausalConv1D: insufficient history for taking last seq_len outputs");
    const size_t out_start = full_out_len - T;

    parallel_for(C, [&](size_t c) {
        const float* weight = get_weight_ptr(t_weight, c);
        const float bias = t_bias ? t_bias->ptr<float>(c)[0] : 0.0f;
        for (size_t b = 0; b < B; ++b) {
            const float* cache = t_cache.ptr<float>(b, c, 0);
            const float* hidden = t_hidden.ptr<float>(b, c, 0);
            float* out = t_out.ptr<float>(b, c, 0);

            for (size_t t = 0; t < T; ++t) {
                const size_t win_start = out_start + t;
                float sum = 0.0f;
                if (win_start >= S) {
                    sum = dot_product(hidden + (win_start - S), weight, K);
                } else if (win_start + K <= S) {
                    sum = dot_product(cache + win_start, weight, K);
                } else {
                    const size_t left = S - win_start;
                    sum += dot_product(cache + win_start, weight, left);
                    sum += dot_product(hidden, weight + left, K - left);
                }
                out[t] = sum + bias;
            }
        }
    });

    parallel_for(C, [&](size_t c) {
        for (size_t b = 0; b < B; ++b) {
            const float* cache = t_cache.ptr<float>(b, c, 0);
            const float* hidden = t_hidden.ptr<float>(b, c, 0);
            float* state = t_state.ptr<float>(b, c, 0);

            if (T >= S) {
                std::memcpy(state, hidden + (T - S), S * sizeof(float));
            } else {
                const size_t n_cache = S - T;
                if (n_cache > 0) {
                    std::memcpy(state, cache + T, n_cache * sizeof(float));
                }
                if (T > 0) {
                    std::memcpy(state + n_cache, hidden, T * sizeof(float));
                }
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

    OPENVINO_ASSERT(inputs.size() == 3 || inputs.size() == 4,
                    "CausalConv1D expects 3 or 4 inputs, got ",
                    inputs.size());

    PlainTensor t_hidden(inputs[0]);
    PlainTensor t_cache(inputs[1]);
    PlainTensor t_weight(inputs[2]);
    PlainTensor t_out(outputs[0]);
    PlainTensor t_state(outputs[1]);

    OPENVINO_ASSERT(t_hidden.get_precision() == ov::element::f32,
                    "CausalConv1D: input_embeds must be f32 in this implementation");
    OPENVINO_ASSERT(t_cache.get_precision() == ov::element::f32,
                    "CausalConv1D: conv_state must be f32 in this implementation");
    OPENVINO_ASSERT(t_weight.get_precision() == ov::element::f32,
                    "CausalConv1D: weight must be f32 in this implementation");
    OPENVINO_ASSERT(t_out.get_precision() == ov::element::f32,
                    "CausalConv1D: output must be f32 in this implementation");
    OPENVINO_ASSERT(t_state.get_precision() == ov::element::f32,
                    "CausalConv1D: output_conv_state must be f32 in this implementation");

    OPENVINO_ASSERT(t_weight.m_rank == 4 || t_weight.m_rank == 3,
                    "CausalConv1D: only 3D/4D weight is supported in this implementation");

    const size_t C = t_hidden.size(1);
    OPENVINO_ASSERT(t_cache.size(1) == C, "CausalConv1D: conv_state C mismatch");
    OPENVINO_ASSERT(t_out.size(1) == C, "CausalConv1D: output C mismatch");
    OPENVINO_ASSERT(t_state.size(1) == C, "CausalConv1D: output state C mismatch");

    OPENVINO_ASSERT(t_weight.size(0) == C, "CausalConv1D: weight out_channels must equal hidden_size");
    OPENVINO_ASSERT(t_weight.size(1) == 1, "CausalConv1D: only depthwise/group_size==1 weight is supported");

    const size_t K = get_weight_k(t_weight);
    const size_t S = t_cache.size(2);
    const size_t T = t_hidden.size(2);
    OPENVINO_ASSERT(K > 0, "CausalConv1D: kernel size must be > 0");
    OPENVINO_ASSERT(S + T >= K, "CausalConv1D: S + T must be >= kernel_size");

    PlainTensor t_bias;
    const PlainTensor* p_bias = nullptr;
    if (inputs.size() == 4) {
        t_bias = PlainTensor(inputs[3]);
        OPENVINO_ASSERT(t_bias.get_precision() == ov::element::f32,
                        "CausalConv1D: bias must be f32 in this implementation");
        OPENVINO_ASSERT(t_bias.m_rank == 1, "CausalConv1D: bias must be 1D");
        OPENVINO_ASSERT(t_bias.size(0) == C, "CausalConv1D: bias size must equal hidden_size");
        p_bias = &t_bias;
    }

    causal_conv1d_optimized_impl(t_hidden, t_cache, t_weight, p_bias, t_out, t_state);
}

}  // namespace ov::intel_cpu::node
