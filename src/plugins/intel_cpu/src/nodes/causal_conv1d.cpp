// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/reference.h"
#include "openvino/core/except.hpp"
#include "ov_ops/causal_conv1d.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

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

static void causal_conv1d_reference_impl(const PlainTensor& t_cache,
                                         const PlainTensor& t_hidden,
                                         const PlainTensor& t_weight,
                                         const PlainTensor& t_pos,
                                         PlainTensor& t_out,
                                         PlainTensor& t_state) {
    const size_t B = t_hidden.size(0);
    const size_t C = t_hidden.size(1);
    const size_t seq_len = t_hidden.size(2);
    const size_t L = t_cache.size(2);

    if (seq_len == 1) {
        // decoding path
        auto read_pos0 = [&]() -> int64_t {
            return static_cast<int64_t>(t_pos.at<int32_t>({0}));
        };
        auto clamp_pos = [&](int64_t p) -> size_t {
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
                for (size_t k = 0; k < L; ++k) {
                    sum += t_state.at<float>({b, c, k}) * t_weight.at<float>({c, 0, 0, k});
                }
                t_out.at<float>({b, c, 0}) = sum;
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
                for (size_t k = 0; k < L; ++k) {
                    // Conv1d uses cross-correlation: idx = t + k - (L - 1)，represent left padding conv
                    const auto idx = static_cast<int64_t>(t) - static_cast<int64_t>(L) + 1 + static_cast<int64_t>(k);
                    if (idx >= 0) {
                        sum += t_hidden.at<float>({b, c, static_cast<size_t>(idx)}) * t_weight.at<float>({c, 0, 0, k});
                    }
                }
                t_out.at<float>({b, c, t}) = sum;
            }
        }
    }
}

[[maybe_unused]] static void causal_conv1d_optimized_impl(const PlainTensor& t_cache,
                                                          const PlainTensor& t_hidden,
                                                          const PlainTensor& t_weight,
                                                          const PlainTensor& t_pos,
                                                          PlainTensor& t_out,
                                                          PlainTensor& t_state) {
    OPENVINO_THROW_NOT_IMPLEMENTED("CausalConv1D: optimized implementation is not available yet");
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

    const bool has_bias = inputs.size() > 4;
    OPENVINO_ASSERT(!has_bias, "CausalConv1D: bias is not supported in this reference implementation");

    PlainTensor t_cache(inputs[0]);
    PlainTensor t_hidden(inputs[1]);
    PlainTensor t_weight(inputs[2]);
    PlainTensor t_pos(inputs[3]);

    PlainTensor t_out(outputs[0]);
    PlainTensor t_state(outputs[1]);

    const size_t L = t_cache.size(2);
    const size_t K = get_weight_k(t_weight);

    OPENVINO_ASSERT(K == L || K == 0, "CausalConv1D: weight K mismatch with cache length");

    const auto weight_prec = t_weight.get_precision();
    OPENVINO_ASSERT(weight_prec == ov::element::f32,
                    "CausalConv1D: only f32 weight is supported in this reference implementation");
    OPENVINO_ASSERT(t_weight.m_rank == 4, "CausalConv1D: only 4D weight is supported in this reference implementation");

    OPENVINO_ASSERT(t_cache.get_precision() == ov::element::f32,
                    "CausalConv1D: cache must be f32 in this reference implementation");
    OPENVINO_ASSERT(t_hidden.get_precision() == ov::element::f32,
                    "CausalConv1D: hidden_states must be f32 in this reference implementation");
    OPENVINO_ASSERT(t_out.get_precision() == ov::element::f32,
                    "CausalConv1D: output must be f32 in this reference implementation");
    OPENVINO_ASSERT(t_state.get_precision() == ov::element::f32,
                    "CausalConv1D: state must be f32 in this reference implementation");

    const auto pos_prec = t_pos.get_precision();
    OPENVINO_ASSERT(pos_prec == ov::element::i32,
                    "CausalConv1D: cache_position must be i32 in this reference implementation");
    OPENVINO_ASSERT(t_pos.m_rank != 0, "CausalConv1D: cache_position must be 1D in this reference implementation");

    causal_conv1d_reference_impl(t_cache, t_hidden, t_weight, t_pos, t_out, t_state);
}

}  // namespace ov::intel_cpu::node
