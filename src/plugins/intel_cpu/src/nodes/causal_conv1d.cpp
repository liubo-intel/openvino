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
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
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

template <typename T>
void CausalConv1D::executeTyped(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs) const {
    PlainTensor t_cache(inputs[0]);
    PlainTensor t_hidden(inputs[1]);
    PlainTensor t_weight(inputs[2]);
    PlainTensor t_pos(inputs[3]);
    PlainTensor t_bias;
    const bool has_bias = inputs.size() > 4;
    if (has_bias) {
        t_bias.reset(inputs[4]);
    }

    PlainTensor t_out(outputs[0]);
    PlainTensor t_state(outputs[1]);

    const size_t B = t_hidden.size(0);
    const size_t C = t_hidden.size(1);
    const size_t seq_len = t_hidden.size(2);
    const size_t L = t_cache.size(2);
    const size_t K = get_weight_k(t_weight);

    OPENVINO_ASSERT(K == L || K == 0, "CausalConv1D: weight K mismatch with cache length");

    auto read_weight = [&](size_t c, size_t k) -> float {
        auto read_as = [&](auto dummy) -> float {
            using DT = decltype(dummy);
            if (t_weight.m_rank == 4) {
                return static_cast<float>(t_weight.at<DT>({c, 0, 0, k}));
            }
            if (t_weight.m_rank == 3) {
                return static_cast<float>(t_weight.at<DT>({c, 0, k}));
            }
            if (t_weight.m_rank == 2) {
                return static_cast<float>(t_weight.at<DT>({c, k}));
            }
            return 0.0f;
        };
        const auto prec = t_weight.get_precision();
        if (prec == ov::element::f16) {
            return read_as(ov::float16{});
        }
        if (prec == ov::element::bf16) {
            return read_as(ov::bfloat16{});
        }
        return read_as(float{});
    };

    auto read_bias = [&](size_t c) -> float {
        if (!has_bias) {
            return 0.0f;
        }
        const auto prec = t_bias.get_precision();
        if (prec == ov::element::f16) {
            return static_cast<float>(t_bias.at<ov::float16>({c}));
        }
        if (prec == ov::element::bf16) {
            return static_cast<float>(t_bias.at<ov::bfloat16>({c}));
        }
        return static_cast<float>(t_bias.at<float>({c}));
    };

    if (seq_len == 1) {
        // decoding path
        auto read_pos0 = [&]() -> int64_t {
            const auto prec = t_pos.get_precision();
            if (t_pos.m_rank == 0) {
                if (prec == ov::element::i32) {
                    return static_cast<int64_t>(t_pos.at<int32_t>({}));
                }
                return t_pos.at<int64_t>({});
            }
            if (prec == ov::element::i32) {
                return static_cast<int64_t>(t_pos.at<int32_t>({0}));
            }
            return t_pos.at<int64_t>({0});
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
                    t_state.at<T>({b, c, k}) = t_cache.at<T>({b, c, k + 1});
                }
                t_state.at<T>({b, c, pos}) = t_hidden.at<T>({b, c, 0});

                float sum = 0.0f;
                for (size_t k = 0; k < L; ++k) {
                    sum += static_cast<float>(t_state.at<T>({b, c, k})) * read_weight(c, k);
                }
                sum += read_bias(c);
                t_out.at<T>({b, c, 0}) = static_cast<T>(sum);
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
                t_state.at<T>({b, c, k}) = static_cast<T>(0);
            }
            for (size_t t = 0; t < copy_len; ++t) {
                t_state.at<T>({b, c, pad + t}) = t_hidden.at<T>({b, c, start + t});
            }

            for (size_t t = 0; t < seq_len; ++t) {
                float sum = 0.0f;
                for (size_t k = 0; k < L; ++k) {
                    // Conv1d uses cross-correlation: idx = t + k - (L - 1)，represent left padding conv
                    const auto idx = static_cast<int64_t>(t) - static_cast<int64_t>(L) + 1 +
                                     static_cast<int64_t>(k);
                    if (idx >= 0) {
                        sum += static_cast<float>(t_hidden.at<T>({b, c, static_cast<size_t>(idx)})) *
                               read_weight(c, k);
                    }
                }
                sum += read_bias(c);
                t_out.at<T>({b, c, t}) = static_cast<T>(sum);
            }
        }
    }
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
    if (precision == ov::element::f16) {
        executeTyped<ov::float16>(inputs, outputs);
    } else if (precision == ov::element::bf16) {
        executeTyped<ov::bfloat16>(inputs, outputs);
    } else {
        executeTyped<float>(inputs, outputs);
    }
}

}  // namespace ov::intel_cpu::node
