// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

// Gemma 3n / Gemma-4 per-layer PLE block:
//   gate     = GELU(MatMul(input, gate_w^T))                  // [B,T,hidden_per_layer]
//   gated    = gate * per_layer_input                         // [B,T,hidden_per_layer]
//   proj_out = MatMul(gated, proj_w^T)                        // [B,T,hidden_size]
//   normed   = RMSNorm(proj_out, norm_gamma, eps)             // [B,T,hidden_size]
//   output   = input + normed                                 // [B,T,hidden_size]
//
// Inputs:
//   0: input (residual)        [B,T,hidden_size]      bf16
//   1: per_layer_input         [B,T,hidden_per_layer] bf16 (already gathered for this layer)
//   2: gate_w                  [hidden_per_layer, hidden_size]  f16 (compressed weights)
//   3: proj_w                  [hidden_size, hidden_per_layer]  f16 (compressed weights)
//   4: norm_gamma              [hidden_size] / [1,1,hidden_size] / etc.   f32
class Gemma4PLEBlockNode : public ov::op::Op {
public:
    OPENVINO_OP("Gemma4PLEBlock", "cpu_plugin_opset");

    Gemma4PLEBlockNode() = default;

    struct Config {
        int hidden_size;        // e.g. 1536
        int hidden_per_layer;   // e.g. 256
        float eps;              // RMSNorm epsilon
        // Per-layer LayerScale scalar (Gemma4 layer_scalar). 1.0 means no extra scaling.
        // Folded into the epi2 JIT pipe so it costs 0 extra DRAM traffic.
        float layer_scalar = 1.0F;
    };

    Gemma4PLEBlockNode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
        validate_and_infer_types();
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

private:
    Config m_config{};
};

}  // namespace intel_cpu
}  // namespace ov
