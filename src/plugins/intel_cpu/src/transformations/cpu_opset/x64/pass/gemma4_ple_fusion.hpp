// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace intel_cpu {

// Fuses the per-layer Gemma 3n / Gemma-4 PLE pathway:
//   gate = GELU(MatMul(input, gate_w)) * per_layer_input
//   proj = MatMul(gate, proj_w)
//   normed = RMSNorm(proj, gamma, eps)
//   output = input + normed
// into a single Gemma4PLEBlockNode (cpu_plugin_opset).
class Gemma4PLEFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Gemma4PLEFusion");
    Gemma4PLEFusion();
};

}  // namespace intel_cpu
}  // namespace ov
