// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

static constexpr const char* causal_conv1d_variable_id_rt_key = "gpu_causal_conv1d_variable_id";

class CausalConv1DVariableFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CausalConv1DVariableFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::intel_gpu
