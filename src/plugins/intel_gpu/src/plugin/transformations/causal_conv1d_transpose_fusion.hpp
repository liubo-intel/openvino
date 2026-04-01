// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/manager.hpp"

namespace ov::intel_gpu {

static constexpr const char* causal_conv1d_transposed_io_rt_key = "gpu_causal_conv1d_transposed_io";

class CausalConv1DTransposeFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CausalConv1DTransposeFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::intel_gpu