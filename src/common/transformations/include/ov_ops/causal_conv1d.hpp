// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

///
/// \brief Causal 1D convolution with cache update.
/// Internal operation which may change in the future.
///
/// Inputs:
/// 0: cache [B, C, L]
/// 1: hidden_states (Bx) [B, C, T]
/// 2: weight [C, 1, 1, L] (or [C, 1, L])
/// 3: cache_position [T] (or scalar)
/// 4: bias [C] (optional)
///
/// Outputs:
/// 0: conv_out [B, C, T]
/// 1: new_conv_state [B, C, L]
class TRANSFORMATIONS_API CausalConv1D : public Op {
public:
    OPENVINO_OP("CausalConv1D", "ie_internal_opset", Op);

    CausalConv1D() = default;

    explicit CausalConv1D(const OutputVector& args);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
