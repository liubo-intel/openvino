// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/causal_conv1d.hpp"

#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace op {
namespace internal {

CausalConv1D::CausalConv1D(const OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

bool CausalConv1D::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void CausalConv1D::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() >= 3 && get_input_size() <= 4,
                    "CausalConv1D expects 3..4 inputs, got ",
                    get_input_size());

    const auto hidden_pshape = get_input_partial_shape(0);
    const auto cache_pshape = get_input_partial_shape(1);

    // Output 0: conv_out has the same shape as input_embeds
    set_output_type(0, get_input_element_type(0), hidden_pshape);

    // Output 1: new_conv_state has the same shape as conv_state
    set_output_type(1, get_input_element_type(1), cache_pshape);
}

std::shared_ptr<Node> CausalConv1D::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<CausalConv1D>(new_args);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
