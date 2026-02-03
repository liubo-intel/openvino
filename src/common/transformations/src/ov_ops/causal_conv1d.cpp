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
    OPENVINO_ASSERT(get_input_size() == 4 || get_input_size() == 5,
                    "CausalConv1D expects 4 or 5 inputs, got ",
                    get_input_size());

    const auto cache_pshape = get_input_partial_shape(0);
    const auto hidden_pshape = get_input_partial_shape(1);

    // Output 0: conv_out has the same shape as hidden_states
    set_output_type(0, get_input_element_type(1), hidden_pshape);

    // Output 1: new_conv_state has the same shape as cache
    set_output_type(1, get_input_element_type(0), cache_pshape);
}

std::shared_ptr<Node> CausalConv1D::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<CausalConv1D>(new_args);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
