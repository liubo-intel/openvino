// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemma4_ple_block.hpp"

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "transformations/itt.hpp"

namespace ov {
namespace intel_cpu {

bool Gemma4PLEBlockNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Gemma4PLEBlockNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("hidden_per_layer", m_config.hidden_per_layer);
    visitor.on_attribute("eps", m_config.eps);
    visitor.finish_structure();
    return true;
}

void Gemma4PLEBlockNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Gemma4PLEBlockNode_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 5, "Gemma4PLEBlock expects 5 inputs");

    const auto& ishape = get_input_partial_shape(0);
    const auto& itype = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this, ishape.rank().is_static() && ishape.rank() == 3, "input rank must be 3");
    NODE_VALIDATION_CHECK(this, itype.is_real(), "input data type must be real");

    set_output_type(0, itype, ishape);
}

std::shared_ptr<Node> Gemma4PLEBlockNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Gemma4PLEBlockNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Gemma4PLEBlockNode>(new_args, m_config);
}

}  // namespace intel_cpu
}  // namespace ov
