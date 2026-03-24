// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d_inst.h"

#include "json_object.h"
#include "primitive_type_base.h"

#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(causal_conv1d)

layout causal_conv1d_inst::calc_output_layout(const causal_conv1d_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> causal_conv1d_inst::calc_output_layouts(const causal_conv1d_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<causal_conv1d>();
    const auto num_outputs = desc->output_size();

    OPENVINO_ASSERT(impl_param.input_layouts.size() == 3 || impl_param.input_layouts.size() == 4,
                    "causal_conv1d must have 3 or 4 inputs");

    const auto& hidden_layout = impl_param.get_input_layout(0);
    const auto& state_layout = impl_param.get_input_layout(1);

    std::vector<layout> output_layouts;
    const auto out0_type = desc->output_data_types[0].value_or(hidden_layout.data_type);
    output_layouts.emplace_back(hidden_layout.get_partial_shape(), out0_type, hidden_layout.format);

    if (num_outputs > 1) {
        const auto out1_type = desc->output_data_types[1].value_or(state_layout.data_type);
        output_layouts.emplace_back(state_layout.get_partial_shape(), out1_type, state_layout.format);
    }

    return output_layouts;
}

template std::vector<layout> causal_conv1d_inst::calc_output_layouts<ov::PartialShape>(const causal_conv1d_node& node,
                                                                                         const kernel_impl_params& impl_param);

std::string causal_conv1d_inst::to_string(const causal_conv1d_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite causal_conv_info;
    causal_conv_info.add("input_embeds", node.input(0).id());
    causal_conv_info.add("conv_state", node.input(1).id());
    causal_conv_info.add("conv_weight", node.input(2).id());
    if (node.get_dependencies().size() > 3) {
        causal_conv_info.add("conv_bias", node.input(3).id());
    }

    node_info->add("causal_conv1d_info", causal_conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

causal_conv1d_inst::typed_primitive_inst(network& network, const causal_conv1d_node& node) : parent(network, node) {}

}  // namespace cldnn
