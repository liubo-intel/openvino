// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/causal_conv1d.hpp"

#include <optional>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/causal_conv1d.hpp"
#include "plugin/transformations/causal_conv1d_variable_fusion.hpp"
#include "openvino/op/assign.hpp"

namespace ov {
namespace op {
namespace internal {
using CausalConv1D = ov::op::internal::CausalConv1D;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

namespace {

std::optional<std::string> get_variable_id_from_rt_info(const std::shared_ptr<op::internal::CausalConv1D>& op) {
    const auto& rt_info = op->get_rt_info();
    auto it = rt_info.find(causal_conv1d_variable_id_rt_key);
    if (it == rt_info.end()) {
        return std::nullopt;
    }

    try {
        const auto& variable_id = it->second.as<std::string>();
        if (variable_id.empty()) {
            return std::nullopt;
        }
        return variable_id;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<ov::op::util::VariableInfo> get_variable_info_from_assign(const std::shared_ptr<op::internal::CausalConv1D>& op) {
    if (op->get_output_size() < 2) {
        return std::nullopt;
    }

    const auto& output_state = op->output(1);
    for (const auto& target : output_state.get_target_inputs()) {
        auto node = target.get_node()->shared_from_this();
        if (auto assign_v6 = ov::as_type_ptr<ov::op::v6::Assign>(node)) {
            return ov::op::util::VariableInfo{output_state.get_partial_shape(), output_state.get_element_type(), assign_v6->get_variable_id()};
        }
        if (auto assign_v3 = ov::as_type_ptr<ov::op::v3::Assign>(node)) {
            return ov::op::util::VariableInfo{output_state.get_partial_shape(), output_state.get_element_type(), assign_v3->get_variable_id()};
        }
    }

    return std::nullopt;
}

}  // namespace

static void CreateCausalConv1DOp(ProgramBuilder& p, const std::shared_ptr<op::internal::CausalConv1D>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);

    std::optional<ov::op::util::VariableInfo> variable_info = std::nullopt;
    auto rt_variable_id = get_variable_id_from_rt_info(op);
    if (rt_variable_id.has_value()) {
        const auto& output_state = op->output(1);
        variable_info = ov::op::util::VariableInfo{output_state.get_partial_shape(), output_state.get_element_type(), rt_variable_id.value()};
    } else {
        variable_info = get_variable_info_from_assign(op);
    }

    cldnn::causal_conv1d prim(layer_type_name_ID(op),
                              inputs,
                              variable_info.has_value() ? variable_info.value() : ov::op::util::VariableInfo{});
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, CausalConv1D);

}  // namespace ov::intel_gpu
