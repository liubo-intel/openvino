// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d_variable_fusion.hpp"

#include "openvino/op/assign.hpp"
#include "openvino/op/convert.hpp"
#include "ov_ops/causal_conv1d.hpp"

namespace ov::intel_gpu {

bool CausalConv1DVariableFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool changed = false;
    ov::SinkVector sinks = m->get_sinks();

    for (auto& sink : sinks) {
        std::shared_ptr<ov::op::Op> assign_op;
        std::string variable_id;

        if (auto assign_v6 = ov::as_type_ptr<ov::op::v6::Assign>(sink)) {
            assign_op = assign_v6;
            variable_id = assign_v6->get_variable_id();
        } else if (auto assign_v3 = ov::as_type_ptr<ov::op::v3::Assign>(sink)) {
            assign_op = assign_v3;
            variable_id = assign_v3->get_variable_id();
        } else {
            continue;
        }

        auto producer = assign_op->get_input_node_shared_ptr(0);
        auto producer_output = assign_op->input_value(0);
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(producer)) {
            producer = convert->input_value(0).get_node_shared_ptr();
            producer_output = convert->input_value(0);
        }

        auto causal = ov::as_type_ptr<ov::op::internal::CausalConv1D>(producer);
        if (!causal) {
            continue;
        }

        // output 1 of CausalConv1D is state output; output 0 is conv output.
        if (producer_output.get_index() != 1) {
            continue;
        }

        causal->get_rt_info()[causal_conv1d_variable_id_rt_key] = variable_id;
        m->remove_sink(sink);
        changed = true;
    }

    return changed;
}

}  // namespace ov::intel_gpu
