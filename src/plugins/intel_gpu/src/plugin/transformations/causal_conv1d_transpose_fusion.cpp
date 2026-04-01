// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d_transpose_fusion.hpp"

#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_ops/causal_conv1d.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

bool has_021_order(const std::shared_ptr<ov::Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
        return false;
    }

    return constant->cast_vector<int64_t>() == std::vector<int64_t>{0, 2, 1};
}

bool is_rank3(const ov::Output<ov::Node>& output) {
    const auto rank = output.get_partial_shape().rank();
    return rank.is_static() && rank.get_length() == 3;
}

}  // namespace

bool CausalConv1DTransposeFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool changed = false;

    for (const auto& node : m->get_ordered_ops()) {
        auto transpose_out = ov::as_type_ptr<ov::op::v1::Transpose>(node);
        if (!transpose_out) {
            continue;
        }

        if (!has_021_order(transpose_out->get_input_node_shared_ptr(1))) {
            continue;
        }

        auto swish = ov::as_type_ptr<ov::op::v4::Swish>(transpose_out->get_input_node_shared_ptr(0));
        auto causal = swish ?
                          ov::as_type_ptr<ov::op::internal::CausalConv1D>(swish->get_input_node_shared_ptr(0)) :
                          ov::as_type_ptr<ov::op::internal::CausalConv1D>(transpose_out->get_input_node_shared_ptr(0));

        if (!causal || transformation_callback(causal)) {
            continue;
        }

        if (transpose_out->input_value(0).get_index() != 0) {
            continue;
        }

        if (swish && swish->input_value(0).get_index() != 0) {
            continue;
        }

        auto transpose_in = ov::as_type_ptr<ov::op::v1::Transpose>(causal->get_input_node_shared_ptr(0));
        if (!transpose_in || !has_021_order(transpose_in->get_input_node_shared_ptr(1))) {
            continue;
        }

        if (!is_rank3(transpose_in->output(0)) || !is_rank3(transpose_in->input_value(0))) {
            continue;
        }

        if (transpose_in->output(0).get_target_inputs().size() != 1) {
            continue;
        }

        if (causal->output(0).get_target_inputs().size() != 1) {
            continue;
        }

        if (swish && swish->output(0).get_target_inputs().size() != 1) {
            continue;
        }

        ov::OutputVector causal_inputs;
        causal_inputs.reserve(causal->get_input_size());
        causal_inputs.push_back(transpose_in->input_value(0));
        for (size_t i = 1; i < causal->get_input_size(); i++) {
            causal_inputs.push_back(causal->input_value(i));
        }

        auto causal_new = std::make_shared<ov::op::internal::CausalConv1D>(causal_inputs);
        causal_new->get_rt_info() = causal->get_rt_info();
        causal_new->get_rt_info()[causal_conv1d_transposed_io_rt_key] = true;

        if (causal->get_output_size() > 1) {
            causal->output(1).replace(causal_new->output(1));
        }

        ov::NodeVector copy_info_src = {transpose_in, causal, transpose_out};
        if (swish) {
            copy_info_src.push_back(swish);
            auto swish_new = std::make_shared<ov::op::v4::Swish>(causal_new->output(0));
            swish_new->set_friendly_name(transpose_out->get_friendly_name());
            ov::copy_runtime_info(copy_info_src, {causal_new, swish_new});
            ov::replace_node(transpose_out, swish_new);
        } else {
            causal_new->set_friendly_name(transpose_out->get_friendly_name());
            ov::copy_runtime_info(copy_info_src, causal_new);
            ov::replace_node(transpose_out, causal_new);
        }
        changed = true;
    }

    return changed;
}

}  // namespace ov::intel_gpu