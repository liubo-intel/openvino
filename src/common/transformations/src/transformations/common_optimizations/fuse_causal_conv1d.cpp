// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_causal_conv1d.hpp"

#include <queue>
#include <string>
#include <unordered_set>

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/causal_conv1d.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::op;

namespace {

bool is_conv_cache_variable_id(const std::string& variable_id) {
    if (variable_id.empty()) {
        return false;
    }

    if (variable_id.find("conv") == std::string::npos) {
        return false;
    }

    const bool has_cache_tag = variable_id.find("cache") != std::string::npos ||
                               variable_id.find("past") != std::string::npos ||
                               variable_id.find("present") != std::string::npos;
    if (!has_cache_tag) {
        return false;
    }

    if (variable_id.find("key") != std::string::npos || variable_id.find("value") != std::string::npos) {
        return false;
    }

    return true;
}

std::shared_ptr<Node> skip_broadcast(const std::shared_ptr<Node>& node) {
    auto cur = node;
    while (cur && (ov::is_type<v1::Broadcast>(cur) || ov::is_type<v3::Broadcast>(cur))) {
        cur = cur->input_value(0).get_node_shared_ptr();
    }
    return cur;
}

bool has_upstream_node(const std::shared_ptr<Node>& start, const Node* target) {
    if (!start || !target) {
        return false;
    }
    std::queue<std::shared_ptr<Node>> q;
    std::unordered_set<Node*> visited;
    q.push(start);
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        if (!node || visited.count(node.get())) {
            continue;
        }
        if (node.get() == target) {
            return true;
        }
        visited.insert(node.get());
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            q.push(node->input_value(i).get_node_shared_ptr());
        }
    }
    return false;
}

bool has_upstream_read_value(const std::shared_ptr<Node>& start) {
    std::queue<std::shared_ptr<Node>> q;
    std::unordered_set<Node*> visited;
    q.push(start);
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        if (!node || visited.count(node.get())) {
            continue;
        }
        visited.insert(node.get());
        if (ov::is_type<v6::ReadValue>(node) || ov::is_type<v3::ReadValue>(node)) {
            return true;
        }
        if (ov::is_type<v6::Assign>(node)) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            q.push(node->input_value(i).get_node_shared_ptr());
        }
    }
    return false;
}

bool is_supported_slice(const std::shared_ptr<Node>& node) {
    return ov::is_type<v8::Slice>(node) || ov::is_type<v1::StridedSlice>(node);
}

std::shared_ptr<Node> find_slice_user(const Output<Node>& output) {
    for (const auto& target : output.get_target_inputs()) {
        auto node = target.get_node()->shared_from_this();
        if (is_supported_slice(node)) {
            return node;
        }
    }
    return nullptr;
}

std::shared_ptr<v1::Add> find_add_bias_user(const std::shared_ptr<Node>& node) {
    if (!node) {
        return nullptr;
    }
    for (const auto& target : node->output(0).get_target_inputs()) {
        auto add = ov::as_type_ptr<v1::Add>(target.get_node()->shared_from_this());
        if (!add) {
            continue;
        }
        const auto in0 = add->input_value(0).get_node_shared_ptr();
        const auto in1 = add->input_value(1).get_node_shared_ptr();
        if (ov::is_type<v0::Constant>(in0) || ov::is_type<v0::Constant>(in1) || ov::is_type<v0::Convert>(in0) ||
            ov::is_type<v0::Convert>(in1)) {
            return add;
        }
    }
    return nullptr;
}

std::shared_ptr<Node> extract_bias(const std::shared_ptr<v1::Add>& add) {
    if (!add) {
        return nullptr;
    }
    auto in0 = add->input_value(0).get_node_shared_ptr();
    auto in1 = add->input_value(1).get_node_shared_ptr();
    if (ov::is_type<v0::Constant>(in0) || ov::is_type<v0::Convert>(in0)) {
        return in0;
    }
    if (ov::is_type<v0::Constant>(in1) || ov::is_type<v0::Convert>(in1)) {
        return in1;
    }
    return nullptr;
}

}  // namespace

ov::pass::CausalConv1DFusion::CausalConv1DFusion() {
    const std::string matcher_name = "CausalConv1DFusion";

    auto assign = pattern::wrap_type<v6::Assign>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto assign_node = ov::as_type_ptr<v6::Assign>(m.get_match_root());
        if (!assign_node) {
            return false;
        }

        const auto& var_id = assign_node->get_variable_id();
        if (!is_conv_cache_variable_id(var_id)) {
            return false;
        }

        auto assign_input = assign_node->input_value(0).get_node_shared_ptr();
        auto state_slice = skip_broadcast(assign_input);
        if (!is_supported_slice(state_slice)) {
            return false;
        }

        auto concat = ov::as_type_ptr<v0::Concat>(state_slice->input_value(0).get_node_shared_ptr());
        if (!concat || concat->get_input_size() != 2) {
            return false;
        }

        auto in0 = concat->input_value(0).get_node_shared_ptr();
        auto in1 = concat->input_value(1).get_node_shared_ptr();
        if (!in0 || !in1) {
            return false;
        }

        std::shared_ptr<Node> conv_state;
        std::shared_ptr<Node> hidden;
        if (has_upstream_read_value(in0)) {
            conv_state = in0;
            hidden = in1;
        } else if (has_upstream_read_value(in1)) {
            conv_state = in1;
            hidden = in0;
        } else {
            return false;
        }

        std::shared_ptr<Node> group_conv_node;
        for (const auto& target : concat->output(0).get_target_inputs()) {
            auto node = target.get_node()->shared_from_this();
            if (ov::is_type<v1::GroupConvolution>(node)) {
                group_conv_node = node;
                break;
            }
        }
        auto group_conv = ov::as_type_ptr<v1::GroupConvolution>(group_conv_node);
        if (!group_conv) {
            return false;
        }

        auto conv_slice = find_slice_user(group_conv->output(0));
        if (!conv_slice) {
            return false;
        }

        auto conv_out_node = std::static_pointer_cast<Node>(conv_slice);
        std::shared_ptr<Node> bias = nullptr;
        auto add_bias = find_add_bias_user(conv_slice);
        if (add_bias) {
            auto extracted = extract_bias(add_bias);
            if (extracted) {
                bias = extracted;
                conv_out_node = add_bias;
            }
        }

        auto weight = group_conv->input_value(1).get_node_shared_ptr();
        if (!weight) {
            return false;
        }

        const auto* assign_ptr = assign_node.get();
        if (has_upstream_node(conv_state, assign_ptr) || has_upstream_node(hidden, assign_ptr) ||
            has_upstream_node(weight, assign_ptr) || (bias && has_upstream_node(bias, assign_ptr))) {
            return false;
        }

        OutputVector args = {hidden, conv_state, weight};
        if (bias) {
            args.push_back(bias);
        }

        auto causal = std::make_shared<ov::op::internal::CausalConv1D>(args);
        causal->set_friendly_name(conv_out_node->get_friendly_name() + "/CausalConv1D");

        if (bias) {
            copy_runtime_info(
                {conv_out_node, state_slice, assign_node, concat, group_conv, conv_state, hidden, weight, bias},
                causal);
        } else {
            copy_runtime_info({conv_out_node, state_slice, assign_node, concat, group_conv, conv_state, hidden, weight},
                              causal);
        }

        conv_out_node->output(0).replace(causal->output(0));
        assign_node->input(0).replace_source_output(causal->output(1));

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(assign, matcher_name);
    this->register_matcher(matcher, callback);
}
