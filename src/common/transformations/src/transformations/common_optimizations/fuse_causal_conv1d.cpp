// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_causal_conv1d.hpp"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "ov_ops/causal_conv1d.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::op;

namespace {

std::shared_ptr<Node> skip_broadcast(const std::shared_ptr<Node>& node) {
    auto cur = node;
    while (cur && (ov::is_type<v1::Broadcast>(cur) || ov::is_type<v3::Broadcast>(cur))) {
        cur = cur->input_value(0).get_node_shared_ptr();
    }
    return cur;
}

bool is_search_barrier(const std::shared_ptr<Node>& node) {
    if (!node) {
        return false;
    }
    if (ov::is_type<v5::Loop>(node)) {
        return true;
    }
    if (ov::is_type<v6::Assign>(node)) {
        return true;
    }
    if (ov::is_type<v6::ReadValue>(node) || ov::is_type<v3::ReadValue>(node)) {
        return true;
    }
    return false;
}

std::shared_ptr<Node> find_read_value(const std::shared_ptr<Node>& start) {
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
            return node;
        }
        if (is_search_barrier(node)) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            q.push(node->input_value(i).get_node_shared_ptr());
        }
    }
    return nullptr;
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

std::shared_ptr<Node> find_group_conv_user(const Output<Node>& output, const std::shared_ptr<Node>& anchor) {
    std::vector<std::shared_ptr<v1::GroupConvolution>> group_convs;
    for (const auto& target : output.get_target_inputs()) {
        auto node = target.get_node()->shared_from_this();
        if (auto gc = ov::as_type_ptr<v1::GroupConvolution>(node)) {
            group_convs.push_back(gc);
        }
    }
    if (group_convs.empty()) {
        return nullptr;
    }
    std::vector<std::shared_ptr<v1::GroupConvolution>> anchored;
    if (anchor) {
        for (const auto& gc : group_convs) {
            if (has_upstream_node(anchor, gc.get())) {
                anchored.push_back(gc);
            }
        }
        if (!anchored.empty()) {
            group_convs.swap(anchored);
        }
    }
    auto has_conv1d_name = [](const std::shared_ptr<Node>& node) -> bool {
        if (!node) {
            return false;
        }
        if (node->get_friendly_name().find("conv1d") != std::string::npos) {
            return true;
        }
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            const auto& names = node->get_output_tensor(i).get_names();
            for (const auto& name : names) {
                if (name.find("conv1d") != std::string::npos) {
                    return true;
                }
            }
        }
        return false;
    };
    auto has_non_zero_pads = [](const std::shared_ptr<v1::GroupConvolution>& node) -> bool {
        if (!node) {
            return false;
        }
        const auto& pads_begin = node->get_pads_begin();
        const auto& pads_end = node->get_pads_end();
        for (size_t i = 0; i < pads_begin.size(); ++i) {
            if (pads_begin[i] != 0) {
                return true;
            }
        }
        for (size_t i = 0; i < pads_end.size(); ++i) {
            if (pads_end[i] != 0) {
                return true;
            }
        }
        return false;
    };
    for (const auto& gc : group_convs) {
        if (has_conv1d_name(gc)) {
            return gc;
        }
    }
    for (const auto& gc : group_convs) {
        if (has_non_zero_pads(gc)) {
            return gc;
        }
    }
    return group_convs.front();
}

bool is_scalar_like(const Output<Node>& out) {
    const auto pshape = out.get_partial_shape();
    if (pshape.rank().is_dynamic()) {
        return false;
    }
    if (pshape.rank().get_length() == 0) {
        return true;
    }
    if (pshape.rank().get_length() == 1 && pshape[0].is_static() && pshape[0].get_length() == 1) {
        return true;
    }
    return false;
}

std::shared_ptr<Node> get_scalar_input(const std::shared_ptr<v1::Multiply>& mul) {
    if (!mul) {
        return nullptr;
    }
    if (is_scalar_like(mul->input_value(0))) {
        return mul->input_value(0).get_node_shared_ptr();
    }
    if (is_scalar_like(mul->input_value(1))) {
        return mul->input_value(1).get_node_shared_ptr();
    }
    return nullptr;
}

std::shared_ptr<Node> get_data_input(const std::shared_ptr<v1::Multiply>& mul) {
    if (!mul) {
        return nullptr;
    }
    if (!is_scalar_like(mul->input_value(0))) {
        return mul->input_value(0).get_node_shared_ptr();
    }
    return mul->input_value(1).get_node_shared_ptr();
}

std::shared_ptr<Node> find_is_decoding(const std::shared_ptr<Node>& scalar_node) {
    auto convert = ov::as_type_ptr<v0::Convert>(scalar_node);
    if (!convert) {
        return nullptr;
    }
    const auto& eq_node = convert->input_value(0).get_node_shared_ptr();
    const bool is_equal =
        ov::is_type<v1::Equal>(eq_node) || (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(eq_node) &&
                                            eq_node->get_type_name() == std::string("Equal"));
    if (!is_equal) {
        return nullptr;
    }
    return convert;
}

std::shared_ptr<Node> find_one_minus(const std::shared_ptr<Node>& is_decoding) {
    if (!is_decoding) {
        return nullptr;
    }
    auto is_subtract_like = [](const std::shared_ptr<Node>& node) -> bool {
        return ov::is_type<v1::Subtract>(node) || (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node) &&
                                                   node->get_type_name() == std::string("Subtract"));
    };
    auto is_add_like = [](const std::shared_ptr<Node>& node) -> bool {
        return ov::is_type<v1::Add>(node) || (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node) &&
                                              node->get_type_name() == std::string("Add"));
    };
    for (const auto& target : is_decoding->output(0).get_target_inputs()) {
        auto node = target.get_node()->shared_from_this();
        if (is_subtract_like(node)) {
            return node;
        }
        auto mul = ov::as_type_ptr<v1::Multiply>(node);
        if (mul) {
            const auto mul_in0 = mul->input_value(0);
            const auto mul_in1 = mul->input_value(1);
            if (ov::op::util::is_constant_and_all_values_equal_int(mul_in0, -1) ||
                ov::op::util::is_constant_and_all_values_equal_int(mul_in1, -1)) {
                for (const auto& add_target : mul->output(0).get_target_inputs()) {
                    auto add = add_target.get_node()->shared_from_this();
                    if (!is_add_like(add)) {
                        continue;
                    }
                    const auto add_in0 = add->input_value(0);
                    const auto add_in1 = add->input_value(1);
                    if (ov::op::util::is_constant_and_all_values_equal_int(add_in0, 1) ||
                        ov::op::util::is_constant_and_all_values_equal_int(add_in1, 1)) {
                        return add;
                    }
                }
            }
        }
    }
    return nullptr;
}

std::shared_ptr<v1::Add> find_conv_out_add(const std::shared_ptr<Node>& is_decoding,
                                           const std::shared_ptr<Node>& one_minus,
                                           const std::shared_ptr<Node>& hidden_states,
                                           const std::shared_ptr<Node>& new_state_add) {
    const auto hidden_pshape = hidden_states->get_output_partial_shape(0);
    for (const auto& target : is_decoding->output(0).get_target_inputs()) {
        auto mul = ov::as_type_ptr<v1::Multiply>(target.get_node()->shared_from_this());
        if (!mul) {
            continue;
        }
        for (const auto& add_target : mul->output(0).get_target_inputs()) {
            auto add = ov::as_type_ptr<v1::Add>(add_target.get_node()->shared_from_this());
            if (!add || add.get() == new_state_add.get()) {
                continue;
            }
            auto mul0 = ov::as_type_ptr<v1::Multiply>(add->input_value(0).get_node_shared_ptr());
            auto mul1 = ov::as_type_ptr<v1::Multiply>(add->input_value(1).get_node_shared_ptr());
            if (!mul0 || !mul1) {
                continue;
            }
            auto scalar0 = get_scalar_input(mul0);
            auto scalar1 = get_scalar_input(mul1);
            if (!scalar0 || !scalar1) {
                continue;
            }
            if (!((scalar0.get() == is_decoding.get() && scalar1.get() == one_minus.get()) ||
                  (scalar1.get() == is_decoding.get() && scalar0.get() == one_minus.get()))) {
                continue;
            }
            const auto data0 = get_data_input(mul0);
            const auto data1 = get_data_input(mul1);
            if (!data0 || !data1) {
                continue;
            }
            if (!has_upstream_node(data0, hidden_states.get()) && !has_upstream_node(data1, hidden_states.get())) {
                continue;
            }
            const auto p0 = data0->get_output_partial_shape(0);
            const auto p1 = data1->get_output_partial_shape(0);
            if (p0.rank().is_dynamic() || p1.rank().is_dynamic()) {
                return add;
            }
            if (p0.rank().get_length() == 3 && p1.rank().get_length() == 3 && hidden_pshape.rank().is_static() &&
                hidden_pshape.rank().get_length() == 3) {
                const auto hidden_t = hidden_pshape[2];
                const auto p0_t = p0[2];
                if (hidden_t.is_dynamic() || p0_t.is_dynamic() || hidden_t == p0_t) {
                    return add;
                }
            }
        }
    }
    return nullptr;
}

std::shared_ptr<Node> find_bias_from_reduce(const std::shared_ptr<Node>& reduce_sum) {
    for (const auto& target : reduce_sum->output(0).get_target_inputs()) {
        auto add = ov::as_type_ptr<v1::Add>(target.get_node()->shared_from_this());
        if (!add) {
            continue;
        }
        for (size_t i = 0; i < add->get_input_size(); ++i) {
            auto input_node = add->input_value(i).get_node_shared_ptr();
            if (ov::is_type<v0::Constant>(input_node) || ov::is_type<v0::Convert>(input_node)) {
                return input_node;
            }
        }
    }
    return nullptr;
}

bool is_swish_like(const std::shared_ptr<Node>& node) {
    if (!node) {
        return false;
    }
    if (ov::is_type<v4::Swish>(node)) {
        return true;
    }
    if (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node) && node->get_type_name() == std::string("Swish")) {
        return true;
    }
    return false;
}

bool has_cache_position_name(const std::shared_ptr<Node>& node) {
    if (!node) {
        return false;
    }
    if (node->get_friendly_name().find("cache_position") != std::string::npos) {
        return true;
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        const auto& names = node->get_output_tensor(i).get_names();
        for (const auto& name : names) {
            if (name.find("cache_position") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<Node> find_cache_position(const std::shared_ptr<Node>& start) {
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
        if (ov::is_type<v0::Clamp>(node)) {
            if (has_cache_position_name(node)) {
                return node;
            }
        }
        if (ov::is_type<v4::Range>(node)) {
            if (has_cache_position_name(node)) {
                return node;
            }
        }
        if (is_search_barrier(node)) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            q.push(node->input_value(i).get_node_shared_ptr());
        }
    }
    return nullptr;
}

}  // namespace

ov::pass::CausalConv1DFusion::CausalConv1DFusion() {
    MATCHER_SCOPE(CausalConv1DFusion);

    auto assign = pattern::wrap_type<v6::Assign>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto assign_node = ov::as_type_ptr<v6::Assign>(m.get_match_root());
        if (!assign_node) {
            return false;
        }
        const auto& var_id = assign_node->get_variable_id();
        if (var_id.find("cache_params.past.conv") == std::string::npos) {
            return false;
        }

        auto assign_input = assign_node->input_value(0).get_node_shared_ptr();
        auto new_state_add = ov::as_type_ptr<v1::Add>(skip_broadcast(assign_input));
        if (!new_state_add) {
            return false;
        }

        auto mul0 = ov::as_type_ptr<v1::Multiply>(new_state_add->input_value(0).get_node_shared_ptr());
        auto mul1 = ov::as_type_ptr<v1::Multiply>(new_state_add->input_value(1).get_node_shared_ptr());
        if (!mul0 || !mul1) {
            return false;
        }

        auto scalar0 = get_scalar_input(mul0);
        auto scalar1 = get_scalar_input(mul1);
        if (!scalar0 || !scalar1) {
            return false;
        }

        auto is_decoding = find_is_decoding(scalar0) ? find_is_decoding(scalar0) : find_is_decoding(scalar1);
        if (!is_decoding) {
            return false;
        }
        auto one_minus = find_one_minus(is_decoding);
        if (!one_minus) {
            return false;
        }

        auto data0 = get_data_input(mul0);
        auto data1 = get_data_input(mul1);
        if (!data0 || !data1) {
            return false;
        }

        std::shared_ptr<Node> pad0 = ov::as_type_ptr<v12::Pad>(data0);
        std::shared_ptr<Node> pad1 = ov::as_type_ptr<v12::Pad>(data1);
        if (!pad0) {
            pad0 = ov::as_type_ptr<v1::Pad>(data0);
        }
        if (!pad1) {
            pad1 = ov::as_type_ptr<v1::Pad>(data1);
        }
        std::shared_ptr<Node> conv_state_prefill = nullptr;
        std::shared_ptr<Node> conv_state_dec = nullptr;
        if (pad0) {
            conv_state_prefill = data0;
            conv_state_dec = data1;
        } else if (pad1) {
            conv_state_prefill = data1;
            conv_state_dec = data0;
        } else {
            return false;
        }

        auto hidden_states = conv_state_prefill->input_value(0).get_node_shared_ptr();
        if (!hidden_states) {
            return false;
        }

        auto read_value = find_read_value(conv_state_dec);
        if (!read_value) {
            return false;
        }

        // Optional bias detection (decode path)
        std::shared_ptr<Node> bias = nullptr;
        for (const auto& target : conv_state_dec->output(0).get_target_inputs()) {
            auto mul = ov::as_type_ptr<v1::Multiply>(target.get_node()->shared_from_this());
            if (!mul) {
                continue;
            }
            for (const auto& next : mul->output(0).get_target_inputs()) {
                auto reduce = ov::as_type_ptr<v1::ReduceSum>(next.get_node()->shared_from_this());
                if (!reduce) {
                    continue;
                }
                bias = find_bias_from_reduce(reduce);
                if (bias) {
                    break;
                }
            }
            if (bias) {
                break;
            }
        }

        auto conv_out_add = find_conv_out_add(is_decoding, one_minus, hidden_states, new_state_add);
        if (!conv_out_add) {
            return false;
        }

        auto group_conv = find_group_conv_user(hidden_states->output(0), conv_out_add);
        if (!group_conv) {
            return false;
        }
        auto weight = group_conv->input_value(1).get_node_shared_ptr();
        if (!weight) {
            return false;
        }

        bool has_silu = false;
        {
            auto co_mul0 = ov::as_type_ptr<v1::Multiply>(conv_out_add->input_value(0).get_node_shared_ptr());
            auto co_mul1 = ov::as_type_ptr<v1::Multiply>(conv_out_add->input_value(1).get_node_shared_ptr());
            if (co_mul0 && co_mul1) {
                auto co_data0 = get_data_input(co_mul0);
                auto co_data1 = get_data_input(co_mul1);
                has_silu = is_swish_like(co_data0) || is_swish_like(co_data1);
            }
        }

        std::shared_ptr<Node> cache_position_node = nullptr;
        if (has_silu) {
            cache_position_node = v0::Constant::create(element::i32, Shape{}, {-1});
        } else {
            cache_position_node = find_cache_position(conv_state_dec);
            if (!cache_position_node) {
                return false;
            }
        }

        const auto* assign_ptr = assign_node.get();
        const bool rv_dep = has_upstream_node(read_value, assign_ptr);
        const bool hs_dep = has_upstream_node(hidden_states, assign_ptr);
        const bool w_dep = has_upstream_node(weight, assign_ptr);
        const bool cp_dep = has_upstream_node(cache_position_node, assign_ptr);
        const bool b_dep = bias && has_upstream_node(bias, assign_ptr);
        if (rv_dep || hs_dep || w_dep || cp_dep || b_dep) {
            return false;
        }

        const bool rv_down = has_upstream_node(read_value, conv_out_add.get());
        const bool w_down = has_upstream_node(weight, conv_out_add.get());
        const bool cp_down = has_upstream_node(cache_position_node, conv_out_add.get());
        const bool b_down = bias && has_upstream_node(bias, conv_out_add.get());
        if (rv_down || w_down || cp_down || b_down) {
            return false;
        }

        Output<Node> cache_position_output = cache_position_node->output(0);

        OutputVector args = {read_value, hidden_states, weight, cache_position_output};
        if (bias) {
            args.push_back(bias);
        }
        const int32_t activation_val = has_silu ? 1 : 0;
        auto activation_const = v0::Constant::create(element::i32, Shape{}, {activation_val});
        args.push_back(activation_const);
        auto causal = std::make_shared<ov::op::internal::CausalConv1D>(args);
        causal->set_friendly_name(conv_out_add->get_friendly_name() + "/CausalConv1D");

        copy_runtime_info({conv_out_add,
                           new_state_add,
                           assign_node,
                           group_conv,
                           read_value,
                           hidden_states,
                           weight,
                           cache_position_node,
                           activation_const},
                          causal);

        // Replace conv_out
        conv_out_add->output(0).replace(causal->output(0));

        // Replace new_conv_state path (bypass Broadcast to avoid keeping Roll/ShapeOf alive)
        assign_node->input(0).replace_source_output(causal->output(1));

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(assign, matcher_name);
    this->register_matcher(m, callback);
}
