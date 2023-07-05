// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_reorder_for_past_key_value.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset10.hpp>

#include "dnnl.hpp"
#include "itt.hpp"
#include "transformations/common_optimizations/concat_reduce_fusion.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"

ov::intel_cpu::ReduceReorderForPastKeyValue::ReduceReorderForPastKeyValue() {
    MATCHER_SCOPE(ReduceReorderForPastKeyValue);

    ngraph::element::TypeVector param_precisions{element::f32, element::i8, element::u8};
    auto input_m =
        ngraph::pattern::wrap_type<opset10::Parameter>(ov::pass::pattern::type_matches_any(param_precisions));
    auto concat_m = ngraph::pattern::wrap_type<opset10::Concat>({input_m, ov::pass::pattern::any_input()});

    auto result_m = ngraph::pattern::wrap_type<opset10::Result>({concat_m});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& concat_node = pattern_map.at(concat_m).get_node_shared_ptr();

        size_t concat_output_size = concat_node->get_output_size();
        for (size_t output_index = 0; output_index < concat_output_size; ++output_index) {
            const auto output_node = concat_node->output(output_index);
            for (const auto& out_inputs : output_node.get_target_inputs()) {
                auto out_node = out_inputs.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<opset10::Result>(out_node)) {
                    continue;
                    // TODO: place this pass in more suitable place to ignore this 'PowerStatic' node
                } else if (std::dynamic_pointer_cast<opset10::MatMul>(out_node) ||
                           std::dynamic_pointer_cast<ov::intel_cpu::PowerStaticNode>(out_node)) {
                    auto element_type = out_node->get_element_type();
                    const auto& input_node = pattern_map.at(input_m).get_node_shared_ptr();
                    const auto& result_node = pattern_map.at(result_m).get_node_shared_ptr();
                    // TODO: add 'enableBF16' condition check
                    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
                        element_type == element::f32) {
                        input_node->output(0).get_tensor().set_element_type(element::bf16);
                        result_node->output(0).get_tensor().set_element_type(element::bf16);
                    } else if ((element_type != input_node->get_element_type()) ||
                               (element_type != result_node->get_element_type())) {
                        input_node->output(0).get_tensor().set_element_type(element_type);
                        result_node->output(0).get_tensor().set_element_type(element_type);
                    }
                    return true;
                } else {
                    continue;
                }
            }
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(result_m, matcher_name);

    this->register_matcher(m, callback);
}