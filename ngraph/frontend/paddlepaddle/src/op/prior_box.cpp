// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
namespace detail {
namespace {
std::shared_ptr<opset8::StridedSlice> make_slice(std::shared_ptr<ngraph::Node> node, int64_t start, int64_t end) {
    return std::make_shared<opset8::StridedSlice>(
        node,
        opset8::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{start}),
        opset8::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{end}),
        std::vector<int64_t>{0},   // begin mask
        std::vector<int64_t>{0});  // end mask
}
}  // namespace
}  // namespace detail
NamedOutputs prior_box(const NodeContext& node) {
    auto input = node.get_ng_input("Input");
    auto Image = node.get_ng_input("Image");
    auto input_shape = std::make_shared<opset8::ShapeOf>(input);
    auto Image_shape = std::make_shared<opset8::ShapeOf>(Image);
    auto output_shape_slice = detail::make_slice(input_shape, 2, 4);
    auto image_shape_slice = detail::make_slice(Image_shape, 2, 4);

    ngraph::op::PriorBoxAttrs attrs;
    attrs.min_size = node.get_attribute<std::vector<float>>("min_sizes", {});
    if (node.has_attribute<int32_t>("max_sizes"))
        attrs.max_size = node.get_attribute<std::vector<float>>("max_sizes", {});
    attrs.aspect_ratio = node.get_attribute<std::vector<float>>("aspect_ratios", {1.0});
    attrs.flip = node.get_attribute<bool>("flip", false);
    attrs.clip = node.get_attribute<bool>("clip", false);
    attrs.step = node.get_attribute<float>("step_w", 0);

    attrs.offset = node.get_attribute<float>("offset", 0.5);
    attrs.variance = node.get_attribute<std::vector<float>>("variances", {0.1, 0.1, 0.2, 0.2});

    if (node.has_attribute<bool>("min_max_aspect_ratios_order"))
        attrs.min_max_aspect_ratios_order = node.get_attribute<bool>("min_max_aspect_ratios_order", false);

    auto ov_prior_box_node = std::make_shared<opset8::PriorBox>(output_shape_slice, image_shape_slice, attrs);

    auto split_axis_node = opset8::Constant::create(element::i64, ngraph::Shape{}, {0});
    auto node_prior_box_split = std::make_shared<opset8::Split>(ov_prior_box_node, split_axis_node, 2);

    auto node_boxes_origin = node_prior_box_split->output(0);
    auto node_variances_origin = node_prior_box_split->output(1);

    PartialShape input_partial_shape = input.get_partial_shape();
    auto H = input_partial_shape[2].get_length();
    auto W = input_partial_shape[3].get_length();
    auto out_shape = opset8::Constant::create(element::i64, Shape{4}, {H, W, int64_t(-1), int64_t(4)});

    auto node_boxes_reshape = std::make_shared<opset8::Reshape>(node_boxes_origin, out_shape, true);
    auto node_variances_reshape = std::make_shared<opset8::Reshape>(node_variances_origin, out_shape, true);

    NamedOutputs outputs;
    outputs["Boxes"] = {node_boxes_reshape};
    outputs["Variances"] = {node_variances_reshape};
    return outputs;
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph