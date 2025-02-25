// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/opsets/opset8.hpp"

using namespace ov::opset8;

void ov::frontend::tensorflow::tf_shape_to_ov_shape(const ::tensorflow::TensorShapeProto& tf_shape,
                                                    ov::PartialShape* ng_shape) {
    std::vector<ov::Dimension> dims;
    for (int i = 0; i < tf_shape.dim_size(); i++) {
        dims.emplace_back(tf_shape.dim(i).size());
    }
    *ng_shape = ov::PartialShape(dims);
}

void ov::frontend::tensorflow::set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node) {
    const auto& outputs = node->outputs();
    node->set_friendly_name(node_name);
    if (outputs.size() == 1) {
        set_out_name(node_name, outputs[0]);
    }
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        set_out_name({node_name + ":" + std::to_string(idx)}, outputs[idx]);
    }
}

void ov::frontend::tensorflow::set_out_name(const std::string& out_name, const ov::Output<ov::Node>& output) {
    output.get_tensor().add_names({out_name});
}

ov::op::PadType ov::frontend::tensorflow::convert_tf_padding(const ov::frontend::tensorflow::NodeContext& node,
                                                             const std::string& tf_padding) {
    std::set<std::string> supported_ops = {"Conv2D",
                                           "Conv2DBackpropInput",
                                           "Conv3D",
                                           "Conv3DBackpropInputV2",
                                           "MaxPool",
                                           "MaxPoolV2",
                                           "MaxPool3D",
                                           "ExtractImagePatches"};
    auto op_type = node.get_op_type();

    TENSORFLOW_OP_VALIDATION(node,
                             supported_ops.count(op_type),
                             "Conversion of padding mode for " + op_type + " is not supported.");
    TENSORFLOW_OP_VALIDATION(
        node,
        tf_padding == "VALID" || tf_padding == "SAME" || tf_padding == "EXPLICIT",
        "The deconvolutional operation must have one of the padding type: VALID, SAME, and EXPLICIT.");

    if (tf_padding == "VALID") {
        return ov::op::PadType::VALID;
    }
    if (op_type == "Conv2DBackpropInput" || op_type == "Conv3DBackpropInputV2") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // ConvBackpropData layer in the Operation specification,
            // the SAME_LOWER value matches to the SAME value in TensorFlow
            return ov::op::PadType::SAME_LOWER;
        }
    } else if (op_type == "Conv2D" || op_type == "Conv3D" || op_type == "MaxPool" || op_type == "MaxPoolV2" ||
               op_type == "MaxPool3D" || op_type == "ExtractImagePatches") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // Conv layer in the Operation specification,
            // the SAME_UPPER value matches to the SAME value in TensorFlow
            return ov::op::PadType::SAME_UPPER;
        }
    }

    return ov::op::PadType::EXPLICIT;
}

void ov::frontend::tensorflow::fill_explicit_pads_vectors(const ov::frontend::tensorflow::NodeContext& node,
                                                          bool is_nhwc,
                                                          size_t spatial_dims_num,
                                                          const std::vector<int64_t>& tf_explicit_paddings,
                                                          ov::CoordinateDiff& pads_begin,
                                                          ov::CoordinateDiff& pads_end) {
    auto fullfill_pads = [&](ov::CoordinateDiff& pads, const std::vector<int64_t>& indexes) {
        pads.resize(indexes.size());
        for (int i = 0; i < indexes.size(); ++i) {
            pads[i] = tf_explicit_paddings[indexes[i]];
        }
    };

    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_explicit_paddings.size() == 8,
                                 "Conv2D expects 8 padding values for EXPLICIT padding mode.");
        // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
        if (is_nhwc) {
            // For NHWC layout, explicit paddings has the following form:
            // [0, 0, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
            fullfill_pads(pads_begin, {2, 4});
            fullfill_pads(pads_end, {3, 5});
        } else {
            // For NCHW layout, explicit paddings has the following form:
            // [0, 0, 0, 0, pad_h1, pad_h2, pad_w1, pad_w2]
            fullfill_pads(pads_begin, {4, 6});
            fullfill_pads(pads_end, {5, 7});
        }
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_explicit_paddings.size() == 10,
                                 "Conv3D expects 10 padding values for EXPLICIT padding mode.");
        // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
        if (is_nhwc) {
            // For NDHWC layout, explicit paddings has the following form:
            // [0, 0, pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
            fullfill_pads(pads_begin, {2, 4, 6});
            fullfill_pads(pads_end, {3, 5, 7});
        } else {
            // For NCDHW layout, explicit paddings has the following form:
            // [0, 0, 0, 0, pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2]
            fullfill_pads(pads_begin, {4, 6, 8});
            fullfill_pads(pads_end, {5, 7, 9});
        }
    }
}

ov::OutputVector ov::frontend::tensorflow::translate_convolution_op(const ov::frontend::tensorflow::NodeContext& node,
                                                                    size_t spatial_dims_num) {
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dims_num == 2 || spatial_dims_num == 3,
                             "Conv2D or Conv3D are supported only.");
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 2, "Convolution must have at least two inputs.");
    auto input = node.get_input(0);
    auto filter = node.get_input(1);

    // retrieve attributes for Conv2D
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    // retrieve optional attributes
    auto tf_data_format = node.get_attribute<std::string>("data_format", spatial_dims_num == 2 ? "NHWC" : "NDHWC");
    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }
    std::vector<int64_t> dilation_2d = {1, 1, 1, 1};
    std::vector<int64_t> dilation_3d = {1, 1, 1, 1, 1};
    auto tf_dilations =
        node.get_attribute<std::vector<int64_t>>("dilations", spatial_dims_num == 2 ? dilation_2d : dilation_3d);

    bool is_nhwc = true;
    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NHWC" || tf_data_format == "NCHW",
                                 "Conv2D data format is neither NHWC nor NCHW");
        is_nhwc = (tf_data_format == "NHWC");
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
                                 "Conv3D data format is neither NDHWC nor NCDHW");
        is_nhwc = (tf_data_format == "NDHWC");
    }

    // prepare attributes for OpenVINO Convolution operation
    ov::Strides strides(spatial_dims_num);
    ov::Strides dilations(spatial_dims_num);
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_dilations, dilations);

    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num, tf_explicit_paddings, pads_begin, pads_end);
    }

    // prepare inputs to Convolution
    ov::frontend::tensorflow::convert_nhwc_to_nchw(is_nhwc, input);
    ov::AxisVector permutation_2d = {3, 2, 0, 1};
    ov::AxisVector permutation_3d = {4, 3, 0, 1, 2};
    filter = ov::frontend::tensorflow::make_transpose(filter, spatial_dims_num == 2 ? permutation_2d : permutation_3d);

    ov::Output<ov::Node> conv =
        std::make_shared<Convolution>(input, filter, strides, pads_begin, pads_end, dilations, auto_pad);

    ov::frontend::tensorflow::convert_nchw_to_nhwc(is_nhwc, conv);
    ov::frontend::tensorflow::set_node_name(node.get_name(), conv.get_node_shared_ptr());
    return {conv};
}

bool ov::frontend::tensorflow::is_conditional_edge(const std::string& input_tensor_name) {
    return input_tensor_name.length() > 0 && input_tensor_name[0] == '^';
}
