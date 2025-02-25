// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conv_2d_backprop_input_op(const NodeContext& node) {
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 3, "Conv2DBackpropInput must have at least three inputs.");
    auto input_sizes = node.get_input(0);
    auto filter = node.get_input(1);
    auto out_backprop = node.get_input(2);

    // retrieve attributes for Conv2DBackpropInput
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    // retrieve optional attributes
    auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations", {1, 1, 1, 1});
    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }
    auto tf_data_format = node.get_attribute<std::string>("data_format", "NHWC");

    TENSORFLOW_OP_VALIDATION(node,
                             tf_data_format == "NHWC" || tf_data_format == "NCHW",
                             "Conv2DBackpropInput data format is neither NHWC nor NCHW");
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_explicit_paddings.size() == 8,
                                 "Conv2DBackpropInput expects 8 padding values for EXPLICIT padding mode.");
    }
    bool is_nhwc = (tf_data_format == "NHWC");

    // prepare attributes for OpenVINO ConvolutionBackpropData
    Strides strides(2);
    Strides dilations(2);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, dilations);

    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
        if (is_nhwc) {
            // For NHWC layout, explicit paddings has the following form:
            // [0, 0, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
            pads_begin.push_back(tf_explicit_paddings[2]);
            pads_begin.push_back(tf_explicit_paddings[4]);
            pads_end.push_back(tf_explicit_paddings[3]);
            pads_end.push_back(tf_explicit_paddings[5]);
        } else {
            // For NCHW layout, explicit paddings has the following form:
            // [0, 0, 0, 0, pad_h1, pad_h2, pad_w1, pad_w2]
            pads_begin.push_back(tf_explicit_paddings[4]);
            pads_begin.push_back(tf_explicit_paddings[6]);
            pads_end.push_back(tf_explicit_paddings[5]);
            pads_end.push_back(tf_explicit_paddings[7]);
        }
    }

    // prepare inputs to ConvolutionBackpropData
    filter = make_transpose(filter, {3, 2, 0, 1});
    convert_nhwc_to_nchw(is_nhwc, out_backprop);

    // initially think that output shape defined for NCHW layout
    auto ss_begin = make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto ss_end = make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto ss_strides = make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

    // change range of indices for spatial dimensions in case NHWC layout
    if (is_nhwc) {
        ss_begin = make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
        ss_end = make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{3});
    }

    auto spatial_shape = make_shared<StridedSlice>(input_sizes,
                                                   ss_begin,
                                                   ss_end,
                                                   ss_strides,
                                                   std::vector<int64_t>{},
                                                   std::vector<int64_t>{});

    auto conv_backprop = make_shared<ConvolutionBackpropData>(out_backprop,
                                                              filter,
                                                              spatial_shape,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              auto_pad);

    // insert Transpose only if original Conv2DBackpropInput is in NHWC layout
    auto conv_backprop_output = conv_backprop->output(0);
    convert_nchw_to_nhwc(is_nhwc, conv_backprop_output);

    // move the original name to new ConvolutionBackpropData if original layout is NCHW
    // move the original name to Transpose if original layout is NHWC
    set_node_name(node.get_name(), conv_backprop_output.get_node_shared_ptr());
    return {conv_backprop_output};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
