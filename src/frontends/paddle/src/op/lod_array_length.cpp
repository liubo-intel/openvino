// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs lod_array_length(const NodeContext& node) {
    // OV not support array dtype, workaroud for len(array)=1
    return node.default_single_output_mapping({default_opset::Constant::create(element::i64, ov::Shape{}, {0})},
                                              {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov