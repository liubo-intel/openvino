// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs tensor_array_to_tensor(const NodeContext& node) {
    // OV not support array dtype, workaroud for len(array)=1
    auto data = node.get_ng_input("X");
    auto tensor_node = data.get_node_shared_ptr();
    // TODO: FasterRCNN/MaskRCNN Not used 'OutIndex' output, just set Constant for simple debug
    auto outindex_node = default_opset::Constant::create(element::i32, ov::Shape{}, {1});

    NamedOutputs outputs;
    outputs["Out"] = {tensor_node};
    outputs["OutIndex"] = {outindex_node};

    return outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov