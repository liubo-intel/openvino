// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs write_to_array(const NodeContext& node) {
    // OV not support array dtype, workaroud for len(array)=1
    auto data = node.get_ng_input("X");
    return node.default_single_output_mapping({data.get_node_shared_ptr()}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov