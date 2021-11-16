// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/prior_box.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using PriorBoxAttrs = ov::op::v0::PriorBox::Attributes;
using ov::op::v0::PriorBox;
}  // namespace v0
namespace v8 {
using PriorBoxAttrs = ov::op::v8::PriorBox::Attributes;
using ov::op::v8::PriorBox;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
