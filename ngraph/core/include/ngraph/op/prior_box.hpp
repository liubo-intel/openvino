// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/prior_box.hpp"

namespace ngraph {
namespace op {
using PriorBoxAttrs = ov::op::v0::PriorBox::Attributes;  // for plugin default use
namespace v0 {
using PriorBoxAttrs = ov::op::v0::PriorBox::Attributes;
using ov::op::v0::PriorBox;
}  // namespace v0
namespace v8 {
using PriorBoxAttrs = ov::op::v8::PriorBox::Attributes;
using ov::op::v8::PriorBox;
}  // namespace v8
using v0::PriorBox;  // for plugin default use
}  // namespace op
}  // namespace ngraph
