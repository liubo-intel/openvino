// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softsign.hpp"

#include "unary_ops.hpp"

using Types = ::testing::Types<UnaryOperatorType<ov::op::v9::SoftSign, ov::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_atrribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);
