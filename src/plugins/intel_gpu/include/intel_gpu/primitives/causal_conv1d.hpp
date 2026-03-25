// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include "ov_ops/causal_conv1d.hpp"

namespace cldnn {

using CausalConv1D = ov::op::internal::CausalConv1D;

/// @brief causal_conv1d primitive
/// @details Implements internal CausalConv1D op with inputs:
///          0: input_embeds [B, C, T]
///          1: conv_state   [B, C, S]
///          2: conv_weight  [C, 1, L] or [C, 1, 1, L]
///          3: conv_bias    [C] (optional)
///          outputs:
///          0: output_embeds     [B, C, T]
///          1: output_conv_state [B, C, S]
struct causal_conv1d : public primitive_base<causal_conv1d> {
    CLDNN_DECLARE_PRIMITIVE(causal_conv1d)

    causal_conv1d() : primitive_base("", {}) {}

    causal_conv1d(const primitive_id& id, const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<causal_conv1d>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<causal_conv1d>::load(ib);
    }
};

}  // namespace cldnn
