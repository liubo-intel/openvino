// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include "openvino/op/util/variable.hpp"
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

    causal_conv1d(const primitive_id& id,
                  const std::vector<input_info>& inputs,
              const ov::op::util::VariableInfo& variable_info = {},
              bool transposed_io = false)
        : primitive_base(id, inputs),
          variable_info(variable_info),
          transposed_io(transposed_io) {}

    ov::op::util::VariableInfo variable_info;
        bool transposed_io = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, std::hash<std::string>()(variable_info.variable_id));
        seed = hash_combine(seed, variable_info.data_type.hash());
        seed = hash_combine(seed, transposed_io);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const causal_conv1d>(rhs);
        return variable_info == rhs_casted.variable_info && transposed_io == rhs_casted.transposed_io;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<causal_conv1d>::save(ob);
        ov::element::Type_t data_type = variable_info.data_type;
        ob << variable_info.variable_id;
        ob << variable_info.data_shape;
        ob << make_data(&data_type, sizeof(ov::element::Type_t));
        ob << transposed_io;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<causal_conv1d>::load(ib);
        ov::PartialShape data_shape;
        ov::element::Type_t data_type = ov::element::Type_t::dynamic;
        std::string variable_id;
        ib >> variable_id;
        ib >> data_shape;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        ib >> transposed_io;
        variable_info = {data_shape, data_type, variable_id};
    }
};

}  // namespace cldnn
