// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/causal_conv1d.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<causal_conv1d> : public typed_program_node_base<causal_conv1d> {
    using parent = typed_program_node_base<causal_conv1d>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using causal_conv1d_node = typed_program_node<causal_conv1d>;

template <>
class typed_primitive_inst<causal_conv1d> : public typed_primitive_inst_base<causal_conv1d> {
    using parent = typed_primitive_inst_base<causal_conv1d>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const causal_conv1d_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const causal_conv1d_node& node, const kernel_impl_params& impl_param);

    static std::string to_string(const causal_conv1d_node& node);

    typed_primitive_inst(network& network, const causal_conv1d_node& node);
};

using causal_conv1d_inst = typed_primitive_inst<causal_conv1d>;

}  // namespace cldnn
