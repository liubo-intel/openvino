// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/causal_conv1d.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/causal_conv1d.hpp"

namespace ov {
namespace op {
namespace internal {
using CausalConv1D = ov::op::internal::CausalConv1D;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateCausalConv1DOp(ProgramBuilder& p, const std::shared_ptr<op::internal::CausalConv1D>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);

    cldnn::causal_conv1d prim(layer_type_name_ID(op), inputs);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, CausalConv1D);

}  // namespace ov::intel_gpu
