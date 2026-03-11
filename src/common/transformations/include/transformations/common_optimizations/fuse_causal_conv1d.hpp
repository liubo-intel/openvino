// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// clang-format off
/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses unified causal-conv1d subgraph into internal CausalConv1D op.
 *
 * Following graph (simplified):
 *
 *   ReadValue(cache) -> (optional Gather)
 *               \                     \
 *                \                     +--> Concat(cache, hidden) ---> GroupConvolution ---> Slice ---> (optional Add bias) ---> conv_out
 *                 \                                                        \
 *                  \                                                        +--> Slice(last state_len) ---> (optional Broadcast) ---> Assign
 *                   +-----------------------------------------------------------------------------------------------^
 *
 * is transformed to:
 *
 *   hidden, cache, weight, (bias) -> CausalConv1D -> {conv_out, new_state}
 *                                                            |
 *                                                          Assign
 */
// clang-format on
class TRANSFORMATIONS_API CausalConv1DFusion : public ov::pass::MatcherPass {
public:
    CausalConv1DFusion();
};

}  // namespace ov::pass
