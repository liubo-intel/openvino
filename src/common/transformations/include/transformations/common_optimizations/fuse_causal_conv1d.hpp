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
 * @brief This transformation fuses the LFM2 short conv1d cache subgraph into a single
 * CausalConv1D internal op.
 *
 * Following graph (simplified):
 *
 *   ReadValue(cache)   hidden_states(Bx)   weight   (bias)
 *        |                    |             |        |
 *        |                    |             |        |
 *        |                    +-------> GroupConvolution -> Slice/ReduceSum
 *        |                                   |               |
 *        |                                   |               +--> (+ bias) -> conv_out_dec
 *        |                                   |
 *        |                                   +--------------------------> conv_out_prefill
 *        |                    |
 *        |                    +--> Pad -------------------------------> conv_state_prefill
 *        |
 *        +--> Roll -> ScatterNDUpdate( cache_position ) --------------> conv_state_dec
 *
 *   is_decoding = (seq_len == 1)
 *   conv_out  = conv_out_dec * is_decoding + conv_out_prefill * (1 - is_decoding)
 *   new_state = conv_state_dec * is_decoding + conv_state_prefill * (1 - is_decoding)
 *   Assign(new_state)
 *
 * is transformed to:
 *
 *   ReadValue(cache)   hidden_states(Bx)   weight   cache_position   (bias)
 *        |                    |             |           |            |
 *        +--------------------+-------------+-----------+------------+
 *                                CausalConv1D
 *                               /          \
 *                          conv_out     new_state
 *                                            |
 *                                         Assign
 */
// clang-format on
class TRANSFORMATIONS_API CausalConv1DFusion : public ov::pass::MatcherPass {
public:
    CausalConv1DFusion();
};

}  // namespace ov::pass
