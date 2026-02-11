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
 * @brief This transformation fuses the LFM2 short conv1d cache subgraph and the
 * qwen3_next linear-attn conv1d subgraph into a single CausalConv1D internal op.
 *
 * Following graph (simplified, LFM2):
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
 *
 * Following graph (simplified, qwen3_next):
 *
 *   ReadValue(cache)   hidden_states(mixed_qkv)   weight   (bias)   Swish
 *        |                    |             |        |        |
 *        |                    |             |        |        |
 *        |                    +-------> GroupConvolution ----> Swish -> conv_out_prefill
 *        |                                   |
 *        |                                   +--------------------------> conv_out_prefill (pre-activation)
 *        |                    |
 *        |                    +--> Pad -------------------------------> conv_state_prefill
 *        |
 *        +--> concat(cache, hidden) -> GroupConvolution -> Swish ----> conv_out_dec
 *
 *   is_decoding = (seq_len == 1)
 *   conv_out  = conv_out_dec * is_decoding + conv_out_prefill * (1 - is_decoding)
 *   new_state = conv_state_dec * is_decoding + conv_state_prefill * (1 - is_decoding)
 *   Assign(new_state)
 *
 * is transformed to:
 *
 *   ReadValue(cache)   hidden_states(mixed_qkv)   weight   cache_position*  (bias)  activation
 *        |                    |             |           |             |         |
 *        +--------------------+-------------+-----------+-------------+---------+
 *                                CausalConv1D
 *                               /          \
 *                          conv_out     new_state
 *                                            |
 *                                         Assign
 *
 *   * cache_position is optional for qwen3_next (placeholder -1 is used to indicate "no cache_position").
 */
// clang-format on
class TRANSFORMATIONS_API CausalConv1DFusion : public ov::pass::MatcherPass {
public:
    CausalConv1DFusion();
};

}  // namespace ov::pass
