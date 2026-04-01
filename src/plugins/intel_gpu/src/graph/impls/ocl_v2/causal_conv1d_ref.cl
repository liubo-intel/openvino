// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// CausalConv1D kernel for internal op ov::op::internal::CausalConv1D
//
// Inputs:
//   INPUT0 (input_embeds):
//     - TRANSPOSED_IO=0: [B, HIDDEN_SIZE, SEQ_LEN]
//     - TRANSPOSED_IO=1: [B, SEQ_LEN, HIDDEN_SIZE]
//   INPUT1 (conv_state):   [B, HIDDEN_SIZE, CACHE_LEN]
//   INPUT2 (conv_weight):  [HIDDEN_SIZE, 1, KERNEL_SIZE] or [HIDDEN_SIZE, 1, 1, KERNEL_SIZE]
//   INPUT3 (conv_bias):    [HIDDEN_SIZE] (optional)
//
// Outputs:
//   OUTPUT  (output_embeds):
//     - TRANSPOSED_IO=0: [B, HIDDEN_SIZE, SEQ_LEN]
//     - TRANSPOSED_IO=1: [B, SEQ_LEN, HIDDEN_SIZE]
//   OUTPUT1 (output_conv_state): [B, HIDDEN_SIZE, CACHE_LEN]
//
// Dispatch: global = {batch, hidden_size, 1}, local = {1, WG_SIZE, 1}

KERNEL(causal_conv1d_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input_embeds,
    __global INPUT1_TYPE* conv_state,
    __global INPUT2_TYPE* conv_weight,
#if HAS_BIAS
    __global INPUT3_TYPE* conv_bias,
#endif
    __global OUTPUT_TYPE* output_embeds,
    __global OUTPUT1_TYPE* output_conv_state,
    int seq_len
)
{
    const int b = get_global_id(0);
    const int ch = get_global_id(1);

    if (ch >= HIDDEN_SIZE)
        return;

    float w[KERNEL_SIZE];
    for (int k = 0; k < KERNEL_SIZE; k++) {
        const int w_idx = INPUT2_GET_INDEX(ch, 0, 0, k);
        w[k] = convert_float(conv_weight[w_idx]);
    }

    float bias = 0.0f;
#if HAS_BIAS
    const int bias_idx = INPUT3_GET_INDEX(ch, 0, 0, 0);
    bias = convert_float(conv_bias[bias_idx]);
#endif

    const int concat_len = CACHE_LEN + seq_len;
    const int full_out_len = concat_len - KERNEL_SIZE + 1;
    const int out_start = full_out_len - seq_len;

    // output_embeds = conv1d(concat(conv_state, input_embeds))[:, :, -seq_len:]
    // Keep local cache state and update it with x_new at each step.
    float state[KERNEL_SIZE];
    for (int k = 0; k < KERNEL_SIZE; k++) {
        const int state_idx = INPUT1_GET_INDEX(b, ch, k, 0);
        state[k] = convert_float(conv_state[state_idx]);
    }

    for (int t = 0; t < seq_len; t++) {
#if TRANSPOSED_IO
        const int in_idx = INPUT0_GET_INDEX(b, t, ch, 0);
#else
        const int in_idx = INPUT0_GET_INDEX(b, ch, t, 0);
#endif
        const float x_new = convert_float(input_embeds[in_idx]);

        float sum = bias;
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            sum += state[k + 1] * w[k];
        sum += x_new * w[KERNEL_SIZE - 1];

    #if TRANSPOSED_IO
        const int out_idx = OUTPUT_GET_INDEX(b, t, ch, 0);
    #else
        const int out_idx = OUTPUT_GET_INDEX(b, ch, t, 0);
    #endif
        output_embeds[out_idx] = TO_OUTPUT_TYPE(sum);

        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            state[k] = state[k + 1];
        state[KERNEL_SIZE - 1] = x_new;
    }

    // Under CACHE_LEN == KERNEL_SIZE restriction, local state matches updated conv cache.
    for (int s = 0; s < CACHE_LEN; s++) {
        const int out_state_idx = OUTPUT1_GET_INDEX(b, ch, s, 0);
        output_conv_state[out_state_idx] = TO_OUTPUT1_TYPE(state[s]);
    }
}
