// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// CausalConv1D kernel for internal op ov::op::internal::CausalConv1D
//
// Inputs:
//   INPUT0 (input_embeds): [B, HIDDEN_SIZE, SEQ_LEN]
//   INPUT1 (conv_state):   [B, HIDDEN_SIZE, CACHE_LEN]
//   INPUT2 (conv_weight):  [HIDDEN_SIZE, 1, KERNEL_SIZE] or [HIDDEN_SIZE, 1, 1, KERNEL_SIZE]
//   INPUT3 (conv_bias):    [HIDDEN_SIZE] (optional)
//
// Outputs:
//   OUTPUT  (output_embeds):     [B, HIDDEN_SIZE, SEQ_LEN]
//   OUTPUT1 (output_conv_state): [B, HIDDEN_SIZE, CACHE_LEN]
//
// Dispatch: global = {batch, hidden_size, 1}, local = {1, WG_SIZE, 1}

KERNEL(causal_conv1d_ref)(
    __global INPUT0_TYPE* input_embeds,
    __global INPUT1_TYPE* conv_state,
    __global INPUT2_TYPE* conv_weight,
#if HAS_BIAS
    __global INPUT3_TYPE* conv_bias,
#endif
    __global OUTPUT_TYPE* output_embeds,
    __global OUTPUT1_TYPE* output_conv_state,
    int seq_len)
{
    const int b = get_global_id(0);
    const int ch = get_global_id(1);

    if (ch >= HIDDEN_SIZE)
        return;

    const int cache_base = b * HIDDEN_SIZE * CACHE_LEN + ch * CACHE_LEN;
    const int hidden_base = b * HIDDEN_SIZE * seq_len + ch * seq_len;
    const int out_base = b * HIDDEN_SIZE * seq_len + ch * seq_len;
    const int state_base = b * HIDDEN_SIZE * CACHE_LEN + ch * CACHE_LEN;

    float w[KERNEL_SIZE];
    const int w_base = ch * KERNEL_SIZE;
    for (int k = 0; k < KERNEL_SIZE; k++)
        w[k] = convert_float(conv_weight[w_base + k]);

    float bias = 0.0f;
#if HAS_BIAS
    bias = convert_float(conv_bias[ch]);
#endif

    const int concat_len = CACHE_LEN + seq_len;
    const int full_out_len = concat_len - KERNEL_SIZE + 1;
    const int out_start = full_out_len - seq_len;

    // output_embeds = conv1d(concat(conv_state, input_embeds))[:, :, -seq_len:]
    for (int t = 0; t < seq_len; t++) {
        const int win_start = out_start + t;

        float sum = bias;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            const int idx = win_start + k;
            const float v = idx < CACHE_LEN
                                ? convert_float(conv_state[cache_base + idx])
                                : convert_float(input_embeds[hidden_base + (idx - CACHE_LEN)]);
            sum += v * w[k];
        }

        output_embeds[out_base + t] = TO_OUTPUT_TYPE(sum);
    }

    // output_conv_state = last CACHE_LEN values of concat(conv_state, input_embeds)
    for (int s = 0; s < CACHE_LEN; s++) {
        const int idx = concat_len - CACHE_LEN + s;
        const float v = idx < CACHE_LEN
                            ? convert_float(conv_state[cache_base + idx])
                            : convert_float(input_embeds[hidden_base + (idx - CACHE_LEN)]);
        output_conv_state[state_base + s] = TO_OUTPUT1_TYPE(v);
    }
}
