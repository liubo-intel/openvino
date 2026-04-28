// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_causal_conv1d.hpp"

namespace ov::test {

TEST_P(PagedCausalConv1DLayerTest, Inference) {
    run();
}

// PagedCausalConv1DLayerParams fields:
//   hidden_size, kernel_size, has_bias,
//   seq_lengths_sets (multiple sets for dynamic shapes),
//   cache_intervals_sets (matching intervals),
//   element_type, target_device

std::vector<PagedCausalConv1DLayerParams> gpu_paged_conv1d_test_cases = {
    // --- Static shape tests (single seq_lengths set) ---
    // Basic: single sequence, interval=1
    {1, 3, true, {{2}}, {{1}}, ov::element::f32, "GPU"},
    // Single sequence, interval=0 (only final flush)
    {1, 3, true, {{2}}, {{0}}, ov::element::f32, "GPU"},
    // Multiple sequences
    {1, 2, true, {{2, 1}}, {{1, 1}}, ov::element::f32, "GPU"},
    // Kernel size 4 with bias
    {1, 4, true, {{3}}, {{1}}, ov::element::f32, "GPU"},
    // No bias
    {1, 2, false, {{2}}, {{2}}, ov::element::f32, "GPU"},
    // Larger hidden size
    {8, 3, true, {{3, 2}}, {{2, 3}}, ov::element::f32, "GPU"},
    {64, 4, true, {{4, 2, 3}}, {{3, 2, 5}}, ov::element::f32, "GPU"},
    // past_lens offset affecting flush schedule
    {1, 2, false, {{3}}, {{2}}, ov::element::f32, "GPU"},
    // Large sequence lengths
    {32, 4, true, {{15, 32, 33}}, {{16, 16, 16}}, ov::element::f32, "GPU"},
    {128, 3, true, {{15, 32, 33}}, {{16, 16, 16}}, ov::element::f32, "GPU"},

    // --- Dynamic shape tests (multiple seq_lengths sets per test case) ---
    // Two iterations with different token counts and sequence counts
    {1, 3, true, {{2}, {3, 1}}, {{1}, {1, 0}}, ov::element::f32, "GPU"},
    // Different number of sequences between iterations
    {8, 3, true, {{3, 2}, {5, 1, 3}}, {{2, 3}, {1, 0, 2}}, ov::element::f32, "GPU"},
    // Varying total tokens significantly
    {64, 4, true, {{4, 2}, {15, 32, 33}}, {{3, 2}, {16, 16, 16}}, ov::element::f32, "GPU"},
    // Three iterations with progressively different shapes
    {32, 3, true, {{2}, {5, 3}, {1, 1, 1, 1}}, {{1}, {2, 0}, {1, 1, 1, 1}}, ov::element::f32, "GPU"},
    // Dynamic with no bias
    {8, 4, false, {{3}, {2, 2}}, {{1}, {1, 1}}, ov::element::f32, "GPU"},

    // --- f16 tests (static) ---
    {1, 3, true, {{2}}, {{1}}, ov::element::f16, "GPU"},
    {8, 3, true, {{3, 2}}, {{2, 3}}, ov::element::f16, "GPU"},
    {64, 4, true, {{4, 2, 3}}, {{3, 2, 5}}, ov::element::f16, "GPU"},
    {128, 3, true, {{15, 32, 33}}, {{16, 16, 16}}, ov::element::f16, "GPU"},

    // --- f16 tests with dynamic shapes ---
    {1, 3, true, {{2}, {3, 1}}, {{1}, {1, 0}}, ov::element::f16, "GPU"},
    {8, 3, true, {{3, 2}, {5, 1, 3}}, {{2, 3}, {1, 0, 2}}, ov::element::f16, "GPU"},
    {64, 4, true, {{4, 2}, {15, 32, 33}}, {{3, 2}, {16, 16, 16}}, ov::element::f16, "GPU"},
    {32, 3, true, {{2}, {5, 3}, {1, 1, 1, 1}}, {{1}, {2, 0}, {1, 1, 1, 1}}, ov::element::f16, "GPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedCausalConv1D,
                         PagedCausalConv1DLayerTest,
                         ::testing::ValuesIn(gpu_paged_conv1d_test_cases),
                         PagedCausalConv1DLayerTest::getTestCaseName);

}  // namespace ov::test
