// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/lstm_cell.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LSTMCellTest;
using ov::test::utils::InputLayerType;

std::vector<bool> should_decompose{true};
std::vector<size_t> batch{5};
std::vector<size_t> hidden_size{1};
std::vector<size_t> input_size{1};
std::vector<float> clip{0.f};

std::vector<std::vector<std::string>> activations = {{"relu", "relu", "relu"}};

std::vector<InputLayerType> layer_types = {
    InputLayerType::CONSTANT
};

std::vector<ov::element::Type> model_types = {
    ov::element::f16};
    // ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellCommon, LSTMCellTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(layer_types),
                                ::testing::ValuesIn(layer_types),
                                ::testing::ValuesIn(layer_types),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LSTMCellTest::getTestCaseName);

}  // namespace
