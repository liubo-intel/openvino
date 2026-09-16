// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/dynamic_quant/dynamic_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "cpu_parallel.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"

namespace {

using ov::Extensions::Cpu::XARCH::quant_src_grouped_i8;

// Scalar reference matching the kernel's own math: scale = max_abs(group) / 127, symmetric,
// round-to-nearest, clamp to [-128, 127], zero-scale fallback 1e-4.
template <typename T>
void referenceQuantize(const std::vector<T>& src, size_t M, size_t K, size_t groupSize,
                       std::vector<int8_t>& qdst, std::vector<float>& scales) {
    const size_t numGroups = K / groupSize;
    qdst.resize(M * K);
    scales.resize(M * numGroups);
    for (size_t m = 0; m < M; m++) {
        for (size_t g = 0; g < numGroups; g++) {
            float maxAbs = 0.0F;
            for (size_t k = 0; k < groupSize; k++) {
                maxAbs = std::max(maxAbs, std::abs(static_cast<float>(src[m * K + g * groupSize + k])));
            }
            float scale = maxAbs / 127.0F;
            if (scale == 0.0F) {
                scale = 0.0001F;
            }
            scales[m * numGroups + g] = scale;
            for (size_t k = 0; k < groupSize; k++) {
                float v = static_cast<float>(src[m * K + g * groupSize + k]) / scale;
                auto q = static_cast<int>(std::round(v));
                q = std::max(q, -128);
                q = std::min(q, 127);
                qdst[m * K + g * groupSize + k] = static_cast<int8_t>(q);
            }
        }
    }
}

template <typename T>
void fillRandom(std::vector<T>& data, float lo, float hi, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : data) {
        v = static_cast<T>(dist(gen));
    }
}

struct DynamicQuantParams {
    ov::element::Type_t precision;
    size_t M;
    size_t K;
    size_t groupSize;
};

class DynamicQuantKernelTest : public ::testing::TestWithParam<DynamicQuantParams> {};

template <typename T>
void runCase(size_t M, size_t K, size_t groupSize, ov::element::Type_t precision) {
    std::vector<T> src(M * K);
    fillRandom(src, -10.0F, 10.0F, 12345);

    std::vector<int8_t> refQdst;
    std::vector<float> refScales;
    referenceQuantize(src, M, K, groupSize, refQdst, refScales);

    std::vector<int8_t> qdst(M * K, 0);
    std::vector<float> scales(M * (K / groupSize), 0.0F);
    auto cpuParallel = std::make_shared<ov::intel_cpu::CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);

    quant_src_grouped_i8(precision, src.data(), qdst.data(), scales.data(), M, K, groupSize, cpuParallel);

    for (size_t i = 0; i < scales.size(); i++) {
        EXPECT_NEAR(scales[i], refScales[i], std::abs(refScales[i]) * 1e-5F + 1e-8F) << "scale mismatch at " << i;
    }
    for (size_t i = 0; i < qdst.size(); i++) {
        EXPECT_EQ(qdst[i], refQdst[i]) << "qdst mismatch at " << i;
    }
}

TEST_P(DynamicQuantKernelTest, MatchesReference) {
    const auto& p = GetParam();
    if (p.precision == ov::element::Type_t::bf16) {
        runCase<ov::bfloat16>(p.M, p.K, p.groupSize, p.precision);
    } else {
        runCase<float>(p.M, p.K, p.groupSize, p.precision);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Precisions,
    DynamicQuantKernelTest,
    ::testing::Values(DynamicQuantParams{ov::element::Type_t::f32, 1, 16, 16},
                      DynamicQuantParams{ov::element::Type_t::bf16, 1, 16, 16},
                      DynamicQuantParams{ov::element::Type_t::f32, 7, 32, 16},
                      DynamicQuantParams{ov::element::Type_t::bf16, 7, 32, 16},
                      DynamicQuantParams{ov::element::Type_t::f32, 16, 128, 32},
                      DynamicQuantParams{ov::element::Type_t::bf16, 33, 256, 64},
                      DynamicQuantParams{ov::element::Type_t::f32, 128, 512, 128},
                      DynamicQuantParams{ov::element::Type_t::f32, 1, 4096, 128},
                      DynamicQuantParams{ov::element::Type_t::f32, 4, 4096, 256}));

TEST(DynamicQuantKernelTest, AllZeroGroupUsesFallbackScale) {
    const size_t M = 2;
    const size_t K = 16;
    const size_t groupSize = 16;
    std::vector<float> src(M * K, 0.0F);
    std::vector<int8_t> qdst(M * K, 0);
    std::vector<float> scales(M * (K / groupSize), 0.0F);
    auto cpuParallel = std::make_shared<ov::intel_cpu::CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);

    quant_src_grouped_i8(ov::element::Type_t::f32, src.data(), qdst.data(), scales.data(), M, K, groupSize,
                         cpuParallel);

    for (const auto s : scales) {
        EXPECT_FLOAT_EQ(s, 0.0001F);
    }
    for (const auto q : qdst) {
        EXPECT_EQ(q, 0);
    }
}

TEST(DynamicQuantKernelTest, ClampsToInt8Range) {
    // A single huge outlier in an otherwise-small group must clamp to [-128, 127], not overflow.
    const size_t M = 1;
    const size_t K = 16;
    const size_t groupSize = 16;
    std::vector<float> src(M * K, 1.0F);
    src[0] = -1000.0F;
    src[K - 1] = 1000.0F;
    std::vector<int8_t> qdst(M * K, 0);
    std::vector<float> scales(M * (K / groupSize), 0.0F);
    auto cpuParallel = std::make_shared<ov::intel_cpu::CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);

    quant_src_grouped_i8(ov::element::Type_t::f32, src.data(), qdst.data(), scales.data(), M, K, groupSize,
                         cpuParallel);

    EXPECT_LE(qdst[0], -127);
    EXPECT_GE(qdst[K - 1], 126);
    for (const auto q : qdst) {
        EXPECT_GE(q, -128);
        EXPECT_LE(q, 127);
    }
}

}  // namespace
