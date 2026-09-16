// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quant.hpp"

#include <cstddef>
#include <cstdint>

#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::XARCH {

template <typename T>
static void quant_src_grouped_i8_impl(const T* src,
                                      int8_t* qdst,
                                      float* scales,
                                      size_t M,
                                      size_t K,
                                      size_t group_size,
                                      const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const size_t numGroups = K / group_size;
    cpu_parallel->parallel_for(M, [&](size_t m) {
        const T* srcRow = src + m * K;
        int8_t* qdstRow = qdst + m * K;
        float* scaleRow = scales + m * numGroups;
        for (size_t g = 0; g < numGroups; g++) {
            quant_i8<T>(srcRow + g * group_size, qdstRow + g * group_size, group_size, scaleRow[g]);
        }
    });
}

void quant_src_grouped_i8(ov::element::Type_t srcPrecision,
                          const void* src,
                          int8_t* qdst,
                          float* scales,
                          size_t M,
                          size_t K,
                          size_t group_size,
                          const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    switch (srcPrecision) {
    case ov::element::Type_t::f32:
        quant_src_grouped_i8_impl(static_cast<const float*>(src), qdst, scales, M, K, group_size, cpu_parallel);
        break;
    case ov::element::Type_t::bf16:
        quant_src_grouped_i8_impl(static_cast<const ov::bfloat16*>(src), qdst, scales, M, K, group_size, cpu_parallel);
        break;
    default:
        OPENVINO_THROW("quant_src_grouped_i8: unsupported src precision ", srcPrecision);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
