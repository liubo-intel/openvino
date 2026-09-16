// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Symmetric s8 quantization of a row-major [M, K] f32/bf16 tensor, grouped along the innermost
// (K) dimension: scale = max_abs(group) / 127, no zero-point. Used by the AMX int8
// grouped-quantization dynamic-quant matmul path (DnnlExecutor's m_dynQuant branch) to quantize
// activations at runtime before feeding them to a dnnl::matmul primitive whose src is s8.
//
// qdst:   [M, K], s8, contiguous.
// scales: [M, K / group_size], f32, contiguous.
// K must be a multiple of group_size (checked by the caller).
void quant_src_grouped_i8(ov::element::Type_t srcPrecision,
                          const void* src,
                          int8_t* qdst,
                          float* scales,
                          size_t M,
                          size_t K,
                          size_t group_size,
                          const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
