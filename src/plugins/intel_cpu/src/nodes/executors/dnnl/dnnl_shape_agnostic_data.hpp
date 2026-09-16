// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct DnnlShapeAgnosticData {
    explicit DnnlShapeAgnosticData(DnnlPrimitiveAttrs primAttrs,
                                   impl_desc_type implType = impl_desc_type::undef,
                                   bool dynQuant = false,
                                   uint64_t dynQuantGroupSize = 0)
        : m_primAttrs(std::move(primAttrs)),
          m_implType(implType),
          m_dynQuant(dynQuant),
          m_dynQuantGroupSize(dynQuantGroupSize) {}

    DnnlPrimitiveAttrs m_primAttrs;
    // implementation type is a part of shape agnostic data to allow using
    // the same implementation for different shapes to avoid dealing with
    // multiple packed weights based on different implementations even it
    // may be not optimal from a performance perspective
    impl_desc_type m_implType;
    // AMX int8 grouped-quantization dynamic-quant path (DnnlMatMulPrimitive only): whether the
    // primitive's src is s8 (quantized internally by the executor) rather than the node's own
    // f32/bf16 ARG_SRC, and the K-group size used for that quantization. Inert (false/0) for
    // every other primitive/executor.
    bool m_dynQuant;
    uint64_t m_dynQuantGroupSize;
};

using DnnlShapeAgnosticDataPtr = std::shared_ptr<DnnlShapeAgnosticData>;

}  // namespace ov::intel_cpu
