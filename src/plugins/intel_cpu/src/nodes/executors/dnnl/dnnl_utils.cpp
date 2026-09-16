// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/reorder.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "thread_pool_imp.hpp"
#include "utils/general_utils.h"
#include "weights_cache.hpp"

namespace ov::intel_cpu::utils {

namespace {

// Returns the tensor's dim indices ordered from outermost (largest stride) to
// innermost (smallest stride), i.e. the same permutation a format tag encodes.
std::vector<size_t> physicalDimOrder(const dnnl::memory::dims& strides) {
    std::vector<size_t> order(strides.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return strides[a] > strides[b];
    });
    return order;
}

// True if dst's physical layout is exactly src's with the last two (logical) dims
// transposed and everything else unchanged - the specific pattern
// DnnlMatMulPrimitive::makeTransposedWeightDescriptor() can produce for an FC weight
// (e.g. "acb" vs the primitive-requested "abc") when the activation/output carry a
// leading batch dim that the (always at-most-2D) weight constant doesn't have.
bool isLastTwoDimsTransposed(const dnnl::memory::desc& src, const dnnl::memory::desc& dst) {
    const auto dims = src.get_dims();
    if (dims != dst.get_dims() || dims.size() < 2) {
        return false;
    }
    auto srcOrder = physicalDimOrder(src.get_strides());
    const auto dstOrder = physicalDimOrder(dst.get_strides());
    std::swap(srcOrder[dims.size() - 1], srcOrder[dims.size() - 2]);
    return srcOrder == dstOrder;
}

// oneDNN/OV generic reorder has no kernel to permute sub-byte (u4/i4, 2-per-byte
// packed) data - element boundaries don't align with byte boundaries once
// transposed. Do the nibble-level transpose by hand for the "last two dims
// transposed" case (see isLastTwoDimsTransposed() above).
//
// Both src and dst are addressed via their *actual* strides (in elements, not
// bytes) rather than assumed to be densely row-major: makeTransposedWeightDescriptor()
// builds `src` as a zero-copy reinterpretation of the constant's original flat
// buffer (non-trivial strides, no data movement), while `dst` is oneDNN's
// requested (dense) layout - only the addressing differs, so a naive
// row-major `r * cols + c` index is wrong for src.
void transposeSubByteLastTwoDims(const uint8_t* src,
                                 uint8_t* dst,
                                 const dnnl::memory::dims& dims,
                                 const dnnl::memory::dims& srcStrides,
                                 const dnnl::memory::dims& dstStrides,
                                 size_t srcBufBytes,
                                 size_t dstBufBytes) {
    const size_t rank = dims.size();
    OPENVINO_ASSERT(srcStrides.size() == rank && dstStrides.size() == rank,
                    "transposeSubByteLastTwoDims: dims/strides rank mismatch");
    const auto rows = static_cast<size_t>(dims[rank - 2]);
    const auto cols = static_cast<size_t>(dims[rank - 1]);

    size_t batch = 1;
    for (size_t i = 0; i + 2 < rank; i++) {
        batch *= static_cast<size_t>(dims[i]);
    }

    size_t totalElems = batch * rows * cols;

    for (size_t b = 0; b < batch; b++) {
        // decompose the linear batch index into per-leading-dim indices to compute
        // the base element offset for this slice on both sides
        size_t rem = b;
        int64_t srcBase = 0;
        int64_t dstBase = 0;
        for (size_t i = rank - 2; i-- > 0;) {
            const auto extent = static_cast<size_t>(dims[i]);
            const size_t idx = rem % extent;
            rem /= extent;
            srcBase += static_cast<int64_t>(idx) * srcStrides[i];
            dstBase += static_cast<int64_t>(idx) * dstStrides[i];
        }

        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                const int64_t srcElem =
                    srcBase + (static_cast<int64_t>(r) * srcStrides[rank - 2]) + (static_cast<int64_t>(c) * srcStrides[rank - 1]);
                const int64_t dstElem =
                    dstBase + (static_cast<int64_t>(r) * dstStrides[rank - 2]) + (static_cast<int64_t>(c) * dstStrides[rank - 1]);
                OPENVINO_ASSERT(srcElem >= 0 && static_cast<size_t>(srcElem) < totalElems &&
                                     static_cast<size_t>(srcElem) / 2 < srcBufBytes,
                                "transposeSubByteLastTwoDims: srcElem out of range: ",
                                srcElem,
                                " totalElems=",
                                totalElems,
                                " srcBufBytes=",
                                srcBufBytes);
                OPENVINO_ASSERT(dstElem >= 0 && static_cast<size_t>(dstElem) < totalElems &&
                                     static_cast<size_t>(dstElem) / 2 < dstBufBytes,
                                "transposeSubByteLastTwoDims: dstElem out of range: ",
                                dstElem,
                                " totalElems=",
                                totalElems,
                                " dstBufBytes=",
                                dstBufBytes);
                const uint8_t nibble = (src[srcElem / 2] >> ((srcElem % 2) * 4)) & 0x0FU;
                const uint8_t shift = (dstElem % 2) * 4;
                dst[dstElem / 2] = (dst[dstElem / 2] & ~(0x0FU << shift)) | (nibble << shift);
            }
        }
    }
}

}  // namespace

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const ExecutorContext::CPtr& context,
                               const bool needShiftSignedToUnsigned) {
    const auto privateWeightCache = context->getPrivateWeightCache();
    OPENVINO_ASSERT(privateWeightCache, "privateWeightCache is nullptr");

    return prepareWeightsMemory(srcWeightDesc,
                                dstWeightDesc,
                                weightsMem,
                                context->getEngine(),
                                context->getRuntimeCache(),
                                context->getWeightsCache(),
                                privateWeightCache,
                                context->getThreadPool(),
                                needShiftSignedToUnsigned);
}

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const dnnl::engine& eng,
                               const MultiCachePtr& rtCache,
                               const WeightsSharing::Ptr& globalWeightCache,
                               const std::shared_ptr<std::unordered_map<std::string, MemoryPtr>>& privateWeightCache,
                               const std::shared_ptr<ThreadPool>& threadPool,
                               bool needShiftSignedToUnsigned) {
    const auto format = dstWeightDesc->serializeFormat();
    if (privateWeightCache) {
        auto itr = privateWeightCache->find(format);
        if (privateWeightCache->end() != itr) {
            return itr->second;
        }
    }

    auto create = [&]() {
        // https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html?highlight=128#inputs-of-the-same-type-s8
        auto src_wdt = srcWeightDesc->getPrecision();
        auto dst_wdt = dstWeightDesc->getPrecision();
        if (needShiftSignedToUnsigned && src_wdt.is_integral_number() && src_wdt.is_signed() &&
            dst_wdt.is_integral_number() && !dst_wdt.is_signed()) {
            assert(src_wdt.bitwidth() == dst_wdt.bitwidth());

            // prevent reorderData from doing conversion
            Memory srcMemory{eng, srcWeightDesc->cloneWithNewPrecision(dst_wdt), weightsMem->getData()};
            MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
            node::Reorder::reorderData(srcMemory, *_ptr, rtCache, threadPool);

            // do shift
            auto count = _ptr->getSize() / _ptr->getDesc().getPrecision().size();
            if (dst_wdt == ov::element::u8) {
                auto* data = _ptr->getDataAs<uint8_t>();
                for (size_t i = 0; i < count; i++) {
                    data[i] = data[i] + 128;
                }
            } else if (dst_wdt == ov::element::u4) {
                auto* data = _ptr->getDataAs<uint8_t>();
                for (size_t i = 0; i < count; i++) {
                    auto low = (data[i] & 0xF) + 8;
                    auto high = (data[i] >> 4) + 8;
                    data[i] = (high << 4) | (low & 0xF);
                }
            } else {
                OPENVINO_THROW("Unsupported data type for shiftting sign to unsign");
            }
            return _ptr;
        }

        if (src_wdt == dst_wdt && any_of(src_wdt, ov::element::u4, ov::element::i4) &&
            isLastTwoDimsTransposed(srcWeightDesc->getDnnlDesc(), dstWeightDesc->getDnnlDesc())) {
            MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
            std::memset(_ptr->getData(), 0, _ptr->getSize());
            transposeSubByteLastTwoDims(static_cast<const uint8_t*>(weightsMem->getData()),
                                        _ptr->getDataAs<uint8_t>(),
                                        srcWeightDesc->getDnnlDesc().get_dims(),
                                        srcWeightDesc->getDnnlDesc().get_strides(),
                                        dstWeightDesc->getDnnlDesc().get_strides(),
                                        weightsMem->getSize(),
                                        _ptr->getSize());
            return _ptr;
        }

        Memory srcMemory{eng, srcWeightDesc, weightsMem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
        node::Reorder::reorderData(srcMemory, *_ptr, rtCache, threadPool);

        return _ptr;
    };

    MemoryPtr ptr;
    if (globalWeightCache && dnnl::memory::format_kind::blocked == dstWeightDesc->getDnnlDesc().get_format_kind()) {
        ptr = MemoryPtr(
            *globalWeightCache->findOrCreate(DnnlExtensionUtils::computeWeightsStringHash(weightsMem, dstWeightDesc),
                                             create));
    } else {
        ptr = create();
    }

    if (privateWeightCache) {
        (*privateWeightCache)[format] = ptr;
    }

    return ptr;
}

}  // namespace ov::intel_cpu::utils
