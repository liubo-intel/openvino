// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
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
#include "openvino/core/parallel.hpp"
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
    // Only plain layouts: the strides of a blocked layout (e.g. the inner_product's AB16b32a4b) describe the outer
    // blocks, not the elements, and such layouts are left to the regular reorder.
    if (dims != dst.get_dims() || dims.size() < 2 || src.get_format_kind() != dnnl::memory::format_kind::blocked ||
        dst.get_format_kind() != dnnl::memory::format_kind::blocked || src.get_inner_nblks() != 0 ||
        dst.get_inner_nblks() != 0) {
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
//
// dst is expected to be zero-initialized.
void transposeSubByteLastTwoDims(const uint8_t* src,
                                 uint8_t* dst,
                                 const dnnl::memory::dims& dims,
                                 const dnnl::memory::dims& srcStrides,
                                 const dnnl::memory::dims& dstStrides,
                                 size_t srcBufBytes,
                                 size_t dstBufBytes,
                                 const std::shared_ptr<ThreadPool>& threadPool) {
    const size_t rank = dims.size();
    OPENVINO_ASSERT(rank > 0 && srcStrides.size() == rank && dstStrides.size() == rank,
                    "transposeSubByteLastTwoDims: dims/strides rank mismatch");

    // The bounds are checked once for the largest offsets instead of per element.
    size_t total = 1;
    int64_t srcMax = 0;
    int64_t dstMax = 0;
    for (size_t i = 0; i < rank; i++) {
        OPENVINO_ASSERT(dims[i] > 0 && srcStrides[i] >= 0 && dstStrides[i] >= 0,
                        "transposeSubByteLastTwoDims: unexpected dims/strides");
        total *= static_cast<size_t>(dims[i]);
        srcMax += (dims[i] - 1) * srcStrides[i];
        dstMax += (dims[i] - 1) * dstStrides[i];
    }
    OPENVINO_ASSERT(static_cast<size_t>(srcMax) / 2 < srcBufBytes && static_cast<size_t>(dstMax) / 2 < dstBufBytes,
                    "transposeSubByteLastTwoDims: offsets out of range, src max ",
                    srcMax,
                    " (",
                    srcBufBytes,
                    " bytes), dst max ",
                    dstMax,
                    " (",
                    dstBufBytes,
                    " bytes)");

    const auto nibble = [src](int64_t elem) -> uint8_t {
        return (src[elem / 2] >> ((elem % 2) * 4)) & 0x0FU;
    };

    // Work is split by rows along dst's innermost dim.
    const auto inner =
        static_cast<size_t>(std::distance(dstStrides.begin(), std::min_element(dstStrides.begin(), dstStrides.end())));
    const int64_t innerLen = dims[inner];
    const size_t rows = total / static_cast<size_t>(innerLen);
    const auto rowBases = [&](size_t row, int64_t& srcBase, int64_t& dstBase) {
        srcBase = 0;
        dstBase = 0;
        for (size_t i = rank; i-- > 0;) {
            if (i == inner) {
                continue;
            }
            const auto extent = static_cast<size_t>(dims[i]);
            const auto idx = static_cast<int64_t>(row % extent);
            row /= extent;
            srcBase += idx * srcStrides[i];
            dstBase += idx * dstStrides[i];
        }
    };

    // In a dense dst with an even innermost extent every row starts on a byte boundary, so rows can be produced in
    // parallel, a whole byte at a time.
    const bool byteAlignedRows =
        dstStrides[inner] == 1 && innerLen % 2 == 0 && static_cast<size_t>(dstMax) + 1 == total;
    if (!byteAlignedRows) {
        for (size_t row = 0; row < rows; row++) {
            int64_t srcBase = 0;
            int64_t dstBase = 0;
            rowBases(row, srcBase, dstBase);
            for (int64_t i = 0; i < innerLen; i++) {
                const int64_t dstElem = dstBase + i * dstStrides[inner];
                const uint8_t shift = (dstElem % 2) * 4;
                dst[dstElem / 2] =
                    (dst[dstElem / 2] & ~(0x0FU << shift)) | (nibble(srcBase + i * srcStrides[inner]) << shift);
            }
        }
        return;
    }

    const int64_t srcInnerStride = srcStrides[inner];
    const auto body = [&](int ithr, int nthr) {
        size_t start = 0;
        size_t end = 0;
        ov::splitter(rows, nthr, ithr, start, end);
        for (size_t row = start; row < end; row++) {
            int64_t srcBase = 0;
            int64_t dstBase = 0;
            rowBases(row, srcBase, dstBase);
            for (int64_t i = 0; i < innerLen; i += 2) {
                const uint8_t lo = nibble(srcBase + i * srcInnerStride);
                const uint8_t hi = nibble(srcBase + (i + 1) * srcInnerStride);
                dst[(dstBase + i) / 2] = lo | (hi << 4);
            }
        }
    };
    // Use only as many threads as keep at least minElemsPerThread elements each: the parallel dispatch costs ~5-15 us,
    // about the serial time of 1-2K elements (measured on GNR, TBB). Weights below 2 * minElemsPerThread are done on
    // the calling thread.
    constexpr size_t minElemsPerThread = 1U << 13;
    const auto maxThreads = threadPool ? threadPool->get_num_threads() : parallel_get_max_threads();
    const auto nthr = static_cast<int>(std::min<size_t>(maxThreads, std::max<size_t>(1, total / minElemsPerThread)));
    if (nthr == 1) {
        body(0, 1);
    } else if (threadPool) {
        threadPool->parallel_for(nthr, body);
    } else {
        ov::parallel_nt(nthr, body);
    }
}

}  // namespace

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const ExecutorContext::CPtr& context,
                               const bool needShiftSignedToUnsigned,
                               const bool interleaveAcrossNumaNodes) {
    const auto privateWeightCache = context->getPrivateWeightCache();
    OPENVINO_ASSERT(privateWeightCache, "privateWeightCache is nullptr");

    return prepareWeightsMemory(
        srcWeightDesc,
        dstWeightDesc,
        weightsMem,
        context->getEngine(),
        context->getRuntimeCache(),
        context->getWeightsCache(),
        privateWeightCache,
        context->getThreadPool(),
        needShiftSignedToUnsigned,
        interleaveAcrossNumaNodes ? context->getWeightsInterleaveNumaNodes() : std::vector<int>{});
}

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const dnnl::engine& eng,
                               const MultiCachePtr& rtCache,
                               const WeightsSharing::Ptr& globalWeightCache,
                               const std::shared_ptr<std::unordered_map<std::string, MemoryPtr>>& privateWeightCache,
                               const std::shared_ptr<ThreadPool>& threadPool,
                               bool needShiftSignedToUnsigned,
                               const std::vector<int>& interleaveNumaNodes) {
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
                                        _ptr->getSize(),
                                        threadPool);
            return _ptr;
        }

        Memory srcMemory{eng, srcWeightDesc, weightsMem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
        node::Reorder::reorderData(srcMemory, *_ptr, rtCache, threadPool);

        return _ptr;
    };

    // Weights that all the threads of a stream read are interleaved over the NUMA nodes of the stream: pages read from
    // several nodes are otherwise migrated by the kernel one by one, and every migration stalls all the readers.
    // Small buffers are not worth it. A failure is not an error, the weights are fine with any placement.
    constexpr size_t minInterleaveBytes = 1U << 20;
    auto createAndPlace = [&]() {
        auto weights = create();
        if (!interleaveNumaNodes.empty() && weights->getSize() >= minInterleaveBytes) {
            mbind_interleave_move(weights->getData(), weights->getSize(), interleaveNumaNodes);
        }
        return weights;
    };

    MemoryPtr ptr;
    if (globalWeightCache && dnnl::memory::format_kind::blocked == dstWeightDesc->getDnnlDesc().get_format_kind()) {
        ptr = MemoryPtr(
            *globalWeightCache->findOrCreate(DnnlExtensionUtils::computeWeightsStringHash(weightsMem, dstWeightDesc),
                                             createAndPlace));
    } else {
        ptr = createAndPlace();
    }

    if (privateWeightCache) {
        (*privateWeightCache)[format] = ptr;
    }

    return ptr;
}

}  // namespace ov::intel_cpu::utils
