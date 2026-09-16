// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/dynamic_quant/dynamic_quant.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

template <typename Primitive, typename Attrs, typename ShapeAgnosticData, typename Instantiator>
class DnnlExecutor : public Executor {
public:
    using PrimitivePtr = std::shared_ptr<Primitive>;
    DnnlExecutor(Attrs attrs,
                 const MemoryArgs& memory,
                 ExecutorContext::CPtr context,
                 const bool cacheWeights,
                 const bool fc3Das2D = false)
        : m_attrs(std::move(attrs)),
          m_context(std::move(context)),
          m_shapeAgnosticData(Primitive::createShapeAgnosticData(m_attrs, memory, m_context, cacheWeights)),
          m_primArgs(m_shapeAgnosticData->m_primAttrs.dnnlArgs),
          m_fc3Das2D(fc3Das2D),
          m_dynQuant(m_shapeAgnosticData->m_dynQuant),
          m_dynQuantGroupSize(m_shapeAgnosticData->m_dynQuantGroupSize) {}
    bool update(const MemoryArgs& memory) override {
        const auto primitive = createPrimitive(memory, m_attrs);
        if (!primitive) {
            return false;
        }
        updateMemory(m_primitive, primitive, memory);
        m_primitive = primitive;
        return true;
    }

    void execute(const MemoryArgs& memory) override {
        if (m_dynQuant) {
            quantizeSrc(memory);
        } else {
            if (resetSrcMemoryDataHandle) {
                m_primArgs[DNNL_ARG_SRC].set_data_handle(memory.at(ARG_SRC)->getData());
            }
        }
        if (resetDstMemoryDataHandle) {
            m_primArgs[DNNL_ARG_DST].set_data_handle(memory.at(ARG_DST)->getData());
        }

        m_primitive->execute(m_primArgs);
    }

    void execute() override {
        OPENVINO_ASSERT(!m_dynQuant, "DnnlExecutor: the no-argument execute() overload cannot quantize the src for "
                                     "the dynamic-quant path - execute(memory) must be used instead");
        m_primitive->execute(m_primArgs);
    }

    void execute() const override {
        OPENVINO_ASSERT(!m_dynQuant, "DnnlExecutor: the no-argument execute() overload cannot quantize the src for "
                                     "the dynamic-quant path - execute(memory) must be used instead");
        m_primitive->execute(m_primArgs);
    }

    [[nodiscard]] impl_desc_type implType() const override {
        // to satisfy functional tests logic, implementation type should be shape agnostic
        if (m_shapeAgnosticData->m_implType != impl_desc_type::undef) {
            return m_shapeAgnosticData->m_implType;
        }

        return m_primitive ? m_primitive->implType() : impl_desc_type::undef;
    }

    void moveMemToNumaNode(int numaNodeID) override {
        if (curNumaNode == numaNodeID) {
            return;
        }
        if (m_dynQuant) {
            // The combined scratch (qsrc | src_scale | mm-scratch, see updateDynQuantScratch())
            // must be re-partitioned as a whole on the new NUMA node - a plain
            // createScratchPadMem(mm-scratch-only-size) here would alias/overwrite the qsrc and
            // src_scale views. The qsrc/src_scale descriptors themselves are shape-derived and
            // unchanged by a NUMA move, so they are reused as-is; only the mm-scratch size can
            // legitimately differ (a NUMA move never happens without an update() first, but the
            // primitive's own scratchpad size is re-read defensively rather than cached).
            updateDynQuantScratch(m_primitive->scratchPadDesc()->getCurrentMemSize());
        } else {
            const auto newPrimMemDesc = m_primitive->scratchPadDesc();
            m_scratchPadMemory = m_context->getScratchPad()->createScratchPadMem(newPrimMemDesc);
            m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();
        }

        if (auto it = m_primArgs.find(DNNL_ARG_WEIGHTS); it != m_primArgs.end()) {
            if (!mbind_move(it->second, numaNodeID)) {
                DEBUG_LOG("[FullyConnected] move DNNL_ARG_WEIGHTS to node ", numaNodeID, " failed");
            }
        }

        if (auto it = m_primArgs.find(DNNL_ARG_BIAS); it != m_primArgs.end()) {
            if (!mbind_move(it->second, numaNodeID)) {
                DEBUG_LOG("[FullyConnected] move DNNL_ARG_BIAS to node ", numaNodeID, " failed");
            }
        }
        curNumaNode = numaNodeID;
    }

private:
    void updateSrcMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr& memory) {
        const auto& primMemDesc = primitive->srcDesc();
        if (memDesc->isCompatible(*primMemDesc)) {
            m_primArgs[DNNL_ARG_SRC] = memory->getPrimitive();
        } else {
            resetSrcMemoryDataHandle = true;
            // create 2D memory without underlying buffer and reset to the actual memory in scope of 'execute' call
            m_primArgs[DNNL_ARG_SRC] =
                dnnl::memory(primMemDesc->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        }
    }

    void updateDstMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr& memory) {
        const auto& primMemDesc = primitive->dstDesc();
        if (memDesc->isCompatible(*primMemDesc)) {
            m_primArgs[DNNL_ARG_DST] = memory->getPrimitive();
        } else {
            resetDstMemoryDataHandle = true;
            // create 2D memory without underlying buffer and reset to the actual memory in scope of 'execute' call
            m_primArgs[DNNL_ARG_DST] =
                dnnl::memory(primMemDesc->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        }
    }

    void updateWeightsMemory(DnnlMemoryDescPtr originalMemDesc,
                             const PrimitivePtr currentPrimitive,
                             const PrimitivePtr newPrimitive,
                             const MemoryPtr& memory) {
        if (!m_attrs.constantWeights) {  // non constant weights are handled by the primitive
            m_primArgs[DNNL_ARG_WEIGHTS] = memory->getPrimitive();
            return;
        }

        const auto newPrimMemDesc = newPrimitive->weightsDesc();

        if (currentPrimitive && currentPrimitive->weightsDesc()->isCompatible(*newPrimMemDesc)) {
            return;
        }

        originalMemDesc = Primitive::makeTransposedWeightDescriptor(originalMemDesc, newPrimMemDesc, m_attrs);

        const auto weiMemory = utils::prepareWeightsMemory(originalMemDesc, newPrimMemDesc, memory, m_context, true);
        m_primArgs[DNNL_ARG_WEIGHTS] = weiMemory->getPrimitive();
    }

    void updateBiasMemory(const MemoryPtr& memory) {
        m_primArgs[DNNL_ARG_BIAS] = memory->getPrimitive();
    }

    void updatePostOpsMemory(const MemoryArgs& memory) {
        auto update = [&memory, this](int cpuMemoryArg, int dnnlMemoryArg) {
            if (const auto arg = memory.find(cpuMemoryArg); arg != memory.end()) {
                const auto& memory = arg->second;
                m_primArgs[dnnlMemoryArg] = memory->getPrimitive();
            }
        };

        update(ARG_ATTR_POST_OP_DW | ARG_WEI, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        update(ARG_ATTR_POST_OP_DW | ARG_BIAS, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

        if (m_shapeAgnosticData->m_primAttrs.legacyZeroPoints) {
            update(ARG_ATTR_ZERO_POINTS | ARG_SRC, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
            update(ARG_ATTR_ZERO_POINTS | ARG_WEI, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
            update(ARG_ATTR_ZERO_POINTS | ARG_DST, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
        } else {
            update(ARG_ATTR_ZERO_POINTS | ARG_SRC_3, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
        }
    }

    void updateScratchPadMem(const PrimitivePtr currentPrimitive, const PrimitivePtr newPrimitive) {
        const auto newPrimMemDesc = newPrimitive->scratchPadDesc();
        // @todo should we compare dnnl::memory::desc directly to avoid any overhead?
        if (currentPrimitive && currentPrimitive->scratchPadDesc()->isCompatible(*newPrimMemDesc)) {
            return;
        }

        m_scratchPadMemory = m_context->getScratchPad()->createScratchPadMem(newPrimMemDesc);
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();
    }

    // Builds the qsrc (s8) and src_scale (f32) dnnl::memory descriptors for the current shape
    // and (re)builds the combined scratch (qsrc | src_scale | mm-scratch) from scratch. M/K are
    // recomputed from the *node*-level ARG_SRC shape every call - the block backing the
    // combined scratch may grow and its base pointer may move, so nothing here (nor in
    // moveMemToNumaNode()) may cache a raw pointer across update() calls.
    void updateDynQuantSrcMemory(const PrimitivePtr& newPrimitive, const MemoryArgs& memory) {
        const auto& srcDims = memory.at(ARG_SRC)->getShape().getDims();
        m_dynQuantK = srcDims.back();
        m_dynQuantM = 1;
        for (size_t i = 0; i + 1 < srcDims.size(); i++) {
            m_dynQuantM *= srcDims[i];
        }

        // The primitive's own compiled src descriptor is already s8 and already at whatever
        // rank createDescriptorInternalAsFc() normalized to - reuse it verbatim rather than
        // rebuilding dims/strides by hand.
        m_qsrcDnnlDesc = newPrimitive->srcDesc()->getDnnlDesc();

        // Src scale dims mirror the primitive's src dims with the innermost (K) dim divided by
        // the group size, matching the {1, GK} groups set on DNNL_ARG_SRC in
        // createPrimitiveAttrs() (group 1 - i.e. unchanged - for every other dim, including M).
        auto scaleDims = m_qsrcDnnlDesc.get_dims();
        scaleDims.back() /= static_cast<dnnl::memory::dim>(m_dynQuantGroupSize);
        dnnl::memory::dims denseStrides(scaleDims.size());
        dnnl::memory::dim stride = 1;
        for (int i = static_cast<int>(scaleDims.size()) - 1; i >= 0; --i) {
            denseStrides[i] = stride;
            stride *= scaleDims[i];
        }
        m_srcScaleDnnlDesc = dnnl::memory::desc(scaleDims, dnnl::memory::data_type::f32, denseStrides);

        updateDynQuantScratch(newPrimitive->scratchPadDesc()->getCurrentMemSize());
    }

    // (Re)builds the single combined scratch block and rebinds the qsrc/src_scale/mm-scratch
    // views + their DNNL_ARG_* entries in m_primArgs. Uses the cached m_qsrcDnnlDesc /
    // m_srcScaleDnnlDesc, so it is safe to call from moveMemToNumaNode() (no shape change)
    // as well as from updateDynQuantSrcMemory() (fresh shape).
    void updateDynQuantScratch(size_t mmScratchSize) {
        const size_t sizeQsrc = rnd_up(m_qsrcDnnlDesc.get_size(), 64);
        const size_t sizeScale = rnd_up(m_srcScaleDnnlDesc.get_size(), 64);
        const size_t sizeMmScratch = rnd_up(mmScratchSize, 64);
        const size_t total = sizeQsrc + sizeScale + sizeMmScratch;

        const auto combinedDesc = std::make_shared<DnnlBlockedMemoryDesc>(ov::element::u8, Shape({total}));
        m_combinedScratch = m_context->getScratchPad()->createScratchPadMem(combinedDesc);

        auto* base = static_cast<uint8_t*>(m_combinedScratch->getData());
        const auto& engine = m_context->getEngine();

        m_qsrcMem = dnnl::memory(m_qsrcDnnlDesc, engine, DNNL_MEMORY_NONE);
        m_qsrcMem.set_data_handle(base);
        m_srcScaleMem = dnnl::memory(m_srcScaleDnnlDesc, engine, DNNL_MEMORY_NONE);
        m_srcScaleMem.set_data_handle(base + sizeQsrc);
        m_mmScratchMem = dnnl::memory(dnnl::memory::desc({static_cast<dnnl::memory::dim>(mmScratchSize)},
                                                         dnnl::memory::data_type::u8,
                                                         dnnl::memory::format_tag::a),
                                      engine,
                                      DNNL_MEMORY_NONE);
        m_mmScratchMem.set_data_handle(base + sizeQsrc + sizeScale);

        m_primArgs[DNNL_ARG_SRC] = m_qsrcMem;
        m_primArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = m_srcScaleMem;
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_mmScratchMem;
    }

    void quantizeSrc(const MemoryArgs& memory) {
        const auto& srcMem = memory.at(ARG_SRC);
        ov::Extensions::Cpu::XARCH::quant_src_grouped_i8(srcMem->getPrecision(),
                                                         srcMem->getData(),
                                                         static_cast<int8_t*>(m_qsrcMem.get_data_handle()),
                                                         static_cast<float*>(m_srcScaleMem.get_data_handle()),
                                                         m_dynQuantM,
                                                         m_dynQuantK,
                                                         m_dynQuantGroupSize,
                                                         m_context->getCpuParallel());
    }

    void updateMemory(const PrimitivePtr currentPrimitive, const PrimitivePtr newPrimitive, const MemoryArgs& memory) {
        const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
        const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

        if (m_dynQuant) {
            updateDynQuantSrcMemory(newPrimitive, memory);
        } else if (m_fc3Das2D) {
            const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
            updateSrcMemory(srcDesc, newPrimitive, memory.at(ARG_SRC));
        } else {
            m_primArgs[DNNL_ARG_SRC] = memory.at(ARG_SRC)->getPrimitive();
        }

        if (m_fc3Das2D) {
            updateDstMemory(dstDesc, newPrimitive, memory.at(ARG_DST));
        } else {
            m_primArgs[DNNL_ARG_DST] = memory.at(ARG_DST)->getPrimitive();
        }

        updateWeightsMemory(weiDesc, currentPrimitive, newPrimitive, memory.at(ARG_WEI));
        updateBiasMemory(memory.at(ARG_BIAS));
        updatePostOpsMemory(memory);

        if (m_dynQuant) {
            // Already handled by updateDynQuantSrcMemory() above (single combined scratch, see
            // its comment for why this cannot reuse the plain updateScratchPadMem() path).
        } else {
            updateScratchPadMem(currentPrimitive, newPrimitive);
        }
    }

    PrimitivePtr createPrimitive(const MemoryArgs& memory, const Attrs& attrs) {
        return Instantiator{}(memory, attrs, m_context, m_shapeAgnosticData);
    }
    // @todo there is no real reason to store attrs. Better to just pass as api argument
    Attrs m_attrs;
    const ExecutorContext::CPtr m_context;
    std::shared_ptr<ShapeAgnosticData> m_shapeAgnosticData;
    dnnl_primitive_args& m_primArgs;
    bool resetSrcMemoryDataHandle = false;
    bool resetDstMemoryDataHandle = false;
    MemoryPtr m_scratchPadMemory;
    PrimitivePtr m_primitive;
    int curNumaNode = -1;
    bool m_fc3Das2D = false;

    // AMX int8 grouped-quantization dynamic-quant path (DnnlMatMulPrimitive only). Inert
    // (m_dynQuant == false) for every other primitive/executor instantiation.
    bool m_dynQuant = false;
    uint64_t m_dynQuantGroupSize = 0;
    size_t m_dynQuantM = 0;
    size_t m_dynQuantK = 0;
    dnnl::memory::desc m_qsrcDnnlDesc;
    dnnl::memory::desc m_srcScaleDnnlDesc;
    MemoryPtr m_combinedScratch;
    dnnl::memory m_qsrcMem;
    dnnl::memory m_srcScaleMem;
    dnnl::memory m_mmScratchMem;
};

}  // namespace ov::intel_cpu
