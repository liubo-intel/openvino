// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "transformations/cpu_opset/x64/op/gemma4_ple_block.hpp"

namespace ov::intel_cpu {
class BrgemmKernel;
class GateMulCombineKernel;
class RmsResidualBf16Kernel;
}

namespace ov::intel_cpu::node {

class Gemma4PLEBlock : public Node {
public:
    Gemma4PLEBlock(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::Gemma4PLEBlock;
    }
    bool needPrepareParams() const override {
        return false;
    }
    void createPrimitive() override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    Gemma4PLEBlockNode::Config m_config{};

    // bf16-decompressed weights (kept for re-packing if needed).
    std::vector<ov::bfloat16> m_gate_w_bf16;  // [Hp, H]
    std::vector<ov::bfloat16> m_proj_w_bf16;  // [H, Hp]
    std::vector<float> m_norm_gamma_f32;      // [H]

    // Per N-shard packed weights. Each shard is [N_shard, K] packed via copy_buffer_b.
    // Stage1 sharded along Hp; Stage2 sharded along H.
    std::vector<uint8_t> m_packed_gate_w;  // shards laid back-to-back, each `gate_shard_bytes`
    std::vector<uint8_t> m_packed_proj_w;
    size_t m_gate_shard_bytes = 0;
    size_t m_proj_shard_bytes = 0;
    bool m_weights_packed = false;

    // Shared workspace for one execute() call (resized lazily).
    std::vector<ov::bfloat16> m_gated_bf;   // [M, Hp] -- only intermediate that lands in DRAM
    std::vector<float> m_C_proj;            // [M, H] (still f32, fed into JIT epi2)

    // Per-thread brgemm scratch + small f32 staging tile (kept hot in L1/L2).
    std::vector<uint8_t> m_thread_wsp;
    size_t m_nthr = 0;
    size_t m_per_thread_bytes = 0;
    size_t m_wsp_bytes = 0;
    size_t m_scratchA_bytes = 0;
    size_t m_stage_bytes = 0;  // bytes of the per-thread f32 staging tile (kMblk * kNshard * 4)

    // JIT kernels (created once on AVX-512 hosts).
    std::shared_ptr<GateMulCombineKernel> m_gate_combine;
    std::shared_ptr<RmsResidualBf16Kernel> m_rms_combine;

    // Cached BRGEMM kernels per kernel_M (one entry per (M_blk + tail) value).
    // N is fixed = N_SHARD inside the kernel; we slide B base pointer to cover all shards.
    std::unordered_map<size_t, std::shared_ptr<BrgemmKernel>> m_gemm1_cache;
    std::unordered_map<size_t, std::shared_ptr<BrgemmKernel>> m_gemm2_cache;

    static constexpr size_t kMblk = 32;     // matches BrgemmKernel::matmulOptimalM
    static constexpr size_t kNshard = 32;   // columns per parallel N-shard
};

}  // namespace ov::intel_cpu::node
