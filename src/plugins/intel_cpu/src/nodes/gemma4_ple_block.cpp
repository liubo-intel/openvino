// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemma4_ple_block.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "transformations/cpu_opset/x64/op/gemma4_ple_block.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

// JIT epi1 kernel:  out[bf16] = ConvertFp32toBf16( gelu_tanh(gate_f32) * pli_bf16 )
// Processes (rows x cols) elements; cols must be a multiple of 16. The gate buffer has stride
// gate_stride_in_floats per row; the pli/dst buffers have stride pli_stride_in_bf16 per row.
class GateMulCombineKernel : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(GateMulCombineKernel)

    struct CallArgs {
        const float* gate;      // f32 [rows, cols], stride = gate_stride
        const int16_t* pli;     // bf16 [rows, cols], stride = pli_stride
        int16_t* dst;           // bf16 [rows, cols], stride = pli_stride
        int64_t gate_stride;    // in floats
        int64_t pli_stride;     // in bf16 elements
        int64_t rows;
        int64_t cols;
    };

    GateMulCombineKernel() : jit_generator_t(jit_name()) {
        create_kernel();
    }

    void generate() override {
        using namespace Xbyak;
        using namespace dnnl::impl::cpu::x64;

        auto reg_args = abi_param1;
        Reg64 reg_gate = r8;
        Reg64 reg_pli = r9;
        Reg64 reg_dst = r10;
        Reg64 reg_gate_stride = r11;
        Reg64 reg_pli_stride = r12;
        Reg64 reg_rows = r13;
        Reg64 reg_cols = r14;
        Reg64 reg_col = rax;

        auto injector = std::make_shared<jit_uni_eltwise_injector_t<avx512_core>>(
            this,
            dnnl_eltwise_gelu_tanh,
            0.F,
            0.F,
            1.F,
            dnnl::impl::data_type::f32,
            true,            // save_state
            Reg64(Operand::R15),
            Opmask(1),
            true,            // is_fwd
            false,           // use_dst
            false,           // preserve_vmm
            false);          // preserve_p_table

        // Preserve callee-saved regs we clobber.
        push(r12);
        push(r13);
        push(r14);
        push(r15);
        push(rbx);  // align stack to 16

        mov(reg_gate, ptr[reg_args + offsetof(CallArgs, gate)]);
        mov(reg_pli, ptr[reg_args + offsetof(CallArgs, pli)]);
        mov(reg_dst, ptr[reg_args + offsetof(CallArgs, dst)]);
        mov(reg_gate_stride, ptr[reg_args + offsetof(CallArgs, gate_stride)]);
        mov(reg_pli_stride, ptr[reg_args + offsetof(CallArgs, pli_stride)]);
        mov(reg_rows, ptr[reg_args + offsetof(CallArgs, rows)]);
        mov(reg_cols, ptr[reg_args + offsetof(CallArgs, cols)]);

        injector->load_table_addr();

        const Zmm zmm_gate = zmm5;
        const Zmm zmm_act = zmm6;
        const Zmm zmm_pli_f32 = zmm7;

        Label row_loop, col_loop, row_end;
        L(row_loop);
        {
            test(reg_rows, reg_rows);
            jz(row_end, T_NEAR);

            xor_(reg_col, reg_col);
            align(64);
            L(col_loop);
            {
                // load 16 floats from gate
                vmovups(zmm_gate, ptr[reg_gate + reg_col * 4]);
                // gelu_tanh(gate)
                vmovups(zmm_act, zmm_gate);
                injector->compute_vector(zmm_act.getIdx());
                // load 16 bf16 from pli, expand to f32 (shift left 16)
                vpmovzxwd(zmm_pli_f32, ptr[reg_pli + reg_col * 2]);
                vpslld(zmm_pli_f32, zmm_pli_f32, 16);
                // multiply
                vmulps(zmm_act, zmm_act, zmm_pli_f32);
                // f32 -> bf16 (RNE) and store 16 bf16 = 32 bytes
                vcvtneps2bf16(ymm6, zmm_act);
                vmovups(ptr[reg_dst + reg_col * 2], ymm6);

                add(reg_col, 16);
                cmp(reg_col, reg_cols);
                jl(col_loop, T_NEAR);
            }

            // advance row pointers
            lea(reg_gate, ptr[reg_gate + reg_gate_stride * 4]);
            lea(reg_pli, ptr[reg_pli + reg_pli_stride * 2]);
            lea(reg_dst, ptr[reg_dst + reg_pli_stride * 2]);
            dec(reg_rows);
            jmp(row_loop, T_NEAR);
        }
        L(row_end);

        pop(rbx);
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        ret();

        injector->prepare_table();
    }

    void call(const float* gate,
              size_t gate_stride,
              const ov::bfloat16* pli,
              size_t pli_stride,
              ov::bfloat16* dst,
              size_t rows,
              size_t cols) const {
        CallArgs args{};
        args.gate = gate;
        args.pli = reinterpret_cast<const int16_t*>(pli);
        args.dst = reinterpret_cast<int16_t*>(dst);
        args.gate_stride = static_cast<int64_t>(gate_stride);
        args.pli_stride = static_cast<int64_t>(pli_stride);
        args.rows = static_cast<int64_t>(rows);
        args.cols = static_cast<int64_t>(cols);
        (*this)(&args);
    }
};

// JIT epi2 kernel for one row at a time:
//   inv_rms = 1 / sqrt( sum(p^2) / H + eps )
//   out_bf16[c] = ConvertFp32toBf16( layer_scalar * (residual_bf16[c] + p[c] * inv_rms * gamma[c]) )
// Note: layer_scalar (Gemma4 per-layer LayerScale) is folded into the same f32 ZMM pipe so it
// costs one extra vmulps per 16 elements with no additional load/store.
class RmsResidualBf16Kernel : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(RmsResidualBf16Kernel)

    struct CallArgs {
        const float* p;            // f32 [cols]
        const float* gamma;        // f32 [cols]
        const int16_t* residual;   // bf16 [cols]
        int16_t* dst;              // bf16 [cols]
        float inv_h;               // 1.0 / H
        float eps;
        float layer_scalar;        // per-layer LayerScale (defaults to 1.0)
        int64_t cols;
    };

    RmsResidualBf16Kernel() : jit_generator_t(jit_name()) {
        create_kernel();
    }

    void generate() override {
        using namespace Xbyak;
        using namespace dnnl::impl::cpu::x64;

        auto reg_args = abi_param1;
        Reg64 reg_p = r8;
        Reg64 reg_gamma = r9;
        Reg64 reg_res = r10;
        Reg64 reg_dst = r11;
        Reg64 reg_cols = r12;
        Reg64 reg_col = rax;
        Reg64 reg_tmp = rdx;

        push(rbx);
        push(r12);
        push(r13);

        mov(reg_p, ptr[reg_args + offsetof(CallArgs, p)]);
        mov(reg_gamma, ptr[reg_args + offsetof(CallArgs, gamma)]);
        mov(reg_res, ptr[reg_args + offsetof(CallArgs, residual)]);
        mov(reg_dst, ptr[reg_args + offsetof(CallArgs, dst)]);
        mov(reg_cols, ptr[reg_args + offsetof(CallArgs, cols)]);

        // Pass 1: sum(p^2) using 4 accumulators, 16 f32 each.
        const Zmm zmm_acc0 = zmm0;
        const Zmm zmm_acc1 = zmm1;
        const Zmm zmm_acc2 = zmm2;
        const Zmm zmm_acc3 = zmm3;
        const Zmm zmm_v0 = zmm4;
        const Zmm zmm_v1 = zmm5;
        const Zmm zmm_v2 = zmm6;
        const Zmm zmm_v3 = zmm7;

        vpxord(zmm_acc0, zmm_acc0, zmm_acc0);
        vpxord(zmm_acc1, zmm_acc1, zmm_acc1);
        vpxord(zmm_acc2, zmm_acc2, zmm_acc2);
        vpxord(zmm_acc3, zmm_acc3, zmm_acc3);

        Label sumsq_main, sumsq_tail_check, sumsq_tail, sumsq_done;
        // process 64 floats per iteration
        xor_(reg_col, reg_col);
        mov(reg_tmp, reg_cols);
        and_(reg_tmp, ~static_cast<int64_t>(63));
        align(64);
        L(sumsq_main);
        {
            cmp(reg_col, reg_tmp);
            jge(sumsq_tail_check, T_NEAR);
            vmovups(zmm_v0, ptr[reg_p + reg_col * 4 + 0]);
            vmovups(zmm_v1, ptr[reg_p + reg_col * 4 + 64]);
            vmovups(zmm_v2, ptr[reg_p + reg_col * 4 + 128]);
            vmovups(zmm_v3, ptr[reg_p + reg_col * 4 + 192]);
            vfmadd231ps(zmm_acc0, zmm_v0, zmm_v0);
            vfmadd231ps(zmm_acc1, zmm_v1, zmm_v1);
            vfmadd231ps(zmm_acc2, zmm_v2, zmm_v2);
            vfmadd231ps(zmm_acc3, zmm_v3, zmm_v3);
            add(reg_col, 64);
            jmp(sumsq_main, T_NEAR);
        }
        L(sumsq_tail_check);
        // process remaining in 16-wide chunks (cols is divisible by 16)
        align(64);
        L(sumsq_tail);
        {
            cmp(reg_col, reg_cols);
            jge(sumsq_done, T_NEAR);
            vmovups(zmm_v0, ptr[reg_p + reg_col * 4]);
            vfmadd231ps(zmm_acc0, zmm_v0, zmm_v0);
            add(reg_col, 16);
            jmp(sumsq_tail, T_NEAR);
        }
        L(sumsq_done);

        // reduce 4 accumulators -> scalar
        vaddps(zmm_acc0, zmm_acc0, zmm_acc1);
        vaddps(zmm_acc2, zmm_acc2, zmm_acc3);
        vaddps(zmm_acc0, zmm_acc0, zmm_acc2);
        // horizontal reduce zmm_acc0 -> xmm
        const Ymm ymm_acc0 = ymm0;
        const Xmm xmm_acc0 = xmm0;
        const Ymm ymm_hi = ymm1;
        const Xmm xmm_hi = xmm1;
        vextractf64x4(ymm_hi, zmm_acc0, 1);
        vaddps(ymm_acc0, ymm_acc0, ymm_hi);
        vextractf128(xmm_hi, ymm_acc0, 1);
        vaddps(xmm_acc0, xmm_acc0, xmm_hi);
        // shuffle to reduce within xmm
        vpermilps(xmm_hi, xmm_acc0, 0x4E);
        vaddps(xmm_acc0, xmm_acc0, xmm_hi);
        vpermilps(xmm_hi, xmm_acc0, 0xB1);
        vaddss(xmm_acc0, xmm_acc0, xmm_hi);

        // mean(x^2) = sum * inv_h
        vmovss(xmm1, ptr[reg_args + offsetof(CallArgs, inv_h)]);
        vmulss(xmm_acc0, xmm_acc0, xmm1);
        // + eps
        vmovss(xmm1, ptr[reg_args + offsetof(CallArgs, eps)]);
        vaddss(xmm_acc0, xmm_acc0, xmm1);
        // sqrt
        vsqrtss(xmm_acc0, xmm_acc0, xmm_acc0);
        // 1 / sqrt(...)
        mov(reg_tmp.cvt32(), float2int(1.0F));
        vmovd(xmm1, reg_tmp.cvt32());
        vdivss(xmm_acc0, xmm1, xmm_acc0);
        // broadcast inv_rms to a zmm
        const Zmm zmm_inv = zmm8;
        vbroadcastss(zmm_inv, xmm_acc0);

        // broadcast layer_scalar into a zmm once (constant for all 16-element iterations)
        const Zmm zmm_scale = zmm9;
        vbroadcastss(zmm_scale, ptr[reg_args + offsetof(CallArgs, layer_scalar)]);

        // Pass 2: out = bf16( layer_scalar * (residual + p * inv_rms * gamma) ),
        //         16 elements per iteration
        Label out_loop, out_done;
        xor_(reg_col, reg_col);
        align(64);
        L(out_loop);
        {
            cmp(reg_col, reg_cols);
            jge(out_done, T_NEAR);
            // load p and gamma (f32)
            vmovups(zmm_v0, ptr[reg_p + reg_col * 4]);
            vmovups(zmm_v1, ptr[reg_gamma + reg_col * 4]);
            vmulps(zmm_v0, zmm_v0, zmm_inv);
            vmulps(zmm_v0, zmm_v0, zmm_v1);
            // residual (bf16 -> f32)
            vpmovzxwd(zmm_v2, ptr[reg_res + reg_col * 2]);
            vpslld(zmm_v2, zmm_v2, 16);
            vaddps(zmm_v0, zmm_v0, zmm_v2);
            // fold the trailing per-layer LayerScale into the same f32 pipe (free vs DRAM round-trip)
            vmulps(zmm_v0, zmm_v0, zmm_scale);
            // store as bf16
            vcvtneps2bf16(ymm4, zmm_v0);
            vmovups(ptr[reg_dst + reg_col * 2], ymm4);

            add(reg_col, 16);
            jmp(out_loop, T_NEAR);
        }
        L(out_done);

        pop(r13);
        pop(r12);
        pop(rbx);
        ret();
    }

    void call(const float* p,
              const float* gamma,
              const ov::bfloat16* residual,
              ov::bfloat16* dst,
              float inv_h,
              float eps,
              float layer_scalar,
              size_t cols) const {
        CallArgs args{};
        args.p = p;
        args.gamma = gamma;
        args.residual = reinterpret_cast<const int16_t*>(residual);
        args.dst = reinterpret_cast<int16_t*>(dst);
        args.inv_h = inv_h;
        args.eps = eps;
        args.layer_scalar = layer_scalar;
        args.cols = static_cast<int64_t>(cols);
        (*this)(&args);
    }
};

}  // namespace ov::intel_cpu

namespace ov::intel_cpu::node {

namespace {

inline float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float x3 = x * x * x;
    const float t = std::tanh(kSqrt2OverPi * (x + 0.044715f * x3));
    return 0.5f * x * (1.0f + t);
}

// GEMMA4_PLE_TRACE=1 enables per-pass timing, dumped per ~32 calls.
struct PleTrace {
    static bool enabled() {
        static const bool on = []() {
            const char* v = std::getenv("GEMMA4_PLE_TRACE");
            return v && v[0] != '\0' && v[0] != '0';
        }();
        return on;
    }
    enum Kind { GEMM1 = 0, EPI1 = 1, GEMM2 = 2, EPI2 = 3, K_NUM = 4 };
    std::array<std::atomic<uint64_t>, K_NUM> sum_ns{};
    std::array<std::atomic<uint64_t>, K_NUM> n{};
    std::atomic<uint64_t> calls{0};
    void add(Kind k, uint64_t ns) {
        sum_ns[k].fetch_add(ns, std::memory_order_relaxed);
        n[k].fetch_add(1, std::memory_order_relaxed);
    }
    void maybe_print(const std::string& tag, size_t M) {
        const auto c = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if (c % 64 != 0) {
            return;
        }
        auto avg = [&](Kind k) {
            const auto s = sum_ns[k].load(std::memory_order_relaxed);
            const auto cnt = n[k].load(std::memory_order_relaxed);
            return cnt ? (s / cnt) : 0;
        };
        std::cerr << "[Gemma4PLE trace] " << tag << " M=" << M << " calls=" << c
                  << " avg_us: gemm1=" << (avg(GEMM1) / 1000.0)
                  << " epi1=" << (avg(EPI1) / 1000.0)
                  << " gemm2=" << (avg(GEMM2) / 1000.0)
                  << " epi2=" << (avg(EPI2) / 1000.0) << std::endl;
    }
};
// Zero-overhead when GEMMA4_PLE_TRACE is unset: clock_now() is only called when enabled.
struct ScopedTimer {
    PleTrace& t;
    PleTrace::Kind k;
    bool active;
    std::chrono::steady_clock::time_point t0;
    ScopedTimer(PleTrace& tt, PleTrace::Kind kk) : t(tt), k(kk), active(PleTrace::enabled()) {
        if (active) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    ~ScopedTimer() {
        if (active) {
            const auto dt =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            t.add(k, static_cast<uint64_t>(dt));
        }
    }
};

}  // namespace

Gemma4PLEBlock::Gemma4PLEBlock(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    auto node = ov::as_type_ptr<const Gemma4PLEBlockNode>(op);
    m_config = node->get_config();
}

void Gemma4PLEBlock::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    if (rtPrecision == ov::element::f32) {
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
            rtPrecision = ov::element::bf16;
        }
    }
    OPENVINO_ASSERT(rtPrecision == ov::element::bf16,
                    "Gemma4PLEBlock currently only supports bf16 inference precision, got ",
                    rtPrecision);

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);          // input
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(1), false, -1);          // per_layer_input
    inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f16, getInputShapeAtPort(2), false, -1);     // gate_w
    inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f16, getInputShapeAtPort(3), false, -1);     // proj_w
    inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f32, getInputShapeAtPort(4), false, -1);     // norm_gamma

    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::brgemm_avx512_amx);
}

void Gemma4PLEBlock::createPrimitive() {
    // Decompress f16 weights to bf16 once.
    const int H = m_config.hidden_size;
    const int Hp = m_config.hidden_per_layer;
    const size_t Hu = static_cast<size_t>(H);
    const size_t Hpu = static_cast<size_t>(Hp);
    OPENVINO_ASSERT(Hpu % kNshard == 0,
                    "Gemma4PLEBlock requires Hp divisible by ", kNshard, " (got Hp=", Hp, ")");
    OPENVINO_ASSERT(Hu % kNshard == 0,
                    "Gemma4PLEBlock requires H divisible by ", kNshard, " (got H=", H, ")");

    auto gate_mem = getSrcMemoryAtPort(2);
    auto proj_mem = getSrcMemoryAtPort(3);
    auto gamma_mem = getSrcMemoryAtPort(4);

    OPENVINO_ASSERT(gate_mem->getDesc().getPrecision() == ov::element::f16,
                    "Gemma4PLEBlock gate_w precision must be f16");
    OPENVINO_ASSERT(proj_mem->getDesc().getPrecision() == ov::element::f16,
                    "Gemma4PLEBlock proj_w precision must be f16");
    OPENVINO_ASSERT(gamma_mem->getDesc().getPrecision() == ov::element::f32,
                    "Gemma4PLEBlock norm_gamma precision must be f32");

    const auto* gate_f16 = gate_mem->getDataAs<ov::float16>();
    const auto* proj_f16 = proj_mem->getDataAs<ov::float16>();
    const auto* gamma_f32 = gamma_mem->getDataAs<float>();

    m_gate_w_bf16.resize(Hpu * Hu);
    m_proj_w_bf16.resize(Hu * Hpu);
    m_norm_gamma_f32.assign(gamma_f32, gamma_f32 + H);

    parallel_for(Hpu * Hu, [&](size_t i) {
        m_gate_w_bf16[i] = static_cast<ov::bfloat16>(static_cast<float>(gate_f16[i]));
    });
    parallel_for(Hu * Hpu, [&](size_t i) {
        m_proj_w_bf16[i] = static_cast<ov::bfloat16>(static_cast<float>(proj_f16[i]));
    });

    // Pack weights once now. The packed-B layout depends only on (N=kNshard, K, transposed, dtype,
    // with_amx) -- it is independent of M -- so a single ``packing kernel'' produces bytes that are
    // valid for every kernel_M we'll ever build later.
    auto pack_kernel_gate = std::make_shared<BrgemmKernel>(/*M=*/kMblk,
                                                           /*N=*/kNshard,
                                                           /*K=*/Hu,
                                                           /*lda=*/Hu,
                                                           /*ldb=*/Hu,
                                                           /*ldc=*/Hpu,
                                                           /*b_transposed=*/true,
                                                           ov::element::bf16,
                                                           /*b_accumulate=*/false);
    auto pack_kernel_proj = std::make_shared<BrgemmKernel>(/*M=*/kMblk,
                                                           /*N=*/kNshard,
                                                           /*K=*/Hpu,
                                                           /*lda=*/Hpu,
                                                           /*ldb=*/Hpu,
                                                           /*ldc=*/Hu,
                                                           /*b_transposed=*/true,
                                                           ov::element::bf16,
                                                           /*b_accumulate=*/false);

    m_gate_shard_bytes = pack_kernel_gate->get_scratch_b_size();
    m_proj_shard_bytes = pack_kernel_proj->get_scratch_b_size();
    const size_t n_gate_shards = Hpu / kNshard;
    const size_t n_proj_shards = Hu / kNshard;
    m_packed_gate_w.assign(m_gate_shard_bytes * n_gate_shards, 0);
    m_packed_proj_w.assign(m_proj_shard_bytes * n_proj_shards, 0);

    parallel_for(n_gate_shards, [&](size_t s) {
        ov::bfloat16* src = m_gate_w_bf16.data() + s * kNshard * Hu;
        uint8_t* dst = m_packed_gate_w.data() + s * m_gate_shard_bytes;
        pack_kernel_gate->copy_buffer_b(src, dst);
    });
    parallel_for(n_proj_shards, [&](size_t s) {
        ov::bfloat16* src = m_proj_w_bf16.data() + s * kNshard * Hpu;
        uint8_t* dst = m_packed_proj_w.data() + s * m_proj_shard_bytes;
        pack_kernel_proj->copy_buffer_b(src, dst);
    });

    // Once packed, raw bf16 weights are no longer needed.
    m_gate_w_bf16.clear();
    m_gate_w_bf16.shrink_to_fit();
    m_proj_w_bf16.clear();
    m_proj_w_bf16.shrink_to_fit();

    m_weights_packed = true;
    m_nthr = parallel_get_max_threads();

    // JIT epilogue kernels (avx512_core_amx host -> avx512_core guaranteed).
    m_gate_combine = std::make_shared<GateMulCombineKernel>();
    m_rms_combine = std::make_shared<RmsResidualBf16Kernel>();
}

void Gemma4PLEBlock::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto in_mem = getSrcMemoryAtPort(0);
    auto pli_mem = getSrcMemoryAtPort(1);
    auto out_mem = getDstMemoryAtPort(0);

    const auto& in_dims = in_mem->getStaticDims();
    OPENVINO_ASSERT(in_dims.size() == 3, "Gemma4PLEBlock input must be rank-3");
    const int H = m_config.hidden_size;
    const int Hp = m_config.hidden_per_layer;
    const size_t B = in_dims[0];
    const size_t T = in_dims[1];
    OPENVINO_ASSERT(static_cast<int>(in_dims[2]) == H, "Gemma4PLEBlock input last dim mismatch");

    const auto& pli_dims = pli_mem->getStaticDims();
    OPENVINO_ASSERT(pli_dims.size() == 3 && pli_dims[0] == B && pli_dims[1] == T &&
                        static_cast<int>(pli_dims[2]) == Hp,
                    "Gemma4PLEBlock per_layer_input shape mismatch");

    const auto* in_bf = in_mem->getDataAs<ov::bfloat16>();
    const auto* pli_bf = pli_mem->getDataAs<ov::bfloat16>();
    auto* out_bf = out_mem->getDataAs<ov::bfloat16>();

    const float eps = m_config.eps;
    const size_t M = B * T;
    const size_t Hu = static_cast<size_t>(H);
    const size_t Hpu = static_cast<size_t>(Hp);

    // Canonicalize kernel_M.
    const size_t M_blk = BrgemmKernel::get_mblk_size();  // 32
    const size_t tail = M % M_blk;
    const size_t kernel_M = (M < M_blk) ? M : (M_blk + tail);

    // gemm1 writes to a thread-local f32 staging tile (kMblk x kNshard, ldc=kNshard).
    // gemm2 keeps its old behavior (writes to m_C_proj at ldc=Hu) so the row-wise RMS
    // reducer can read a contiguous row of length H.
    auto get_or_create = [&](std::unordered_map<size_t, std::shared_ptr<BrgemmKernel>>& cache,
                             size_t Mv,
                             size_t Kv,
                             size_t lda,
                             size_t ldc) {
        auto it = cache.find(Mv);
        if (it != cache.end()) {
            return it->second;
        }
        auto k = std::make_shared<BrgemmKernel>(Mv,
                                                kNshard,
                                                Kv,
                                                lda,
                                                Kv,
                                                ldc,
                                                /*b_transposed=*/true,
                                                ov::element::bf16,
                                                /*b_accumulate=*/false);
        cache.emplace(Mv, k);
        return k;
    };

    auto gemm1 = get_or_create(m_gemm1_cache, kernel_M, Hu, Hu, /*ldc=*/kNshard);
    auto gemm2 = get_or_create(m_gemm2_cache, kernel_M, Hpu, Hpu, /*ldc=*/Hu);

    // Workspace allocation (lazy / monotonic grow). m_C_gate is gone -- gemm1 output now
    // lives in a per-thread staging tile that stays hot in L1.
    if (m_gated_bf.size() < M * Hpu) {
        m_gated_bf.assign(M * Hpu, ov::bfloat16(0));
    }
    if (m_C_proj.size() < M * Hu) {
        m_C_proj.assign(M * Hu, 0.0f);
    }

    auto round_up_64 = [](size_t v) { return (v + 63) & ~size_t(63); };
    const size_t wsp_per = round_up_64(BrgemmKernel::get_wsp_size());
    const size_t scratchA_per = round_up_64(std::max(gemm1->get_scratch_a_size(), gemm2->get_scratch_a_size()));
    const size_t stage_per = round_up_64(kMblk * kNshard * sizeof(float));  // 4 KB, already 64-aligned
    // 64 byte head padding so the per-thread base is also cache-line aligned regardless of allocator.
    const size_t per_thread = round_up_64(wsp_per + scratchA_per + stage_per + 64);
    if (m_thread_wsp.size() < m_nthr * per_thread) {
        m_thread_wsp.assign(m_nthr * per_thread, 0);
        m_per_thread_bytes = per_thread;
        m_wsp_bytes = wsp_per;
        m_scratchA_bytes = scratchA_per;
        m_stage_bytes = stage_per;
    }

    const bool small_M = (M < M_blk);
    const size_t n_body_blocks = small_M ? 0 : (M / M_blk);
    const bool has_tail = small_M || (tail != 0);
    const size_t n_m_blocks = n_body_blocks + (has_tail ? 1 : 0);
    const size_t n_gate_shards = Hpu / kNshard;
    const size_t n_proj_shards = Hu / kNshard;

    static PleTrace s_trace;

    // Fused pass 1+2: per shard-tile, run gemm1 into a thread-local f32 staging tile
    // (kMblk x kNshard, ~4 KB, stays in L1), then immediately apply epi1 (gelu_tanh * pli -> bf16)
    // and write the bf16 result into m_gated_bf. m_C_gate (M*Hp f32, ~128 KB for M=128) is gone.
    {
        ScopedTimer _t(s_trace, PleTrace::GEMM1);
        const auto* gate_combine = m_gate_combine.get();
        parallel_for2d(n_m_blocks, n_gate_shards, [&](size_t mi, size_t ns) {
            const int tid = parallel_get_thread_num();
            uint8_t* wsp = m_thread_wsp.data() + tid * m_per_thread_bytes;
            uint8_t* scratch_a = wsp + wsp_per;
            float* stage = reinterpret_cast<float*>(scratch_a + scratchA_per);

            const bool is_tail = has_tail && (mi == n_body_blocks);
            const size_t m_start = is_tail ? (n_body_blocks * M_blk) : (mi * M_blk);
            const size_t m_rows = is_tail ? (M - m_start) : M_blk;

            const size_t n0 = ns * kNshard;
            // GEMM into thread-local 32x32 f32 staging tile.
            gemm1->executeGemm(is_tail,
                               const_cast<ov::bfloat16*>(in_bf + m_start * Hu),
                               m_packed_gate_w.data() + ns * m_gate_shard_bytes,
                               stage,
                               nullptr,
                               nullptr,
                               wsp,
                               scratch_a);

            // Inline epi1 over the staging tile (still in L1) -> bf16 m_gated_bf shard column.
            gate_combine->call(stage,
                               /*gate_stride=*/kNshard,
                               pli_bf + m_start * Hpu + n0,
                               /*pli_stride=*/Hpu,
                               m_gated_bf.data() + m_start * Hpu + n0,
                               m_rows,
                               kNshard);
        });
    }

    // Pass 3: Stage2 GEMM (gated * proj_w^T -> C_proj). Unchanged: still lands in m_C_proj.
    {
        ScopedTimer _t(s_trace, PleTrace::GEMM2);
        parallel_for2d(n_m_blocks, n_proj_shards, [&](size_t mi, size_t ns) {
            const int tid = parallel_get_thread_num();
            uint8_t* wsp = m_thread_wsp.data() + tid * m_per_thread_bytes;
            uint8_t* scratch_a = wsp + wsp_per;

            const bool is_tail = has_tail && (mi == n_body_blocks);
            const size_t m_start = is_tail ? (n_body_blocks * M_blk) : (mi * M_blk);

            const size_t n0 = ns * kNshard;
            gemm2->executeGemm(is_tail,
                               m_gated_bf.data() + m_start * Hpu,
                               m_packed_proj_w.data() + ns * m_proj_shard_bytes,
                               m_C_proj.data() + m_start * Hu + n0,
                               nullptr,
                               nullptr,
                               wsp,
                               scratch_a);
        });
    }

    // Pass 4: per-row RMSNorm + residual + LayerScale + cast to bf16 (JIT).
    {
        ScopedTimer _t(s_trace, PleTrace::EPI2);
        const float inv_h = 1.0f / static_cast<float>(H);
        const float layer_scalar = m_config.layer_scalar;
        const auto* rms_combine = m_rms_combine.get();
        parallel_for(M, [&](size_t m) {
            rms_combine->call(m_C_proj.data() + m * Hu,
                              m_norm_gamma_f32.data(),
                              in_bf + m * Hu,
                              out_bf + m * Hu,
                              inv_h,
                              eps,
                              layer_scalar,
                              Hu);
        });
    }

    if (PleTrace::enabled()) {
        s_trace.maybe_print(getName(), M);
    }
}

bool Gemma4PLEBlock::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        auto node = ov::as_type_ptr<const Gemma4PLEBlockNode>(op);
        if (!node) {
            errorMessage = "Not Gemma4PLEBlockNode";
            return false;
        }
        const auto& cfg = node->get_config();
        if (cfg.hidden_size <= 0 || cfg.hidden_per_layer <= 0) {
            errorMessage = "Gemma4PLEBlock invalid hidden sizes";
            return false;
        }
        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
            errorMessage = "Gemma4PLEBlock requires AMX-bf16";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
#else
    (void)op;
    errorMessage = "Gemma4PLEBlock is x86_64 only";
    return false;
#endif
}

}  // namespace ov::intel_cpu::node
