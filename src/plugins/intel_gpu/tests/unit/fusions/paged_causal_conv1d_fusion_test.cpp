// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_causal_conv1d.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstdint>

using namespace cldnn;
using namespace ::tests;

namespace {

struct paged_causal_conv1d_fusion_test_params {
    int32_t tokens;
    int32_t num_sequences;
    int32_t hidden_size;
    int32_t kernel_size;
    data_types data_type;
    bool with_bias;
    bool with_reshape;  // reshape between conv1d and swish (tests fusion through reshape)
};

struct paged_causal_conv1d_fusion_test : public ::testing::TestWithParam<paged_causal_conv1d_fusion_test_params> {
    tests::random_generator rg;
    cldnn::engine& engine = get_test_engine();

    struct paging_desc {
        std::vector<int32_t> subsequence_begins;
        std::vector<int32_t> block_indices;
        std::vector<int32_t> block_indices_begins;
        std::vector<int32_t> past_lens;
        std::vector<int32_t> cache_interval;
        int32_t num_blocks = 0;
    };

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    static paging_desc make_paging_desc(int32_t tokens, int32_t num_sequences) {
        paging_desc d;
        d.subsequence_begins.push_back(0);
        d.block_indices_begins.push_back(0);

        int32_t consumed = 0;
        int32_t total_blocks = 0;
        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t rem = tokens - consumed;
            const int32_t seq_tokens = rem / (num_sequences - seq);
            consumed += seq_tokens;
            d.subsequence_begins.push_back(consumed);

            const int32_t past_len = 1 + (seq % 3);
            const int32_t interval = 2 + (seq % 2);
            d.past_lens.push_back(past_len);
            d.cache_interval.push_back(interval);

            const int32_t prev_nums = past_len % interval;
            const int32_t total_cached = prev_nums + seq_tokens;
            const int32_t max_slot = 1 + (total_cached - 1) / interval;
            const int32_t required_slots = 1 + std::max<int32_t>(1, max_slot);
            for (int32_t i = 0; i < required_slots; i++) {
                d.block_indices.push_back(total_blocks + i);
            }
            total_blocks += required_slots;
            d.block_indices_begins.push_back(total_blocks);
        }
        d.num_blocks = total_blocks;
        return d;
    }

    topology create_topology(const paged_causal_conv1d_fusion_test_params& p,
                             const paging_desc& page,
                             data_types dt) {
        const layout input_layout_desc({p.tokens, p.hidden_size}, dt, format::bfyx);
        const layout state_layout_desc({page.num_blocks, p.hidden_size, p.kernel_size}, dt, format::bfyx);
        const layout weight_layout_desc({p.hidden_size, 1, p.kernel_size}, dt, format::bfyx);
        const layout bias_layout_desc({p.with_bias ? p.hidden_size : 0}, dt, format::bfyx);
        const layout subseq_layout_desc({static_cast<int32_t>(page.subsequence_begins.size())}, data_types::i32, format::bfyx);
        const layout block_idx_layout_desc({static_cast<int32_t>(page.block_indices.size())}, data_types::i32, format::bfyx);
        const layout block_begins_layout_desc({static_cast<int32_t>(page.block_indices_begins.size())}, data_types::i32, format::bfyx);
        const layout past_lens_layout_desc({static_cast<int32_t>(page.past_lens.size())}, data_types::i32, format::bfyx);
        const layout interval_layout_desc({static_cast<int32_t>(page.cache_interval.size())}, data_types::i32, format::bfyx);

        topology topo;
        topo.add(input_layout("input_embeds", input_layout_desc));
        topo.add(input_layout("conv_state_table", state_layout_desc));
        topo.add(input_layout("conv_weight", weight_layout_desc));
        topo.add(input_layout("conv_bias", bias_layout_desc));
        topo.add(input_layout("subsequence_begins", subseq_layout_desc));
        topo.add(input_layout("block_indices", block_idx_layout_desc));
        topo.add(input_layout("block_indices_begins", block_begins_layout_desc));
        topo.add(input_layout("past_lens", past_lens_layout_desc));
        topo.add(input_layout("cache_interval", interval_layout_desc));

        topo.add(paged_causal_conv1d("conv1d",
                                     {input_info("input_embeds"),
                                      input_info("conv_state_table"),
                                      input_info("conv_weight"),
                                      input_info("conv_bias"),
                                      input_info("subsequence_begins"),
                                      input_info("block_indices"),
                                      input_info("block_indices_begins"),
                                      input_info("past_lens"),
                                      input_info("cache_interval")}));

        primitive_id last = "conv1d";

        if (p.with_reshape) {
            topo.add(reshape("reshape", input_info(last), tensor(1, p.tokens, p.hidden_size, 1)));
            last = "reshape";
        }

        topo.add(activation("swish", input_info(last), activation_func::swish));
        last = "swish";

        if (p.with_reshape) {
            topo.add(reshape("reshape_back", input_info(last), tensor(p.tokens, p.hidden_size, 1, 1)));
            last = "reshape_back";
        }

        topo.add(reorder("output", input_info(last), format::bfyx, dt));
        return topo;
    }

    void set_network_inputs(network& net,
                            cldnn::memory::ptr input_mem,
                            cldnn::memory::ptr state_mem,
                            cldnn::memory::ptr weight_mem,
                            cldnn::memory::ptr bias_mem,
                            cldnn::memory::ptr subseq_mem,
                            cldnn::memory::ptr block_idx_mem,
                            cldnn::memory::ptr block_begins_mem,
                            cldnn::memory::ptr past_lens_mem,
                            cldnn::memory::ptr interval_mem) {
        net.set_input_data("input_embeds", input_mem);
        net.set_input_data("conv_state_table", state_mem);
        net.set_input_data("conv_weight", weight_mem);
        net.set_input_data("conv_bias", bias_mem);
        net.set_input_data("subsequence_begins", subseq_mem);
        net.set_input_data("block_indices", block_idx_mem);
        net.set_input_data("block_indices_begins", block_begins_mem);
        net.set_input_data("past_lens", past_lens_mem);
        net.set_input_data("cache_interval", interval_mem);
    }

    // Reference: conv1d + swish on CPU
    template <typename T>
    static void run_reference_with_swish(const std::vector<T>& input_embeds,
                                         const std::vector<T>& conv_weight,
                                         const std::vector<T>& conv_bias,
                                         std::vector<T>& conv_state_table,
                                         const paging_desc& page,
                                         int32_t hidden_size,
                                         int32_t kernel_size,
                                         std::vector<T>& output) {
        const int32_t token_count = static_cast<int32_t>(input_embeds.size()) / hidden_size;
        output.resize(static_cast<size_t>(token_count) * hidden_size);

        auto state_off = [hidden_size, kernel_size](int32_t block, int32_t h, int32_t k) {
            return (block * hidden_size + h) * kernel_size + k;
        };

        for (int32_t seq = 0; seq < static_cast<int32_t>(page.subsequence_begins.size()) - 1; seq++) {
            const int32_t token_begin = page.subsequence_begins[seq];
            const int32_t token_end = page.subsequence_begins[seq + 1];
            const int32_t blk_begin = page.block_indices_begins[seq];
            const int32_t blk_end = page.block_indices_begins[seq + 1];
            const int32_t block_span = blk_end - blk_begin;
            if (block_span <= 1)
                continue;

            const int32_t seq_interval = page.cache_interval[seq];
            const int32_t prev_nums = (seq_interval > 0) ? (page.past_lens[seq] % seq_interval) : 0;
            const int32_t seq_tokens = token_end - token_begin;
            const int32_t read_block = page.block_indices[blk_begin];

            for (int32_t h = 0; h < hidden_size; h++) {
                std::vector<float> state(static_cast<size_t>(kernel_size));
                for (int32_t k = 0; k < kernel_size; k++)
                    state[k] = static_cast<float>(conv_state_table[state_off(read_block, h, k)]);
                const float bias_val = conv_bias.empty() ? 0.0f : static_cast<float>(conv_bias[h]);

                for (int32_t t = 0; t < seq_tokens; t++) {
                    for (int32_t k = 0; k + 1 < kernel_size; k++)
                        state[k] = state[k + 1];
                    const int32_t tidx = token_begin + t;
                    state[kernel_size - 1] = static_cast<float>(input_embeds[tidx * hidden_size + h]);

                    float sum = bias_val;
                    for (int32_t k = 0; k < kernel_size; k++)
                        sum = std::fma(state[k], static_cast<float>(conv_weight[h * kernel_size + k]), sum);

                    sum = sum / (1.0f + std::exp(-sum));
                    output[tidx * hidden_size + h] = static_cast<T>(sum);

                    const int32_t cached = prev_nums + (t + 1);
                    const bool hit = (seq_interval > 0) && ((cached % seq_interval) == 0);
                    if (hit || t == seq_tokens - 1) {
                        const int32_t slot = (seq_interval > 0) ? (1 + (cached - 1) / seq_interval) : 1;
                        if (slot >= 1 && slot < block_span) {
                            const int32_t pb = page.block_indices[blk_begin + slot];
                            for (int32_t k = 0; k < kernel_size; k++)
                                conv_state_table[state_off(pb, h, k)] = static_cast<T>(state[k]);
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void execute_t(const paged_causal_conv1d_fusion_test_params& p, data_types dt, float tolerance) {
        const auto page = make_paging_desc(p.tokens, p.num_sequences);

        const layout input_l({p.tokens, p.hidden_size}, dt, format::bfyx);
        const layout state_l({page.num_blocks, p.hidden_size, p.kernel_size}, dt, format::bfyx);
        const layout weight_l({p.hidden_size, 1, p.kernel_size}, dt, format::bfyx);
        const layout bias_l({p.with_bias ? p.hidden_size : 0}, dt, format::bfyx);
        const layout subseq_l({static_cast<int32_t>(page.subsequence_begins.size())}, data_types::i32, format::bfyx);
        const layout block_l({static_cast<int32_t>(page.block_indices.size())}, data_types::i32, format::bfyx);
        const layout begins_l({static_cast<int32_t>(page.block_indices_begins.size())}, data_types::i32, format::bfyx);
        const layout past_l({static_cast<int32_t>(page.past_lens.size())}, data_types::i32, format::bfyx);
        const layout interval_l({static_cast<int32_t>(page.cache_interval.size())}, data_types::i32, format::bfyx);

        auto input_mem = engine.allocate_memory(input_l);
        auto weight_mem = engine.allocate_memory(weight_l);
        auto bias_mem = engine.allocate_memory(bias_l);
        auto subseq_mem = engine.allocate_memory(subseq_l);
        auto block_mem = engine.allocate_memory(block_l);
        auto begins_mem = engine.allocate_memory(begins_l);
        auto past_mem = engine.allocate_memory(past_l);
        auto interval_mem = engine.allocate_memory(interval_l);

        auto input_data = rg.generate_random_1d<T>(ov::shape_size(input_l.get_shape()), -1.0f, 1.0f, 256);
        auto state_data = rg.generate_random_1d<T>(ov::shape_size(state_l.get_shape()), -1.0f, 1.0f, 256);
        auto weight_data = rg.generate_random_1d<T>(ov::shape_size(weight_l.get_shape()), -1.0f, 1.0f, 256);
        std::vector<T> bias_data;
        if (p.with_bias)
            bias_data = rg.generate_random_1d<T>(ov::shape_size(bias_l.get_shape()), -0.5f, 0.5f, 256);

        set_values(input_mem, input_data);
        set_values(weight_mem, weight_data);
        if (p.with_bias)
            set_values(bias_mem, bias_data);
        set_values(subseq_mem, page.subsequence_begins);
        set_values(block_mem, page.block_indices);
        set_values(begins_mem, page.block_indices_begins);
        set_values(past_mem, page.past_lens);
        set_values(interval_mem, page.cache_interval);

        // CPU reference: conv1d + swish
        auto ref_state = state_data;
        std::vector<T> ref_output;
        run_reference_with_swish(input_data, weight_data, bias_data, ref_state,
                                 page, p.hidden_size, p.kernel_size, ref_output);

        // GPU fused network: topology has conv1d → swish, optimize_data fuses them
        auto state_mem = engine.allocate_memory(state_l);
        set_values(state_mem, state_data);

        auto topo = create_topology(p, page, dt);
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::optimize_data(true));
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);

        set_network_inputs(*net, input_mem, state_mem, weight_mem, bias_mem,
                           subseq_mem, block_mem, begins_mem, past_mem, interval_mem);
        auto outputs = net->execute();
        auto out_mem = outputs.at("output").get_memory();

        // Verify fusion happened: swish should not be in executed primitives
        {
            auto exec = net->get_executed_primitives();
            ASSERT_TRUE(exec.find("swish") == exec.end())
                << "Swish should be fused into paged_causal_conv1d, not executed separately";
        }

        // Compare GPU fused output vs CPU reference
        ASSERT_EQ(out_mem->count(), ref_output.size());
        {
            cldnn::mem_lock<T, mem_lock_type::read> gpu_out(out_mem, get_test_stream());
            for (size_t i = 0; i < ref_output.size(); i++) {
                ASSERT_NEAR(static_cast<float>(gpu_out[i]), static_cast<float>(ref_output[i]), tolerance)
                    << "Mismatch at index " << i;
            }
        }
    }

    void execute(const paged_causal_conv1d_fusion_test_params& p) {
        if (p.data_type == data_types::f16) {
            execute_t<ov::float16>(p, p.data_type, 0.05f);
        } else {
            execute_t<float>(p, p.data_type, 1e-4f);
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<paged_causal_conv1d_fusion_test_params>& info) {
        const auto& p = info.param;
        return "tokens_" + std::to_string(p.tokens) +
               "_seq_" + std::to_string(p.num_sequences) +
               "_hidden_" + std::to_string(p.hidden_size) +
               "_kernel_" + std::to_string(p.kernel_size) +
               "_" + (p.data_type == data_types::f16 ? "f16" : "f32") +
               (p.with_bias ? "_bias" : "_no_bias") +
               (p.with_reshape ? "_reshape" : "");
    }
};

class paged_causal_conv1d_fuse_swish : public paged_causal_conv1d_fusion_test {};
TEST_P(paged_causal_conv1d_fuse_swish, basic) {
    execute(GetParam());
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                         paged_causal_conv1d_fuse_swish,
                         ::testing::Values(
                             // f32 with bias, direct
                             paged_causal_conv1d_fusion_test_params{8, 2, 16, 4, data_types::f32, true, false},
                             paged_causal_conv1d_fusion_test_params{12, 3, 32, 3, data_types::f32, true, false},
                             // f32 without bias
                             paged_causal_conv1d_fusion_test_params{8, 2, 16, 4, data_types::f32, false, false},
                             // f32 with reshape between conv1d and swish
                             paged_causal_conv1d_fusion_test_params{8, 2, 16, 4, data_types::f32, true, true},
                             paged_causal_conv1d_fusion_test_params{12, 3, 32, 3, data_types::f32, true, true},
                             // f16 with bias
                             paged_causal_conv1d_fusion_test_params{8, 2, 16, 4, data_types::f16, true, false},
                             paged_causal_conv1d_fusion_test_params{12, 3, 32, 3, data_types::f16, true, false},
                             // f16 with reshape
                             paged_causal_conv1d_fusion_test_params{8, 2, 16, 4, data_types::f16, true, true}
                         ),
                         paged_causal_conv1d_fusion_test::PrintToStringParamName);

}  // namespace
