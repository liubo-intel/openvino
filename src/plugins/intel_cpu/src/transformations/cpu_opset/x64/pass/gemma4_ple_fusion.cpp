// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemma4_ple_fusion.hpp"

#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/cpu_opset/x64/op/gemma4_ple_block.hpp"

using namespace ov::pass::pattern;

ov::intel_cpu::Gemma4PLEFusion::Gemma4PLEFusion() {
    MATCHER_SCOPE(Gemma4PLEFusion);
    // Set GEMMA4_PLE_FUSE=0 to bypass this fusion pass for A/B perf debugging.
    static const bool s_fusion_disabled = []() {
        const char* v = std::getenv("GEMMA4_PLE_FUSE");
        return v && v[0] == '0';
    }();

    using ov::op::v0::Constant;
    using ov::op::v0::Convert;
    using ov::op::v0::MatMul;
    using ov::op::v1::Add;
    using ov::op::v1::Multiply;
    using ov::op::v7::Gelu;

    // residual / input  [B, T, hidden_size]
    auto input = any_input(rank_equals(3));

    // per_layer_input  [B, T, hidden_per_layer] (already gathered)
    auto per_layer_input = any_input(rank_equals(3));

    // gate_w (compressed FP16) -> Convert -> MatMul
    auto gate_w_compressed = wrap_type<Constant>(type_matches(ov::element::f16) && rank_equals(2));
    auto gate_w_f32 = wrap_type<Convert>({gate_w_compressed}, {{"destination_type", "f32"}});
    auto gate_matmul = wrap_type<MatMul>({input, gate_w_f32}, {{"transpose_a", false}, {"transpose_b", true}});

    // GELU(TANH approximation)
    auto gate_act = wrap_type<Gelu>({gate_matmul}, {{"approximation_mode", "TANH"}});

    // gate * per_layer_input  (commutative)
    auto gated = wrap_type<Multiply>({gate_act, per_layer_input}, {{"auto_broadcast", "numpy"}});

    // proj_w (compressed FP16) -> Convert -> MatMul
    auto proj_w_compressed = wrap_type<Constant>(type_matches(ov::element::f16) && rank_equals(2));
    auto proj_w_f32 = wrap_type<Convert>({proj_w_compressed}, {{"destination_type", "f32"}});
    auto proj_matmul = wrap_type<MatMul>({gated, proj_w_f32}, {{"transpose_a", false}, {"transpose_b", true}});

    // post_per_layer_input_norm: RMS internal op (already fused).
    // gamma must be an f32 constant: the JIT epi2 kernel reads it directly as f32 from the
    // packed `m_norm_gamma_f32` buffer prepared in createPrimitive().
    auto norm_gamma = wrap_type<Constant>(type_matches(ov::element::f32));
    auto rms = wrap_type<ov::op::internal::RMS>({proj_matmul, norm_gamma});

    // residual + RMS output. The optional trailing per-layer LayerScale Multiply is detected
    // manually in the callback: if we encoded it as a second `Or` branch, the matcher would fire
    // on Add first (topo order) and replace it before Multiply is visited, leaving the LayerScale
    // un-fused. Peeking at the consumer side of Add lets us absorb both nodes atomically.
    auto add_out = wrap_type<Add>({input, rms}, {{"auto_broadcast", "numpy"}});

    matcher_pass_callback callback = [=](Matcher& m) {
        if (s_fusion_disabled) {
            return false;
        }
        const auto& pm = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(pm.at(rms).get_node_shared_ptr());
        if (!rms_node) {
            return false;
        }

        const auto input_pshape = pm.at(input).get_partial_shape();
        const auto pli_pshape = pm.at(per_layer_input).get_partial_shape();
        if (!input_pshape.rank().is_static() || input_pshape.rank().get_length() != 3) {
            return false;
        }
        if (!pli_pshape.rank().is_static() || pli_pshape.rank().get_length() != 3) {
            return false;
        }
        const auto& in_last = input_pshape[2];
        const auto& pli_last = pli_pshape[2];
        if (!in_last.is_static() || !pli_last.is_static()) {
            return false;
        }

        // gate_w: [hidden_per_layer, hidden_size]
        // proj_w: [hidden_size, hidden_per_layer]
        const auto gate_w_pshape = pm.at(gate_w_compressed).get_partial_shape();
        const auto proj_w_pshape = pm.at(proj_w_compressed).get_partial_shape();
        if (!gate_w_pshape.is_static() || !proj_w_pshape.is_static()) {
            return false;
        }
        const auto& gate_shape = gate_w_pshape.get_shape();
        const auto& proj_shape = proj_w_pshape.get_shape();
        if (gate_shape.size() != 2 || proj_shape.size() != 2) {
            return false;
        }

        const int hidden_size = static_cast<int>(in_last.get_length());
        const int hidden_per_layer = static_cast<int>(pli_last.get_length());

        // Sanity: weight shapes must match the deduced dimensions.
        if (static_cast<int>(gate_shape[0]) != hidden_per_layer || static_cast<int>(gate_shape[1]) != hidden_size) {
            return false;
        }
        if (static_cast<int>(proj_shape[0]) != hidden_size || static_cast<int>(proj_shape[1]) != hidden_per_layer) {
            return false;
        }

        // The fused kernel shards N along a fixed `kNshard = 32` tile and reuses one BRGEMM
        // instance for every shard. There is no N-tail path: shards are not allowed to be
        // partial. Reject any geometry where H or Hp is not a multiple of 32 so we fall
        // back to the original RMS+MatMul+Gelu subgraph instead of throwing later in
        // Gemma4PLEBlock::createPrimitive().
        constexpr int kNshard = 32;
        if (hidden_size % kNshard != 0 || hidden_per_layer % kNshard != 0) {
            return false;
        }

        Gemma4PLEBlockNode::Config cfg{};
        cfg.hidden_size = hidden_size;
        cfg.hidden_per_layer = hidden_per_layer;
        cfg.eps = static_cast<float>(rms_node->get_epsilon());
        cfg.layer_scalar = 1.0F;

        // Peek at Add's consumer chain for an optional `* layer_scalar`. PyTorch export emits
        //   Add -> Multiply(Add, Constant<f16>[1,1,1] -> Convert<f32>)
        // for `hidden_states *= self.layer_scalar`. If that exact pattern is the *only* consumer,
        // fold the scalar into the fused op and replace both Add and Multiply.
        std::shared_ptr<ov::Node> scale_mul;
        const auto add_node = root;
        if (add_node->get_output_size() == 1 && add_node->output(0).get_target_inputs().size() == 1) {
            auto consumer = add_node->output(0).get_target_inputs().begin()->get_node();
            auto* mul = ov::as_type<Multiply>(consumer);
            if (mul) {
                // Find the operand that is *not* `add_out`.
                ov::Output<ov::Node> scale_op;
                bool first_is_add = (mul->input_value(0).get_node_shared_ptr() == add_node);
                bool second_is_add = (mul->input_value(1).get_node_shared_ptr() == add_node);
                if (first_is_add ^ second_is_add) {
                    scale_op = first_is_add ? mul->input_value(1) : mul->input_value(0);
                    auto scale_node = scale_op.get_node_shared_ptr();
                    // Walk through optional Convert(f32).
                    if (auto cvt = ov::as_type_ptr<Convert>(scale_node)) {
                        if (cvt->get_destination_type() == ov::element::f32) {
                            scale_node = cvt->input_value(0).get_node_shared_ptr();
                        }
                    }
                    auto scale_const = ov::as_type_ptr<Constant>(scale_node);
                    if (scale_const && (scale_const->get_element_type() == ov::element::f32 ||
                                        scale_const->get_element_type() == ov::element::f16)) {
                        const auto& scalar_shape = scale_const->get_shape();
                        const size_t scalar_count = std::accumulate(scalar_shape.begin(),
                                                                    scalar_shape.end(),
                                                                    size_t{1},
                                                                    std::multiplies<size_t>());
                        if (scalar_count == 1) {
                            const auto values = scale_const->cast_vector<float>();
                            if (values.size() == 1) {
                                cfg.layer_scalar = values[0];
                                scale_mul = consumer->shared_from_this();
                            }
                        }
                    }
                }
            }
        }

        ov::OutputVector new_args{
            pm.at(input),
            pm.at(per_layer_input),
            pm.at(gate_w_compressed),
            pm.at(proj_w_compressed),
            rms_node->input_value(1),  // gamma
        };

        auto new_node = std::make_shared<Gemma4PLEBlockNode>(new_args, cfg);
        // Friendly name follows the outermost replaced node so downstream RT info stays predictable.
        auto outer = scale_mul ? scale_mul : root;
        new_node->set_friendly_name(outer->get_friendly_name());

        ov::NodeVector replaced_nodes = m.get_matched_nodes();
        if (scale_mul) {
            replaced_nodes.push_back(scale_mul);
        }
        ov::copy_runtime_info(replaced_nodes, new_node);

        // Plugin support gate.
        if (!transformation_callback(new_node)) {
            return false;
        }

        ov::replace_node(outer, new_node);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(add_out, matcher_name);
    register_matcher(matcher, callback);
}
