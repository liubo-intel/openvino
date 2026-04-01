// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d_ref.hpp"

#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/primitives/causal_conv1d.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

size_t get_hidden_size(const cldnn::layout& hidden_layout, bool transposed_io) {
    const auto& pshape = hidden_layout.get_partial_shape();
    OPENVINO_ASSERT(pshape.rank().is_static() && pshape.rank().get_length() >= 3, "[GPU] causal_conv1d expects rank >= 3 for input_embeds");

    const auto& second_last_dim = pshape[pshape.size() - 2];
    const auto& last_dim = pshape[pshape.size() - 1];

    if (transposed_io) {
        OPENVINO_ASSERT(last_dim.is_static(), "[GPU] causal_conv1d expects static hidden size (last dim) for transposed input_embeds");
        return last_dim.get_length();
    }

    OPENVINO_ASSERT(second_last_dim.is_static(), "[GPU] causal_conv1d expects static hidden size (second-last dim) for non-transposed input_embeds");
    return second_last_dim.get_length();
}

size_t get_seq_len(const cldnn::layout& hidden_layout, bool transposed_io) {
    const auto& pshape = hidden_layout.get_partial_shape();
    OPENVINO_ASSERT(pshape.rank().is_static() && pshape.rank().get_length() >= 3, "[GPU] causal_conv1d expects rank >= 3 for input_embeds");

    const auto& second_last_dim = pshape[pshape.size() - 2];
    const auto& last_dim = pshape[pshape.size() - 1];

    if (transposed_io) {
        OPENVINO_ASSERT(second_last_dim.is_static(),
                        "[GPU] causal_conv1d expects static seq_len (second-last dim) at execution time for transposed input_embeds");
        return second_last_dim.get_length();
    }

    OPENVINO_ASSERT(last_dim.is_static(), "[GPU] causal_conv1d expects static seq_len (last dim) at execution time for non-transposed input_embeds");
    return last_dim.get_length();
}

size_t get_last_static_dim(const ov::PartialShape& pshape, const char* tensor_name) {
    OPENVINO_ASSERT(pshape.rank().is_static() && pshape.rank().get_length() > 0, "[GPU] ", tensor_name, " rank must be static and non-zero");
    const auto& last_dim = pshape[pshape.size() - 1];
    OPENVINO_ASSERT(last_dim.is_static(), "[GPU] ", tensor_name, " trailing dimension must be static");
    return last_dim.get_length();
}

class CausalConv1DRefGenerator : public KernelGenerator {
public:
    CausalConv1DRefGenerator() : KernelGenerator("causal_conv1d_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<cldnn::causal_conv1d>();
        const bool transposed_io = desc->transposed_io;

        const auto& hidden_layout = params.get_input_layout(0);
        const auto& cache_layout = params.get_input_layout(1);
        const auto& weight_layout = params.get_input_layout(2);

        const size_t hidden_size = get_hidden_size(hidden_layout, transposed_io);
        const size_t cache_len = get_last_static_dim(cache_layout.get_partial_shape(), "conv_state");
        const size_t kernel_size = get_last_static_dim(weight_layout.get_partial_shape(), "conv_weight");
        const size_t has_bias = params.input_layouts.size() == 4 ? 1 : 0;

        OPENVINO_ASSERT(cache_len == kernel_size,
                        "[GPU] causal_conv1d currently supports CACHE_LEN == KERNEL_SIZE only, but got CACHE_LEN=",
                        cache_len,
                        ", KERNEL_SIZE=",
                        kernel_size);

        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("CACHE_LEN", cache_len);
        jit.make("KERNEL_SIZE", kernel_size);
        jit.make("HAS_BIAS", has_bias);
        jit.make("TRANSPOSED_IO", transposed_io ? 1 : 0);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        // seq_len scalar
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& prim_desc = params.typed_desc<cldnn::causal_conv1d>();
            const bool transposed_io = prim_desc->transposed_io;

            const auto& hidden_layout = params.get_input_layout(0);
            const size_t batch = extract_channel(ChannelName::BATCH, hidden_layout);
            const size_t hidden_size = get_hidden_size(hidden_layout, transposed_io);
            const size_t seq_len = get_seq_len(hidden_layout, transposed_io);

            wgs.global = {batch, hidden_size, 1};
            wgs.local = {1, 256, 1};

            if (wgs.local[1] > hidden_size) {
                wgs.local[1] = hidden_size;
            }

            kd.params.scalars.clear();
            scalar_desc seq_len_scalar;
            seq_len_scalar.t = scalar_desc::Types::INT32;
            seq_len_scalar.v.s32 = static_cast<int32_t>(seq_len);
            kd.params.scalars.push_back(seq_len_scalar);
        }};
    }
};

class CausalConv1DRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::CausalConv1DRefImpl)

    Stage::Ptr causal_conv1d_stage = make_stage<CausalConv1DRefGenerator>();

    CausalConv1DRefImpl() : PrimitiveImplOCL(CausalConv1DRef::get_type_info_static()) {}
    CausalConv1DRefImpl(const program_node& node, const RuntimeParams& params) : CausalConv1DRefImpl() {
        add_stage(causal_conv1d_stage, params);
    }

    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        auto args = PrimitiveImplOCL::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<cldnn::causal_conv1d>();

        if (desc->variable_info.variable_id.empty()) {
            return args;
        }

        auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);

        OPENVINO_ASSERT(args.inputs.size() >= 2, "[GPU] causal_conv1d expects at least 2 inputs");
        OPENVINO_ASSERT(args.outputs.size() >= 2, "[GPU] causal_conv1d expects 2 outputs");

        // Reuse variable state once initialized; otherwise use conv_state input.
        if (variable.is_set()) {
            args.inputs[1] = variable.get_memory();
        }
        // Write updated state directly to variable memory and avoid explicit Assign copy.
        args.outputs[1] = variable.get_memory();

        return args;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        const auto& desc = instance.get_typed_desc<cldnn::causal_conv1d>();

        if (!desc->variable_info.variable_id.empty()) {
            auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);

            // Keep variable layout in sync with runtime state shape.
            variable.set_layout(instance.input_memory_ptr(1)->get_layout());

            // State source can change from conv_state input to variable after first iteration.
            for (const auto& stage_id : _order) {
                _stages[stage_id]->kd.need_args_update = true;
            }

            auto ev = PrimitiveImplOCL::execute(events, instance);
            variable.set();
            return ev;
        }

        return PrimitiveImplOCL::execute(events, instance);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<CausalConv1DRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> CausalConv1DRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<causal_conv1d>());
    return std::make_unique<CausalConv1DRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::causal_conv1d)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::CausalConv1DRefImpl)
