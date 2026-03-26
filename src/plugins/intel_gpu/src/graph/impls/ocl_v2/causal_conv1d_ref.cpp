// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_conv1d_ref.hpp"

#include "intel_gpu/primitives/causal_conv1d.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class CausalConv1DRefGenerator : public KernelGenerator {
public:
    CausalConv1DRefGenerator() : KernelGenerator("causal_conv1d_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto& hidden_shape = params.get_input_layout(0).get_partial_shape();
        const auto& cache_shape = params.get_input_layout(1).get_partial_shape();
        const auto& weight_shape = params.get_input_layout(2).get_partial_shape();

        const size_t hidden_size = hidden_shape[1].get_length();
        const size_t cache_len = cache_shape[2].get_length();
        const size_t kernel_size = weight_shape[weight_shape.size() - 1].get_length();
        const size_t has_bias = params.input_layouts.size() == 4 ? 1 : 0;

        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("CACHE_LEN", cache_len);
        jit.make("KERNEL_SIZE", kernel_size);
        jit.make("HAS_BIAS", has_bias);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

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

            const auto& hidden_shape = params.get_input_layout(0).get_partial_shape();
            const size_t batch = hidden_shape[0].get_length();
            const size_t hidden_size = hidden_shape[1].get_length();
            const size_t seq_len = hidden_shape[2].get_length();

            wgs.global = {batch, hidden_size, 1};
            wgs.local = {1, 256, 1};

            if (wgs.local[1] > hidden_size) {
                wgs.local[1] = hidden_size;
            }

            kd.params.scalars.clear();
            scalar_desc desc;
            desc.t = scalar_desc::Types::INT32;
            desc.v.s32 = static_cast<int32_t>(seq_len);
            kd.params.scalars.push_back(desc);
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

std::unique_ptr<primitive_impl> CausalConv1DRef::create_impl(const program_node& node,
                                                             const RuntimeParams& params) const {
    assert(node.is_type<causal_conv1d>());
    return std::make_unique<CausalConv1DRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::causal_conv1d)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::CausalConv1DRefImpl)
