// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "causal_conv1d_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct CausalConv1DRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::causal_conv1d::ref")

    explicit CausalConv1DRef(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;

    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        // Kernel code relies on compile-time constants for cache/kernel dimensions.
        // Keep manager visible for dynamic-shape graph preparation, but compile only static params.
        return !params.is_dynamic();
    }

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<causal_conv1d>());

        const bool transposed_io = node.as<causal_conv1d>().get_primitive()->transposed_io;

        static constexpr std::array supported_fmts = {
            format::bfyx,
            format::byxf,
        };

        static constexpr std::array supported_state_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            data_types::f16,
            data_types::f32,
        };

        if (node.get_dependencies().size() < 3 || node.get_dependencies().size() > 4) {
            return false;
        }

        const auto& hidden_in = node.get_input_layout(0);
        if (!one_of(hidden_in.data_type, supported_types)) {
            return false;
        }
        if (transposed_io) {
            if (!one_of(hidden_in.format, supported_fmts)) {
                return false;
            }
        } else {
            if (hidden_in.format != format::bfyx) {
                return false;
            }
        }

        for (size_t i = 1; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!one_of(in_layout.format, supported_state_fmts) || !one_of(in_layout.data_type, supported_types)) {
                return false;
            }
        }

        const auto& hidden_out = node.get_output_layout(0);
        if (!one_of(hidden_out.data_type, supported_types)) {
            return false;
        }
        if (transposed_io) {
            if (!one_of(hidden_out.format, supported_fmts)) {
                return false;
            }
        } else {
            if (hidden_out.format != format::bfyx) {
                return false;
            }
        }

        for (size_t i = 1; i < node.get_output_layouts().size(); i++) {
            const auto& out_layout = node.get_output_layout(i);
            if (!one_of(out_layout.format, supported_state_fmts) || !one_of(out_layout.data_type, supported_types)) {
                return false;
            }
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
