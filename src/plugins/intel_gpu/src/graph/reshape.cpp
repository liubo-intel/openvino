// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

#include "shape_nodes.hpp"

namespace cldnn {

primitive_type_id reshape::type_id() {
    static primitive_type_base<reshape> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto input_layout = impl_param.get_non_padded_input_layout();
    auto desc = impl_param.typed_desc<reshape>();
    auto sizes = desc->output_shape.sizes();
    auto input_sizes = input_layout.get_tensor().sizes();
    size_t need_recalc = 0;
    uint32_t shape_count = 1;

    for (size_t i = 0; i < sizes.size(); i++) {
        if (sizes[i] == -1) {
            if (need_recalc) {
                CLDNN_ERROR_MESSAGE(desc->id, "Only one dimension of the new shape can be -1");
            }
            need_recalc = i;
            continue;
        }
        if (sizes[i] == 0) {
            sizes[i] = input_sizes[i];
        }
        shape_count *= sizes[i];
    }
    if (need_recalc)
        sizes[need_recalc] = static_cast<int>(input_layout.count()) / shape_count;

    return layout{input_layout.data_type, input_layout.format, tensor(sizes)};
}

template<typename ShapeType>
std::vector<layout> reshape_inst::calc_output_layouts(reshape_node const& node, const kernel_impl_params& impl_param) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto prim = impl_param.typed_desc<reshape>();
    auto input_layout = impl_param.get_input_layout(0);

    auto& memory_deps = impl_param.memory_deps;
    // On program build stage for the cases with pattern being stored in a runtime tensor
    // we return output_partial_shape taken from the original model intead of something like PartialShape::dynamic(rank)
    // as ngraph may refine output shape using interval arithmetic
    if (memory_deps.empty() && prim->output_pattern.empty()) {
        return { layout{prim->output_partial_shape, input_layout.data_type, format::adjust_to_rank(input_layout.format, prim->output_partial_shape.size())} };
    }

    ov::op::v1::Reshape op;
    op.set_special_zero(prim->special_zero);

    ShapeType pattern_shape = impl_param.input_layouts.size() == 2 ? impl_param.get_input_layout(1).get<ShapeType>()
                                                           : ShapeType(ov::Shape{ prim->output_pattern.size() });
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        pattern_shape,
    };


    if (!memory_deps.empty()) {
        auto pattern_mem = memory_deps.at(1);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> pattern_lock(pattern_mem, node.get_program().get_stream());

        auto pattern_ptr = pattern_lock.data();
        auto pattern_tensor = make_host_tensor(pattern_mem->get_layout(), pattern_ptr);

        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, pattern_tensor},
        };

        shape_infer(&op, input_shapes, output_shapes, const_data);
    } else {
        auto pattern_data = prim->output_pattern;
        auto pattern_tensor = make_host_tensor({pattern_shape, data_types::i64, format::bfyx}, static_cast<void*>(pattern_data.data()));
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, pattern_tensor},
        };

        shape_infer(&op, input_shapes, output_shapes, const_data);
    }

    return { layout{output_shapes[0], input_layout.data_type, format::adjust_to_rank(input_layout.format, output_shapes[0].size())} };
}

std::string reshape_inst::to_string(reshape_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite reshape_info;
    reshape_info.add("input id", input.id());
    reshape_info.add("output shape", desc->output_shape);

    node_info->add("reshape info", reshape_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reshape_inst::typed_primitive_inst(network& network, reshape_node const& node) : parent(network, node, false) {
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                    "Input layout data typr",
                                    input_layout.data_type,
                                    "output layout data type",
                                    output_layout.data_type,
                                    "");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Output layout count",
                          output_layout.count(),
                          "input layout count",
                          input_layout.count(),
                          "Output layout of reshape primitive changes size of input buffer");

    // if reshape operated in-place, postpone creation of the output until network run,
    // then create new memory object as the reinterpreted output of the previous primitive
    if (!node.can_be_optimized())
        _output = allocate_output();
    else
        reuse_input();
}

void reshape_inst::on_execute() {
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void reshape_inst::reuse_input() {
    build_deps();  // reshape need deps
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}

}  // namespace cldnn
