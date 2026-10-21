// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_mvn_last_axis_to_3d.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::intel_cpu::ReshapeMVNLastAxisTo3D::ReshapeMVNLastAxisTo3D() {
    MATCHER_SCOPE(ReshapeMVNLastAxisTo3D);
    using namespace ov::pass::pattern;

    auto data_m = any_input(has_static_rank());
    auto axes_m = wrap_type<ov::op::v0::Constant>();
    auto mvn_m = wrap_type<ov::op::v6::MVN>({data_m, axes_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto mvn = ov::as_type_ptr<ov::op::v6::MVN>(pattern_map.at(mvn_m).get_node_shared_ptr());
        if (!mvn || transformation_callback(mvn)) {
            return false;
        }
        // Target LayerNorm-like MVN only: mean-only MVN is decomposed into cheap ReduceMean + Subtract anyway
        if (!mvn->get_normalize_variance()) {
            return false;
        }

        const auto data = pattern_map.at(data_m);
        const auto& data_pshape = data.get_partial_shape();
        const auto rank = static_cast<int64_t>(data_pshape.size());
        // rank <= 3 with last axis is already supported by CPU MVN node
        if (rank != 4 && rank != 5) {
            return false;
        }
        const auto& channels = data_pshape[rank - 1];
        if (channels.is_dynamic()) {
            return false;
        }
        // Reshape with -1 can't infer dimension in case of empty tensor
        for (const auto& dim : data_pshape) {
            if (dim.is_static() && dim.get_length() == 0) {
                return false;
            }
        }

        auto axes_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(axes_m).get_node_shared_ptr());
        auto axes = axes_const->cast_vector<int64_t>();
        if (axes.size() != 1) {
            return false;
        }
        const auto axis = axes[0] < 0 ? axes[0] + rank : axes[0];
        if (axis != rank - 1) {
            return false;
        }

        const auto in_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                           ov::Shape{3},
                                                           std::vector<int64_t>{0, -1, channels.get_length()});
        const auto reshape_in = std::make_shared<ov::op::v1::Reshape>(data, in_shape, true);
        const auto new_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
        const auto new_mvn = std::make_shared<ov::op::v6::MVN>(reshape_in,
                                                               new_axes,
                                                               mvn->get_normalize_variance(),
                                                               mvn->get_eps(),
                                                               mvn->get_eps_mode());

        std::shared_ptr<ov::Node> out_shape;
        if (data_pshape.is_static()) {
            out_shape =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{data_pshape.size()}, data_pshape.to_shape());
        } else {
            out_shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
        }
        const auto reshape_out = std::make_shared<ov::op::v1::Reshape>(new_mvn, out_shape, false);

        new_mvn->set_friendly_name(mvn->get_friendly_name() + "/MVN3D");
        reshape_out->set_friendly_name(mvn->get_friendly_name());
        ov::copy_runtime_info(mvn, {in_shape, reshape_in, new_axes, new_mvn, out_shape, reshape_out});
        ov::replace_node(mvn, reshape_out);
        return true;
    };

    auto m = std::make_shared<Matcher>(mvn_m, matcher_name);
    register_matcher(m, callback);
}
