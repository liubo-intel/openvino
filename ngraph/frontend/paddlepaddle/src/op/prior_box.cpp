// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
namespace detail {
namespace {
std::shared_ptr<opset8::StridedSlice> make_slice(std::shared_ptr<ngraph::Node> node, int64_t start, int64_t end) {
    return std::make_shared<opset8::StridedSlice>(
        node,
        opset8::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{start}),
        opset8::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{end}),
        std::vector<int64_t>{0},   // begin mask
        std::vector<int64_t>{0});  // end mask
}
}  // namespace
}  // namespace detail

static inline float clip_great(float x, float threshold) {
    return x < threshold ? x : threshold;
}

static inline float clip_less(float x, float threshold) {
    return x > threshold ? x : threshold;
}

struct PaddlePriorBoxAttrs {
    // H                Feature map Height
    // W                Feature map Width
    // IH               Input Image Height
    // IW               Input Image Width
    // min_size         Desired min_size of prior boxes
    // max_size         Desired max_size of prior boxes
    // aspect_ratio     Aspect ratios of prior boxes
    // clip             Clip output to [0,1]
    // flip             Flip aspect ratios
    // step             Distance between prior box centers
    // offset           Box offset relative to top center of image
    // variance         Values to adjust prior boxes with
    // scale_all_sizes  Scale all sizes
    // min_max_aspect_ratios_order output prior box is in order of [min, max, aspect_ratios], which is consistent with
    // Caffe.
    int64_t H;
    int64_t W;
    int64_t IH;
    int64_t IW;
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> aspect_ratio;
    bool clip = false;
    bool flip = false;
    float step = 0.0f;
    float offset = 0.0f;
    std::vector<float> variance;
    bool min_max_aspect_ratios_order = true;
};

std::vector<float> paddle_normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip) {
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio) {
        unique_ratios.insert(std::round(ratio * 1e6) / 1e6);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6) / 1e6);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

int64_t paddle_number_of_priors(const PaddlePriorBoxAttrs& attrs) {
    // Starting with 0 number of prior and then various conditions on attributes will contribute
    // real number of prior boxes as PriorBox is a fat thing with several modes of
    // operation that will be checked in order in the next statements.
    int64_t num_priors = 0;

    // Total number of boxes around each point; depends on whether flipped boxes are included
    // plus one box 1x1.
    int64_t total_aspect_ratios = paddle_normalized_aspect_ratio(attrs.aspect_ratio, attrs.flip).size();

    num_priors = total_aspect_ratios * attrs.min_size.size() + attrs.max_size.size();

    return num_priors;
}

void paddle_prior_box(std::vector<float>& dst_data,
                      std::vector<float>& variance_data,
                      int64_t& num_priors,
                      const PaddlePriorBoxAttrs& attrs) {
    const int64_t W = attrs.W;
    const int64_t H = attrs.H;
    const int64_t IW = attrs.IW;
    const int64_t IH = attrs.IH;
    std::vector<float> aspect_ratios = {1.0f};
    for (const auto& aspect_ratio : attrs.aspect_ratio) {
        bool exist = false;
        for (const auto existed_value : aspect_ratios)
            exist |= std::fabs(aspect_ratio - existed_value) < 1e-6;

        if (!exist) {
            aspect_ratios.push_back(aspect_ratio);
            if (attrs.flip) {
                aspect_ratios.push_back(1.0f / aspect_ratio);
            }
        }
    }
    std::vector<float> variance = attrs.variance;
    num_priors = paddle_number_of_priors(attrs);

    float step = attrs.step;
    auto min_size = attrs.min_size;

    // int64_t idx = 0;
    float center_x, center_y, box_width, box_height, step_x, step_y;
    float IWI = 1.0f / static_cast<float>(IW);
    float IHI = 1.0f / static_cast<float>(IH);

    if (step == 0) {
        step_x = static_cast<float>(IW) / W;
        step_y = static_cast<float>(IH) / H;
    } else {
        step_x = step;
        step_y = step;
    }

    auto calculate_data =
        [&dst_data, &IWI, &IHI](float center_x, float center_y, float box_width, float box_height, bool clip) {
            if (clip) {
                // order: xmin, ymin, xmax, ymax
                dst_data.push_back(clip_less((center_x - box_width) * IWI, 0));
                dst_data.push_back(clip_less((center_y - box_height) * IHI, 0));
                dst_data.push_back(clip_great((center_x + box_width) * IWI, 1));
                dst_data.push_back(clip_great((center_y + box_height) * IHI, 1));
            } else {
                dst_data.push_back((center_x - box_width) * IWI);
                dst_data.push_back((center_y - box_height) * IHI);
                dst_data.push_back((center_x + box_width) * IWI);
                dst_data.push_back((center_y + box_height) * IHI);
            }
        };

    for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
            if (step == 0) {
                center_x = (w + 0.5f) * step_x;
                center_y = (h + 0.5f) * step_y;
            } else {
                center_x = (attrs.offset + w) * step;
                center_y = (attrs.offset + h) * step;
            }

            for (size_t ms_idx = 0; ms_idx < min_size.size(); ms_idx++) {
                box_width = min_size[ms_idx] * 0.5f;
                box_height = min_size[ms_idx] * 0.5f;
                calculate_data(center_x, center_y, box_width, box_height, false);

                if (attrs.min_max_aspect_ratios_order) {
                    if (attrs.max_size.size() > ms_idx) {
                        box_width = box_height = std::sqrt(min_size[ms_idx] * attrs.max_size[ms_idx]) * 0.5f;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }

                    for (float ar : aspect_ratios) {
                        if (std::fabs(ar - 1.0f) < 1e-6) {
                            continue;
                        }

                        ar = std::sqrt(ar);
                        box_width = min_size[ms_idx] * 0.5f * ar;
                        box_height = min_size[ms_idx] * 0.5f / ar;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }

                } else {
                    for (float ar : aspect_ratios) {
                        if (std::fabs(ar - 1.0f) < 1e-6) {
                            continue;
                        }

                        ar = std::sqrt(ar);
                        box_width = min_size[ms_idx] * 0.5f * ar;
                        box_height = min_size[ms_idx] * 0.5f / ar;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }

                    if (attrs.max_size.size() > ms_idx) {
                        box_width = box_height = std::sqrt(min_size[ms_idx] * attrs.max_size[ms_idx]) * 0.5f;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }
                }
            }
        }
    }

    if (attrs.clip) {
        for (int64_t i = 0; i < H * W * num_priors * 4; ++i) {
            dst_data[i] = (std::min)((std::max)(dst_data[i], 0.0f), 1.0f);
        }
    }

    for (int64_t i = 0; i < H * W * num_priors; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            variance_data.push_back(variance[j]);
        }
    }
}

NamedOutputs prior_box(const NodeContext& node) {
    auto input = node.get_ng_input("Input");
    auto image = node.get_ng_input("Image");

    PaddlePriorBoxAttrs attrs;
    attrs.min_size = node.get_attribute<std::vector<float>>("min_sizes", {});
    if (node.has_attribute<int32_t>("max_sizes"))
        attrs.max_size = node.get_attribute<std::vector<float>>("max_sizes", {});
    attrs.aspect_ratio = node.get_attribute<std::vector<float>>("aspect_ratios", {1.0});
    attrs.flip = node.get_attribute<bool>("flip", false);
    attrs.clip = node.get_attribute<bool>("clip", false);
    attrs.step = node.get_attribute<float>("step_w", 0);

    attrs.offset = node.get_attribute<float>("offset", 0.5);
    attrs.variance = node.get_attribute<std::vector<float>>("variances", {0.1, 0.1, 0.2, 0.2});

    if (node.has_attribute<bool>("min_max_aspect_ratios_order"))
        attrs.min_max_aspect_ratios_order = node.get_attribute<bool>("min_max_aspect_ratios_order", false);

    PartialShape input_partial_shape = input.get_partial_shape();
    attrs.H = input_partial_shape[2].get_length();
    attrs.W = input_partial_shape[3].get_length();
    PartialShape image_partial_shape = image.get_partial_shape();
    attrs.IH = image_partial_shape[2].get_length();
    attrs.IW = image_partial_shape[3].get_length();

    std::vector<float> box_data;
    std::vector<float> variance_data;
    int64_t num_priors;

    paddle_prior_box(box_data, variance_data, num_priors, attrs);

    auto node_boxes = opset8::Constant::create(ngraph::element::f32,
                                               Shape{uint64_t(attrs.H), uint64_t(attrs.W), uint64_t(num_priors), 4},
                                               box_data);
    auto node_variances = opset8::Constant::create(element::f32,
                                                   Shape{uint64_t(attrs.H), uint64_t(attrs.W), uint64_t(num_priors), 4},
                                                   variance_data);

    NamedOutputs outputs;
    outputs["Boxes"] = {node_boxes};
    outputs["Variances"] = {node_variances};
    return outputs;
    ////////////////////////////////////////////////////////////////////
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph
