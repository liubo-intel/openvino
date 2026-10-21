// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ReshapeMVNLastAxisTo3D transformation detects MVN-6 operations (with variance normalization) with rank 4 or 5
 *     input normalized over the last axis only (e.g. LayerNorm2d after NCHW->NHWC Transpose),
 *     which are not supported by the CPU MVN node and would be decomposed otherwise.
 *     Such MVN is replaced with Reshape([N, -1, C]) -> MVN(axes=[2]) -> Reshape(original shape).
 *
 *     Before:
 *         Data [N, H, W, C] -> MVN(axes=[3])
 *     After:
 *         Data [N, H, W, C] -> Reshape([N, H*W, C]) -> MVN(axes=[2]) -> Reshape([N, H, W, C])
 */

namespace ov::intel_cpu {

class ReshapeMVNLastAxisTo3D : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeMVNLastAxisTo3D");
    ReshapeMVNLastAxisTo3D();
};

}  // namespace ov::intel_cpu
