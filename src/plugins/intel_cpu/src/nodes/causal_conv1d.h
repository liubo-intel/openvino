// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "ov_ops/causal_conv1d.hpp"

namespace ov::intel_cpu::node {

class CausalConv1D : public Node {
public:
    CausalConv1D(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::CausalConv1D;
    }
    bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename T>
    void executeTyped(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs) const;
};

}  // namespace ov::intel_cpu::node
