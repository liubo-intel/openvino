// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cldnn {
namespace gpu_trace_dump {

struct trace_event {
    std::string name;
    std::string category;
    uint64_t timestamp_us = 0;
    uint64_t duration_us = 0;
    uint32_t thread_id = 0;
};

inline std::string get_env(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

inline bool is_truthy(std::string value) {
    for (auto& ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }

    return value == "1" || value == "true" || value == "on" || value == "yes" || value == "dump";
}

inline bool is_enabled() {
    static const bool enabled = []() {
        if (is_truthy(get_env("OV_GPU_TRACE_DUMP"))) {
            return true;
        }

        if (is_truthy(get_env("CLI_ChromePerformanceTiming"))) {
            return true;
        }

#ifdef __linux__
        if (is_truthy(get_env("LINUX_PERF"))) {
            return true;
        }
#endif

        return false;
    }();

    return enabled;
}

inline std::string get_output_path() {
    std::string dump_dir = get_env("OV_GPU_TRACE_DUMP_DIR");
    if (dump_dir.empty()) {
        dump_dir = get_env("CLI_DumpDir");
    }

    if (dump_dir.empty()) {
        return "ov_gpu_trace_dump.json";
    }

    const char separator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif

    if (dump_dir.back() != '/' && dump_dir.back() != '\\') {
        dump_dir.push_back(separator);
    }

    return dump_dir + "ov_gpu_trace_dump.json";
}

inline std::string escape_json(const std::string& src) {
    std::string escaped;
    escaped.reserve(src.size());
    for (const auto ch : src) {
        if (ch == '\\') {
            escaped += "\\\\";
        } else if (ch == '"') {
            escaped += "\\\"";
        } else if (ch == '\n') {
            escaped += "\\n";
        } else if (ch == '\r') {
            escaped += "\\r";
        } else if (ch == '\t') {
            escaped += "\\t";
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

class trace_collector {
public:
    trace_collector() = default;

    ~trace_collector() {
        finalize();
    }

    uint64_t get_timestamp_us() const {
        const auto now = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - m_start_time).count());
    }

    void add(trace_event event) {
        if (!is_enabled()) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.emplace_back(std::move(event));
    }

    void finalize() {
        if (!is_enabled()) {
            return;
        }

        bool expected = false;
        if (!m_finalized.compare_exchange_strong(expected, true)) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_events.empty()) {
            return;
        }

        std::ofstream out(get_output_path());
        if (!out.is_open()) {
            return;
        }

        out << "{\n";
        out << "  \"schemaVersion\": 1,\n";
        out << "  \"traceEvents\": [\n";

        for (size_t i = 0; i < m_events.size(); ++i) {
            const auto& e = m_events[i];
            out << "    {\"name\":\"" << escape_json(e.name)
                << "\",\"cat\":\"" << escape_json(e.category)
                << "\",\"ph\":\"X\",\"pid\":1,\"tid\":" << e.thread_id
                << ",\"ts\":" << e.timestamp_us
                << ",\"dur\":" << e.duration_us
                << "}";
            if (i + 1 != m_events.size()) {
                out << ",";
            }
            out << "\n";
        }

        out << "  ]\n";
        out << "}\n";
    }

private:
    const std::chrono::steady_clock::time_point m_start_time = std::chrono::steady_clock::now();
    std::mutex m_mutex;
    std::vector<trace_event> m_events;
    std::atomic<bool> m_finalized{false};
};

inline trace_collector& get_collector() {
    static trace_collector collector;
    return collector;
}

class scoped_trace {
public:
    scoped_trace(std::string name, std::string category)
        : m_enabled(is_enabled())
        , m_name(std::move(name))
        , m_category(std::move(category)) {
        if (m_enabled) {
            m_start_us = get_collector().get_timestamp_us();
        }
    }

    ~scoped_trace() {
        if (!m_enabled) {
            return;
        }

        trace_event event;
        event.name = m_name;
        event.category = m_category;
        event.timestamp_us = m_start_us;
        event.duration_us = get_collector().get_timestamp_us() - m_start_us;
        event.thread_id = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        get_collector().add(std::move(event));
    }

private:
    bool m_enabled = false;
    uint64_t m_start_us = 0;
    std::string m_name;
    std::string m_category;
};

inline scoped_trace make_scope(const char* name, const char* category) {
    return scoped_trace{name, category};
}

inline scoped_trace make_scope(const std::string& name, const char* category) {
    return scoped_trace{name, category};
}

}  // namespace gpu_trace_dump
}  // namespace cldnn
