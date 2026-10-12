#pragma once

#include "ggml-backend.h"
#include "json.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct server_memory_samples {
    size_t minimum = std::numeric_limits<size_t>::max();
    size_t final = 0;
    size_t total = 0;
    size_t count = 0;
    size_t failures = 0;
    unsigned providers = 0;

    void add(ggml_backend_memory_status status, size_t free_bytes, size_t total_bytes) {
        ++count;
        const auto provider = status >= GGML_BACKEND_MEMORY_FAILURE && status <= GGML_BACKEND_MEMORY_TOTAL_AS_FREE
                            ? status : GGML_BACKEND_MEMORY_FAILURE;
        providers |= 1u << provider;
        const bool exact = status == GGML_BACKEND_MEMORY_EXACT_LEVEL_ZERO || status == GGML_BACKEND_MEMORY_EXACT_SYCL;
        if (!exact || total_bytes == 0 || free_bytes > total_bytes) {
            ++failures;
            return;
        }
        minimum = std::min(minimum, free_bytes);
        final = free_bytes;
        total = total_bytes;
    }
};

inline bool server_memory_interval(const char * value, unsigned & interval_ms) {
    interval_ms = 50;
    if (!value) {
        return true;
    }
    unsigned n = 0;
    if (!*value) {
        return false;
    }
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || n > 1000) {
            return false;
        }
        n = n * 10 + unsigned(*p - '0');
    }
    if (n < 10 || n > 1000) {
        return false;
    }
    interval_ms = n;
    return true;
}

class server_memory_sampler {
    ggml_backend_dev_t device = nullptr;
    FILE * file = nullptr;
    std::string launch_id;
    std::string device_name;
    unsigned interval_ms = 50;
    server_memory_samples samples;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable wake;
    bool stopping = false;
    bool finished = false;

    void sample() {
        size_t free_bytes = 0, total_bytes = 0;
        const auto status = ggml_backend_dev_memory_status(device, &free_bytes, &total_bytes);
        samples.add(status, free_bytes, total_bytes);
    }

public:
    server_memory_sampler() = default;
    server_memory_sampler(const server_memory_sampler &) = delete;
    server_memory_sampler & operator=(const server_memory_sampler &) = delete;

    bool start(const std::vector<ggml_backend_dev_t> & configured) {
        const char * path = std::getenv("LLAMA_BENCH_MEM_LOG");
        if (!path) {
            return true;
        }
        const char * interval = std::getenv("LLAMA_BENCH_MEM_INTERVAL_MS");
        if (!server_memory_interval(interval, interval_ms)) {
            LOG_ERR("LLAMA_BENCH_MEM_INTERVAL_MS: invalid value '%s'\n", interval);
            return false;
        }
        const char * sysman = std::getenv("ZES_ENABLE_SYSMAN");
        if (!sysman || std::strcmp(sysman, "1") != 0) {
            LOG_ERR("LLAMA_BENCH_MEM_LOG requires ZES_ENABLE_SYSMAN=1\n");
            return false;
        }
        auto consider = [&](ggml_backend_dev_t candidate) {
            if (!candidate) {
                return true;
            }
            auto reg = ggml_backend_dev_backend_reg(candidate);
            if (!reg || std::strcmp(ggml_backend_reg_name(reg), "SYCL") != 0) {
                return true;
            }
            if (device && device != candidate) {
                return false;
            }
            device = candidate;
            return true;
        };
        if (configured.empty()) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                if (!consider(ggml_backend_dev_get(i))) {
                    LOG_ERR("LLAMA_BENCH_MEM_LOG requires one configured SYCL device\n");
                    return false;
                }
            }
        } else {
            for (auto candidate : configured) {
                if (!consider(candidate)) {
                    LOG_ERR("LLAMA_BENCH_MEM_LOG requires one configured SYCL device\n");
                    return false;
                }
            }
        }
        if (!device) {
            LOG_ERR("LLAMA_BENCH_MEM_LOG: no configured SYCL device\n");
            return false;
        }
        device_name = ggml_backend_dev_name(device);
        const char * identity = std::getenv("LLAMA_BENCH_MEM_LAUNCH_ID");
        launch_id = identity ? identity : std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        file = std::fopen(path, "a");
        if (!file) {
            LOG_ERR("LLAMA_BENCH_MEM_LOG: cannot open '%s'\n", path);
            return false;
        }
        sample();
        try {
            worker = std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex);
                while (!wake.wait_for(lock, std::chrono::milliseconds(interval_ms), [this] { return stopping; })) {
                    sample();
                }
            });
        } catch (const std::exception & error) {
            LOG_ERR("memory sampler thread: %s\n", error.what());
            finish(false);
            return false;
        }
        return true;
    }

    bool finish(bool completed) {
        if (!file || finished) {
            return true;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
        sample();
        finished = true;
        common_json providers = common_json::array();
        if (samples.providers & (1u << GGML_BACKEND_MEMORY_EXACT_LEVEL_ZERO)) {
            providers.push_back("exact_level_zero");
        }
        if (samples.providers & (1u << GGML_BACKEND_MEMORY_EXACT_SYCL)) {
            providers.push_back("exact_sycl");
        }
        if (samples.providers & (1u << GGML_BACKEND_MEMORY_TOTAL_AS_FREE)) {
            providers.push_back("total_as_free");
        }
        if (samples.providers & (1u << GGML_BACKEND_MEMORY_FAILURE)) {
            providers.push_back("query_failure");
        }
        const common_json record = {
            {"schema_version", 1}, {"launch_id", launch_id}, {"device", device_name},
            {"interval_ms", interval_ms}, {"sample_count", samples.count},
            {"min_free_bytes", samples.minimum == std::numeric_limits<size_t>::max() ? 0 : samples.minimum},
            {"final_free_bytes", samples.final}, {"total_bytes", samples.total},
            {"providers", providers}, {"query_failure_count", samples.failures},
            {"valid", completed && samples.failures == 0 && samples.count > 0},
            {"completed", completed},
        };
        const auto line = record.dump() + "\n";
        bool ok = std::fwrite(line.data(), 1, line.size(), file) == line.size();
        ok = std::fflush(file) == 0 && ok;
        ok = std::fclose(file) == 0 && ok;
        file = nullptr;
        if (!ok) {
            LOG_ERR("memory sampler: terminal record write/flush failed\n");
        }
        return ok;
    }

    ~server_memory_sampler() {
        finish(false);
    }
};
