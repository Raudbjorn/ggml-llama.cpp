#undef NDEBUG
#include "../tools/server/server-memory.h"

#include <cassert>
#include <atomic>
#include "../ggml/src/ggml-sycl/mem-query.hpp"
#include <fstream>
#include "../ggml/src/ggml-backend-impl.h"

#ifndef _WIN32
#include <unistd.h>

static std::atomic<size_t> fake_free{100};
static std::atomic<size_t> fake_calls{0};
static ggml_backend_memory_status fake_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = fake_free.load();
    *total = 100;
    ++fake_calls;
    return GGML_BACKEND_MEMORY_EXACT_SYCL;
}

static void observe(size_t value) {
    const auto before = fake_calls.load();
    fake_free = value;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (fake_calls.load() <= before + 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(fake_calls.load() > before + 1);
}

static void lifecycle() {
    ggml_backend_reg reg{};
    reg.iface.get_name = [](ggml_backend_reg_t) { return "SYCL"; };
    reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        return std::strcmp(name, "ggml_backend_dev_memory_status") == 0 ? reinterpret_cast<void *>(fake_memory) : nullptr;
    };
    ggml_backend_device device{};
    device.reg = &reg;
    device.iface.get_name = [](ggml_backend_dev_t) { return "SYCL_TEST"; };
    char path[] = "/tmp/test-server-memory-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    unsetenv("LLAMA_BENCH_MEM_LOG");
    {
        server_memory_sampler inactive;
        assert(inactive.start({&device}));
        assert(inactive.finish(true));
        assert(fake_calls == 0);
    }
    setenv("LLAMA_BENCH_MEM_LOG", path, 1);
    setenv("LLAMA_BENCH_MEM_INTERVAL_MS", "10", 1);
    setenv("ZES_ENABLE_SYSMAN", "1", 1);
    {
        server_memory_sampler sampler;
        assert(sampler.start({&device}));
        observe(60); // peak before model sleep
        observe(100); // model released
        observe(80); // model reloaded
        assert(sampler.finish(true));
        assert(sampler.finish(true));
    }
    std::ifstream stream(path);
    std::string line, extra;
    assert(std::getline(stream, line));
    assert(!std::getline(stream, extra));
    const auto row = common_json::parse(line);
    assert(row["valid"] == true && row["min_free_bytes"] == 60 && row["final_free_bytes"] == 80);
    stream.close();
    std::remove(path);
    setenv("LLAMA_BENCH_MEM_LOG", "/dev/full", 1);
    {
        server_memory_sampler sampler;
        assert(sampler.start({&device}));
        assert(!sampler.finish(true));
    }
    setenv("LLAMA_BENCH_MEM_LOG", "/nonexistent/test-server-memory/file", 1);
    {
        server_memory_sampler sampler;
        assert(!sampler.start({&device}));
    }
    unsetenv("LLAMA_BENCH_MEM_LOG");
    unsetenv("LLAMA_BENCH_MEM_INTERVAL_MS");
}
#endif

int main() {
    size_t free_bytes = 0, total_bytes = 0;
    assert(ggml_sycl_sum_memory_modules(2, [](size_t, size_t & f, size_t & t) {
        f = 60; t = 100; return true;
    }, free_bytes, total_bytes));
    assert(free_bytes == 120 && total_bytes == 200);
    for (int failure = 0; failure < 4; ++failure) {
        assert(!ggml_sycl_sum_memory_modules(2, [failure](size_t i, size_t & f, size_t & t) {
            f = 60; t = 100;
            if (i == 1 && failure == 0) { return false; }
            if (i == 1 && failure == 1) { f = 101; }
            if (i == 1 && failure == 2) { t = 0; }
            if (i == 1 && failure == 3) { t = std::numeric_limits<size_t>::max(); }
            return true;
        }, free_bytes, total_bytes));
        assert(free_bytes == 0 && total_bytes == 0);
    }
    unsigned interval = 0;
    assert(server_memory_interval(nullptr, interval) && interval == 50);
    for (const char * value : {"10", "50", "1000"}) {
        assert(server_memory_interval(value, interval));
    }
    for (const char * value : {"", "0", "9", "1001", "-50", "+50", " 50", "50ms", "9999999999999999999999999"}) {
        assert(!server_memory_interval(value, interval));
    }
    for (size_t last : {60, 80}) {
        server_memory_samples samples;
        for (size_t value : {size_t(100), size_t(60), last}) {
            samples.add(GGML_BACKEND_MEMORY_EXACT_LEVEL_ZERO, value, 100);
        }
        assert(samples.minimum == 60 && samples.final == last && samples.count == 3 && samples.failures == 0);
    }
    for (auto status : {GGML_BACKEND_MEMORY_TOTAL_AS_FREE, GGML_BACKEND_MEMORY_FAILURE}) {
        server_memory_samples samples;
        samples.add(status, 100, 100);
        samples.add(GGML_BACKEND_MEMORY_EXACT_SYCL, 60, 100);
        assert(samples.failures == 1 && samples.count == 2);
        assert(samples.providers & (1u << status));
    }
    server_memory_samples invalid;
    invalid.add(GGML_BACKEND_MEMORY_EXACT_SYCL, 0, 0);
    invalid.add(GGML_BACKEND_MEMORY_EXACT_SYCL, 101, 100);
    invalid.add(static_cast<ggml_backend_memory_status>(99), 50, 100);
    assert(invalid.failures == 3);
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            size_t free_bytes = 7, total_bytes = 9;
            assert(ggml_backend_dev_memory_status(device, &free_bytes, &total_bytes) == GGML_BACKEND_MEMORY_FAILURE);
            assert(free_bytes == 0 && total_bytes == 0);
        }
    }
    #ifndef _WIN32
    lifecycle();
    #endif
    return 0;
}
