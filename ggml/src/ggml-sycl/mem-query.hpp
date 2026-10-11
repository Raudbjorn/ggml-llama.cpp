#pragma once

#include <cstddef>
#include <limits>

// Exact telemetry requires every enumerated module to contribute a valid sample.
template <typename Query>
bool ggml_sycl_sum_memory_modules(size_t count, Query query, size_t & free_bytes, size_t & total_bytes) {
    free_bytes = total_bytes = 0;
    if (count == 0) {
        return false;
    }
    size_t sum_free = 0, sum_total = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t module_free = 0, module_total = 0;
        if (!query(i, module_free, module_total) || module_total == 0 || module_free > module_total ||
            module_free > std::numeric_limits<size_t>::max() - sum_free ||
            module_total > std::numeric_limits<size_t>::max() - sum_total) {
            return false;
        }
        sum_free += module_free;
        sum_total += module_total;
    }
    free_bytes = sum_free;
    total_bytes = sum_total;
    return true;
}
