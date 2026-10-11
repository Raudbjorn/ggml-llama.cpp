#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif

#include "base.hpp"
#include "mem.hpp"
#include "mem-query.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

const char * mem_api_int2str(int mem_api) {
    if (mem_api == MEMORY_API_TYPE_SYCL) {
        return "SYCL API";
    } else if (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        return "Level Zero API";
    } else {
        return "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
bool query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_count = 0;

#if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#else
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    try {

        if (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            return false;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        if (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            return false;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_count, nullptr);
        if (r != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            return false;
        }

        std::vector<zes_mem_handle_t> modules(module_count);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_count, modules.data());
        if (r != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            return false;
        }

        if (module_count > modules.size()) {
            return false;
        }
        return ggml_sycl_sum_memory_modules(module_count,
            [&](size_t i, size_t & module_free, size_t & module_total) {
                zes_mem_state_t state = {};
                state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
                if (zesMemoryGetState(modules[i], &state) != ZE_RESULT_SUCCESS) {
                    return false;
                }
                module_free = state.free;
                module_total = state.size;
                return true;
            }, free_bytes, total_bytes);

    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        return false;
    }
}
#endif

bool get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    if (dev.has(sycl::aspect::ext_intel_free_memory)) {
        try {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            return true;
        } catch (const sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            return false;
        }
    } else {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#else
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    return false;
}

ggml_backend_memory_status get_memory_size_status(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    if (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        if (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            return GGML_BACKEND_MEMORY_EXACT_LEVEL_ZERO;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    if(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        return GGML_BACKEND_MEMORY_EXACT_SYCL;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    return GGML_BACKEND_MEMORY_TOTAL_AS_FREE;
}

bool get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {
    return get_memory_size_status(dev, free_bytes, total_bytes, api_type) != GGML_BACKEND_MEMORY_FAILURE;
}
