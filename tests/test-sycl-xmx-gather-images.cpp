#include "../ggml/src/ggml-sycl/fused-gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

// Standalone link harness: satisfy the gather object's ggml imports without a backend or queue.
extern int g_ggml_sycl_xmx_gather_types;
int g_ggml_sycl_xmx_gather_types = 0;
int g_ggml_sycl_xmx_gather_shapes = 0;
int g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
int g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;

extern "C" void ggml_log_internal(enum ggml_log_level, const char *, ...) {}
extern "C" const char * ggml_type_name(enum ggml_type) { return "unused"; }

extern "C" [[noreturn]] void ggml_abort(const char *, int, const char *, ...);
extern "C" [[noreturn]] void ggml_abort(const char *, int, const char *, ...) {
    std::abort();
}


int main() {
    // A static consumer pulls the archive member through an ordinary public entry point.
    auto * volatile device_ok = &ggml_sycl_fused_dequant_gemm;
    if (!device_ok) { return 1; }
    int named = 0;
    for (const auto & id : sycl::get_kernel_ids()) {
        if (std::string(id.get_name()).find("fused_dequant_gemm_kernel") != std::string::npos) {
            ++named;
        }
    }
    // Each compiled matrix combination contributes nine IQ kernels; retain at least one complete set.
    std::printf("registered gather kernels: %d (expected a positive multiple of 9)\n", named);
    return named >= 9 && named % 9 == 0 ? 0 : 1;
}
