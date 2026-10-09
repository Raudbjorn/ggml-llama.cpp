#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl;

int main() {
    queue q;
    size_t copy_size = 860672;
    size_t gb = 1024 * 1024 * 1024;
    size_t vram_limit = 12LL * gb; // Target ~12GB pressure on 16GB A770
    
    // Allocate garbage to induce pressure
    printf("Allocating VRAM pressure pool...\n");
    void* pressure_pool = malloc_device(vram_limit, q);
    
    // Actual test payload
    int num_experts = 32;
    void* host_ptr = aligned_alloc(4096, copy_size * num_experts);
    void* dev_ptr = malloc_device(copy_size * num_experts, q);
    
    printf("Starting stress loop under pressure...\n");
    for (int iter = 0; iter < 500; ++iter) {
        for (int i = 0; i < num_experts; ++i) {
            event e_copy = q.memcpy(static_cast<char*>(dev_ptr) + i * copy_size, 
                                    static_cast<char*>(host_ptr) + i * copy_size, 
                                    copy_size);
            q.submit([&](handler& h) {
                h.depends_on(e_copy);
                h.single_task([=](){ volatile int x = 0; });
            });
        }
        q.wait(); 
    }
    printf("Done.\n");
    free(dev_ptr, q);
    free(pressure_pool, q);
    free(host_ptr);
    return 0;
}
