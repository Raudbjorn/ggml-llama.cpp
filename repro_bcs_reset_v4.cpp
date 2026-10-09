#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl;

int main() {
    // Default queue (out-of-order)
    queue q;
    printf("Device: %s\n", q.get_device().get_info<info::device::name>().c_str());
    
    size_t copy_size = 860672;
    int num_experts = 32;
    
    void* host_ptr = aligned_alloc(4096, copy_size * num_experts);
    void* dev_ptr = malloc_device(copy_size * num_experts, q);
    
    printf("Starting stress loop...\n");
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
    free(host_ptr);
    return 0;
}
