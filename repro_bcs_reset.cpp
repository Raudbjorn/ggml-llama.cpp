#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl;

int main() {
    size_t copy_size = 860672;
    int batch_size = 10;
    int iterations = 100;
    
    // Default queue is out-of-order
    queue q{};
    
    void* host_ptr = aligned_alloc(4096, copy_size);
    void* dev_ptr = malloc_device(copy_size, q);
    
    printf("Starting batched loop...\n");
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<event> events;
        for (int i = 0; i < batch_size; ++i) {
            event e = q.memcpy(dev_ptr, host_ptr, copy_size);
            q.submit([&](handler& h) {
                h.depends_on(e);
                h.single_task([=]() { volatile int x = 0; });
            });
            events.push_back(e);
        }
        sycl::event::wait(events);
    }
    printf("Done.\n");
    free(dev_ptr, q);
    free(host_ptr);
    return 0;
}
