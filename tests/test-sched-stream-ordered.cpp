#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>

static ggml_backend_buffer_type destination_buft;
static int uploads;
static int synchronizations;

static bool stream_ordered(ggml_backend_dev_t) { return true; }

static bool run_case(bool parallel) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t destination = ggml_backend_cpu_init();
    GGML_ASSERT(cpu && destination);
    ggml_backend_cpu_set_n_threads(cpu, 1);
    ggml_backend_cpu_set_n_threads(destination, 1);

    // CPU storage/compute with distinct buffer support forces a real scheduler copy.
    // The destination advertises stream ordering but cannot allocate events.
    ggml_backend_device device = *destination->device;
    ggml_backend_reg reg = *device.reg;
    reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        return std::strcmp(name, "ggml_backend_async_is_stream_ordered") == 0 ?
            reinterpret_cast<void *>(stream_ordered) : nullptr;
    };
    device.reg = &reg;
    destination_buft = *ggml_backend_cpu_buffer_type();
    destination_buft.device = &device;
    destination_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) {
        auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        if (buffer) { buffer->buft = buft; }
        return buffer;
    };
    device.iface.get_buffer_type = [](ggml_backend_dev_t) { return &destination_buft; };
    device.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
        return buft == &destination_buft;
    };
    device.iface.event_new = [](ggml_backend_dev_t) -> ggml_backend_event_t { return nullptr; };
    destination->device = &device;
    destination->iface.cpy_tensor_async = nullptr;
    destination->iface.set_tensor_async = [](ggml_backend_t, ggml_tensor * t,
                                              const void * data, size_t offset, size_t size) {
        ++uploads;
        ggml_backend_tensor_set(t, data, offset, size);
    };
    destination->iface.synchronize = [](ggml_backend_t) { ++synchronizations; };

    ggml_backend_t backends[] = { destination, cpu };
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, parallel, false);
    auto ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
    GGML_ASSERT(sched && ctx);
    auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    auto intermediate = ggml_scale(ctx, input, 2.0f);
    auto output = ggml_scale(ctx, intermediate, 3.0f);
    ggml_set_output(output);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(sched, input, cpu);
    ggml_backend_sched_set_tensor_backend(sched, intermediate, cpu);
    ggml_backend_sched_set_tensor_backend(sched, output, destination);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) == 2);
    uploads = synchronizations = 0;
    const int runs = 2 * ggml_backend_sched_get_n_copies(sched) + 1;
    bool ok = true;
    for (int step = 0; step < runs; ++step) {
        float values[] = {float(step), 1.0f, -2.0f, 3.0f};
        float actual[4];
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) { ok &= actual[i] == 6.0f * values[i]; }
    }
    // Output equality alone cannot detect accidental async selection in parallel mode.
    ok &= parallel ? (uploads == 0 && synchronizations >= runs) : uploads == runs;
    std::printf("parallel=%d, runs=%d, async uploads=%d, syncs=%d: %s\n",
                parallel, runs, uploads, synchronizations, ok ? "PASS" : "FAIL");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(destination);
    ggml_backend_free(cpu);
    return ok;
}

int main() {
    const bool serial = run_case(false);
    const bool parallel = run_case(true);
    return serial && parallel ? 0 : 1;
}
