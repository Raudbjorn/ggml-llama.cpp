# Stream-ordered split input copies (Andrei-Dr 0018/0020 port) - 2026-09-28

Development log for porting Andrei-Dr local-ai patches 0018 and 0020 (skip the host sync
before a stream-ordered split input copy). It records the choices, the obstacles and the
evidence.

Evidence labels: **measured** means tool output on this host; **source** means read from code;
**author** means Andrei-Dr's numbers on their hardware.

## What the patches do

Source: github.com/Andrei-Dr/local-ai, `research/patches/mainline-series/0018-*.patch` and
`0020-*.patch`.

In `ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`), a split input from
another backend used to be handled like this when there are no pipeline-parallel events
(single-GPU):

1. the host synchronizes the split backend fully, so the device drains;
2. SYCL has no `cpy_tensor_async`, so the generic branch synchronizes again and runs a blocking
   `ggml_backend_tensor_copy`. For a SYCL destination that is
   `ggml_backend_sycl_buffer_set_tensor`, which runs `queues_wait_and_throw()` on the device
   and then waits on the memcpy.

In decode with host-resident experts (`--n-cpu-moe`, `--fit`), every MoE layer returns its CPU
expert output to the GPU through this path. 0018 notices that a copy enqueued with
`set_tensor_async` on the split backend's own stream is already ordered after that stream's
earlier reads of the destination, so steps 1 and 2 become one async enqueue. 0020 makes this
the default and adds `GGML_SCHED_COPY_SYNC=1` to restore the wait. Author: host overhead per
MoE layer went from 89.5 to 62.7 us (nsys, GTX 1650 SUPER, CUDA). Output was IDENTICAL x4.
Decode t/s did not change beyond noise.

## Choices

1. **0018 + 0020 only, without 0010.** 0018's second hunk is written against 0010's version of
   the generic branch, which master does not have. Its `stream_ordered` branch already contains
   the async upload that 0010 introduces, so for single-GPU (no events) nothing else is needed.
   0010's remaining change affects only the pipeline-parallel event path, which this fork does
   not run. That path is excluded explicitly: `stream_ordered` also requires that no event
   exists for the split backend (`sched->events[b][cur_copy] == NULL`), so with
   `n_copies > 1` the code is exactly as before. (The first push of this PR missed that. Review
   caught the new branch replacing the event path's blocking fallback.)
2. **Only backends that declare stream order, currently SYCL only** (the user's choice).
   - A new proc-address hook, `ggml_backend_async_is_stream_ordered`, with its typedef in
     `ggml-backend.h`. The scheduler caches its answer per backend in `ggml_backend_sched_new`.
   - It also requires the scheduler buffer type for that backend to be the backend's default:
     tensor copies live there, and SYCL's `set_tensor_async` asserts on any other buffer type.
   - Andrei's patches apply to every backend with `set_tensor_async`. Here Vulkan keeps the
     synchronize. Vulkan records async writes into its compute context, or into a separate
     transfer queue on AMD dGPUs (`ggml-vulkan.cpp`, `ggml_backend_vk_set_tensor_2d_async`).
     Whether such a write is barriered against an earlier dispatch still reading the buffer was
     not audited. Its `synchronize` also runs `ggml_vk_graph_cleanup`.
3. **Evidence that SYCL is stream-ordered** (source, grep of `ggml/src/ggml-sycl/`):
   - every op submits to `ctx.stream()`;
   - the buffer set/get/copy calls, the host allocator (`common.cpp`, `malloc_host`) and the
     MoE cache (`moe-cache.cpp`, `sctx->stream()`) all use the device's one dpct in-order default
     queue;
   - the only other queue is PR 72's private stream, which belongs to the prefetch backend and
     is never a scheduler split backend;
   - the in-order property itself was confirmed on the A770 in
     `sycl-prefetch-second-queue-2026-09-28.md`.
   - **This holds for one device only.** With several SYCL devices, tensor-split `MUL_MAT`
     submits to other devices' queues (`ctx.stream(i, is)`) and joins them back onto the main
     queue with barriers only at the end of the op. Stream order would then depend on every such
     path always joining before it returns, which has not been audited, so the hook returns
     `ggml_sycl_info().device_count == 1`. Review caught this on the first push.
4. **A host-source lifetime guard, which is new relative to the patches.** The device reads the
   host source when the async memcpy executes, not at submit time. 0010's comment argues the
   source cannot be overwritten early, because any later host split first synchronizes the
   device backend. That argument does not cover two cases:
   - a host split whose inputs are all user inputs, which synchronizes only itself;
   - the next graph. `ggml_gallocr_free_node` (`ggml-alloc.c`) protects only OUTPUT tensors, so
     an intermediate may reuse a graph input's memory. The next ubatch's `set_inputs` then
     writes there while the previous graph's last upload may still be queued.

   The guard works like this:
   - `pending_host_reads[b]` is set when a stream-ordered upload reads a non-WEIGHTS host
     source. Weights are immutable, so they never set it.
   - Before a host split (`ggml_backend_buft_is_host(bufts[b])`, which covers CPU and BLAS)
     starts, including its own input copies, every flagged backend is synchronized.
   - After every `ggml_backend_sched_compute_splits`, whatever its status (in
     `ggml_backend_sched_graph_compute_async`), the same happens, and `ggml_backend_sched_synchronize`
     clears the flags.
   - The existing scheduler syncs of a backend clear its flag too, so the flush adds no new sync
     on the common decode path. There, a CPU split needs the GPU's output anyway, so the flush
     only moves that wait ahead of its input loop.
5. **MoE-weight and prefetch inputs.** When the stream-ordered condition holds for these, the
   step-1 wait is skipped as well. This is safe because the SYCL stream is in-order: the routed
   copy into the destination runs after every earlier read of it. The ids readback is not the
   argument, because it does not always synchronize the split backend. The weight sources are
   immutable. Prefetch writes a staging slot, not the destination's own memory.
6. **Counters.** The number of stream-ordered copies, flushes at a host split and flushes at graph
   end are printed as one `GGML_LOG_DEBUG` line when the scheduler is freed, visible with `-v`.
   In decode with CPU experts, expect roughly one host-split flush per CPU MoE layer. As noted
   above, that is the wait the CPU split needed anyway.

## Obstacles

- **0018 is not self-contained.** Its context lines are 0010's code, so it had to be applied by
  hand against the pre-0010 generic branch.
- **The author's lifetime argument has gaps**, as described in choice 4. This was found while
  reviewing the patch, not from a failure.
- **The GPU was reserved for another session's Ornith benchmark** during development. The
  runtime checks below were done only once the user released it.

## Verification

(Filled in as runs complete.)

## Not claimed

- Vulkan and OpenVINO behaviour is unchanged by construction: they do not export the hook. That
  is verified from source only.
- The pipeline-parallel event path (`n_copies > 1`) is unchanged by construction: the
  `stream_ordered` condition requires no events. Multi-device SYCL keeps the old syncs, because
  the hook returns false there. Neither configuration was run.
