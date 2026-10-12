# Static MoE expert packs

A static expert pack keeps the most-routed experts of every CPU-resident MoE
layer in a real weight tensor on the GPU that runs the layer, chosen once from
a routing profile. The graph routes each selected expert either to the pack or
to the CPU-resident tensor and sums the two FFN chains, so the result equals
the single-chain result. There is no runtime fill traffic, no eviction and no
per-token admission decision.

It is the alternative to the dynamic [`--moe-cache`](MOE-CACHE.md), not an
addition to it: a run uses one or the other.

| | static pack (`--moe-pack-*`) | dynamic cache (`--moe-cache`) |
| --- | --- | --- |
| Level | graph: a second `MUL_MAT_ID` chain over GPU weight tensors | hook inside the CPU `MUL_MAT_ID` |
| Residency | fixed at load from a profile | admitted and evicted at run time |
| Prefill | packed experts run on the GPU at every batch size | batches above `max_batch` (8) are not cached |
| Memory | exact, counted by `--fit` | provider budget, not wired into `--fit` |
| Needs | a routing profile of the intended workload | nothing |
| Supported layers | plain SiLU-gated experts (below) | also scaled and biased experts |

## Workflow

```sh
# 1. record routing on a workload that represents the deployment
llama-moe-trace -m model.gguf -ngl 99 --n-cpu-moe 99 -f work.txt -n 512 --temp 0.7 --seed 1 -o work.csv

# 2. pick a slot count; score a held-out workload against the work profile
tools/moe-trace/simulate.py heldout.csv --profile work.csv --budgets 0.0625,0.125,0.25

# 3. run with the pack
llama-server -m model.gguf -ngl 99 --n-cpu-moe 99 --fit on \
    --moe-pack-profile work.csv --moe-pack-slots 16
```

See [tools/moe-trace/README.md](../../tools/moe-trace/README.md) for the trace
format and the simulator's policies. Keep the profile and the evaluation
workload separate: a profile scored against its own trace (`static-oracle`) is
optimistic.

## Options

| Flag | Environment | Meaning |
| --- | --- | --- |
| `--moe-pack-profile FNAME` | `LLAMA_ARG_MOE_PACK_PROFILE` | llama-moe-trace CSV; decode rows (`pos >= 0`) select the experts |
| `--moe-pack-slots N` | `LLAMA_ARG_MOE_PACK_SLOTS` | experts per layer, 1..4096 |

Both are required together. An explicit `--moe-cache` other than `off` (or
`LLAMA_ARG_MOE_CACHE`) with a profile is a parse error, and libllama turns the
dynamic cache off for any model that has packs. A speculative draft model never
gets the main model's pack. The library fields are
`llama_model_params::moe_pack_profile` and `moe_pack_slots`.

Each layer packs `min(N, n_expert, experts the profile routes in that layer)`
experts, ranked by decode routing count, ties to the lower expert id.

## Which layers get a pack

A profiled layer gets a pack only when all of these hold; otherwise the load log
counts it under a reason (`-lv 4`):

- the layer has `ffn_down_exps` and either separate `ffn_gate_exps` and
  `ffn_up_exps` or a merged `ffn_gate_up_exps`;
- no per-expert scales or biases, no SwiGLU clamp, and no pre-FFN weighting
  (Llama 4);
- the expert tensors are CPU-resident (host buffers, including the CPU repack
  buffer);
- the layer itself runs on a GPU or iGPU device, which is where its pack goes;
- the split mode is not `tensor`.

The graph then uses a pack only when `build_moe_ffn` receives exactly the
pack's source tensors with `LLM_FFN_SILU` and no LoRA adapter on them.
`sched_reserve` logs how many packs the graph uses; fewer than built means the
architecture's graph does not qualify and the pack memory is wasted.

## Profile checks

The profile is rejected as a whole, and the model loads without packs, when it
cannot be opened, has a malformed or negative field, a line over 64 KiB, more
than 1 GiB, no decode rows, a `# arch=`, `# n_layer=` or `# n_expert=` line
that disagrees with the model, a layer the model does not have, or an expert id
outside a layer's range.

## Memory

Pack tensors are registered with the model's buffers, also in the memory-fit
pass, so `--fit` places layers with the pack bytes included. Allocation is all
or nothing per device; a failure logs the requested size, the device's reported
free memory and the slot count that would fit, and leaves that device's layers
unpacked. Some backends report free memory equal to total (SYCL without Level
Zero Sysman), so the hint can be optimistic.

## Implementation notes

- Expert slabs are read from the GGUF file in their stored layout. The
  CPU-side tensors may have been repacked and are never read back, so packs
  work with CPU weight repacking left on.
- The pack graph sets no extra `op_params` on its `MUL_MAT_ID` nodes. The source
  fork's `op_params[0] = 1` "may contain -1" hint would disable the SYCL grouped
  GEMM, which requires `GGML_PREC_DEFAULT` there. The `-1` skip is
  unconditional on CPU (including repack), SYCL and Vulkan.
- The two id maps of a layer (expert to pack slot or `-1`, expert to itself or
  `-1`) live in host memory and the remap `GET_ROWS` is pinned to the CPU; only
  the router's contiguous top-k ids cross to it. OpenVINO accepts no I32 weight
  tensor, and it would otherwise take the remap because it accepts host buffers.
  Devices that run `GET_ROWS` on I32, except OpenVINO, also get a copy of the
  hot map in the pack buffer (`n_expert` x 4 bytes per layer), which only
  `--sched-async-cpu` uses.
- Each pack tensor is assembled in host memory and written in one call, since
  OpenVINO converts weights only on a full-tensor write. The load therefore holds
  one pack tensor (slots x one expert slab) of transient host memory.
- Every packed layer adds about ten nodes (id remap, a second gate/up/down
  chain, the add); the graph node budget grows by 16 per packed layer.

## Overlapping the cold chain (`--sched-async-cpu`, experimental)

Without it, the cold chain's CPU split and the pack's GPU split of a layer run
one after the other: the scheduler computes CPU splits on its own thread and
returns only when they are done. `--sched-async-cpu` (library:
`llama_context_params::sched_async_cpu`, ggml: `ggml_backend_sched_set_async_cpu`)
gives each scheduler a worker thread for CPU splits. The main thread keeps
launching later splits until one touches the running CPU split's memory, so the
hot chain is computed beside the cold one.

For that the graph changes in batches of at most 8 tokens: the hot remap reads
the device copy of the hot map, and the merge `cold + hot` is pinned to the CPU,
so the pack's split takes no input from the cold split. The price is two
activation copies per layer (the hot result to the CPU and the merged result
back). The 8-token bound is the source fork's CUDA measurement, where prefill
regressed with the CPU merge; it was not measured here. On OpenVINO the hot
remap stays on the CPU, so nothing overlaps.

| | |
| --- | --- |
| Waits for the CPU split | splits on CPU-device or host-buffer backends, splits sharing its buffer type, and splits that read a tensor root the CPU split writes or write one it uses |
| Runs without the worker | graphs with an eval callback (`llama-moe-trace`, `LLAMA_ENABLE_INNERQ=1`), splits with a prefetch slot |
| On return | every graph compute has joined the worker, as without it |
| Errors | a failed or aborted CPU split is the graph's status; `llama_decode` returns 2 on an abort |
| Counters | `ggml_backend_sched_free: async CPU splits: N, splits launched beside one: M` at `-lv 5` (completion) or `-v` (bench) |

Hazards to measure before relying on it on the A770:

- The CPU backend's abort callback runs on the worker thread.
- Thread hand-off costs one condition-variable round trip per CPU split. That is
  about two per offloaded layer per token, and it is paid whether or not anything
  overlaps.
- SYCL runs an in-order queue. A `--moe-cache` dispatch from a CPU split on the
  worker is queued behind the main thread's launches, and `--moe-cache` with the
  worker has not been run.
- With `--cpu-tp`, the meta device's CPU member runs on the main thread beside
  the worker, so the CPU is oversubscribed. Not tested.
- Measure decode with and without the worker (`llama-bench -sac 0,1` with the
  pack arm), and check the `M` counter. If `M` is 0, the hot remap did not move
  to the device.

Validation: with the worker, `test-moe-pack` gives logits bitwise equal to the
synchronous run on the CPU, on Vulkan (llvmpipe) with and without a pack, and on
OpenVINO. On Vulkan all 8 decode-layer pack splits of the test ran beside the
cold split. An abort raised on the worker returns 2, and the next decode gives
the reference logits on the CPU and on Vulkan. On OpenVINO, decoding after an
abort fails in OpenVINO's ScatterUpdate with or without the worker, so the test
skips that case there. Qwen3-0.6B Q8_0 on Vulkan with `-ngl 14` gives
perplexity 51.0296 in both modes (85 CPU splits on the worker). No speed
measurement exists: llvmpipe runs on the CPU.

## Validation status

`tests/test-moe-pack.cpp` (ctest `test-moe-pack`) runs on random 8-expert
qwen3moe (separate gate/up) and qwen35moe (merged gate_up, hybrid) fixtures
with their expert scale tensors stripped:

- CPU placement through the test hook `LLAMA_MOE_PACK_TEST_CPU=1`, which allows
  packs on CPU-placed layers: F32 logits with a pack are bit-identical to the
  logits without one for mixed hot/cold layers, an all-hot layer (the cold chain
  sees only `-1`), a pack on one layer, and slot counts capped by the profile.
  Q4_0 with the CPU experts in the repack buffer and the pack uploaded from the
  file matches within an NMSE bound of 1e-4.
- Malformed, foreign-arch, out-of-range, prefill-only and missing profiles load
  the model without packs and unchanged logits.
- The memory-fit pass counts exactly the pack bytes a real load allocates.
- With a GPU device, every layer is offloaded and the experts pinned to the
  CPU, so the pack lands on the GPU as with `-ngl 99 --n-cpu-moe 99`. Run on
  Vulkan with Mesa llvmpipe (`GGML_VK_VISIBLE_DEVICES=0`): F32 NMSE about 3e-8
  and Q4_0 about 7e-6 against the same placement without a pack.
  `GGML_SCHED_DEBUG=2` shows the CPU split holding only the cold chain and the
  hot chain and merge in the following Vulkan split.

- OpenVINO (2026.4.1 wheel, `GGML_OPENVINO_DEVICE=CPU`) in the same placement:
  with every layer packed the logits match the CPU within NMSE 4e-7 (qwen3moe)
  and 9e-6 (qwen35moe), while the same placement without a pack deviates from the
  CPU by 7e-3 and 1.3e-3. That deviation is the backend's own: it sits in the
  prompt logits of a few tokens, needs no pack, and disappears when a pack in the
  last layer splits that layer's graph. With only layer 0 packed the result
  follows the unpacked OpenVINO run (NMSE 2e-5). The test therefore accepts a
  device run that matches either the unpacked device run or the CPU within 1e-4.
  It needed two backend fixes, both in this tree: `MUL_MAT_ID` and `ADD_ID`
  honor `-1` ids, and a view read by a later split gets its source written back
  (before, the pack's hot chain read a stale `ffn_norm`).

`llama-bench --moe-pack off,PROFILE:SLOTS` compares the arms in one run and
reports a `moe-pack` column; an instance with a pack runs with `moe-cache off`.

Not yet run: the SYCL backend on the Arc A770, and any throughput measurement.
The same test covers a SYCL device unchanged (`ctest -R test-moe-pack` in a
SYCL build). Before adopting packs, compare at equal VRAM: no cache, `--moe-cache`
at the pack's budget, and the pack, for prefill and decode separately, with
`-np 1`, `-np 2` and `--spec-type draft-mtp`. Two- to eight-token batches take
the blocking sorted `MUL_MAT_ID` path on SYCL, and a packed layer has two of
them per projection instead of one; that is where a regression would show.
