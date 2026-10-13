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
| Supported layers | gated experts without scales or biases (below) | also scaled and biased experts |

## Quick start

```sh
# 0. build the server, the bench and the trace tool (SYCL: see SYCL.md for the oneAPI setup)
cmake --build build --target llama-server llama-bench llama-moe-trace

# 1. baseline: experts on the CPU, without and with the dynamic cache
llama-bench -m model.gguf -ngl 99 --n-cpu-moe 99 -p 512 -n 128 --moe-cache off,on

# 2. record routing on a workload that represents the deployment, and on a held-out one
llama-moe-trace -m model.gguf -ngl 99 --n-cpu-moe 99 -f work.txt    -n 512 --temp 0.7 --seed 1 -o work.csv
llama-moe-trace -m model.gguf -ngl 99 --n-cpu-moe 99 -f heldout.txt -n 512 --temp 0.7 --seed 2 -o heldout.csv

# 3. pick a slot count: held-out hit rate with 1/16, 1/8 and 1/4 of each layer's experts packed
tools/moe-trace/simulate.py heldout.csv --profile work.csv --budgets 0.0625,0.125,0.25

# 4. A/B at equal VRAM: the pack against the dynamic cache with the pack's budget in MiB
#    (the load log prints it: "static expert packs on DEV: ... MiB")
llama-bench -m model.gguf -ngl 99 --n-cpu-moe 99 -p 512 -n 128 --moe-pack off,work.csv:16 --moe-cache off,MIB

# 5. serve; --fit places the layers with the pack bytes included
llama-server -m model.gguf -ngl 99 --n-cpu-moe 99 --fit on \
    --moe-pack-profile work.csv --moe-pack-slots 16
```

A slot costs one expert slab of each of the layer's expert tensors (gate, up and
down, or gate_up and down) on the layer's GPU. See
[tools/moe-trace/README.md](../../tools/moe-trace/README.md) for the trace format,
the simulator's policies and the expert prediction diagnostics. Keep the profile
and the evaluation workload separate: a profile scored against its own trace
(`static-oracle`) is optimistic.

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
- no expert tensor is Hadamard-folded (`prism.hadamard.weight_names`): the
  graph rotates the input only for the source tensor, not for a pack copy;
- the expert tensors are CPU-resident (host buffers, including the CPU repack
  buffer);
- the layer itself runs on a GPU or iGPU device, which is where its pack goes;
- the device runs `MUL_MAT_ID` on every pack tensor of its layers, probed with
  each tensor's type and shape for a decode step and for a 512-token batch, the
  batch the loader probes expert weights with (OpenVINO, for one, takes no `q5_0`
  experts, and on a GPU no MXFP4 batch whose expert temporary exceeds 1 GiB);
  otherwise none of that device's layers gets a pack, since a prompt batch the
  device refuses would copy the pack on every ubatch;
- the split mode is not `tensor`.

The device MoE cache (`--moe-cache-mib`) skips a layer that has a pack; the
other host-resident layers keep it.

The graph then uses a pack only when `build_moe_ffn` receives exactly the
pack's source tensors and no LoRA adapter on them. The hot and the cold chain
apply the layer's own activation (SiLU, GELU, ReLU, and the others
`build_moe_ffn` knows), which acts on each (token, expert) row alone, so the
split is exact whatever it is.
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
| Runs without the worker | graphs with an eval callback (`llama-moe-trace`, `LLAMA_ENABLE_INNERQ=1`), schedulers with a `--moe-cache` session, splits with a prefetch slot; BLAS splits and the `--cpu-tp` meta device's CPU member stay on the main thread |
| On return | every graph compute has joined the worker, as without it |
| Errors | a failed or aborted CPU split is the graph's status; `llama_decode` returns 2 on an abort |
| Counters | `ggml_backend_sched_free: async CPU splits: N, splits launched without waiting for one: M` at `-lv 5` (completion) or `-v` (bench); `M` counts the splits the scheduler launched while a CPU split was on the worker and not joined, not how long the two ran together (a short CPU split may have finished already): time decode for that |

Hazards to measure before relying on it on the A770:

- The CPU backend's abort callback runs on the worker thread.
- Thread hand-off costs one condition-variable round trip per CPU split. That is
  about two per offloaded layer per token, and it is paid whether or not anything
  overlaps.
- A `--moe-cache` provider submits to the device's own queue from inside the CPU
  split (Vulkan without the backend's submit lock, SYCL on the backend's in-order
  stream), so a scheduler with a cache session refuses the worker
  (`ggml_backend_sched_set_async_cpu` returns false). Packs turn the cache off,
  so this costs the pack path nothing.
- In builds without OpenMP the worker is thread 0 of the CPU threadpool, and the
  threadpool applies `--prio` and `-C` to the calling thread only when a paused pool
  resumes, so they may not reach the worker. Performance only.
- With `--cpu-tp`, the meta device's CPU member runs on the main thread beside
  the worker, so the CPU is oversubscribed. Not tested.
- Measure decode with and without the worker (`llama-bench -sac 0,1` with the
  pack arm), and check the `M` counter. If `M` is 0, the hot remap did not move
  to the device. The `sched_async_cpu` column is the effective state
  (`llama_get_sched_async_cpu`): 0 in the `-sac 1` arm means nothing ran on the
  worker (an eval callback, `LLAMA_ENABLE_INNERQ`, or a MoE cache session; the
  bench's `--moe-cache` defaults to `auto`).

Validation: with the worker, `test-moe-pack` gives logits bitwise equal to the
synchronous run on the CPU, on Vulkan (llvmpipe) with and without a pack, and on
OpenVINO. On Vulkan the counter reports 8 splits launched without waiting for the
CPU split, as many as the test's packed decode layers (4 steps x 2 layers); the
test asserts only that it is non-zero. With a `--moe-cache` session the worker computes
no split (`test-moe-cache`, case `cache-hit-async-cpu`). An abort raised on the worker returns 2, and the next decode gives
the reference logits on the CPU and on Vulkan. On OpenVINO, decoding after an
abort fails in OpenVINO's ScatterUpdate with or without the worker, so the test
skips that case there. Qwen3-0.6B Q8_0 on Vulkan with `-ngl 14` gives
perplexity 51.0296 in both modes (85 CPU splits on the worker). No speed
measurement exists: llvmpipe runs on the CPU.

## Troubleshooting

| Log line or symptom | Cause | What to do |
| --- | --- | --- |
| `profile '...' selected no layer for a static expert pack` | no profiled layer qualifies | `-lv 4` counts the layers per reason, see [Which layers get a pack](#which-layers-get-a-pack) |
| `profiled layer(s) without a pack: experts not CPU-resident` | the experts already sit on the GPU | packs only serve offloaded experts: `--n-cpu-moe` or `-ot` |
| `profiled layer(s) without a pack: the layer does not run on a GPU` | the layer is on the CPU | offload the layer (`-ngl`) |
| `was traced on arch`, `routes expert`, `routes layer`, `has n_layer=` | the profile belongs to another model | retrace with the deployed model |
| `has no decode rows (pos >= 0)`, `malformed row` | not a decode trace, or edited by hand | retrace with `-n` > 0 |
| `allocating ... MiB of expert packs on DEV failed` | not enough device memory | fewer slots (the line prints how many fit) or `--fit on`; SYCL without Level Zero Sysman reports free = total |
| `the graph uses M of N static expert packs` | the architecture's FFN graph does not qualify (LoRA, other activation) | `-lv 5` logs the reason per layer; the pack memory of those layers is wasted, drop them from the profile |
| `--moe-pack-profile and --moe-cache are mutually exclusive` | both were given | choose one |
| `static expert packs are not supported with split mode tensor` | `-sm tensor` | use layer split |
| decode not faster | low held-out hit rate, or 2-8 token batches on the blocking sorted SYCL `MUL_MAT_ID` path (two per projection with a pack) | check `simulate.py` on held-out traces; compare `-np 1` decode with and without |
| `--sched-async-cpu`: `splits launched without waiting for one: 0` | no pack in use, OpenVINO or another device without I32 `GET_ROWS` (no device hot map), batches above 8 tokens, an eval callback, or a `--moe-cache` session | expected: see [Overlapping the cold chain](#overlapping-the-cold-chain---sched-async-cpu-experimental) |

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
The column shows the pack that was built: an arm whose profile selects no layer,
or whose allocation fails, is reported as `off`, with a warning on stderr.

Not yet run: the SYCL backend on the Arc A770, and any throughput measurement.
The same test covers a SYCL device unchanged (`ctest -R test-moe-pack` in a
SYCL build). Before adopting packs, compare at equal VRAM: no cache, `--moe-cache`
at the pack's budget, and the pack, for prefill and decode separately, with
`-np 1`, `-np 2` and `--spec-type draft-mtp`. Two- to eight-token batches take
the blocking sorted `MUL_MAT_ID` path on SYCL, and a packed layer has two of
them per projection instead of one; that is where a regression would show.
