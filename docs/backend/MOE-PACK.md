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
- the device is not OpenVINO, which lowers expert id `-1` to the last expert;
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
- Every packed layer adds about ten nodes (id remap, a second gate/up/down
  chain, the add); the graph node budget grows by 16 per packed layer.

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

Not yet run: the SYCL backend on the Arc A770, and any throughput measurement.
The same test covers a SYCL device unchanged (`ctest -R test-moe-pack` in a
SYCL build). Before adopting packs, compare at equal VRAM: no cache, `--moe-cache`
at the pack's budget, and the pack, for prefill and decode separately, with
`-np 1`, `-np 2` and `--spec-type draft-mtp`. Two- to eight-token batches take
the blocking sorted `MUL_MAT_ID` path on SYCL, and a packed layer has two of
them per projection instead of one; that is where a regression would show.
