# MoE offload integration: context for LLM contributors

This document gives coding agents the context needed to work on integrating
features from the `thecodacus/llama.cpp` fork into this repository. It summarizes
the source review dated 2026-10-11, and [Implementation status](#implementation-status)
records what has since landed, with its validation and its open items. The rest
of the document describes the reviewed baseline; where the status section
disagrees with it, the status section and the current source win.

Read [AGENTS.md](../../AGENTS.md) and the [repository field guide](../GUIDE.md)
first. Repository instructions govern the work; this document supplies project
context. Read the relevant implementation and tests before changing behavior.
The baseline below will age: confirm the current checkout and feature status.

## Purpose and backend priorities

The project aims to improve inference when a model, particularly its MoE expert
weights, does not fit entirely in GPU memory. Useful outcomes include faster
prefill, better decode latency, greater usable model capacity, and reliable
CPU/GPU cooperation without changing model results.

Prioritize Linux, Intel GPUs through SYCL, OpenVINO, and the generic CPU backend
on AMD processors. The target has focused on SYCL, but substantial OpenVINO and
AMD CPU improvements are in scope. CUDA, Windows, and macOS support have low
priority for this integration. Do not interpret the AMD CPU priority as evidence
that OpenVINO provides native AMD acceleration.

The reference workstation is an AMD Ryzen 9 7900X3D, Intel Arc A770, and 64 GB
DDR5. Record the actual machine, GPU memory availability, driver, runtime, and
build configuration for each experiment; workstation specifications alone do
not establish a working backend or a performance result.

## What was already implemented at the reviewed baseline

- **Dynamic expert caching:** scheduler-owned cache providers, including SYCL,
  cooperate with CPU expert execution. GPU hits and CPU misses have explicit
  dispatch, collection, fallback, and cleanup behavior. The shared small-batch
  limit is eight tokens at the reviewed baseline. See the
  [MoE cache guide](../backend/MOE-CACHE.md).
- **Expert prefetch:** staging slots, allocation sizing, transfer draining, and
  a private-stream contract already exist. Source prefetch code is not a new
  feature to copy over the target implementation.
- **CPU and SYCL skipped-expert semantics:** `-1` expert IDs can produce zero
  contributions. The scheduler also protects its all-skipped scan. OpenVINO
  did not provide equivalent semantics at the baseline (now implemented, see
  the status section).
- **OpenVINO graph translation, MoE fusion, and compiled-model caching:** these
  are existing integration constraints, not missing foundations to rebuild.
  OpenVINO has no provider for the target's dynamic MoE cache.
- **TurboQuant, MTP/speculative decoding, and relevant upstream fixes:** the
  target already has its own implementations. The reviewed BF16 CPU work, x86
  flash-attention tail fixes, and several SYCL/OpenVINO changes also have target
  equivalents. Compare semantics before declaring a missing feature.
- **A specialized decision endpoint:** `/v1/systemone` already exists. The
  general `/v1/decision` engine was a separate proposed capability (now
  implemented beside it, see the status section).

Source `master` is an ancestor of the reviewed target. A broad merge is not the
integration strategy. Neither commit counts nor differing patch IDs prove that
a feature is absent. Whole-file replacements risk losing target-specific work.

## Recommended work, in order

Scores are qualitative: impact and relevance run from 1 (low) to 5 (high);
complexity runs from 1 (small isolated change) to 5 (research-level redesign).
They describe incremental target value, not measured speedups. These were the
review's recommendations; the next section records what was done with them.

| Priority | Candidate | Impact / complexity / relevance | Direction |
| --- | --- | --- | --- |
| 1 | Routing trace tool and corrected offline cache simulator | 3 / 2 / 5 for tracing; 3 / 2 / 4 for simulation | Use existing graph callbacks; honor sampling flags; collect evidence before selecting a cache policy. |
| 2 | Profile-selected static hot expert packs on SYCL | 4 / 3 / 4 | Keep hot experts on the layer's GPU and cold experts on CPU; compare against the existing dynamic cache. |
| 3 | Apply static packs to OpenVINO | 4 / 4 / 4 | A proposed adaptation, not a complete source feature. Implement safe skipped IDs first, then measure fusion and compilation effects. |
| 4 | Backend-neutral constrained-decision engine | 4 / 3 / 4 for engine; 4 / 4 / 3 for server | Harden numeric handling and engine/context lifetime; preserve the existing server API. Validate hybrid-model batching. |
| 5 | AMD CPU membership in CPU/GPU tensor parallelism | 3 / 3 / 3 | Test capacity and latency separately. Automatic split sizing is a larger follow-up, scored 2 / 4 / 3. |
| 6 | Async CPU graph splits and predictive expert refill | 2 / 4 / 3 for worker; 2 / 5 / 2 for ring | Defer scheduler concurrency until serialization is measured. Treat prediction and ring refill as research. |

The CPU tensor-parallel proposal concerns CPU plus GPU execution; it is not a
demonstrated CPU-only kernel improvement. The audit found no new AMD CPU kernel
gap among the examined BF16 and x86 tail changes. Extract useful tracing tools
independently of speculative runtime machinery.

## Implementation status

Branch `MoE-offload-prefill-optimizations` (2026-10-12), one commit per change.
"Validated" means the named tests pass on the cloud test host, which has
the CPU backend (Xeon with AMX), Vulkan on Mesa llvmpipe, and OpenVINO 2026.4.1
on its CPU device. None of it has run on SYCL or the Arc A770, and none of it
has a throughput measurement: llvmpipe and OpenVINO CPU run on the host CPU.

| Item | Status | Implementation | Validation | Open |
| --- | --- | --- | --- | --- |
| 1. Routing trace | done | `63c9853` [llama-moe-trace](../../tools/moe-trace/README.md): eval-callback capture, sampling flags, per-run session and model provenance | `test-moe-trace` | - |
| 1. Simulator | done | `5567504` `simulate.py`: per-(layer, expert) skew, expert count from the header, sessions never interleaved, held-out `static` policy, shared-pool `dynamic` model of `--moe-cache` (per-plan fill limit and pins, `9303a9a`), costs only with stated assumptions | `test-moe-trace-simulate` | calibrate the cost model on the A770 |
| 2. Static packs | done, opt-in | `bf671e8`, `43c945a`, `e078fb2` ([guide](../backend/MOE-PACK.md)): per-layer device, `--fit` accounting, eligibility checks (a LoRA on the experts disables the pack), the `build_lora_mm_id` precision path, no source `op_params` flag, exclusive with `--moe-cache`, `llama-bench --moe-pack`; since the `master` merge `d7ff063`, no pack for Hadamard-folded experts, and the device cache (`--moe-cache-mib`) skips packed layers | `test-moe-pack` (`010aaf9` fixtures): bit-exact on the CPU, NMSE 3e-8 on Vulkan; a Hadamard fixture, and the cached layer count beside a pack on Vulkan | SYCL run; A/B against `--moe-cache` at equal VRAM |
| 3. Packs on OpenVINO | done, opt-in | `a6c59d1` (`-1` in `MUL_MAT_ID`/`ADD_ID`), `a6529c2` and `3a4ebce` (view write-back), `43c945a` (host id maps) | `test-moe-pack`; on OpenVINO `test-backend-ops` `MUL_MAT_ID` 757/757, `ADD_ID` 72/72 | fusion and compile-time cost, GPU device |
| 4. Decision engine | done | `7957865` [llama-schema-decision](../../tools/schema-decision/README.md): errors as values, overflow-free bounds, joint-text tokenization, hybrid padding bounded by `n_ubatch` | `test-schema-decision-vocab`, `test-schema-decision` (oracle within 4e-8) on CPU and Vulkan, e2e on three real models | task accuracy and latency on real workloads; OpenVINO cannot run it (multi-sequence batches fail, below) |
| 4. `/v1/decision` | done, opt-in (`--decision-seqs`) | `81e2bcf`: engine reset before context destruction, cancellation, 400/503/500 mapping; `/v1/systemone` unchanged | `tools/server/tests/unit/test_schema_decision.py` (19 tests: sleep/wake, embeddings, per-request LoRA, cell reclaim) | - |
| 5. CPU tensor parallel | done, experimental | `3277180` `--cpu-tp`, `--cpu-tp-margin` ([guide](../backend/CPU-TP.md)): explicit membership, one split source, sizing over all split files | `test-arg-parser`; perplexity on Vulkan+CPU | SYCL has no `cpy_tensor_async`; latency and capacity on the A770; `--cpu-tp-margin` sizes as if every tensor were split (mirrored tensors and the CPU-resident embedding are not separated) |
| 6. Async CPU splits | done, default off | `4e992bd` `--sched-async-cpu` ([section](../backend/MOE-PACK.md#overlapping-the-cold-chain---sched-async-cpu-experimental)): job joined on every return, memory-root dependency checks, serial with a MoE cache session, status propagation | `test-moe-pack`: logits bitwise equal with and without on CPU, Vulkan, OpenVINO; 8 splits launched beside the CPU split on Vulkan (4 steps x 2 packed layers); abort recovery; `test-moe-cache`: no split on the worker with a cache session | speed; oversubscription with `--cpu-tp` |
| 6. Expert prediction | diagnostics only | `3872d7e` `llama-moe-trace --spec-router`, `--hidden-out` (versioned header, bounded) | `test-moe-trace` with a host recomputation oracle on CPU, Vulkan, OpenVINO | recall on real models; lead time |
| 6. Dynamic ring refill | not built | [reasons](../../tools/moe-trace/README.md#why-there-is-no-dynamic-ring) | - | build only as a `--moe-cache` admission policy if recall justifies it |

Defects found on the way and fixed in their own commits:

- `b3ead50`: the AMX `mul_mat` read and wrote wrong rows for batched `src1`.
  Every sequence after the first in a multi-sequence Qwen3.5 batch got wrong
  logits on AMX hosts, off by up to 82.
- `3a4ebce`: OpenVINO did not write back the source of a view that only an eval
  callback reads, so the callback read stale memory.

Pre-existing failures seen on this host, unchanged by the branch (checked against
`master` where noted):

- `test-grammar-integration` aborts and `test-quant-type-selection` fails the
  Qwen3.6-27B `MXFP4_MOE` snapshot (both also on `master` `ee4b97c`, built and
  run). `test-chat` aborted before the `master` merge and passes since.
- `test-tokenizers-ggml-vocabs` reads Git LFS pointer files.
- On OpenVINO:
  - the random dense llama fixture fails a shape check (also on `master`);
  - every batch with more than one sequence fails in shape inference
    (`llama-batched-bench -npl 2`, also on `master`), so `test-schema-decision`
    fails there;
  - `test-recurrent-state-rollback-bailingmoe3` aborts in a Reshape (also on
    `master`);
  - decoding after an aborted decode fails in ScatterUpdate, with or without the
    CPU worker;
  - `master`'s device MoE cache (`--moe-cache-mib`) fails at compute: OpenVINO
    rejects its I32 slot map as a weight (seen after the `master` merge).
- With `LLAMA_BUILD_SERVER=OFF` the `llama` app does not link: `app/CMakeLists.txt`,
  which this branch does not change, always links `llama-server-impl`.
- `llama-server --embedding --pooling mean` aborts at load with
  `GGML_ASSERT(ggml_can_mul_mat(a, b))` when the sequence count does not divide the
  batch size (`-np 9 -c 2048 -b 256` on stories260K, also on `master`); with
  `--decision-seqs` the count is `-np` plus the decision sequences.

## Code map

Paths below belong to the target. Use symbols and the current source rather
than copying line numbers from the older review.

| Area | Start reading here |
| --- | --- |
| CLI, configuration, memory fitting | [common/arg.cpp](../../common/arg.cpp), [common/common.h](../../common/common.h), [common/common.cpp](../../common/common.cpp), [common/fit.cpp](../../common/fit.cpp), [include/llama.h](../../include/llama.h) |
| Model loading and graph construction | [llama-model.cpp](../../src/llama-model.cpp), [llama-graph.cpp](../../src/llama-graph.cpp), [model builders](../../src/models/) |
| Context lifetime and evaluation | [llama-context.cpp](../../src/llama-context.cpp) |
| Scheduler, prefetch, transfer ownership | [ggml-backend.cpp](../../ggml/src/ggml-backend.cpp); search for `ggml_backend_sched_set_prefetch_experts_slots`, `ggml_backend_sched_set_async_cpu` and `ggml_backend_init_private_stream` |
| Dynamic cache interface and providers | [ggml-backend-moe-cache.h](../../ggml/src/ggml-backend-moe-cache.h), [SYCL moe-cache.cpp](../../ggml/src/ggml-sycl/moe-cache.cpp), [CPU integration](../../ggml/src/ggml-cpu/ggml-cpu.c) |
| Static packs | `init_moe_packs` in [llama-model.cpp](../../src/llama-model.cpp), the pack branch of `build_moe_ffn` in [llama-graph.cpp](../../src/llama-graph.cpp) |
| Routing traces and prediction diagnostics | [tools/moe-trace](../../tools/moe-trace/), `build_moe_spec_router` in [llama-graph.cpp](../../src/llama-graph.cpp) |
| SYCL expert dispatch | [ggml-sycl.cpp](../../ggml/src/ggml-sycl/ggml-sycl.cpp), [mmvq.cpp](../../ggml/src/ggml-sycl/mmvq.cpp) |
| OpenVINO expert translation and fusion | [mul_mat_id.cpp](../../ggml/src/ggml-openvino/openvino/op/mul_mat_id.cpp), [compressed MoE fusion](../../ggml/src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp), [router fusion](../../ggml/src/ggml-openvino/openvino/pass/fuse_moe_router.cpp) |
| Tensor-parallel membership and model eligibility | [ggml-backend-meta.cpp](../../ggml/src/ggml-backend-meta.cpp), [llama-arch.cpp](../../src/llama-arch.cpp); search for `llm_arch_supports_sm_tensor`; `common_cpu_tp_prepare` in [common.cpp](../../common/common.cpp) |
| Decisions | [schema-decision engine](../../tools/schema-decision/), [server-context.cpp](../../tools/server/server-context.cpp) (`process_schema_decision`); existing [server-decision.cpp](../../tools/server/server-decision.cpp), [server-decision.h](../../tools/server/server-decision.h) |
| Quantized format compatibility | [ggml-common.h](../../ggml/src/ggml-common.h), [ggml.h](../../ggml/include/ggml.h), [ggml-turbo-quant.c](../../ggml/src/ggml-turbo-quant.c) |

Read the [SYCL guide](../backend/SYCL.md), [OpenVINO guide](../backend/OPENVINO.md),
[MoE cache guide](../backend/MOE-CACHE.md) and [static pack guide](../backend/MOE-PACK.md)
for the backend or feature being changed. Use [scripts/README.md](../../scripts/README.md)
for maintained helpers and [tests/CMakeLists.txt](../../tests/CMakeLists.txt) to check
test registration.

## Integration constraints that must survive a port

1. **Keep cache ownership and fallback explicit.** In the target, CPU rows are
   skipped only after successful GPU dispatch; collection failure triggers CPU
   recomputation. Preserve session ownership and cleanup on every exit. Initially
   make static packs and the dynamic cache mutually exclusive for affected
   tensors, unless the task explicitly implements shared ownership and budgets.
2. **Account for the whole memory budget.** Include model weights, KV state,
   scratch space, draft models, staging, and all cache allocations. The target's
   fitter/provider budget disconnect is a known issue. Source fixed slot counts
   and the CPU-TP free-memory margin are not sufficient automatic sizing rules.
3. **Preserve exact static-pack graph semantics.** The source routes each expert
   ID to either its packed hot index or its original cold index, using `-1` on
   the other side, then sums complete FFN contributions. Its basic path assumes
   separate gate/up/down tensors and SILU. Scales, biases, clamping, preweighting,
   and fused variants need explicit eligibility checks. Direct source
   `ggml_mul_mat_id` calls bypass the normal `build_lora_mm_id` path: adapt LoRA
   handling or reject that combination clearly.
4. **Do not copy the source operation flag.** Source `op_params[0] = 1` marks
   negative IDs, but the target SYCL backend uses that field for precision and
   its grouped GEMM path requires `GGML_PREC_DEFAULT`. Retain the target's
   precision contract and bounded all-skipped handling.
5. **OpenVINO needs an explicit zero contribution.** Its reviewed translator
   treats `-1` as the last expert, not as a skipped expert. Use a safe index plus
   a mask and zero selection; verify GatherMatmul, compressed/MXFP4 paths, and
   all-skipped inputs. Check fused graphs and compiled-model reuse as well as
   the unfused operator before claiming static-pack support.
6. **Use the layer's actual device.** The source multi-GPU correction allocates
   packs on each layer's GPU instead of putting every pack on GPU zero. Preserve
   this even if the first experiment uses one GPU; isolate allocation fallback
   per device.
7. **A second backend is not proof of a private stream.** Retain the target's
   `ggml_backend_init_private_stream` contract, staging lifetimes, synchronization,
   and error draining. SYCL has no source-style host registration hook at the
   baseline. CUDA's registration environment variable is not a SYCL control, and
   `mlock` is not DMA registration. Prefer supported owned pinned buffers.
8. **Async CPU execution changes scheduler ownership.** Audit dependencies,
   allocator aliases, cancellation/reset, callback behavior, thread-local cache
   sessions, and CPU thread oversubscription. Do not import the source's global
   worker or small-batch threshold as a validated AMD/SYCL default.
9. **Prediction is not ready for direct adoption.** The reviewed prediction tip
   has a malformed CLI insertion, unsafe asynchronous staging reuse in fallback
   cases, and model-owned mutable ring state shared by contexts. Its predictor
   reuses pre-attention router weights for Qwen3.5 MoE; it is not a separately
   trained predictor. Establish transfer ownership, per-context state, bounds,
   useful lead time, and miss behavior before evaluating throughput.
10. **Decisions need lifecycle and numeric hardening.** Source cached engines
    can retain a context pointer across server context destruction/recreation.
    Reset them before destroying the context. Check integer range arithmetic,
    non-finite inputs, rounding bounds, and grid semantics. The engine supports
    independent finite fields, not arbitrary JSON Schema relationships; its
    constrained probabilities are not calibrated truth probabilities. Validate
    recurrent-state isolation for hybrid padding and separate candidate-count
    heuristics from actual batch/microbatch capacity.
11. **Preserve quantized format identity.** Source turbo4 blocks are 66 bytes;
    target blocks are 68 bytes at the baseline. Adopting the source layout under
    the same type identity requires an explicit compatibility/migration design.
    Existing target codecs and backend kernels must stay consistent.

## Evidence and validation expected from future work

Routing traces should record model identity, quantization, workload, seed and
sampling settings, layer IDs, and session boundaries. Keep profile-building
and evaluation workloads separate. Callback synchronization also changes timing,
so tracing runs are not clean throughput measurements.

For a runtime change, first compare results with the existing target fallback.
Exercise mixed hits/misses, all-skipped inputs, unsupported layouts, allocation
failure, reset/reuse, and concurrent contexts where applicable. Relevant existing
tests include [backend ops](../../tests/test-backend-ops.cpp),
[MoE cache](../../tests/test-moe-cache.cpp),
[cache fitting](../../tests/test-moe-cache-fit.cpp),
[stream ordering](../../tests/test-sched-stream-ordered.cpp),
[static packs](../../tests/test-moe-pack.cpp),
[routing traces](../../tests/test-moe-trace.sh),
[Qwen4exp MTP](../../tests/test-qwen4exp-mtp.cpp), and
[SYCL TurboQuant correctness](../../tests/test-sycl-turbo-correctness.cpp).
Server work should also inspect the existing
[SystemOne tests](../../tools/server/tests/unit/test_systemone.py).

Measure prefill and decode separately, across prompt lengths and batch sizes.
For static packs, compare uncached CPU fallback, the existing dynamic cache, and
the pack at comparable memory budgets (`llama-bench --moe-pack off,P:S
--moe-cache off,MiB`). Include workload shifts, peak host/device memory,
cold-start costs, and OpenVINO compilation costs where relevant. Report CPU-only
and CPU/GPU tensor-parallel results separately.

Every result should distinguish source inspection, successful compilation,
correctness tests, and hardware measurements. Record exact commits, model and
quantization, command/configuration, backend/device, runtime/driver, and sampling
conditions. Source CUDA speedups and toy bandwidth models do not establish
target performance.

## Source provenance and branch coverage

Source repository: `https://github.com/thecodacus/llama.cpp.git`. The full review
is the optional workspace-level `MoE-offload-prefill-optimizations.md`, outside
the target Git checkout. This document is self-contained so it can travel with
the target repository without that sibling artifact or source clone.

| Reviewed tree/ref | Revision |
| --- | --- |
| Target `master` | `2f623c6607c078d1069dac23e9d8af9cb146e771` |
| Source `master` | `8f5ab832ca7d8a7b4f23687693fb8b0ecbc227e7` |
| Source `perf`: primary offload optimizations and tracing | `27c54b4bbcefadedcec6397477cc2e866c1db716` |
| Source `fable5/expert-prediction` | `011c0ab74a4279c50a71c612aca8d19febdd4be0` |
| Source `fable5/cpu-tensor-parallel` | `10f8aff0c56494ac8eb83913b5cf161d1dd29b17` |
| Source `decision-upstream` | `a8d96e948f06da54fc3f4eddca0a272e346d640a` |
| Source `perf-multigpu-cache` | `f201e552799760e5286f6fe12f3e803105cd8313` |
| Source `parallel-decision` | `ad129b08d9f134cd298d1f8a85efc52b1b66e18e` |

The exact unprefixed branch `cpu-tensor-parallel` was absent from both local refs
and source remote heads during the review. Do not silently equate it with the
prefixed branch. The two decision tips have identical engine-directory content,
but different surrounding server baselines; the port used the consolidated
`decision-upstream` feature diff as a reference, adapted to the current target.

For source navigation, start with `tools/moe-trace`, `tools/parallel-decision`,
the relevant branch's `common/arg.cpp`, model loading/graph builders, and
`ggml/src/ggml-backend.cpp`. Inspect pinned files with `git show <revision>:<path>`
without switching a working checkout. Update the status section when further
integration lands.
