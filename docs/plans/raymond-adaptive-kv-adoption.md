# Adaptive KV fork: future adoption

Date: 2026-10-11. Status: proposed evaluation sequence, not an implementation
commitment or evidence that the deferred features work on SYCL.

## Source and starting point

Evaluate selected changes from
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming).
The source audit used `feature/adaptive-kv-stream-v2` at
`1240f34c77441d8afc1afecf34f60c702d436c61`; source behavior below refers to
that pin, not the latest branch. The target audit base was
`2f623c6607c078d1069dac23e9d8af9cb146e771`.

The current extraction adds server UTF-8 repair and complete streamed token
delivery, retained buffer views and bounded allocator/scheduler workspaces,
SYCL VMM lifetime fixes, and Meta-backend ownership/initialization fixes.
See [the extraction report](../research/adaptive-kv-extraction-2026-10-11.md)
for source commits, adaptations, observed tests, and limitations. These
primitives do not enable automatic arenas, KV streaming, or projector lending.
They change the backend API version and public allocator layout; consumers and
backend modules must be rebuilt together.

At the audited pin, source `master` is an upstream mirror; `device-memory-infra`
is the first eight V2 commits, and the `compat/*` lineage is superseded by V2.
V1 was already merged and stripped from this target. Do not import whole
branches, reintroduce removed backends, or resurrect V1. Pin and re-audit every
future candidate, including its fixes, tests, transitive dependencies, and
overlap with target changes since this report.

## Decision sequence

| Stage | Candidate | Entry condition | Required predecessor |
| --- | --- | --- | --- |
| A | Host-USM KV experiment | A named dense-model workload needs more KV capacity | Existing SYCL allocation and attention paths |
| B1 | Projector unload/reload and vision tests | Vision memory pressure is measured | Backend graph retirement and lifetime proof |
| B2 | MTP across image chunks | A named model needs MTP with vision | Model-specific embedding/hidden-state contract |
| B3 | Host-spilled recurrent rollback | Measured device rollback planes limit useful draft depth | Architecture audit and copy/event lifetime proof |
| C1 | Execution buffers | A selected consumer needs owner-executed graph nodes | Backend capability, dispatch, and capture contract |
| C2 | Arena planner, leases, and compute workspace | Two measured consumers need to share a device allocation | Extracted views/ranges; transactional failure tests |
| C3 | Phase coordinator | A named consumer can use reclaimed phase memory | C2 and real backend quiescence/graph retirement |
| D | SYCL KV streaming | A supports a useful capacity/latency tradeoff, and simpler placement is insufficient | C1-C3, qualified copy path, attention oracle |
| E1 | MTP on streamed KV | D passes with non-speculative decoding | D and qualified recurrent rollback where required |
| E2 | Vision arena lending | B1 alone cannot meet the vision memory budget | B1, C2-C3, and D's KV suspend/resume contract |

B1-B3 are conditional, independently justified workstreams. B2 can retain an
eager projector; B3 should start with the simplest non-staged spill path.
C1-C3 are not justified merely because the source implements them. Split-KV
attention contracts may be extracted independently when a concrete SYCL
split-attention consumer exists. Admission diagnostics can also proceed
independently when they address a measured allocation failure.

## Common readiness gates

A feature progresses from experiment to opt-in implementation to default only
with a recorded decision. Each decision must include:

1. **Need and scope.** Name the model, architecture, workload, backend, KV
   types, context depths, concurrency, and user-visible limit being removed.
   Record live configuration; the audited production configuration used
   `--fit on`, q8_0/q8_0, ngram speculation, and a hybrid MoE with relatively
   small KV. That is context, not a permanent assumption about deployment.
2. **Reproducible baseline.** Pin source and target revisions, model hash,
   build flags, driver, firmware, runtime/compiler, environment, commands, and
   effective routing. Demonstrate that the intended path actually ran using
   counters or traces; passing CPU fallbacks is not SYCL coverage.
3. **Correctness.** Compare enabled and disabled paths with deterministic
   token IDs and state round trips. Require exact results for pure lifetime
   or placement changes that preserve arithmetic. For changed arithmetic,
   declare numerical and output-quality tolerances before measurement; test
   logits and recurrent-state drift over long decode, not just local attention
   error. Unexpected divergence, stale reads, races, leaks, or OOB access fail
   the gate. Do not loosen assertions after seeing a failure without analysis.
4. **Resources and failure behavior.** Measure peak device allocation, minimum
   free memory, pinned host bytes, process RSS, and fragmentation/headroom
   across all phases. Bound buffers, queues, and in-flight copies. Inject
   allocation/copy/initialization failures, cancellation, and teardown while
   work is pending. Either restore a valid prior state or reject/poison the
   affected session explicitly; never silently continue with stale state.
5. **Performance and capacity.** Before running, record a numeric minimum
   useful capacity gain, acceptable p95 token/request latency, throughput
   regression budget, host/device memory caps, repetitions, and uncertainty
   method for the named workload. Missing budgets mean not ready. Compare
   identical builds/settings except the candidate, on the same boot/driver,
   across prefill, decode depths, short requests, and transitions. Capacity
   gains can justify slower decode only within those declared budgets.
6. **Compatibility.** Exercise `--fit`, prompt/state caching, supported slot
   counts, causal-attention changes, graph modes, and affected speculative
   paths. Unsupported combinations must fail early with a useful diagnostic.
   Check public ABI/API changes, regenerated consumers, backend support
   parity, TurboQuant type/layout invariants, and removed-backend exclusions.
7. **Rollout and evidence.** First ship disabled by default with a tested
   fallback and independent revert boundary. Disabled mode must preserve
   behavior and avoid feature-specific reservations/copy queues. Promotion to
   default requires the complete supported matrix, repeated multi-request
   stress within declared budgets, and no unresolved corruption or hangs.
   Archive commands, raw logs, output comparisons, failures, and a decision
   with explicit limits under `docs/research/`.

Use existing [benchmark runners](../../scripts/README.md), including
`scripts/perf/bench_spec.py`, `scripts/bench-a770-fork-unique.py`, and
`scripts/server-bench.py`. The [minimum-free-memory plan](kmic68-02-min-free-memory.md)
supplies measurement discipline, not a mandatory implementation dependency.
Timing runs require the repository's sole-tenancy, timeout, service restoration,
and xe/i915 fault checks. Correctness runs without sole tenancy cannot establish
performance. A clean kernel log alone cannot establish completion on xe.

## A: host-USM KV experiment before streaming

Build a reversible experiment using existing SYCL host-USM allocation and
attention access, placing selected KV layers on host within an explicit budget.
Start with q8_0/q8_0 and a supported dense model; sweep device-resident,
partly host-resident, and host-resident placement at 32K, 64K, and 128K where
the model and host memory permit. Compare at identical feasible depths; report
OOM as a capacity limit rather than inventing a device-only timing baseline.

The historical 19-23 GB/s pinned-host-read result measured expert-sized reads,
not attention access. Re-measure attention bandwidth and end-to-end behavior on
the current stack. Verify the actual buffer type and access path, output/state
equivalence, memory bounds, cancellation, and release after repeated contexts.

Decision: stop if there is no target capacity need or no placement meets the
predeclared capacity/latency budgets. If direct host reads meet them, prefer
that smaller solution. Consider streaming only if a measured residual bottleneck
can plausibly be removed by staging; quantify transfer and compute time first.
Leave xe copy-engine defaults intact. The
[copy-engine failure evidence](../research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md)
and [second-queue probe](../research/sycl/sycl-prefetch-second-queue-2026-09-28.md)
do not establish safe overlap on the current boot. Qualify any driver/runtime
fix separately before relying on it; otherwise budget serialized copies and
compute. Do not assume simultaneous compute on the single enabled CCS.

## B: conditional vision and speculative features

### B1: projector unload/reload

Audit `tools/mtmd` weight residency and source commits `9eb6b492c`,
`29f0e0ca9`, `7b48b9b45`, and `dbd04ebca`. Source-author savings of about
885 MiB and reload cost are CUDA evidence only. Measure the actual target
projector and reload latency before committing to this work.

Readiness requires a real SYCL graph-cache retirement path, completion of all
work using the weights, and preserved tensor metadata/embedding ownership
through unload and reload. Test eager versus unload/reload for identical
outputs across first, cached, changed, follow-up, and multiple images, prompt
cache restore, repeated cycles, cancellation, and oversized-image recovery.
Adapt the deterministic PNG generator and comparison loop from source
`tools/server/tests/test_adaptive_vision.py`; remove streaming-only assumptions
and use existing target harnesses rather than CUDA telemetry. Unsupported graph
or residency modes must be rejected before freeing memory.

### B2: MTP drafting across image chunks

Audit source `cf079bec8` and `5dc431243` against the target's current MTP
implementation. Preserve the separate image-embedding and shifted-hidden-state
planes and their chunk positions. Qualify each architecture explicitly; an MTP
fix does not establish DFlash or other draft-mode correctness.

Require image/text boundary and multi-chunk tests, correct draft catch-up,
zero/partial/full acceptance, cancellation, prompt-cache restore, and seeded
equivalence to eager non-speculative generation. Run the existing Qwen4Exp MTP
gates, including q8 KV, plus the new model-specific vision cases. Record accepted
draft counts and end-to-end throughput so passing with no accepted drafts
cannot be mistaken for exercising the new path.

### B3: host-spilled recurrent rollback

First measure the current device rollback planes at useful draft depths.
ngram speculation with `n_rs_seq == 0` has no such savings. Source-author
memory/speed figures compare another model and checkpoint strategy, not this
target's on-device rollback planes.

Audit every architecture's capture, publication, restore, and slot addressing
before enabling spill. At the source pin, capture/restore gating is narrower
than the delta-net family, while mamba/lfm2 still address device snapshot rows;
shrinking their tensors appears unsafe. These are source-inferred hazards,
not reproduced crashes. Use a supported-architecture allowlist until each
builder is qualified. Staged publication additionally needs a backend-neutral
stage allocation path; the source accepts CPU/CUDA only.

Require exact state restoration after zero/partial/full acceptance, failed
decode, cancellation, slot reuse, and serialization, plus bounded pinned RAM
and event lifetimes. Stress delayed copies and teardown before enabling async
staging. Reject unsupported architectures before allocating smaller planes.

## C: shared-memory machinery only with a consumer

Execution buffers must integrate `graph_compute`, `supports_op`, `supports_buft`,
allocation sizing, write notifications, and destruction on every claimed
backend. Test managed nodes beside ordinary nodes and views, including disabled
and enabled graph capture. Resolve the attention `op_params` layout against
upstream sparse-attention fields; do not reuse a word by assumption.

Arenas require bounded/non-overlapping reservations, nested retained views,
persistent regions, transaction rollback, and lease lifetime tests. Include
failure after a later view factory fails (source fix `6bfeefe41`), exhaustion,
repeated attach/detach, and destruction order. Reconcile reported compute bytes
with `--fit` and `memory_breakdown` without double-counting parent and views.

Do not import the source's unconditional `prepare_compute_arenas` calls into
`llama-context.cpp`: they add reservation passes for ordinary contexts and
throw on preparation failure. A future integration must be explicitly opt-in,
test no-allocation contexts and causal-attention invalidation, and establish
recoverable failure behavior before changing startup allocation policy.

The phase coordinator needs backend-neutral capability/version queries,
quiescence, capture-state queries, real graph retirement, and tested ordering:
drain work, invalidate graphs, release leases, commit, rebind, activate. Inject
failure at each boundary. A synchronous implementation is acceptable if it
meets the declared budgets; dummy CUDA-named hooks or assumed async waits are
not. Without a consumer that demonstrably uses reclaimed memory, defer it.

## D and E: KV streaming and dependent consumers

The audited source qualifies Qwen35, head dimension 256, one sequence,
`--fit off`, and Q8_0/Q4_0. Its CUDA results do not qualify SYCL, q8_0/q8_0,
TurboQuant, hybrid MoE, or production configuration. First define and enforce
a narrow target support matrix. Unsupported types, architectures, sequence
operations, embeddings, snapshots, or graph modes must fail admission before
state mutation. Expand only with independent evidence for each combination.

Require backend-neutral attention/copy capabilities and workspace queries,
bounded ready/consumed slot protocols, ordered K/V publication, host-authority
coherence, eviction safety, and repeated prefill/decode transitions. A copy
must not overwrite a slot until its last reader completes. Cancellation and
allocation failure must preserve a restorable cache or invalidate the session
explicitly. Validate `--fit` accounting before claiming support for it.

Use the source split-attention contracts and scalar oracle where useful.
Compare split/resumable attention with contiguous attention over empty, partial,
and wrapped spans, independent K/V strides, masks/positions, GQA, and every
claimed feature such as ALiBi, softcap, or sinks. Include all-masked rows and
numerical extremes. Verify long-run recurrent state as well as attention/logits:
small local partial-merge error can amplify in hybrid models. Preserve the
target's WHT and q8_0 layout contracts for any newly supported quantized path.

MTP on streamed KV additionally needs protected draft-layer leases, suffix
eviction guards, target/draft scheduling tests, and rollback qualification.
Vision arena lending additionally needs projector retirement and verified
suspend/resume with host KV remaining authoritative. Test repeated image/text
transitions, restore failures, and simultaneous memory demands; isolated
projector or streaming passes do not qualify their composition.

Admission diagnostics should distinguish unsupported kernels, arena minima,
host allocation/registration failure, and device allocation/binding failure.
Predicted peaks must include every phase, shared allocation accounting, pinned
RAM, and an explicit headroom reserve. A startup free-memory reading is not a
guarantee against fragmentation or later concurrent allocations.

## Exit decision and limits

Each evaluation ends with adopt, retain as opt-in, defer with a named missing
gate, or reject with evidence. Record the smallest independently useful patch,
its dependencies, rollback procedure, and which new evidence would change the
decision. Do not copy source thresholds or advertise CUDA results as A770 gains.

This plan adds no runtime behavior and performs none of the future experiments.
The extraction report is the evidence for the four implemented improvements;
its test failures, sanitizer qualification, ABI break, and untested modes still
apply. Future source revisions and target configuration changes require a new
audit before these source-specific assessments are reused.
