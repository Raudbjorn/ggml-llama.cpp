# Chronology: inherited TurboQuant to the Arc-focused fork

Audit date: 2026-10-11. Fork endpoint: `2f623c6607c078d1069dac23e9d8af9cb146e771`.
Upstream comparison endpoint: `5c8854fd88319125125dc4b836c98b1b215b0f23`.

The main transition is from a broad, initially Metal-led TurboQuant derivative to
an Arc-focused inference and research fork. The Raudbjorn work did not introduce
SYCL to llama.cpp: upstream already had that backend. It imported a separate
TurboQuant implementation into SYCL, discovered that important paths were wrong,
removed them temporarily, restored them with a more precise data contract, and
then expanded into ordinary q8_0/f16 performance, speculative execution, memory
management, server policy, packaging, and preservation of research evidence.

This chapter follows the first-parent history leading to the pinned master tip.
That includes history inherited before the Raudbjorn fork existed. It is not a
claim that every inherited commit was originally developed on Raudbjorn's master,
or that every first-parent commit was individually pushed there. Squashes retain
many intermediate subjects in one message; rebase preserves old author dates while
assigning new committer dates. Where the distinction matters, both are stated.

## Reading the timeline

The first first-parent object absent from the fetched upstream history is
[`0b3744874`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0b3744874), whose
parent is `b3d758750`. It was authored on March 24 and committed on April 15.
It introduces the first TurboQuant KV types in this retained lineage. That is a
better starting point for the feature history than the most recent shared
ancestor, `81bc6b83f`, which reflects subsequent integration.

There are 516 first-parent objects in `b3d758750..2f623c660`. Their committer-month
counts, generated with `git log --first-parent --format='%cs'`, are:

| Committer month | Objects | Interpretation |
| --- | ---: | --- |
| April 2026 | 169 | Mostly inherited March/April codec and GPU work; March author dates were retained in the April 15 chain |
| May 2026 | 18 | Inherited stabilization, model/speculative work, and upstream catch-up |
| June 2026 | 27 | Continued inherited changes, then the June 25 SYCL import and fork pruning |
| July 2026 | 190 | SYCL recovery, correctness gates, quality/capacity work, InnerQ, performance experiments, SDK, security |
| August 2026 | 13 | Default promotions, completed evidence, policy and profiling corrections |
| September 2026 | 84 | Upstream/TheTom reconciliation, MKL layout fix, caches, replay, prefetch, packaging, xe mitigation |
| October 2026, through the pinned tip | 15 | Large-GRF follow-through, MTP, scheduler safety, OpenVINO, docs, and archived evidence |

These are object counts, not independent features or measures of developer
productivity. The full DAG includes donor branches and merge ancestry; counting
those objects as separately authored fork features would substantially overstate
original work. This chapter uses commit diffs and retained research notes for
selected mechanisms; commit messages alone are labeled as historical reports.

## March 24-30, committed April 15: the inherited codec takes shape

The first implementation registered turbo3/turbo4 types, block structures, CPU
routines, and command-line cache choices. Its own message explicitly said Metal
SET_ROWS/dequantization were still missing. The subsequent Metal kernels and
rotation matrices were therefore enabling work, not optimization of an already
complete cross-backend implementation. The original turbo3 block description also
differs from later layouts: old block sizes and compression claims must not be
read as the present ABI. See [the first implementation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0b3744874),
[Metal kernels](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b37e5f0f2), and
[rotation integration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/63e25a58d).

The initial architecture evolved through three meaningful steps:

1. Dense rotation matrices became a fast Walsh-Hadamard transform (WHT), reducing
   the work and stored rotation data. That did not immediately explain the slow
   observed inference. The subsequent fix inlined the WHT header into the Metal
   shader; its diff replaces the include with the implementation, and its message
   identifies CPU fallback as the misleading earlier execution path. This is an
   early example of a performance diagnosis made before proving which backend
   actually ran. See [WHT replacement](https://github.com/Raudbjorn/ggml-llama.cpp/commit/377727552)
   and [shader inclusion fix](https://github.com/Raudbjorn/ggml-llama.cpp/commit/adfbc595b).
2. Rotation moved out of repeated KV dequantization and into graph construction:
   forward rotation of Q, followed by inverse rotation of the attention result.
   A dedicated `GGML_OP_TURBO_WHT` subsequently replaced dense graph matrix
   multiplications. This graph-owned rotation remains the architectural basis
   of the fork, including the later SYCL correction. See [graph-side rotation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4de7c1c88)
   and [the custom operation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/012faec26).
3. Block size, centroid lookup, residual correction, and rotation grouping were
   repeatedly changed. A 32-element block experiment was followed by a return
   to 128-element turbo2/turbo3 blocks. These were format and quality changes,
   not harmless performance knobs. See [32-element experiment](https://github.com/Raudbjorn/ggml-llama.cpp/commit/fff20923b)
   and [return to 128](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b9df4c3d8).

The sequence contains serious correctness failures as well as fast iterations.
A reported turbo3 perplexity blow-up prompted restoration of inverse rotation in
Metal dequantization; the diff explicitly replaces centroid-only reconstruction
with full-block inverse WHT. Later graph-owned rotation made centroid-only
attention dequantization appropriate again. These two changes are not
contradictory when placed in their actual graph contracts. See [quality failure](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b4270ecbb)
and [inverse restoration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ebe0a23bc).

Other concrete repairs included turbo4 SET_ROWS using turbo3 packing, truncated
tail blocks after the block-size refactor, a large worker-stack allocation, and
a CPU turbo3 quantizer that had been a zero-filling stub. The latter is especially
important when interpreting an early claim that CPU support existed: registration
and an entry point did not imply functional quantization. See [packing and tail fixes](https://github.com/Raudbjorn/ggml-llama.cpp/commit/3eca09aec),
[stack allocation fix](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a5a0f7b86),
and [CPU quantizer implementation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/f0601a786).

The Metal lookup experiments also left a useful negative record: deferred norm
multiplication, named-register lookup, and inline block processing were reverted
when slower; a fused block-dot experiment performed worse. Those findings apply
to their recorded Apple hardware and kernels. They are not measurements of Arc
or evidence that a similar-looking SYCL optimization must behave identically.
Representative reversals: [deferred norm](https://github.com/Raudbjorn/ggml-llama.cpp/commit/35034f142),
[named-register lookup](https://github.com/Raudbjorn/ggml-llama.cpp/commit/73d512c37),
and [inline block processing](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b596a5c6c).

CUDA support arrived through Gabe Ortiz's work, followed by vector stride fixes,
MMA/TILE support, mixed K/V types, turbo2, turbo4, InnerQ hooks, and non-aligned
head handling. Non-aligned heads first received fallback/group-size work and then
zero padding, with a corresponding padded-width serialization correction.
HIP adaptations also entered the inherited history. Sources: [CUDA port](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4489686ec),
[Q/K stride correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a17a63a38),
[turbo2](https://github.com/Raudbjorn/ggml-llama.cpp/commit/25a19f223),
[zero padding](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d158db5c6),
[padded state serialization](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ee82c554f),
and [HIP port](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a30a59e98).

Layer-adaptive KV and boundary-V policy are inherited ideas, too. The March chain
already includes boundary-layer choices and subsequent thread-safety/underflow
repairs; these should not be attributed to the later Arc campaigns merely because
those campaigns measured them. See [layer adaptation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/73f00086e),
[review repairs](https://github.com/Raudbjorn/ggml-llama.cpp/commit/701e08531), and
[boundary V](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a3e97d869).

## April-June 23: inherited breadth and stabilization

The April chain added TQ3_1S/TQ4_1S weight compression, separate from the turbo KV
formats, with Metal fused kernels and subsequent CUDA implementations. It also
contains the fixes that made those paths usable across compilers, expert counts,
and MoE execution: heap allocation for large CPU dot products, CUDA graph
exclusions for unsupported TQ MoE execution, shared-memory caps, multi-token
kernels, and HIP-compatible intrinsics. See [TQ types](https://github.com/Raudbjorn/ggml-llama.cpp/commit/3a2fad19c),
[CUDA fused path](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8c2e0d878),
[CPU heap correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b1a6f79b3),
and [MoE graph exclusion](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ec6b8d420).

Vulkan followed its own implementation and correction sequence: an initial
turbo3 shader contribution, completion and backend tests, then TQ4_1S support.
The present fork retains Vulkan, while much of the Metal/CUDA/HIP implementation
later disappears through deliberate backend pruning. Keeping their commits in
history preserves provenance and design evidence; it does not mean these are
currently supported build targets. See [initial Vulkan work](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a494833d0),
[completion](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ff8bb7394),
[tests](https://github.com/Raudbjorn/ggml-llama.cpp/commit/88fcb67e5), and
[Vulkan TQ weights](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8ba9f1288).

The retained April-May first-parent merges include upstream catch-up, hybrid-model
speculative decoding, GQA reshape fixes, automatic asymmetric K/V selection,
Vulkan turbo2/4 stores, and attention-rotation policy changes. One instructive
reversal is disabling upstream attention rotation in April and subsequently
merging a change to enable it by default in May. Sparse V was likewise enabled
and explored before a CUDA-disablement merge. These are backend/policy-specific
changes, not evidence that every sparse-V implementation was removed everywhere.
Sources: [April upstream sync](https://github.com/Raudbjorn/ggml-llama.cpp/commit/67559e580),
[asymmetric policy](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0fa61d724),
[rotation disabled](https://github.com/Raudbjorn/ggml-llama.cpp/commit/c452be605),
[rotation re-enabled](https://github.com/Raudbjorn/ggml-llama.cpp/commit/e0954d1b9),
and [CUDA sparse-V disablement](https://github.com/Raudbjorn/ggml-llama.cpp/commit/11a241d0d).

May and early June were not a dormant gap. The inherited history integrated a
further upstream/MTP sync, Gemma4 multimodal and MTP support, cache-key slot reuse,
router child lifecycle fixes, recurrent sequence-removal corrections, Vulkan
attention work, and checkpoint/restore isolation. Calendar dates can also differ
solely because of timezone offsets: `7d9715f1f`'s June 6 author timestamp and
June 5 committer timestamp denote the same instant. Representative milestones:
[May MTP/upstream merge](https://github.com/Raudbjorn/ggml-llama.cpp/commit/2b61ea24e),
[cache-key reuse](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8e12d35d0),
[router lifecycle](https://github.com/Raudbjorn/ggml-llama.cpp/commit/7d9715f1f),
[Gemma4 MTP](https://github.com/Raudbjorn/ggml-llama.cpp/commit/2e68087aa), and
[checkpoint isolation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a33ef00b1).

## June 25-27: Raudbjorn imports SYCL turbo, then withdraws the broken path

[`3faf37b41`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/3faf37b41)
merged the FellypeMelo SYCL TurboQuant donor at `7eb9efe72` on June 25. Against
its first parent it changed 37 files, adding 2,073 lines and removing 184. The
payload included WHT and codec helpers, attention changes, stores/conversions,
MMVQ/DMMV support, device discovery, and smoke/fuzz/stress tests. Its first parent,
`a33ef00b1`, is the end of the inherited mainline described above. The merge is
the cleanest landmark for the Arc-specific fork's own development phase.

The import immediately required missing template instantiations, a CPU-versus-
SYCL oracle, a WHT predicate correction, and a q8_0 flash-attention repair. The
q8 fix is important: the new turbo work had affected ordinary q8 execution as
well. See [missing instances](https://github.com/Raudbjorn/ggml-llama.cpp/commit/7eee285ab),
[oracle introduction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/de709ee4f),
[WHT predicate](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a7696500b), and
[q8_0 repair](https://github.com/Raudbjorn/ggml-llama.cpp/commit/fc3584d62).

June 27 then narrowed the project deliberately. Backend pruning removed CUDA,
Metal, HIP, MUSA, RPC, WebGPU, VirtGPU, OpenCL, Hexagon, zDNN, ZenDNN, and CANN;
CPU, BLAS, OpenVINO, SYCL, and Vulkan remained. The initial pruning commit disabled
workflows by renaming them, with later housekeeping removing additional inherited
maintenance surfaces. This is a change in product scope, not accidental merge
loss. See [pruning](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b1566523e)
and [later housekeeping](https://github.com/Raudbjorn/ggml-llama.cpp/commit/640610a8a).

More importantly, [`904738505`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/904738505)
removed 527 lines of broken turbo attention/WHT work and inserted hard failures
for turbo execution on SYCL. Its recorded reasons were an IGC hang, garbage from
a forced low-optimization build, and corrupt block-quantized V in the non-FA
path. Thus the June import is not a valid "working turbo on Arc" milestone.
Withdrawal was an intentional safety/correctness response, and restoration
required a separate implementation and gate.

## July 2-12: restore correctness and establish what capacity actually buys

Authored July 2 and committed July 3,
[`dbf32f863`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/dbf32f863)
restored turbo VEC attention with two independent root-cause fixes:

- The KQ dot helper now reads the calling thread's private Q register slice,
  rather than indexing it as a complete head-width row. Turbo K also avoids the
  ordinary q8 Q-packing branch so its Q registers are populated.
- SET_ROWS launches use the source row count, `ne01`, rather than the index
  tensor's `ne11`. The old common case launched only one work-group, leaving
  subsequent cache rows uninitialized even after attention math was repaired.

The actual store diff changes the launch argument and documents why. The commit
records device oracle and live-model smoke observations; this history review did
not rerun them. [The following non-FA fix](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0d518a2ec)
adds turbo-to-f32 copying and dequantizes block-quantized V before the transpose
required by non-flash attention. Together these changes distinguish restoration
of flash attention from repair of the separate non-FA path.

The July 5 turbo4 centroid change widened the codebook in both CPU and SYCL to
address under-reconstruction. The original claim of correctness was therefore
still being refined at the codec level after the backend resumed operation.
See [centroid change](https://github.com/Raudbjorn/ggml-llama.cpp/commit/f14f0ef10).

XMX/DPAS attention was developed in parallel: f16 first, then masks, q8/turbo,
and D=256, with tail and unsupported-feature gates added on review. It remained
an experiment rather than replacing VEC by default. This phase should not be
retold as "SYCL never implemented XMX" or "XMX shipped as the fast default."
The chain establishes an implementation and the guards necessary to contain it:
[scaffold](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9e43bd2a5),
[turbo support](https://github.com/Raudbjorn/ggml-llama.cpp/commit/37f99b591),
[tail correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d4fb1b22c), and
[feature/mask gates](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9a3d8661d).

The July 9 batch combined model-specific perplexity, cache-capacity searches,
synthetic oracle expansion, and InnerQ infrastructure. The results required
several qualifications: corrected corpus chunk counts; automatic boundary-V
rather than pure turbo2 in some capacity numbers; automatic q8 K substitution;
and CPU attention in the early Qwen MoE capacity result. Reports of large
allocation capacity are not proof of equally large GPU attention capacity with
acceptable quality. The retained commits explicitly expose these differences:
[chunk-count correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/058e536d9),
[boundary-V qualification](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4e8a77363),
[MoE instability](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8397d884e), and
[CPU-FA capacity qualification](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9099a4806).

InnerQ work added a host state machine, a K-squared reference and device profile,
recovery hooks, runtime state plumbing, and tests. Many commits explicitly call
the work partial. Subsequent fixes addressed consumer lifetime, hierarchical
state clearing, null outputs, and strict quality-gate behavior. This is a history
of building and repairing infrastructure, not evidence that a beneficial,
automatically calibrated InnerQ policy was deployed. See [state machine](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a2acb02e2),
[lifetime repair](https://github.com/Raudbjorn/ggml-llama.cpp/commit/e2d2690a2),
[clear propagation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/5b867f03b), and
[strict quality gate](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0e5ec0eac).

The same period began separating fork value from inherited SYCL capabilities:
pinned upstream A/B, standard f16/q8 baselines, toolchain provenance, and the
rejection of an unsupported q8 scale-fold attribution. These are milestones in
measurement discipline, not just documentation volume. See [upstream A/B](https://github.com/Raudbjorn/ggml-llama.cpp/commit/3c6c5d73f)
and [attribution correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8b11e9f93).

## July 18-26: standard-SYCL performance, rejected optimizations, SDK and security

The [P5 campaign](../research/sycl/sycl-a770-p5-performance-campaign-2026-07-19.md)
shifted the center of gravity toward ordinary SYCL performance. It retained
per-kernel device splitting, an opt-in quants-first q8 KV layout, route diagnostics,
FA scratch growth before graph capture, and reproducible benchmark tooling. It
rejected global large-GRF, a non-PVC upload shortcut, and the first exact replay
prototype. Per-kernel splitting reduced cold-start latency by 19.94% on Mistral
and 17.01% on Llama-3.1. The report separately records warm pp512 gains of
92.69% and 85.22%, and tg128 gains of 1.74% and 1.05%. These are distinct
endpoints; the startup result must not substitute for the warm measurements.
See the [performance evidence review](07-performance-early.md) for their scope.

Explicit implementation/revert pairs make the abandoned work traceable:

| Attempt | Introduced | Withdrawn | Meaning |
| --- | --- | --- | --- |
| Direct upload bypass on Arc | [8aff564de](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8aff564de) | [1c261d32b](https://github.com/Raudbjorn/ggml-llama.cpp/commit/1c261d32b) | Rejected in P5; not a retained default |
| Exact executable graph replay prototype | [b5e6e4058](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b5e6e4058) | [9b1324500](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9b1324500) | July attempt rejected; September implemented replay again |
| Packed q8 GQA kernel | [4ed65c1cd](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4ed65c1cd) | [81c941664](https://github.com/Raudbjorn/ggml-llama.cpp/commit/81c941664) | Packing reduced traffic but did not deliver the hoped-for speed |
| q8 VEC prefetch | [6e32d0cb5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/6e32d0cb5) | [a1456f1f5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a1456f1f5) | Removed after regression |

The July 25 upstream sync also renumbered fork-private type IDs, an ABI event
rather than an ordinary source refresh. SDK work followed with an optional
network-free common library and a generated library-only branch. The July sync also contains
an unrelated but real logger fix: pausing a full log queue could wrap the queue
into an apparently empty state and deadlock on exit. Broad squash subjects can
hide such behavior changes; the full messages and diffs are necessary. See
[type renumbering](https://github.com/Raudbjorn/ggml-llama.cpp/commit/61eed0aac)
and [SDK changes](https://github.com/Raudbjorn/ggml-llama.cpp/commit/5beb6c3f3).

The most important ordinary-attention change was the July 26 occupancy correction.
The old governor divided maximum work-group size by compute-unit count, producing
an inappropriate cap. Correcting that cap initially regressed prefill because
`parallel_blocks` started at the cap: the cap was effectively a floor. The final
change starts at one and grows splitting only as needed. This preserved prefill
while increasing deep decode parallelism. The retained
[pre-registration and results](../research/sycl/fa-occupancy-prereg-2026-07-25.md)
distinguish early contended observations, rejected intermediate code, and the
later paired campaign. The squash title's large deep-context percentage is a
specific historical cell, not a universal present-day speedup.
[Implementation and development record](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ba478ef9d).

Dense FFN SwiGLU fusion entered the same development arc, first with eligibility
instrumentation and an implementation, then a later default promotion. The
packed-GQA hypothesis was retested against the corrected occupancy baseline and
still lost: split/combine overhead offset the traffic reduction. This is a useful
example of reopening a rejected hypothesis under changed conditions without
turning its hoped-for mechanism into a claimed success.

Speculative decoding also produced a retraction. A global f16 recommendation had
rested partly on a free-prose exactness/speed result that failed repeated testing.
The retained record reports differing speculative token hashes, stable target-only
hashes, and a much smaller speed ratio; warm prompt-cache contamination was a
hypothesis, not a proven cause. The supported conclusion narrowed to measured
prompt classes. The explicit follow-up correction is
[`80d52e708`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/80d52e708).

Finally, July 26 expanded the fork beyond kernels. Server hardening added API-key
handling, token fallback, optional unknown-environment diagnostics, constrained
MCP proxy destinations, and CORS policy. Review then corrected credentials on
actual CORS responses and filtering of authority/framing/hop-by-hop request
headers. This work creates a security surface that later upstream merges must
preserve; it is not incidental UI customization.
[Security integration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/5aa5c34f8).

## August 12-16: default promotions and corrections to the evidence

Quants-first q8_0 KV became the default for eligible SYCL cache rows in
[`754dc99e2`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/754dc99e2).
The device representation places quantized bytes before scales while host/session
bytes remain canonical. The [round-2 report](../research/sycl/sycl-a770-round2-report-2026-07-26.md)
explicitly recorded that protected depths 0 and 2048 had not yet been gated in
that report. Its promotion language therefore must be read with its own stated
coverage gap. Current source still selects this representation by default for
eligible layers; it is not merely the July opt-in experiment.

FFN fusion was promoted separately on August 13:
[`968b451e9`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/968b451e9).
The same day completed missing occupancy cells, closed a split-KV rewrite,
hardened rowless benchmark handling, and reopened dense turbo4 capacity with a
stronger gate. The [turbo4 capacity report](../research/turbo/p4.5a-turbo4-capacity-reopen-2026-08-13.md)
records GO for its two dense low-GQA models with full perplexity and GPU-FA
capacity evidence. It explicitly makes no default change. That is stronger and
narrower evidence than treating every July capacity number as a generally safe
configuration.

The [August follow-up probes](../research/sycl/round2-decode-probes-2026-08-13.md)
found two correctness defects. First, a low-GQA Qwen model still suffered severe
turbo-K quality degradation, disproving a purely GQA-threshold explanation. The
policy was extended to trigger on Qwen-family architecture as well as high GQA.
Despite its subject saying "gate on Qwen family," the diff preserves both
triggers; it does not replace high-GQA handling. Second, FA timing profiling
could synchronize during graph recording and crash. Separate fixes landed
August 15 and 16: [policy correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d74cace9a)
and [profiling/capture correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ca41ef399).

Evidence handling itself needed repair: logs missing because of ignore rules were
restored, and committed environment provenance was changed to an allowlist. These
commits matter because a test claim with absent artifacts, or a raw environment
dump containing unrelated data, is not a sound research record.
[Missing evidence](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ac960c5fc),
[provenance allowlist](https://github.com/Raudbjorn/ggml-llama.cpp/commit/cd8193d25).

## September 3-14: reconciliation, new formats, and a missed layout interaction

The September 3 upstream sync is a large content integration with a long repair
record, including graph wiring, server restore, security preservation, sampling
index bounds, build configuration, and backend signatures. Its presence in the
history should not be equated with a clean, automatic merge. Detailed integration
failure analysis belongs with the companion merge review; the timeline landmark
is [`54d02625c`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/54d02625c).

The September 5 TheTom integration then reconciled compatible donor content
without reintroducing excluded backends or overwriting the fork's SYCL work.
It appended ConvRot weight types, reconciled turbo3 tables across CPU/SYCL/Vulkan,
retained the fork's turbo4 layout, and imported compatible model, checkpoint,
MTP, Vulkan caching, and fitting work. A numeric `general.file_type` correction,
conversion alias restoration, fixture correction, and removal of an unsuitable
foreign benchmark followed. The durable
[integration ledger](../research/turbo/pr51-thetom-integration.md) is explicit that
ancestry and content bases differ after squashed upstream syncs. Sources:
[integration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b1e4170ec),
[conversion alias](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9045f6cc3),
[turbo4 fixture](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0f3e21f53), and
[file-type separation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/5e490fe86).

A particularly clear development mistake was the interaction between the new
MKL attention route and the previously promoted q8 layout. At larger contexts,
MKL staging chose converters using the destination tensor rather than K/V, losing
the quants-first flag. Quant bytes were read as scales and perplexity became NaN.
Existing tests used shapes too small to take MKL, so a green oracle did not cover
the failing path. The correction carries source tensors in the KV descriptor,
uses layout-aware converters, logs MKL routing, and adds explicit route assertions
with both layouts. See [the resolved defect record](../research/sycl/sycl-q8-context-2048-nan.md),
[code fix](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4bb688b3d), and
[oracle expansion](https://github.com/Raudbjorn/ggml-llama.cpp/commit/80a120da0).

The fix was authored September 5 but committed September 14. The note's
"resolved 2026-09-05" names the investigation/fix date, not the date its object
entered this retained first-parent chain. The same distinction applies to the
September turbo3 quality work: its
[report](../research/turbo/turbo3-quality-gate-llama31-8b-2026-09.md) records a failed
5% gate, with symmetric turbo3 at PPL 8.3939 against q8_0 at 7.9861, or +5.11%.
The response was to correct the broad quality claim and retain the configured
threshold/defaults, not silently weaken the gate. This is one model/corpus/context
protocol, not a verdict on every turbo3 workload.

## September 26-30: memory movement becomes the dominant engineering problem

The September 26 re-resync and feature merges repaired and reconciled earlier
integration damage. Immediately afterward, upstream graph-eligibility/device
changes were ported with fork-specific precondition fixes. Their June or earlier
September author dates must not be mistaken for their September 27 integration
date. Milestones: [re-resync](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4ce310f10),
[feature reconciliation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d37dd50bb),
[MoE graph port](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ad0c05fc5), and
[complete fused-path gate](https://github.com/Raudbjorn/ggml-llama.cpp/commit/da0135fbf).

SYCL expert caching was introduced through a provider and then repaired for LRU
state, lifetime, budget, and error handling. The explicit default changed to off;
cache existence must not be reported as automatic use in every process.
[Provider](https://github.com/Raudbjorn/ggml-llama.cpp/commit/52f0bca26),
[first repairs](https://github.com/Raudbjorn/ggml-llama.cpp/commit/005bf5c01),
[default-off decision](https://github.com/Raudbjorn/ggml-llama.cpp/commit/41cd2e985), and
[second-round repairs](https://github.com/Raudbjorn/ggml-llama.cpp/commit/c11157c2e).

Graph record-once/replay-many was implemented again, alongside a DMMV row-tail
and FA allocation fix. Review repaired stale replay, lifetime, comparison,
allocation, and eligibility mistakes. The July replay revert is consequently a
historical rejection of that prototype, not evidence that the pinned fork has no
replay implementation. Conversely, implementation presence does not prove replay
was used in any given run; the October MTP report explicitly records direct
fallback with zero replay calls in its test configuration.
[Replay implementation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d00df4038),
[first review repairs](https://github.com/Raudbjorn/ggml-llama.cpp/commit/cfee30775),
[follow-up repairs](https://github.com/Raudbjorn/ggml-llama.cpp/commit/bb6908513).

Several performance changes in this period were ports, not original algorithms:
Q5_K reordered MMVQ/fused GLU, Q8 DMMV/wide loads, grouped MoE XMX GEMM, Vulkan GDN
tuning, and mixed-compiler SYCL builds. The fork added compatibility changes and
Arc-specific restrictions. Q5_K row pairing, for example, was restricted to the
measured column regime; the wide-load port required naming and alignment fixes.
[Q5_K port](https://github.com/Raudbjorn/ggml-llama.cpp/commit/f045c2e50),
[Arc threshold](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0dca7bcae),
[wide-load corrections](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4eb8c3d03),
[grouped GEMM](https://github.com/Raudbjorn/ggml-llama.cpp/commit/e985ccb9e), and
[ExternalProject build](https://github.com/Raudbjorn/ggml-llama.cpp/commit/73a52b9fd).

Arch packaging also progressed from an initial package to a real systemd/config
layout, corrected runtime dependencies, install behavior, RPATH, and removal of
an unsupported reload action. These repairs describe deployability work rather
than a kernel milestone. [Initial packaging](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b18d04455),
[service layout](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ccd913cc9),
[install/RPATH repairs](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ba6ad4aac),
[runtime dependency/reload correction](https://github.com/Raudbjorn/ggml-llama.cpp/commit/589bf18cb).

Lookahead host-expert prefetch was ported, then given a private copy queue and
device-side waiting. Review repaired multi-device handling, leaked resources,
padding, staging clears, and unwind behavior. The host's move to xe then exposed
a different limit: file-backed uploads through the copy engine could stall/reset.
The fork introduced driver-aware conservative defaults, while a separate local
compute-runtime retry patch and its experiments were archived. These are two
layers of work: application mitigation and external driver/runtime investigation.
[Prefetch port](https://github.com/Raudbjorn/ggml-llama.cpp/commit/de7a0aed4),
[review fixes](https://github.com/Raudbjorn/ggml-llama.cpp/commit/aa410ce4a),
[private queue](https://github.com/Raudbjorn/ggml-llama.cpp/commit/11598778b),
[xe mitigation](https://github.com/Raudbjorn/ggml-llama.cpp/commit/beda936e7),
and [driver-scope refinement](https://github.com/Raudbjorn/ggml-llama.cpp/commit/70b511cae).

The [xe research note](../research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md)
is the evidence record for that incident. July/August measurements were on the
older i915 setup; late September xe observations must not be presented as a
continuous matched benchmark series. The runtime patch is not proof that an
unpatched distributed runtime has the same behavior, and the fork's conservative
copy policy must not be read as a generic claim that all xe copying is broken.

Large-GRF FA work was briefly mixed into the xe change, removed into its own
change, and reintroduced separately. Its tile-and-vector variant was rejected
when decode regressed; the retained design scopes the larger register file to
eligible TILE prefill and adjusts occupancy. This differs materially from July's
rejected global compiler setting. See [split-out](https://github.com/Raudbjorn/ggml-llama.cpp/commit/54c3ecbbd),
[separate change, committed October 1](https://github.com/Raudbjorn/ggml-llama.cpp/commit/5b1eb7c34),
and [campaign](../research/sycl/sycl-fa-large-grf-2026-09-30.md).

## October 1-10: make advanced paths measurable, preserve defaults, archive work

October's MTP integration is larger than "add a model head." It corrects use of
the head's own hyper-connection mixer, validates borrowed target tables, makes
adaptive fitting use an MTP context, preserves hidden outputs across allocator
reuse, constrains chained drafting to valid batches/microbatches, and repairs
on-device state restore sizing for block-quantized KV. It also improves speculative
probabilities and checkpoint placement. [Integrated commit](https://github.com/Raudbjorn/ggml-llama.cpp/commit/001d906a6).

The [MTP correctness report](../research/speculative/qwen4exp-mtp-correctness-2026-10-04.md)
records baseline failures including assertions, segmentation faults, trunk-mixer
dependence, and chain/sequential divergence. Its synthetic CPU/SYCL checks and
real-model attempts have separate limits. Enabling graph mode produced direct
fallback in the cited fixture runs, not demonstrated replay. Later adaptive
acceptance-measurement tests strengthened the measurement layer:
[`d09fb6ba4`](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d09fb6ba4).

Documentation was reorganized on October 5, indexed, and paired with regenerated
op tables and a Vulkan POOL_1D restoration. The fork also adopted a shared agent
roster/runbook and a set of explicitly planned future changes. Those plans are
not implemented features merely because their documents are merged.
[Docs and op restoration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/440e35259),
[plans](https://github.com/Raudbjorn/ggml-llama.cpp/commit/071e5bb13),
[agent contract](https://github.com/Raudbjorn/ggml-llama.cpp/commit/c4b9d6430), and
[later plan refinement](https://github.com/Raudbjorn/ggml-llama.cpp/commit/e519a4833).

AOT XMX-gather selection was made target-aware rather than relying on a blanket
build assumption. OpenVINO REPEAT later received explicit shape and dynamic-
multiplier coverage. These illustrate continued maintenance of build and retained
secondary backends while SYCL remains the focus.
[AOT policy](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a17b8400d),
[OpenVINO REPEAT](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a3af25005).

The October 9 scheduler change is another case where the final code differs from
an early commit-message paragraph. Its initial port removed a host synchronization
before stream-ordered uploads. Review added single-device/single-copy restrictions
and source-lifetime completion handling; the final policy kept synchronous copies
as the default pending model-output equivalence. Current source requires the
literal `GGML_SCHED_COPY_SYNC=0` to opt in. Unset, empty, and other values keep
synchronization. Historical CPU/device gate passes do not establish an output-
equivalent model speedup. [Scheduler integration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/bba3994a9)
and [current contract](../backend/SYCL.md#scheduler-input-copy-synchronization).

The October 9 sanitized-snapshot integration has an inaccurate September 1 date
in its subject: the actual source commit is October 5 upstream `7049ff0cbeb1`.
It is also a single-parent squash on master, despite the word "Merge" in its
title. The companion integration audit treats its tree effects and topology
explicitly. [Snapshot integration](https://github.com/Raudbjorn/ggml-llama.cpp/commit/08d0baaca).

The final two days strengthened evidence provenance: benchmark collection fixes,
firmware/xe research, and a Git LFS-backed benchmark corpus with an indexer.
The endpoint is an archive/documentation milestone. It does not itself rerun or
validate every preserved historical campaign, and LFS pointers are not equivalent
to locally materialized measurement payloads. See [provenance repair](https://github.com/Raudbjorn/ggml-llama.cpp/commit/7830967e1),
[corpus endpoint](https://github.com/Raudbjorn/ggml-llama.cpp/commit/2f623c660), and
[benchmark inventory](../benchmarks/README.md).

## What this history establishes, and what it does not

The durable trajectory is stronger support for the actual Arc deployment and
better evidence around it, rather than a single uninterrupted TurboQuant speedup.
Capacity-oriented KV formats remained important, but ordinary attention occupancy,
q8 layout, model/speculative correctness, expert memory movement, and driver-aware
behavior became equally central. Several highly visible ideas were withdrawn,
reintroduced in narrower form, or corrected after their first landing.

The most recurring correctness failure pattern is a changed contract that one
consumer failed to follow: rotation ownership, thread-private Q layout, row-count
dimensions, serialized type/layout IDs, source-tensor layout flags, graph replay
lifetime, and scheduler host-source lifetime. The most recurring evidence failure
is testing a nearby path rather than proving the claimed one executed. Concrete
counterexamples include early Metal fallback, MKL-unreached oracle shapes, and
SYCL graph-enabled runs that actually took direct fallback.

Coverage is comprehensive at the milestone and retained first-parent chronology
level, supported by the full commit inventory accompanying this report. The
mechanisms discussed in detail were checked against selected diffs/current source
and dated research records. This chapter is not a line-by-line semantic audit of
every historical patch, all donor branches, all deleted experiments, or every
merge resolution. Historical build/test/benchmark statements remain attributed
to their original commits or artifacts; none was rerun for this chapter.
