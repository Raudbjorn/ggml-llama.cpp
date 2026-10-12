# Correctness mistakes, reversals, and remaining findings

Endpoint: `2f623c6607c078d1069dac23e9d8af9cb146e771`, reviewed 2026-10-11.
This chapter distinguishes source-confirmed defects, historical reports,
intentional experiment rejection, and unresolved hypotheses. No historical
build or performance result was rerun for this review.

## 1. Remaining findings at the endpoint

P1 means a concrete memory-safety or live-state correctness failure warrants
priority; P2 means a bounded feature/configuration/resource defect. These are
review priorities, not a formal vulnerability severity rating. The six
findings below remain unpatched in this documentation branch.

### F1. P1: ACC offsets can narrow into a negative signed parameter

Source: `ggml/src/ggml.c:2280`, `ggml/src/ggml-cpu/ops.cpp:1194` and `:1227`.
[`ggml_acc_impl`](../../ggml/src/ggml.c) accepts `size_t offset`, then stores
it in an `int32_t` parameter array without a range assertion. The
[CPU consumer](../../ggml/src/ggml-cpu/ops.cpp) converts that signed value
back to `size_t`; its extent checks use wrapping unsigned addition.

On the ordinary host representation, an offset representing -4 bytes with
contiguous F32 tensors of two elements can narrow to -4, widen to a
huge unsigned offset, and wrap back to zero when the final-element displacement
is added for the bound check. The first accessed address is still before the
allocation. This is an invalid-input API memory-safety scenario established
from the data flow, not a demonstrated server exploit.

Upstream repair
[`8ae386707`](https://github.com/ggml-org/llama.cpp/commit/8ae386707bed05f02354dbfc6fa5ee0cf58e5f0f)
adds `GGML_ASSERT(offset < (size_t)(1 << 30))` before narrowing. It was merged
after the fork's last full snapshot and is absent here. Porting that repair
with an invalid-offset regression is a concrete follow-up.

### F2. P1: explicit requests for busy server slots can mutate their cache

Source: `tools/server/server-context.cpp:1921`, `:2015`, and `:2959`.
[Slot selection](../../tools/server/server-context.cpp) sets `ret` for an
explicit `task.id_slot` without immediately rejecting or returning a busy
slot. The prompt-similarity loop skips it, leaving the selected pointer and
zero best similarity. The later cache-update path can save, load, or clear
the selected slot's prompt. Only after this function returns does the caller
see `is_processing()` and defer the new task.

Trigger: a new completion explicitly pins an actively processing slot with a
nonempty stored prompt, with prompt caching and similarity selection enabled. Deferral does not undo the
earlier mutation of the ongoing task's state. Upstream
[`23b0202a1`](https://github.com/ggml-org/llama.cpp/commit/23b0202a189c44a54625aadcb37a946dd1d6278d)
returns the busy slot before these side effects. That repair is absent here.
The source path is confirmed; no concurrent HTTP reproducer was executed.

### F3. P1: Vulkan expert invalidation omits the expert offset

Source: `ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp:1699`.
The [Vulkan provider](../../ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp)
stores a cache key containing the tensor base and expert index, and copies
weights from `host_base + expert * expert_size`. Its invalidation overlap
test instead always checks the tensor base for one expert's length.

The shared `ggml_backend_moe_cache_invalidate_buffer` path calls registered
providers when host weight buffers change. Updating just expert N, N > 0,
can therefore leave N's old device copy marked valid. Updating expert zero
can unnecessarily invalidate other entries. This can serve stale weights,
not merely lower hit rate.

The provider entered through `b1e4170ec`. The SYCL equivalent was repaired
by `005bf5c01`; current [SYCL invalidation](../../ggml/src/ggml-sycl/moe-cache.cpp)
includes the expert offset. The corresponding Vulkan repair was not carried
across. A public-path partial expert mutation test is the relevant follow-up;
no Vulkan execution was performed here.

### F4. P2: multiple Vulkan pools each receive the full device budget

Source: `ggml/src/ggml-vulkan/ggml-vulkan-moe-cache.cpp:1095` and `:741`.
Each new `(expert_size, weight_type)` pool receives
`session->config.budget_mb << 20`; pool slots derive from that whole amount.
Although allocated bytes are tracked, previous pools do not reduce the amount
available to the next one.

A model with multiple expert shapes/types can exceed the configured total
device cache budget, causing unexpected memory pressure or allocation failure.
This also entered through `b1e4170ec`. SYCL repairs `005bf5c01` and
`c11157c2e` use remaining budget and account for scratch. The Vulkan path
still needs equivalent accounting. This is source-confirmed, not measured
allocation pressure on the A770.

### F5. P2: setting the experimental XMX flag to zero enables it

Source: `ggml/src/ggml-sycl/fattn.cpp:761` and `:836`.
Both [XMX route predicates](../../ggml/src/ggml-sycl/fattn.cpp) use
`getenv("GGML_SYCL_FA_XMX")` as a boolean. Thus `=0` and even an empty value
select the experimental path when type, shape, and feature predicates allow.
An A/B run that uses explicit zero for its baseline can silently run XMX in
both arms. The current workaround is to unset it.

The ALiBi/softcap/sinks/multisequence guard is present; this finding does not
repeat the old, already-fixed unsupported-feature bug. The specific remaining
defect is value parsing. Use the existing `ggml_sycl_get_env` convention and
cover explicit zero rather than adding a second parser.

### F6. P2: cache-key CLI settings no longer have a server implementation

PR 50 removes cache-key request parsing, binding, lookup, and selection.
`--slot-cache-key-similarity` and `--slot-cache-key-min-prefix` still store
parameters in [common/arg.cpp](../../common/arg.cpp) and
[common/common.h](../../common/common.h), but no server code consumes them.
The task's leftover `cache_key` member does not restore the removed behavior.

Requests relying on that affinity contract receive ordinary slot selection
instead. This does not prove the broader similarity cache is broken. The
[integration chapter](02-integration-history.md#confirmed-surviving-loss-cache-key-slot-selection)
gives the exact deletion and later non-restoration evidence. Restore or retire
the public contract deliberately; do not mistake a surviving flag for a
surviving feature.

## 2. Initial SYCL bring-up: multiple independent numerical faults

The June 25 backend merge `3faf37b41` was followed by the CPU/SYCL oracle
`de709ee4f`, fixes to ordinary attention, a hard disable of turbo FA, and then
a corrected reintroduction. The actual patches show why a plausible port and
a readable short response did not establish correctness:

| Fault | Repair | Consequence and final state |
| --- | --- | --- |
| TILE WHT used `if constexpr(type_K)` | `a7696500b` | Nonzero F16/Q8 enum values rotated Q without rotating K; explicit turbo-type checks repair ordinary attention too |
| VEC used `max(default, D)` threads and a workgroup barrier instead of the stock subgroup barrier | `fc3584d62` | Recorded A770 compile/crash problem; restore backend geometry and barrier |
| Turbo dot product indexed a thread-private Q register slice as a full row | `dbf32f863` | Invalid Q reads; use the private half2/float2 slice and keep turbo out of the q8_1 Q conversion branch |
| Turbo SET_ROWS launch counted `ne11`, usually one, rather than source `ne01` | `dbf32f863` | Only one KV row written; arithmetic fixes alone cannot repair unwritten cache rows |
| Non-FA path transposed block-quantized V as though scalar | `0d518a2ec` | Dequantize before the transpose and supply turbo-to-F32 copy |

`904738505` deliberately removed broken turbo FA and made use fail explicitly.
`dbf32f863` restored it after addressing the register and row-count contracts.
This is a temporary safety withdrawal, not proof that TurboQuant was abandoned.

## 3. XMX experiments needed correctness gates before performance mattered

Introduction `9e43bd2a5` expanded through additive masks, q8/turbo staging,
and D256 support. `d4fb1b22c` repaired tail/mask handling;
`9a3d8661d` added an eligibility predicate that rejects ALiBi, softcap, sinks,
and multiple sequences. Before that, explicit XMX selection could compute a
different attention operation from the graph's request.

Historical compiler and performance results explain why this remained opt-in:
SG16 encountered an IGC failure, while the SG8 implementation was functional
but slow in the recorded experiments. Those are dated toolchain observations,
not a fresh test of today's compiler. The current flag-parsing defect in F5
is independent of the repaired mathematical feature gating.

## 4. Format changes and incomplete representation propagation

The July 25 sync `61eed0aac` moved turbo/TQ tensor IDs from 42-46 to 43-47
to make room for upstream Q2_0. This was a serialized ABI break. Today's
no-renumber invariant prevents repeating it without a migration, but cannot
make old artifacts compatible retroactively. Separately, September
`5e490fe86` moved CR file-type metadata into 512-514. The two enum namespaces
must not be conflated; see the integration chapter for exact tables.

The q8 quants-first work is another representation boundary:

1. `77d356251` introduced the experimental layout.
2. `754dc99e2` promoted it by default for eligible cache rows.
3. Upstream integration introduced a oneMKL FA consumer that did not pass
   the actual K/V tensor into layout-aware dequantization.
4. `4bb688b3d` repaired both contiguous and strided conversion paths;
   `80a120da0` expanded route/layout oracle coverage.

The faulty call used the destination tensor in one branch and a converter
without source layout metadata in another. Quant bytes could be read as FP16
scales. The dated report records finite PPL at context 512 but NaN at 1024,
where MKL became reachable. Current
[fattn-mkl.cpp](../../ggml/src/ggml-sycl/fattn-mkl.cpp) passes `d.tensor` to
both converters. This is the reason route coverage belongs beside numeric
assertions: a passing short-context test did not exercise the broken consumer.

Source and historical result:
[q8/MKL diagnosis](../research/sycl/sycl-q8-context-2048-nan.md).

## 5. Quality policy was initially overgeneralized

`d74cace9a` corrected an auto-asymmetric policy based only on GQA >= 6.
The checked-in campaign reports Qwen3-1.7B, only GQA 2:1, at PPL 659.66
with turbo3 K versus 16.64 for q8, and 16.92 after protecting K. These are
historical measurements in that commit's evidence, not new results here.

The repair uses model-family identification as well as GQA. Current source
also has explicit model exceptions. The lesson is narrower than "Qwen never
works with compressed K": the tested predicate failed to protect a sensitive
model. The AGENTS overview describing only the high-GQA condition is stale.

July PPL/capacity records also revised their interpretation: turbo runs made
with FA off were superseded, an estimated corpus chunk count was corrected
from 642 to 564, and apparent pure turbo2 capacity results were qualified as
including auto-selected Boundary V. Relevant commits include `54331e098`,
`058e536d9`, and `4e8a77363`. Analytic bytes/value, initialized cache capacity,
and validated model quality are three separate claims.

## 6. Graph replay: a failed experiment was followed by a different design

The July signature experiment was reverted; September's record-once/replay
implementation was new work. Treating the old "killed" entry as permanent
would erase a major later milestone.

| Repair | What the source change addresses |
| --- | --- |
| `ca41ef399` | Profiling waited on a queue while recording, which SYCL forbids; suppress synchronizing timing during capture |
| `d00df4038` | Add the later record-once/replay-many design, DMMV row-tail guard, and FA allocation repair |
| `cfee30775` | Track scratch generations so executable graphs do not retain freed K/V addresses; clear stale replay state after failure; compare explicit tensor fields instead of padding bytes; tighten WHT gates |
| `bb6908513` | End actual recording on exception rather than only restoring a boolean; fix allocation/reorder predicates missed in the first repair |

Current source retains these mechanisms in
[ggml-sycl.cpp](../../ggml/src/ggml-sycl/ggml-sycl.cpp). It does not follow
that every graph-enabled run replays: compatibility, shape changes, and
CPU/GPU splits can prevent it. The MTP correctness record even contains
enabled runs with zero replay calls, correctly interpreted as direct-path
evidence only.

## 7. MoE cache and prefetch: fixes required ownership discipline

SYCL cache introduction `52f0bca26` was followed by `005bf5c01`,
`41cd2e985`, and `c11157c2e`. Actual repair diffs cover:

- A virtual destructor for objects owned through a base pointer.
- Clearing freed pointers/capacities before allocation can fail.
- Reserving parallel container capacity before device allocation.
- Remaining device budget rather than full budget for each pool.
- Coalescing duplicate fills and preventing duplicate LRU promotion.
- Expert-offset invalidation, exception boundaries, and device identity.
- Scratch reservation, slab floors, and rejecting unmeasurable free memory.

The repair record reports that duplicate-fill coalescing itself first caused
an LRU self-loop, discovered by model generation. That is an incident recorded
inside one repair commit, not a separately surviving master commit to bisect.
The final mode default was changed to off. Vulkan's corresponding budget and
invalidation gaps remain as F3/F4, showing why a fix in one provider cannot
establish cross-provider correctness.

A documented limitation remains in SYCL too: shared pools are capped by one
tensor's expert count. Simply removing the cap previously let the first shape
consume the remaining budget. That is an unresolved allocation tradeoff,
separate from the demonstrable Vulkan budget violation.

Expert prefetch followed a similar progression:

| Commit | Effect |
| --- | --- |
| `de7a0aed4` | Add lookahead host-expert copies |
| `9e8bab375` | Repair ownership, cleanup, allocation sizing, and unwind |
| `aa410ce4a` | Fix mistakes in that repair: device switching could free live events/slots, and host `memset` could target device storage; refuse switching, use backend clearing with synchronization, and temporarily exclude unsafe SYCL use |
| `11598778b` | Add an actual private SYCL copy queue and device event waits |

The routed-ID shape heuristic still cannot distinguish every long-prefill
case from enough simultaneous one-token sequences. Hardware safety also
depends on the xe/Level Zero stack. Neither the source repairs nor an old
i915 speedup establishes stable copy-engine behavior on current xe.

## 8. MTP, restore, and measurement semantics

The actual October master implementation landing is `001d906a6`. Its diffs
substantiate repairs to separately trained MTP mixer tensors, head-only model
validation, chained-draft microbatch constraints, backend-sampler/output
contracts, deferred catch-up preservation on failure, exported hidden-state
buffer replanning, quantized restore byte-to-element conversion, zero-layer
context handling, and outputless catch-up reservation.

The [MTP correctness record](../research/speculative/qwen4exp-mtp-correctness-2026-10-04.md)
lists concrete pre-fix failures, regression tests, counters, and unproven
paths. It is stronger evidence than a commit subject saying MTP is supported,
but still reports historical executions rather than new validation here.

`d09fb6ba4` has a very long squash message repeating earlier MTP and xe work.
Its actual first-parent diff is six files, +546/-22, concerning acceptance
measurement harnesses and tests. It invalidates stale trace files, rejects
missing/invalid draft statistics, and gates publication on successful launch
and fault checks. Its sequential request totals do not prove a throughput
improvement. This review dates implementation changes by their diffs, not by
every repeated claim in that later message.

## 9. Explicit reversals and changed scope

The first three source reversals below were checked by comparing the reverted
files against the tree before the candidate. They restore those files exactly.
Historical measurements explain decisions; they were not reproduced here.

| Candidate | Reversal | Decision and surviving consequence |
| --- | --- | --- |
| `8aff564de` | `1c261d32b` | Direct blocking upload/bounce bypass was slower: reported 2.278841 s baseline vs 2.784785 s candidate |
| `b5e6e4058` | `9b1324500` | Exact-signature replay preserved tested output but regressed deep decode roughly 81-87%; later replay design is separate |
| `4ed65c1cd` | `81c941664` | Packed q8 GQA gained 7.59% at depth 16384 but missed its 20% keep gate and lost 1.84% at 4096 |
| `6e32d0cb5` | `a1456f1f5` | Four engaged prefetch arms missed +3%, deep decode lost about 1%; minor companion edits remain, so not a whole-tree exact reversal |
| `3775efdd2` and follow-ups | `54c3ecbbd`, then `5b1eb7c34` | Large-GRF work moved out of the xe change and returned as its own change; scope separation, not permanent rejection |
| `904738505` safety withdrawal | `dbf32f863` restoration | Broken turbo FA removed, then reintroduced with the register/SET_ROWS fixes |

Inherited Metal history contains additional rejected experiments: deferred
norm multiplication (`35034f142`), named-register alternatives (`73d512c37`),
inline block processing (`b596a5c6c`), and an 8-magnitude turbo4 lookup table
(`c4e98a523`). Those decisions concern removed Metal paths and their recorded
hardware, not measured SYCL behavior. The chronology preserves their role in
the codec's development without promoting them into current performance rules.

TheTom's CUDA-only KV streaming was deliberately removed at PR 56 integration.
It was not an implemented SYCL feature that later disappeared. PR 102's stale
snapshot was repaired on its branch without reaching master. Both distinctions
matter when cataloguing "reverted changes."

Revert messages sometimes refer to original pre-rebase hashes rather than
the reachable candidates in this table. A failed hash lookup is not proof
the experiment never happened; compare the surviving patch and parent tree.

## 10. Retractions and weak evidence that was corrected

### Target-exact speculative output and the implausible f16 speedup

`655005519` initially closed a campaign with a target-exact f16 claim and
an 11.46x free-prose gain. The [findings ledger](../../scripts/perf/FINDINGS.md)
retracts the exactness interpretation: repeated free-prose runs produced three
different hashes at both old and new occupancy settings. `80d52e708`
propagates that correction into the P5 narrative. Warm prompt-cache effects
are a suspected explanation for the speedup, not an established root cause.

### q8 scale-fold attribution was on an unreachable compile-time path

`8b11e9f93` withdraws a proposed mechanism because the changed template branch
is never instantiated for q8 K. The measured 5.3-6.7% difference remained in
the record, while its cause was left unverified. A plausible kernel explanation
was disproved by checking the actual compile-time dispatch. That does not
mean the measurement was fabricated.

### Provenance was incomplete even when result files existed

The history repeatedly repairs ignored logs, stale traces, environment
collection, route labels, and test gates. `7830967e1` recognizes installed
`-git` package variants and MKL/ONEDNN route names that old collectors missed.
`ca41ef399` corrects a wait-site audit whose search missed arrow syntax and
`wait_and_throw`. A provenance file or a grep count is not necessarily a
complete account of the runtime or consumer paths.

The archive import `2f623c660` makes evidence durable and indexed. It does not
retrospectively validate every conclusion in the imported reports. The same
rule applies to benchmark copies, shell transcripts, and LFS-backed artifacts.

### Additional mistakes found while mapping performance progression

The follow-up [early campaign review](07-performance-early.md) found that P1.8
describes configured server context ceilings as per-slot capacity even though
its method only starts the server and completes one token. Historical context
allocation and the report's own caveat do not support its stronger sustained
concurrency claims. It also distinguishes invalid July fusion provenance from
the later valid August promotion campaign.

The [late campaign review](08-performance-late.md) reads all 16 primary GRF
products: six explicitly fail global binary/repository provenance validation,
even though their per-cell invalid count is zero. October's final matrix also
withdraws earlier speculative recommendations and admits missing early fault
logs, a broken fault counter, and a leaked server holding GPU memory.

The [Hindsight crosswalk](09-hindsight-crosswalk.md) finds an internally reversed
MKL conclusion and records that later memories sometimes repeated obsolete
centroid/XMX/LUT claims. These are evidence-quality defects, not newly reproduced
kernel defects. The [dead-end register](11-dead-ends-and-open-questions.md) explains
which require a new test and which are already answerable from source or records.

## 11. What should be carried forward

The most productive controls in this history are concrete and already fit the
repository: preserve numeric ABI and layout contracts; trace changes to every
consumer; prove dispatcher reachability; test error and lifetime paths; compare
both merge parents; and attach provenance and limitations to measurements.

An experiment that misses its keep gate and is reverted is often evidence of
good development discipline. A "fix" that breaks another path, a surviving
option without consumers, or an exactness claim without stable output is a
correctness/accuracy failure. Keeping these categories separate prevents both
unfairly treating every unsuccessful experiment as a mistake and understating
real defects as mere experimentation.

Not run: builds, test suites, hardware gates, live server scenarios, invalid
ACC execution, or old-GGUF compatibility tests. Source review supports the
specific scenarios above; it does not certify all remaining code or prove
remote exploitability. Suggested fixes are not implemented by this report.
