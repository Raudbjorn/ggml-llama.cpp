# Performance and functionality progression: September-October 2026

This chapter follows measured behavior, failed experiments and functionality through
2026-10-10 master `2f623c6607`. It is a historical evidence review, prepared
2026-10-11, not a new benchmark. Dates below are experiment/report dates unless
explicitly called landing dates. Branch build hashes can differ from the squashed
master commit. The [chronology](01-chronology.md) and
[master ledger](05-master-ledger.md) provide the integration dates.

The late history is not a monotonically rising throughput curve. It shows a q8
correctness repair, useful but narrow prefill optimizations, a driver transition
that changed copy safety, model-dependent tuning, and speculative functionality
whose benefit depends strongly on the workload. The common denominator is the
A770, not a common model, driver, compiler or memory placement. Do not multiply
the percentage gains in this chapter into a cumulative fork speedup.

## Evidence and chronology at a glance

All throughput and quality figures below are historical measured results quoted
from the linked sources. Memory sizes calculated from layouts and explanatory
mechanisms are identified separately. No current installed setting is inferred
from an old service report.

| Experiment date | Change / observation | Measured progression or functionality | History anchor and interpretation |
| --- | --- | --- | --- |
| September 3-5; repair committed September 14 | MKL attention must preserve the source q8_0 layout flag | Long-context q8_0 PPL had become NaN at `n_kv >= 1024`; small oracle shapes missed that route | `54d02625c` integration, `4bb688b3d` repair, `80a120da0` route tests; a correctness recovery, not a speedup |
| September 5 | TheTom centroid refresh and policy sweep | Symmetric turbo3 PPL 8.4805 before -> 8.3939 after; still fails the 5% gate against q8_0 7.9861 | `b1e4170ec` integration; policy choice matters more than treating all turbo3 settings as equal |
| September 27 | Dense and MoE kernel ports, new graph replay | Q8 ESIMD matvec probe 177 -> 157 us and 187 -> 172 us; wide MMVQ alone neutral to -7% | `69db254ec`; microkernel evidence does not establish whole-model benefit |
| September 28 | Actual private queue for host-expert prefetch | Qwen3-Coder prefill about +16%, three alternating repetitions; correct output and successful teardown | `11598778b`; i915/v1-adapter experiment, not transferable to xe with copy engines disabled |
| September 29-30 | xe copy failure investigation | Pinned source + compute-queue copies: pp512 325.1 -> 366.0 t/s against mmap; decode essentially unchanged | `beda936e7` safety default; host runtime patch is a separate dependency |
| September 30-October 1 | Per-kernel large-GRF FA | Quiet-host Llama 8B pp512 998.4 -> 1098.3 t/s (+10.0% +/-0.9%); deep decode flat in tile-only mode | `3775efdd2` initial insertion, `54c3ecbbd` split-out, `5b1eb7c34` integrated feature; opt-in, not a global-GRF rehabilitation |
| October 4-5 | Qwen4Exp head loading, chaining, borrowed tables, checkpoint guard | Synthetic correctness tests and bounded real-head/trunk observations; draft fit/reservation and rollback fixes | `001d906a6`; the branch's several review fixes became one master change |
| October 4 | On-device speculative checkpoint A/B | Qwen3.5 self-draft 15.98 -> 16.55 t/s, +3.53% +/-3.62%; workload mismatches prevent causal isolation | PR 90 branch `99db3ec3f` plus change, before guard; checkpoint sites do not execute under tested Qwen4Exp MTP |
| October 6-7 | Multi-axis xe matrix | CCS 1 beats CCS 2/4; WG 32 helps deep Ornith decode; MKL off helps some d=128 prefill but hurts d=256; ngram advice revised | Installed `b12327.c4b9d6430`; a host experiment, not evidence every setting landed as a fork default |
| October 8-9 | Better speculative observation and guarded scheduler copy opt-in | Accepted-token probabilities become observable; asynchronous input-copy lifetime coverage, but no model-equivalence/speed claim | `d09fb6ba4`, `bba3994a9`; functionality and validation progress without proven throughput gain |
| October 10 | Benchmark archive committed | Historical reports/raw results become repository evidence | `2f623c6607`; archival landing is not the experiment date |

Sources for the first rows: [q8 NaN investigation](../research/sycl/sycl-q8-context-2048-nan.md),
[Turbo3 quality gate](../research/turbo/turbo3-quality-gate-llama31-8b-2026-09.md),
[Ornith source and probe research](../research/sycl/ornith-a770-perf-research-2026-09-27.md).
The remaining source trails are expanded below.

## Quality and capability before throughput

The September 5 quality campaign used Llama-3.1-8B-Instruct Q4_K_M,
WikiText-2, context 512 and flash attention. Build label `435f47bb8` is recorded
in the [report](../research/turbo/turbo3-quality-gate-llama31-8b-2026-09.md).
Its decision retained the 5% threshold rather than changing the gate to make the
new codec pass.

| K / V policy | PPL | Delta against q8_0 | Interpretation |
| --- | ---: | ---: | --- |
| f16 / f16 | 7.9844 | -0.02% | Quality reference |
| q8_0 / q8_0 | 7.9861 | 0 | Gate baseline |
| turbo3 / turbo3 | 8.3939 | +5.11% | Fails; threshold 8.385405 |
| turbo3 / turbo3, adaptive 1 | 8.2428 | +3.21% | Passes this gate with q8 boundary layers |
| turbo3 / turbo3, adaptive 2 | 8.3339 | +4.36% | Passes this gate with q8 last eight layers |
| q8_0 / turbo3 | 8.0292 | +0.54% | Asymmetric K preserves quality much better here |
| turbo3 / q8_0 | 8.3444 | +4.49% | Same nominal bytes as the reverse asymmetry, worse PPL |

Adaptive V modes 5, 6 and 7 gave 8.6085, 8.7709 and 8.6415 and all failed
this comparison. The same report's context-scaling stage gave turbo3 335.46 vs
q8_0 318.75 t/s (ratio 1.052), which passed its numeric throughput gate while quality
failed. The report flags GPU contention in that stage, so the timing ratio is
indicative only. This is a direct example of why a successful performance row is not a
successful feature qualification. The report's 5.12x symmetric and 2.75x
asymmetric compression ratios are layout-derived ratios against f16, not a
measurement of total model memory or a long-context quality guarantee.

## September 28: prefetch worked on one queue/driver regime

The [second-queue investigation](../research/sycl/sycl-prefetch-second-queue-2026-09-28.md)
corrected an earlier premise that SYCL had no usable second queue. On i915,
a standalone probe showed 23.5 ms of 23.5 ms copy/kernel overlap with the v1
adapter: 53 ms wall time versus 77 ms single-queue. The v2 adapter serialized
the cross-queue barrier; disabling the copy engine also removed overlap.

The model experiment was Qwen3-Coder-30B-A3B-Instruct UD-Q3_K_XL, 20 CPU MoE
layers, ctx 4096, 2500 prompt tokens and 32 greedy output tokens, build
`~/build-pr-28414`. Slots 0 measured 133.2 / 132.3 / 132.2 t/s prefill;
slots 3 measured 154.7 / 154.2 / 151.9. The report says about +16%, with no
confidence interval. A repeated first cold-page-cache run around 92 t/s was
excluded; this is a warm-cache comparison. Logs confirmed 300 stagings and
byte-identical output for the exercised slot configurations.

Progress was not merely faster submission: the first verification exposed a
SIGSEGV at teardown because the scheduler synchronized already-destroyed
backends. Waiting on scheduler-owned slot events repaired the lifetime problem.
Correct output before process exit had not established correct functionality.

The same idea later gave 44.7 vs 45.5 t/s on the October Ornith workload and
consumed about 850 MiB outside the fit budget. These observations are compatible:
they use different models, workloads, driver/runtime states and placement.
There is no contradiction to resolve by averaging the two results.

## September 29-30: xe changed the safety and memory-copy problem

The [xe investigation](../research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md)
records Ornith-1.5-35B-A3B Q4_K_M, installed fork `4e7400c3a` (b12305), xe
7.3-rc5, compute-runtime 26.35.39758.10, Level Zero 1.32, oneAPI 2026.1.4 and
GuC 70.53.0. Source/trace analysis tied read-only mmap userptr bind failures to
a runtime eviction sweep unbinding a still-pending blitter command buffer.

The clean same-boot control is X4b versus X4d, both `--moe-cache off` and copy
engines off. The distinction is pinned owned weights (`--load-mode none`)
versus mmap, not compiler or driver:

| Test | mmap X4d t/s | pinned X4b t/s | Supported reading |
| --- | ---: | ---: | --- |
| pp512 | 325.1 +/-18.4 | 366.0 +/-7.6 | Reported about +12.6% prefill |
| tg64 | 49.2 +/-0.6 | 49.4 +/-0.4 | Essentially flat |
| pp512 at depth 8192 | 274.4 +/-3.1 | 304.2 +/-2.0 | Prefill benefit persists |
| tg64 at depth 8192 | 41.9 +/-0.3 | 42.3 +/-0.1 | Small difference, not an isolated decode breakthrough |

[Raw X4b](../benchmarks/N-xe-x4b-pinned-ceoff/bench.md) and
[raw X4d](../benchmarks/N-xe-x4d-mmap-ceoff/bench.md) retain printed benchmark
spread; those +/- values are not the paired-launch CIs of the GRF harness.
X4c with pinned memory and the copy engine on gave 363.8 / 47.9 / 302.4 / 41.6,
so restoring the blitter did not reveal an additional gain in that comparison.

The decisive runtime control was successful completion versus stall, not t/s:
X7G, package NEO `be85a8d685` with read-only retry, completed four rows and
239,364 userptr imports with zero failed binds; X7H, the same package with retry
disabled, stalled after three rows, exited 124 at 600 s and logged 20,363 EPERM
bind failures. X7G's throughput is void because host load was 21-36. X7H had no
reset within the window: a clean reset counter does not prove forward progress.

Several paths ended here: forcing blitter direct submission completed one
instrumented run but decoded at 8.67 vs roughly 14.5 t/s; disabling scratch
pages did not change the VM flags and still stalled; kernel rc5 alone did not
fix the problem. The application mitigation disables copy engines on DG2/xe.
The runtime retry patch and the application hook are distinct changes. Neither
an old package installation record nor the archive proves today's runtime is
patched or that all hardware/driver combinations are safe.

## Compiler and host-load archive: evidence, not a clean compiler ranking

The [campaign guide](../benchmarks/CAMPAIGNS.md) maps A to oneAPI 2026.0 without
DNN, C to 2026.1 without DNN and D to 2026.1 with DNN. All six tables below
identify `bb6908513` and Ornith Q4_K_M with q8 KV, fit target 1024, fit context 32768.
The first three run A/C/D; the second three reverse order. These are selected
raw rows, not fresh analysis or a validated compiler-only speed ratio.

| Arm / ordering | pp512 t/s | tg64 t/s | pp512 depth 8192 t/s | tg64 depth 8192 t/s |
| --- | ---: | ---: | ---: | ---: |
| [A forward](../benchmarks/A-final/bench.md) | 53.65 +/-20.67 | 13.58 +/-3.34 | 81.40 +/-11.49 | 12.57 +/-5.36 |
| [C forward](../benchmarks/C-final/bench.md) | 124.90 +/-7.37 | 17.41 +/-0.47 | 100.57 +/-1.36 | 16.91 +/-0.33 |
| [D forward](../benchmarks/D-final/bench.md) | 68.61 +/-40.92 | 14.93 +/-4.70 | 105.11 +/-4.54 | 12.97 +/-6.65 |
| [D reverse](../benchmarks/D-final2/bench.md) | 43.64 +/-30.06 | 17.56 +/-0.17 | 104.48 +/-3.75 | 16.48 +/-0.58 |
| [C reverse](../benchmarks/C-final2/bench.md) | 123.46 +/-5.86 | 17.83 +/-0.11 | 105.90 +/-3.29 | 16.36 +/-0.52 |
| [A reverse](../benchmarks/A-final2/bench.md) | 54.37 +/-22.59 | 16.57 +/-1.08 | 106.69 +/-1.45 | 15.87 +/-0.52 |

C's short-prefill advantage is a repeated observation. Wide spreads, order,
load and missing original C/D build directories limit causal attribution and
reproducibility. The archive's K-aocc label has no corresponding controlled
compiler-only runner. Later b12321/b12327 ABBA trials record CCS 2, so they must
not be silently spliced into the later CCS 1 matrix. Duplicate `oneapi-ab/`
copies add no independent repetitions. Missing runner logs and old environments
cannot be recreated by static analysis of a surviving throughput table.

## September 30-October 1: GRF exploration narrowed into an opt-in feature

The [GRF note](../research/sycl/sycl-fa-large-grf-2026-09-30.md) records xe
7.3-rc5, IGC 2.41.5 and oneAPI 2026.1.4. Twenty-one of 23 spilling kernels in
its 173-kernel dump were flash-attention kernels. This suggested targeted
register capacity, not a global compiler setting.

All products below use q8_0/q8_0 KV, pp512/tg128, four launches per arm with
sample 0 discarded. Percent intervals are the report's paired 95% half-widths.
These are independent stages, not successive gains to multiply.

| Stage / recorded build | Short-prefill result | Deep-decode result / decision |
| --- | --- | --- |
| All FA kernels forced to 256 GRF; `6bd56b372` vs `73398f9f3` | Llama 8B 1004.0 -> 1098.0 (+9.4%); Ornith IQ2_M 266.0 -> 274.9 (+3.3%) | Llama depth 8192 36.5 -> 34.8 (-4.5%); Ornith 46.6 -> 48.3 (+3.6%); cannot make one global rule |
| Tile-only mode 1; same binary `e4bf0b239` | Llama 1005.6 -> 1107.8 (+10.2%); Ornith 267.7 -> 274.5 (+2.5%) | Both essentially flat; avoids VEC decode tradeoff |
| Mode1 plus occupancy adjustment; `bff20e62a` plus dirty change becoming `30533cbb1` | Llama 998.4 -> 1098.3 (+10.0% +/-0.9%); Ornith 266.5 -> 273.0 (+2.4% +/-1.1%) | Both flat; occupancy adjustment itself adds no resolved gain |
| Final tile-only build `33f59efee` | Llama 942.2 -> 1051.7 (+12.2% +/-27.5%); Ornith 249.5 -> 256.6 (+3.0% +/-7.6%) | Provenance accepted, but load 10-12 makes short-prefill intervals inconclusive |

Raw accepted products include [Llama mode 1 occupancy](../benchmarks/grfmode1occ-llama8b/product.json),
[Ornith mode 1 occupancy](../benchmarks/grfmode1occ-ornith-iq2m/product.json) and
[final Llama](../benchmarks/grfmode1final-llama8b/product.json).

All 16 root-level GRF `product.json` files were inspected for their validation
flags. Ten are accepted by the historical harness; six are rejected. Both
`grfmode1pr89-*` and `grfmode2pr89-*` reject sample build `9ae0f9ad9` against
repository `9595c7ca...`; both `grfmode1aocc-*` reject sample `b23424e30` against
repository `5b97abd5...`. `invalid_cell_count=0` does not override the global
`all_cells_valid=false` provenance failure.

The rejected mode 2 table reports deep-decode losses of 7.9% on Ornith and 4.1%
on Llama. It documents why mode 2 was dropped, but it is not valid confirmation
of those effect sizes. A later accepted `grfmode1aoccv` pair does not repair it:
the Llama depth 0 paired pp result is +14.705592% +/-16.252969%, while Ornith is
-3.812315% +/-13.664320%. These broad intervals do not establish improvement.
The same products have small depth 8192 prefill regressions (Llama -0.663164%
+/-0.618639%; Ornith -1.965031% +/-1.653340%). Accepted provenance is not a
license to declare every metric positive. See [Llama raw](../benchmarks/grfmode1aoccv-llama8b/product.json)
and [Ornith raw](../benchmarks/grfmode1aoccv-ornith-iq2m/product.json); decimals
here are rounded displays of their stored paired means and intervals.

At the pinned tip, [CMake](../../ggml/CMakeLists.txt) still defaults large-GRF
variants OFF; [FA routing](../../ggml/src/ggml-sycl/fattn.cpp) restricts mode 1
to tile launches with more than one query row. An installed package can choose
to compile and enable it without changing the fork-wide default. The GRF note's
last "Not claimed" bullet about mode 2 not being remeasured is internally stale:
its earlier October 1 section does contain a later, invalid mode 2 experiment.
The defensible reading is no later *valid* mode 2 performance confirmation.

## October 4-5: speculative correctness before speculative speed

The [Qwen4Exp correctness record](../research/speculative/qwen4exp-mtp-correctness-2026-10-04.md)
follows repeated fixes: the proper MTP mixer, trunkless-head context handling,
borrowing target embedding/output tables, fit measurement beside the target,
chain sampler rejection before cache mutation, quantized state restore, failed
chain catch-up retention, hidden export and buffer reservation. The integrated
master anchor is `001d906a6`; the report's numerous intermediate PR hashes
refer to branch development, not a separate master milestone each.

The final recorded CPU suites have 75/50 passes for f16/q8 KV; A770 suites have
76/51 and zero new i915/xe fault lines. They establish the tested behavior,
not real-model long-context quality or hardware graph replay. The observed
graph profile used direct execution. A real-head catch-up needed 152.0078 MiB
against 141.0157 MiB originally reserved. The added reservation reaches 152.0000
MiB, still 8192 bytes short: one small reallocation remains explicitly documented.

Borrowed-table real-trunk launches moved head GPU weights 3291.18 -> 2647.04 MiB
and trunk GPU weights 9455.72 -> 10134.54 MiB. Accepted drafts were 838/1810
(46.3%) versus 857/1702 (50.4%), and six-request decode ranges 9.63-14.38 versus
11.26-15.64 t/s. These were one launch per head with different quantized shared
tables, not a paired speed or acceptance benchmark. Their strongest result is
successful execution and changed fit placement.

The [checkpoint A/B](../research/speculative/sycl-a770-spec-checkpoint-on-device-ab-2026-10-04.md)
used Qwen3.5-9B Q4_K_M as both target and `draft-simple` draft, ctx 8192,
q8 KV, temperature 0, four paired launches and two repeats. With about 50.25 MiB
per context checkpoint, host 15.98 -> device 16.55 t/s gives +3.53% +/-3.62%.
Nine of 12 paired launch/prompt cells generated different streams. The interval
includes zero and the work differs between arms; the post-hoc stream-matched
+4% subsets cannot replace the planned statistic. These measurements also
predate the device-margin guard and stricter accepted-token verifier.

Under Qwen4Exp `draft-mtp`, all 95 verified rounds in the probe created and
restored zero checkpoints because snapshot rollback and partial removal cover
the path. Benchmarking that flag repeatedly on the same configuration cannot
measure a nonexecuted feature. Under Qwen4Exp ngram-mod, code-edit 13.14 -> 31.35
t/s reflected different accepted draft workloads, not checkpoint efficiency.
The two no-draft prompts differed 1.8% and 5.3% despite functionally identical
arms, quantifying the local noise problem.

## October 6-7: controlled follow-ups changed the advice

The [final matrix report](../benchmarks/matrix-1006/FINAL-REPORT.md) covers
October 6 03:14 through October 7 11:23, installed `b12327.c4b9d6430`, xe and
oneAPI 2026.1. Its "clean" criterion is foreign host CPU <=18%, not an idle
dedicated machine. It supersedes phase 2/phase 4 speculative conclusions and the
early [matrix guide](../benchmarks/MATRIX-GUIDE.md). Raw phase tables are in
[ALL-TABLES.md](../benchmarks/matrix-1006/ALL-TABLES.md).

| Axis and cohort | Final recorded result | Scope / functionality implication |
| --- | --- | --- |
| CCS 1 vs 2 vs 4, Ornith production, four rounds | Mode 2 -12.5% pp, -13.2% tg, -17.5% tg at 8k; mode 4 about -35% | Multiple compute engines do not automatically accelerate one workload; mechanism not proved |
| CCS 1 vs 2/4, dense 8B in VRAM | Mode 2 -29% pp/-18% tg; mode 4 -62%/-52% | Cross-model negative evidence for these settings; not all possible concurrent workloads |
| Two concurrent processes | Aggregate 1525 pp/59 tg vs single 1634/62 at mode 1 | No demonstrated aggregate gain from this concurrency experiment |
| MKL off, Llama 8B d=128 | pp512 +32% at 2k, +57% at 8k, +58% at 16k, +63% at 32k; clean 32k: 63.5 -> 103.7 t/s | Default-on MKL is not universally fastest |
| MKL off, Ornith d=256 | pp -10% at 2k, -24% at 8k, -34% at 16k, -43% at 32k, -53% at 64k; clean 32k: 208 -> 118 | Opposite direction; no universal MKL-off recommendation |
| MKL off, Qwen2.5-7B d=128 GQA 7:1 | +4%, -7% at 4k, +30% at 8k, +38% at 16k | Head dimension alone does not explain every cell |
| WG 32 vs 16, Ornith Q4_K_M, three rotated rounds | tg +5.2% at 16k, +10.3% at 32k, +18.0% at 64k; depth 0 unchanged | Report records service-only wg32 drop-in October 7; source default remains 16 |
| WG 32 real-source decode | +5% near 27.5k, +20% near 61k | More relevant than random-token bench; long greedy checks within observed noise, not bitwise equality |
| WG 32, dense 8B | Within +/-1.7%, two clean rounds | Ornith result does not generalize to dense d=128 |
| Speculation, Ornith ten varied local-file prompts, three rotated launches | None 45.4 t/s at temperature 0, 45.3 at temperature 0.6; ngram-mod -0.1%/-6.2%; ngram-simple -3.3%/-2.8% | No speculative mode wins this mix; earlier ngram-simple advice withdrawn |
| Prompt-dependent ngram-mod | +40% code_comment, +18% csv_to_md, -36% summarize, -25% repeat_code | A single prompt cannot establish product-wide benefit |
| Qwen4Exp MTP | No spec 13.3; depth 6 16.8, depth 3 17.6, depth 6+chain 4 17.4 t/s | Useful workload-specific capability; no matched CI supplied in final report |
| Prefetch slots 4 with Ornith | 44.7 vs 45.5 t/s; free GPU memory 104.7 vs 951 MiB after load | No speed gain in this run; added memory pressure about 850 MiB outside fit |

Other negatives are part of the progression: Ornith at 24 CPU threads lost 10.5%
decode; ubatch 256 lost 33% prefill; ubatch 2048 lost 4% at 8k; pinning 2000 MHz
lost 7% decode and 1500 MHz lost 19%; pinning 2400 matched default, and a verified
power-saving profile had no effect. Disabling immediate command lists lost 23%
Ornith decode, general optimizations 20%, fusion 11%, pinned memory 8%, ESIMD 6%.
Graph-off was neutral on Ornith but -6.7% on dense 8B. These are the final
report's workload-specific controls, not a single portable configuration.

MKL route PPL differences at ctx 8192 were small in these tests (on minus off:
Llama -0.007, Qwen2.5 +0.07, Qwen3.5 +0.008, Ornith within 0.01 noise). They do
not replace an oracle for d=256, GQA 7:1 and the large-KV fallback. The report
explicitly says those oracle shapes were absent.

The report disclosed harness failures: SSH consuming loop input, `setvars.sh`
under nounset, a fault counter always returning 0, a wrong model path, own CPU
counted as foreign load, and a leaked server holding VRAM. Per-cell kernel
fault evidence from 03:14-06:00 is missing. Later monitors cover later runs;
they cannot retrospectively certify that interval. Allocation aborts under
small fit margins remain partly confounded by possible stray GPU holders.

At the pinned source, [SYCL initialization](../../ggml/src/ggml-sycl/ggml-sycl.cpp)
uses WG 16, graph=0 and MKL-FA=1 as defaults. Thus the final report's operational
recommendations and reported service changes are not identical to repository
defaults. [Scheduler source](../../ggml/src/ggml-backend.cpp) keeps synchronous
input copies unless `GGML_SCHED_COPY_SYNC` is exactly `0`; the October 9 change
adds an opt-in path, not an evidenced default speedup.

## Dead ends and what can still answer them

"More benchmarking" is not a universal remedy. Some questions need a missing
artifact, a corrected test, a different execution path, or acceptance of an
irrecoverable historical gap.

| Dead end / unresolved question | What static analysis can answer | What new experiments could answer | What cannot be recovered or claimed |
| --- | --- | --- | --- |
| Symmetric turbo3 fails 5% PPL gate despite speed pass | Verify centroid/policy selection and actual K/V types | Paired quality/capacity sweep across longer contexts and models | A faster timing run cannot turn this recorded quality failure into a pass |
| i915 prefetch gain disappears on xe/Ornith | Confirm adapter/copy-engine decline and scheduler dispatch | Same model/placement/stack A/B, staging counters and bounded output checks | Original +16% cannot be transferred across the driver/model change |
| xe completion vs silent stall | Trace VM_BIND flags, lifetime and runtime retry logic | Reproduce on pinned new runtime, source-memory controls, completion timeout and bind counters | Absence of reset lines never proves a stalled process healthy |
| Blitter direct submission and scratch switch | Determine whether the switch actually changes code/VM flags | Only re-open with changed runtime behavior and controlled timings | One completed instrumented run is not production qualification |
| Global/all-FA GRF hurts decode; mode 2 rejected | Inspect which kernels get 256 GRF and occupancy limits | Correct-provenance rerun after relevant compiler/kernel evidence changes | Invalid products cannot become valid through selective quotation |
| Final GRF short-prefill uncertainty | Verify the same kernel route and build/property mapping | Quiet paired launches can narrow current uncertainty | A rerun cannot improve the old interval or repair the old repository/binary mismatch |
| Compiler/DNN archive lacks clean controls | Recover runner flags, hashes and recorded libraries | Rebuild pinned versions and run matched clean arms | Missing original build directories/logs and historical host state are not reconstructible from tables |
| Checkpoint flag under tested Qwen4Exp MTP never executes | Establish eligibility and snapshot/partial-removal path | Choose a workload that creates checkpoints, confirm counters first | Repeating the same inactive-path benchmark cannot measure the flag |
| Checkpoint +3.53% with mismatched generated streams | Audit checkpoint sizes and verifier blind spots | Workload-matched replay or controlled requests, current verifier, more paired launches | Post-hoc stream selection cannot establish the original causal effect |
| MTP fixture ran with graph enabled but no replay | Read eligibility and profile counters | Instrument an eligible graph and require replay count>0 | A green direct-execution fixture proves no hardware replay correctness |
| CCS 1 wins; why? | Trace queue-to-engine selection and capacity interpretation | Per-engine counters/traces plus same-session controlled workloads | Existing fdinfo is suggestive, not a demonstrated scheduler mechanism |
| MKL sign flips and missing d=256/GQA7 oracle | Verify routing/layout support; add meaningful oracle shapes | Execute oracle before model/depth timing sweep | Small PPL differences alone prove neither exactness nor universal route policy |
| WG 32 long greedy outputs differ | Inspect reduction ordering and exact compared token representation | Repeated same-WG/cross-WG controls, logprob margins, longer quality tests | Near-noise divergence cannot establish bitwise equality |
| Ngram gains vary by prompt; earlier recommendation withdrawn | Trace acceptance/verification accounting | Representative prompt mix and sampling, full acceptance stats, balanced launches | The final negative mix does not disprove gains on repetition-heavy prompts |
| Prefetch/MTP/large-batch OOM at tight fit margin | Audit allocations omitted from fit and reservation growth | Log every GPU holder and per-process VRAM; isolate each feature; forced allocation failures | Earlier runs without holder logs cannot exclude the leaked-server explanation |
| MTP reservation still 8 KiB short | Inspect allocator best-fit behavior and reserved/decode shape differences | Real-head catch-up/peak-allocation tests across shapes | The documented partial fix is not a universal upper-bound proof |
| Early October fault log missing | Audit and repair the fault collector | New complete per-cell logs and completion gates | Neither static review nor rerunning today certifies the lost interval |
| Scheduler stream-ordered copy opt-in | Check source lifetime/event/exclusion rules | Path-counter-confirmed model equivalence and paired performance | Lifetime tests alone do not establish output equivalence or speed |

## Coverage and limits

This chapter reads the named dated notes, the final October report, six A/C/D
raw tables and all 16 primary GRF products. The archive's
[guide](../benchmarks/README.md), [campaign registry](../benchmarks/CAMPAIGNS.md)
and [matrix guide](../benchmarks/MATRIX-GUIDE.md) cover the remaining families,
including noisy plain-decode trials, package ABBA runs, server soaks and traces.
Those artifacts are preserved as supporting evidence, not silently promoted to
independent accepted benchmarks. Copied archive trees are not additional samples.

Hindsight recall was consulted to locate the October transition: early facts
`9c307dde-0bcf-4bba-afb2-a166b9ca64d8` and
`7187487c-293c-42c7-9489-274a83111995` describe missing Ornith axes and unrun
MTP before follow-up; the final report supplies later results. They are
historical gaps, not proof those questions remain unanswered. All numeric
claims above have repository sources; memory is not an independent replicate.

Not run: no archived executable, build, benchmark, GPU test, service command or
new statistical campaign. Risks: historical validity is bounded by each report's
provenance, missing artifacts, configuration and corrected methodology; none of
these results establishes current production performance or safety.
