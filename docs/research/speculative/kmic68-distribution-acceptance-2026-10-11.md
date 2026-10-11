# P06 distribution-recording speculative acceptance: candidate evidence

Date: 2026-10-11. Status: **Accepted and integrated; fresh CPU integration checks passed**.
Scope: [P06 requirements](../../plans/kmic68-06-speculative-sampling.md).

The required optimized statistical corpus, synthetic consumer fixtures, trained
exact-Qwen4Exp OFF/ON campaign and regression floors passed. The fresh final
campaign records actual distribution acceptance in all 287 enabled cycles and
1984 proposal-bound q records, including 369 sampled tokens distinct from top-1.
An independent [acceptance review](/home/svnbjrn/wt/kmic68-evidence/p06-acceptance-review.md) cleared the candidate. Integration commit
`09b203c4744198dd66f72cf6a29bf5fd31cc5164` passed the fresh CPU checks below.
Final combined AOT and P07 interaction checks remain pending.

## Candidate and behavior

The original tested candidate was `codex/kmic68-speculative-sampling` at
`1cd063604d08b2a322a56b4897265e60d45602a2` plus 19 changed files in
`/home/svnbjrn/wt/kmic68-speculative-sampling`. That tested snapshot was uncommitted.
The reviewed implementation and three documentation files were subsequently
committed as `c303c6c5898eba82bbfcfcf9cf433e7266ae1d70` and integrated as
`09b203c4744198dd66f72cf6a29bf5fd31cc5164`; all 19 source/test/harness files
match the accepted snapshot byte for byte. The normal CPU builds use GCC and Ninja; the separate
consumer fixture build is CPU Debug. The normal AOT continuation ran on the
Intel Arc A770 with the xe driver and captured actual executable/project-library
mappings. The completed reports below retain those identities. Sampling itself
is CPU sampler logic in the required optimized SYCL build; mapping SYCL libraries
does not establish GPU arithmetic. Explicit SYCL0 MTP fixtures passed separately.

P06 remains disabled when `LLAMA_SPEC_SAMPLE_TEMP` is unset or zero. A finite
positive value enables eligible non-chained host MTP sampling. Optional
`LLAMA_SPEC_DRAFT_TOPP` defaults to 1 and must be finite in `(0, 1]`.
Unsupported stateful, grammar, backend-selected, or chained sampling retains
whole-block exact-match fallback. These defaults are present in the candidate's
`common/speculative.h` and parser in `common/speculative.cpp`.

The candidate binds each sparse draft distribution to P04's immutable proposal
identity. The sampled token is distinct from top-1 confidence. Complete block
validation and a working sampler/RNG clone precede live state mutation. A new
attempt with a consumed or replaced proposal falls back to exact match. Cache
repair of an already completed distribution decision preserves its selected
tokens and committed sampler/RNG state without sampling again. The acceptor's
`G+1` contract remains intact; consumers discard any suffix after an effective
EOG according to their existing stopping policy.

## Requirement-linked evidence

All paths below refer to retained evidence under
`/home/svnbjrn/wt/kmic68-evidence/`.

| Requirements | Recorded evidence | Limit or pending gate |
| --- | --- | --- |
| R06.1-R06.3 | Candidate source review covers aligned sparse distributions, sampled-token identity and temperature/top-p parsing. CPU sampling fixtures pass in Release. | Trained campaign: 287 enabled proposals, 1984 normalized q records, 369 sampled tokens distinct from top-1. This verifies recorded proposals, not general trained-output distribution equivalence. |
| R06.4-R06.6 | Complete Release corpus: 77 cases, 100000 outputs per arm per case, 231 Pearson/Holm hypotheses; final PASS. | Normative optimized SYCL-build execution also passed all 77 cases/231 hypotheses in 47.16 seconds under the unchanged 180-second bound. Neither corpus proves universal or trained-model equivalence. |
| R06.7-R06.11 | Normal CPU parser/sampling fixtures and independent source review cover disabled mode, constrained/stateful/backend/chained fallbacks, invalid distributions and rejected environment values. | Trained baseline: 327 exact-match cycles, zero distribution attempts; enabled: 287 distribution cycles, zero exact-match attempts. P05 combinations are excluded because P05 was rejected. |
| R06.12, R06.15-R06.16 | Transactional and checkpoint-repair fixtures; 19 executable consumer cases preserve decisions, sampler/history and both RNG domains, including same-length replay repair. | Test-only forced FULL repair does not establish trained-model normal-RS versus FULL equivalence. |
| R06.13 | Nine speculative-simple cases, nine ordinary server cases, and one cancellation/reset/reuse case passed; includes early/final/correction EOG and ignore-EOG behavior. | Generated tiny GGUFs and injected decisions are synthetic. These fixtures alone do not establish trained-model repair or EOG coverage; the separate real campaign establishes the ordinary server route. |
| R06.14 | P04 cycle tests pass; candidate source review covers versioned proposal-bound q records. Tenancy correction passed 23 mocks and 3 generic process lifecycle tests with 90 subtests; later load correction passed 26 mocks and 99 subtests, carrying unchanged lifecycle evidence. | Six matched requests per arm completed. Baseline null q and enabled non-null normalized q records reconcile with every recorded request/proposal. |

Primary sources:

- [CPU retry report](/home/svnbjrn/wt/kmic68-evidence/p06-cpu-retry/report.md), with `release-sampling.log`, `release-statistical-summary.json`, and the preserved Debug timeout.
- [Compiled consumer retry](/home/svnbjrn/wt/kmic68-evidence/p06-consumer-cli-retry/report.md), with 19 per-case results, exact process arguments, cycle logs, fixture identities and cleanup evidence.
- [Final source/CPU review](/home/svnbjrn/wt/kmic68-evidence/p06-final-source-review.md), an independent read-only review, not a build or runtime test.

## Statistical and consumer results

The measured CPU Release test duration was 49.33 seconds. All 77 cases retained
100000 output tokens per arm, arranged as 25000 four-token episodes per arm.
There were 77 complete pooling records and 231 unique hypothesis records.
The predeclared seed formulas, zero-support failure rule, deterministic pooling
with expected counts at least 5, and Holm-Bonferroni family-wise alpha 0.01 were
unchanged. TV and maximum error remain diagnostics. The saved audit and
independent review checked the complete family rather than a passing prefix.

The corresponding Debug run timed out at 180 seconds with exit 124, after 44
completed corpus records and before any Holm decision. Its full corpus is
incomplete, not passing. Its checkpoint-repair fixture completed before the
timeout. This diagnostic timeout neither relaxes nor substitutes for the
plan's separate optimized SYCL 180-second criterion.

Both normal CPU configurations passed the parser/adaptive/cycle group (3/3),
synthetic MTP default and Q8 modes, and Python benchmark/cycle-log modules
(40 and 8 tests). Harness qualification passed 21 mocks, 3 real-process signal
cases, and 13 independent consumer-runner observation/cleanup checks. Those
runner checks are separate from executable consumer coverage.

The corrected consumer run passed all 19 registered cases with zero skips in
8.64 seconds of measured CTest execution time. It reused the successful fresh
`build-cpu-p06-consumers-offline` artifacts: CPU only, fixture macro enabled,
both UI options disabled, and zero embedded UI assets. The test-only
`LLAMA_TEST_SPEC_CONSUMER` option defaults OFF in `tests/CMakeLists.txt`.
The fixtures force FULL checkpoint repair and inject p/q decisions. Their P04
q observations describe the original proposal; they must not be reused as
trained-model distribution evidence. No throughput inference follows from
these test durations.

## Failed attempts and offline qualification

Earlier attempts remain preserved. The first CPU build failed while creating
an abstract model fixture. A later consumer build failed because its observer
accepted `llama_batch` while both production callers provided `llama_batch_ext`.
The successful offline consumer build then encountered an unsupported CLI
argument in the simple consumer. Reviewed fixture construction, observer, and
server-only argv corrections preceded the successful retry; the 19 consumer
assertions were not weakened.

Earlier normal CPU and failed consumer builds supplied `LLAMA_BUILD_UI=OFF`
but left the separate prebuilt-UI option enabled. UI assets were downloaded,
contrary to the campaign restriction. The CPU functional observations remain
recorded, but those builds are not compliant offline/UI-disabled build evidence.
Logs, archives and stamps remain under the earlier evidence directories.
The later fresh consumer build explicitly disabled both UI options. It does
not retroactively correct the earlier violation. The P06 CPU retry report
records no model download in that run; it is not a claim about all earlier
campaign activity.

## AOT verification and retained failures

The [first runtime attempt](/home/svnbjrn/wt/kmic68-evidence/p06-sycl-verification/report.md)
passed normative sampling in a measured 47.16 seconds: 77 cases, 100000 outputs
per arm per case and all 231 Holm hypotheses, with unchanged seeds, pooling and
180-second outer bound. Actual process mappings identify the candidate image and
six project libraries. The [independent statistical audit](/home/svnbjrn/wt/kmic68-evidence/p06-normative-runtime-review.md)
reconciled complete recorded seeds, pooling maps, p-values and Holm thresholds.
The log does not contain full observed-count vectors, so this is not an
independent recomputation of Pearson statistics from samples.

That sequence still failed: all three focused test payloads passed, but the
short-lived cycle process lacked a complete same-PID executable/library capture.
The strict provenance gate stopped execution. A separately reviewed bounded
stop-at-main GDB observation path captured unchanged fast fixture binaries;
it does not retroactively repair the original failed observation or support
performance claims.

The [continuation report](/home/svnbjrn/wt/kmic68-evidence/p06-sycl-verification-retry/report.md)
records seven passing gates: the three focused fixtures with all three process
captures; README MTP default and Q8; README Python 40-test floor; Python cycle-log
8-test module with all five native child captures; and explicit SYCL0 MTP default
and Q8. Normative sampling carried forward only after source/index/binary identity
equality. The trained baseline server then loaded the approved model/head and
reached health, but failed with `GPU holder attribution unavailable` before any
request. The enabled arm and three post-campaign floors did not run. The server
exited cleanly; the enclosing sequence remains failed. Readable fault deltas were
empty, resources were released, and source/index/binary/model metadata remained
unchanged. Model hashes were reused after metadata checks, not recomputed by
this failed continuation.

The [tenancy fix report](/home/svnbjrn/wt/kmic68-evidence/p06-tenancy-fix/report.md)
and [independent review](/home/svnbjrn/wt/kmic68-evidence/p06-tenancy-fix/source-review.md)
record a correction confined to two Python harness files. Structured probes now
retain argv, return code, separate stdout/stderr and exception/partial-timeout
streams in the existing unconditional per-arm evidence path. Positive PID stdout,
recognized node/access-marker stderr and ownership checks remain strict. The old
failure discarded its internal probe streams; rejection of normal access markers
is a source-supported explanation, not an observed cause of that attempt.

The updated harness passed 26 tests and 90 subtests in a measured 1.06 seconds
(2.634 seconds outer wall time), including 23 mocks and three generic process
lifecycle tests. No native P06 binary ran for this qualification. The saved
[identity record](/home/svnbjrn/wt/kmic68-evidence/p06-tenancy-fix/post-test-source-identities.json)
confirms unchanged HEAD/index and the other 17 frozen files. At that intermediate revision the two source identity exceptions were:

| File | Updated SHA256 |
| --- | --- |
| `scripts/perf/verify-spec-distribution.py` | `c92f251a1e476b8f341aec943f55a1d90e4937b72e0233939db077d1beba53ae` |
| `scripts/test_spec_distribution.py` | `5871af0752e54a90ce2f6e2844048ea9709a9a4e3ae03fc0c649e81773501aab` |

The next [attempt](/home/svnbjrn/wt/kmic68-evidence/p06-final-verification/report.md)
completed six baseline requests and all 13 tenancy probes, then failed its
immediate postflight load check. Its exact rejected load sample was discarded;
a later outer observation was 11.2080 against a threshold of 6. The enabled arm
and three post-floors did not run. This attempt remains failed, without a causal
claim about the elevated load.

The [load correction](/home/svnbjrn/wt/kmic68-evidence/p06-load-fix/report.md)
retains the threshold and permits a bounded 180-second settling interval, with
repeated resource checks, saved numeric observations and a fresh load sample
after probes. The unchanged outer 1800-second campaign bound and independent
final resource guards remain binding. An in-flight holder probe may finish after
the settling deadline but cannot pass late. Focused mocks passed 26 tests and
99 subtests; unchanged real-process ownership tests were carried forward.
Both the [source fix](/home/svnbjrn/wt/kmic68-evidence/p06-load-fix/source-review.md)
and [continuation](/home/svnbjrn/wt/kmic68-evidence/p06-load-retry-orchestration-review.md)
passed independent review.

Final Python identity replacements:

| File | SHA256 |
| --- | --- |
| `scripts/perf/verify-spec-distribution.py` | `21de2a7631e1ea4a7d2feae82609847c04f12275a56944daa3413c4119ae7cbc` |
| `scripts/test_spec_distribution.py` | `9ee6f0c51a3663ad56181c16fdc26097b4de16432f2d8a8fea69b6ba8a7dad83` |

## Completed trained campaign and final floors

The [fresh final report](/home/svnbjrn/wt/kmic68-evidence/p06-load-retry/report.md)
records four passing payloads and four passing provenance audits: trained OFF/ON,
README MTP default, README MTP Q8 and Python 40-test floor. No failed arm was reused.

| Arm | Requests | Cycles | Drafted | Accepted | Distribution attempts | Exact-match attempts |
| --- | --- | --- | --- | --- | --- | --- |
| OFF | 6 | 327 | 2266 | 790 | 0 | 327 |
| ON | 6 | 287 | 1984 | 806 | 287 | 0 |

Each arm passed 13 active-holder probes and clean SIGINT shutdown without
escalation. Postflight load settled below the unchanged threshold in both arms.
The 287 enabled proposal rows contain 1984 q records; 369 sampled tokens differ
from top-1, and all sampled/top-1 fields reconcile against sparse q support.
For example, request 48/proposal 51/index 1 sampled token 361 at probability
0.1076947451 while top-1 token 2564 had probability 0.8070554137. The raw
[separation audit](/home/svnbjrn/wt/kmic68-evidence/p06-load-retry/sampled-versus-top1.json)
and q artifact preserve exact values. These counts are route evidence, not
acceptance-rate, speedup or trained-output distribution equivalence claims.

Effective settings were target/draft temperature 0.8, draft top-k 10/top-p 1,
target temperature-only sampling, seed 123, non-chained host MTP with minimum
1/maximum 7 draft tokens, context 16384 and Q8 KV. Models are the validated
local two-shard qwen4exp target and matching Q8 MTP head recorded in the report.
Their accepted full hashes were reused after before/after resolved path, size,
mtime, inode and ctime checks; this run did not recompute full model hashes.
Actual native process mappings and compiled source/binary/HEAD/index identities
passed. Only the two reviewed Python replacements differ from original build
evidence. The normative corpus and seven earlier successful audits carry forward
individually; every failed enclosing sequence remains failed.

Final release found no render holders, surviving payloads, llama processes,
builders or new GPU faults. Kernel, xe driver and boot identities stayed fixed.
Services remained inactive/inactive/failed; restoration belongs to the coordinator.

## Integration checks

The [fresh CPU integration report](/home/svnbjrn/wt/kmic68-evidence/p06-integration-verification/report.md)
records a successful 152-step Release build at `09b203c47`, with both UI options
disabled and production `NDEBUG` preserved while tests enable assertions.
Cycle/adaptive/parser tests passed 3/3; sampling passed all 77 cases and 231
Holm hypotheses in 51.05 seconds; MTP default and Q8 passed; Python suites
passed 40, 8 and 29 tests with zero skips. Source, index, HEAD and binary
identities stayed unchanged. These checks used fresh CPU artifacts and did
not repeat the 19 injected consumer cases or GPU/model execution. Runtime
mapping samples do not guarantee capture of every short-lived fixture.

Final combined AOT regression remains pending. P06/P07 combined coverage remains dependent on P07.
P06/P10 is blocked by P10's recorded calibration protocol blocker; P05 was
rejected and is excluded. No combined pass is claimed for excluded features.

No speedup, universal trained-model distribution equivalence or full typed ABI
compatibility is claimed. Synthetic forced FULL repair is not proof of ordinary
trained-model RS versus FULL equivalence. New state consists of source changes,
build artifacts and retained evidence; no deployment is implied.

## Build and compatibility update

The [normal AOT build](/home/svnbjrn/wt/kmic68-evidence/p06-sycl-build/report.md)
completed successfully in 853.40 seconds with IntelLLVM 2026.1.1, acm-g10,
Release flags, normal consumer mode, both UI options OFF and build/internal link
parallelism capped at four. Effective DNNL is ON. No embedded UI assets or
archives were present. All 19 frozen source files, the index and HEAD matched
after the build. These are build observations, not A770 execution evidence.

The [predecessor floor](/home/svnbjrn/wt/kmic68-evidence/p06-predecessor-floor/report.md)
passed default and Q8 MTP fixtures using the accepted P04 binary, plus the
current integration branch's 40 Python harness tests. The source qualification
establishes unchanged MTP/CPU paths; backend and harness revisions differ, so
this is explicitly split-provenance evidence rather than a whole-head baseline.

The [public C compatibility check](/home/svnbjrn/wt/kmic68-evidence/apiabi-p06/report.md)
passed with 100% source compatibility and no removed/changed C symbols. The
sole added C symbol belongs to previously accepted P02. A separate unsuppressed
C++ audit found five removed and 39 added common exports, including removal of
the old recorder signatures, while SONAME remains libllama-common.so.0. Source
inspection also confirms changed installed common structure layouts. Rebuild
common C++ clients with matching headers/libraries; do not treat the C-only
checker result as drop-in C++ binary compatibility. Full typed ABI remains
unproven because Release binaries lack complete type debug information.
