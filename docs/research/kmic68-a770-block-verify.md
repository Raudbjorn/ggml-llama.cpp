# P07 block verification: rejected by the mode-identity gate

Evidence date: 2026-10-11. **P07 is rejected by its mandatory mode 1/2 token-identity
gate; candidate code is excluded from integration.** All three trained requests
completed 4096 tokens with a limit stop, but modes 1 and 2 first differ at
zero-based output index 1: token 103196 versus 248068. The actual outer launcher
returned 1. The cause is unproven; this result does not establish distribution
bias, a diagnostic RNG defect or a throughput regression.

Source review, CPU verification, an unchanged AOT retry, scoped compatibility
checks and all eleven native/synthetic gates passed. Those scoped passes do not
override the real-model gate. Independent disposition review confirmed rejection.
Final accepted-tree checks and campaign service restoration are complete; see
the [campaign ledger](kmic68-campaign-outcomes-2026-10-11.md). They do not qualify this rejected candidate.

Scope: [P07 plan](../plans/kmic68-07-block-verify.md), extending accepted
[P06](speculative/kmic68-distribution-acceptance-2026-10-11.md).
Worktree: `/home/svnbjrn/wt/kmic68-speculative-block`, branch
`codex/kmic68-speculative-block`. Tested CPU source HEAD:
`218f89837901990a65a6ddcd54ff2b8ea0ff4c46` plus 11 frozen product files. Four
reviewed harness files were subsequently copied without changing those product
bytes. The AOT preservation inventory tracks 27 paths, including unchanged build
and test dependencies; this is not a claim that 27 product files were modified.
No P07 feature commit is integrated. The 15 frozen product/harness files are
preserved in [candidate.tar.gz](/home/svnbjrn/wt/kmic68-evidence/p07-rejected-source/candidate.tar.gz),
154552 bytes, SHA256
`b4b7e8da2ecd1b2833b8020f3ea18af5ec9a889153915e8fde9b2ea949dedb88`.
The coordinator reports independent member rehashing; adjacent manifests,
HEAD/status records and the candidate patch preserve provenance.

Native verification ran on the Arc A770 on xe. The first AOT attempt used
IntelLLVM 2026.1.1, installed IGC 1:2.41.10-1, compute-runtime package
22.43.24558.r13063.gbe85a8d685-1, Level Zero loader
1.34.0.r2.gae3db48-1, and kernel
7.3.0-rc5-273-linux73-tkg-bore-rc5-xe-aocc. These package versions are build/stack
observations. The completed native report separately records actual executable
and library mappings, positive SYCL device routes, and consistent driver/host
identities. The trained campaign separately records actual mappings and preserved
model/source/binary identities, with the qualifications below.

## Excluded candidate behavior and source review

`LLAMA_SPEC_BLOCK_VERIFY` unset or exactly `0` retains P06 per-token acceptance;
`1` selects block verification; `2` selects the same rule with diagnostics.
Other values, including empty and noncanonical numeric strings, fail sampler
initialization. No measurement can promote a default in this plan.

The helper normalizes validated P06 target/draft rows in double precision,
starts `keep[0]=1`, applies the capped probability-ratio recurrence, evaluates an
independent survival draw at every drafted position, and chooses the greatest
passing prefix. Intermediate survival probability is `S/(S+(1-keep))`; terminal
survival is `keep[G]`. A rejected suffix draws from the normalized positive
residual `max(keep[tau]*P[tau]-q[tau],0)`; zero residual retains the independent
target draw. Full acceptance uses target row `P[G]`. Existing P06 eligibility,
whole-block fallback, cloned transaction and commit/consumption ordering remain.

Source review and synthetic same-input fixtures found identical mode 1/2 RNG
decisions; the trained cross-launch identity gate failed as reported below.
Diagnostic calculations draw no additional random values. State is noncopyable and caller-owned in `server_slot`
or the simple consumer's local scope. Reset of the same acceptor/sequence keeps
its identity; actual sampler replacement rotates identity even for an identical
seed or reused pointer address. Replacements discard partial buckets. Different
slots, sequences and cloned acceptors cannot pool contributions.

Only a committed, consumed, recorded eligible proposal contributes once.
Fallback, failed transactions, replay and duplicates contribute nothing. Exactly
256 eligible contributions emit a `spec_block` JSON line identifying launch,
sequence, acceptor and first/last proposal, with `blocks`, both expected accepted
draft-token sums, their signed difference and `drafted_tokens_total`.
The conditional expectations are:

- `E_block = sum_r (1-product_{i=r..G}(1-h[i]))`.
- `E_token = sum_k product_{j=0..k} min(1,P[j](draft[j])/q[j](draft[j]))`.

Neither operand nor denominator includes the correction/bonus token. A signed
conditional gain describes the observed proposals; it is not an unconditional
confidence interval, throughput result or default-eligibility decision. Existing
cycle JSON schema version 1 adds `block_verify_mode`; fallback records carry 0.

The [source handoff](/home/svnbjrn/wt/kmic68-evidence/p07-source-handoff.md) and
[closure review](/home/svnbjrn/wt/kmic68-evidence/p07-source-closure-review.md) retain two corrected findings:
left-associated `S+1-keep` lost a positive `2^-54` residual, fixed by grouping
`S+(1-keep)` with an exact fixture; identity-only recorders had no launch ID when
logging/profiling were absent, fixed by the existing atomic identity allocator
without enabling file output, profiling or phase allocation. Closure review
found no unresolved issue in the 11-file product diff. Review is source evidence,
not execution or full ABI proof.

## Frozen statistical and consumer verification

The matrix was fixed before execution and retained every original P06 case and
seed: 77 cases, two arms each. P07 adds all 14 special-support cases and 16 dense
vocabularies (2,6,10,...,62), each at block lengths 1,2,4,8 and modes 0,1,2.
This is 154 original plus 360 added arms: **514 arms, 437 comparisons and 1311
hypotheses, with 100000 output tokens per arm**. All hypotheses share one
Holm-Bonferroni family at alpha 0.01; bins pool to expected count at least 5;
zero-support outputs fail. TV/max error are diagnostic. The normative timeout
is **240 seconds**, unchanged after results.

P06 seeds are `0x426b1021+i*7919` and `0xd31780a5+i*104729`. P07 base seed is
`0x91772f01+i*65537+G*193`; modes 0/1/2 use base, base XOR `0x13579bdf`, and base
XOR `0x2468ace0`. All 120 added case/length seed sets remain preregistered.

The [CPU report](/home/svnbjrn/wt/kmic68-evidence/p07-cpu-retry/report.md) records normal and separate
consumer-hook Release builds (232 steps each), with production NDEBUG and test
UNDEBUG, four jobs, both UI options OFF and all GPU backends disabled. The full
normative corpus passed in **228.90 seconds**, with exact enumeration, target
joint-law, RNG identity and lifetime checks passing. No case, seed, sample size,
threshold or timeout was reduced. Raw observed-bin counts were not logged, so
this is not an independent recomputation of Pearson statistics from raw samples.

The registered synthetic consumer suite passed **60/60 cases**, no skips, in
23.62 seconds: 19 original cases per mode plus three caller-lifetime cases.
Generated tiny GGUFs and injected probabilities exercise real server/simple
caller state, EOG, checkpoint repair and reset/replacement. They are synthetic;
they do not replace the trained one-acceptor 256-block campaign. The report binds
309 retained consumer artifacts. Focused cycle/adaptive/parser checks passed 3/3,
compiled recorder Python checks passed 8, and P06 distribution checks passed 29.
CPU README MTP default/Q8 and the 40-test Python floor passed before and after. Source,
HEAD/index/status and binary preservation passed. Sampled mappings did not
capture every short-lived process, so their coverage is explicitly bounded.

The Stories-based server suite was not run because its fixtures invoke downloads
and no qualified local mapping was supplied. The 60 synthetic cases are not a
pass of that unavailable suite. An earlier predecessor Python floor failed on
an empty PID-file publication race; its evidence remains preserved. The reviewed
atomic-publication fix is separate integration commit `456db50ba`, present as
local `218f89837`; the successful retry does not erase the first failure.

The [normal AOT verification](/home/svnbjrn/wt/kmic68-evidence/p07-sycl-verification/report.md)
completed all eleven gates and evidence audits, and the coordinator observed
launcher exit status 0. The fresh normative corpus matched 437 comparisons,
1311 hypotheses, 100000 outputs per arm and the frozen seeds. All three focused
fixtures, eight compiled-recorder Python tests (with five native child captures),
and before/after README MTP default/Q8 and 40-test Python floors passed.
Explicit SYCL0 runs exercised default attention modes 0/1 and Q8 attention
mode 1 on the device. An additional verifier assertion initially required Q8
mode 0; source explicitly excludes that
combination. The corrected audit and original assertion failure are retained;
no payload was retried and no numerical threshold changed.

The independent audit after process cleanup confirmed unchanged hashes for all
27 recorded source paths and recorded binaries, unchanged HEAD/index/patches,
and 59 unchanged wrapper dependencies. Native mappings passed; no render-node
holders, llama processes or builders remained, and the whole-sequence fault
delta was empty. The service vector remained inactive/inactive/failed.
The normative sampler runs host logic in the AOT build; explicit SYCL MTP
fixtures supply the GPU evidence. The ordinary README MTP fixtures remain CPU
tests. GDB observation of short focused/recorder fixtures makes no timing claim.
These synthetic ending floors precede the
trained campaign and cannot replace its separate post-real floor.

## Requirement-linked evidence

| Requirement | Existing evidence | Additional qualification or boundary |
| --- | --- | --- |
| R07.1 | Pure helper and CPU corpus evaluate every eligible position. | Normal AOT execution and evidence audit passed. |
| R07.2 | Exact capped recurrence fixtures, including tiny positive mass. | Synthetic fixtures do not establish all trained probabilities. |
| R07.3 | Nonmonotonic survival fixture continues after earlier failure. | Normal AOT execution and evidence audit passed. |
| R07.4 | Exact pass/fail enumeration and 256-contribution caller fixtures check expectations and draft-only denominator. | Three trained 256-block records were observed for one acceptor; retained as raw observations of the failed campaign. |
| R07.5 | Residual target joint-law and exact 2^-54 regression fixture pass. | Full trained-output distribution equivalence is not proven. |
| R07.6 | Inherited P06 fallback/transaction corpus passes under modes 0/1/2 in CPU and normal AOT builds. | Source and synthetic evidence cover constraints, stateful/backend/chained and invalid/missing q; trained coverage is separate. |
| R07.7 | Unset/0 same-seed future decisions and disabled behavior pass. | Candidate excluded; final accepted-tree floor passed separately. |
| R07.8 | Frozen 437-comparison/1311-hypothesis family and exact short joint law pass in CPU and normal AOT builds. | Trained mode 1/2 identity gate failed; this does not itself prove distribution bias or invalidate the scoped synthetic pass. |
| R07.9 | Greatest-passing-prefix checks include all-pass/all-fail/nonmonotonic cases. | Normal AOT execution and evidence audit passed. |
| R07.10 | Strict accepted/rejected mode parsing and startup fixtures pass; empty is directly tested through parser. | Candidate not integrated; no integrated P07 parser claim. |
| R07.11 | Transaction/checkpoint/RNG/EOG cases pass in all modes and executable consumers. | Forced synthetic repair is not trained normal-RS/FULL equivalence. |
| R07.12 | Identity/range and caller fixtures pass, including absent log/profile environment. | Trained mode 2 emitted disjoint ranges 1..256, 257..512 and 513..768 for acceptor 2; enclosing failure invalidated all arm success flags. |
| R07.13 | Single contribution after commit/consumption; replay/duplicate/failure exclusions pass. | Trained counts and three complete buckets are retained and independently reviewed; mandatory mode identity still fails. |
| R07.14 | Reset between two groups of 128 blocks emits once; same-slot replacement, identical/different seeds, sequences/clones remain separate. | Trained records belong to one acceptor, but mandatory mode identity failed; separate acceptors were not pooled to create a pass. |

## Harness and orchestration checks

The [harness retry](/home/svnbjrn/wt/kmic68-evidence/p07-harness-retry/report.md) records 40 benchmark
Python tests, 10 block unittests, 29 inherited distribution tests, and 10 block
pytest cases with 46 subtests passing.
Pytest repeats cases rather than adding an independent population. Strict
response validation now requires exactly 4096 returned tokens, matching
`tokens_predicted`, and `stop_type=limit`; seven negative cases cover premature
completion. The preregistered prompt/payload remained unchanged.

The [runtime mock report](/home/svnbjrn/wt/kmic68-evidence/p07-runtime-mock-verification/report.md) records
11 pure unittest methods passing, with 35 prepared/dependency files unchanged.
They reject malformed coverage/seeds/p-values, missing or foreign modeled
mappings, outer timeout/cancellation/exception results and final identity
failure, withdrawing nested success while preserving original report bytes.
No native process, GDB, model or hardware probe ran in that mock pass. Native
provenance, route and cleanup were subsequently qualified by the separate
eleven-gate report. The later trained execution completed but failed its identity
gate, as recorded below.

The real wrapper's source review corrected a post-worker cancellation gap.
Deferred SIGINT/SIGTERM handlers cover resource, source and model postflight;
root separately checks actual launcher wait status because saved success cannot
be atomic with every later signal. Nonzero or uncertain exit withdraws nested
success and preserves original report bytes. The expected model manifest is
never substituted for observed hashes after interrupted persistence.
[Real wrapper mocks](/home/svnbjrn/wt/kmic68-evidence/p07-real-mock-verification/report.md)
passed ten methods. After mechanical retry-evidence path changes, all eleven
synthetic and ten real methods passed again, with 59 unchanged dependency files:
[mock refresh](/home/svnbjrn/wt/kmic68-evidence/p07-mock-refresh/report.md).
These are mock-only qualification, not live process/device/model proof.

## Failed AOT attempt and successful unchanged retry

The [first build](/home/svnbjrn/wt/kmic68-evidence/p07-sycl-build/report.md) configured and audited successfully:
IntelLLVM 2026.1.1, Release production NDEBUG/test UNDEBUG, normal consumers OFF,
both UI flags OFF, CCACHE OFF, acm-g10/F16, effective DNNL=1, four host/device jobs.
It returned 1 after **848.178268 seconds**, with device generation exit 254 and
`llvm-foreach: Segmentation fault (core dumped)`. It produced no qualifying final
binary identity manifest and launched no native/GPU/model gate. Its 27-path
source, HEAD, index and patch preservation passed after failure.

The [stack diagnosis](/home/svnbjrn/wt/kmic68-evidence/p07-aot-failure-diagnosis/report.md) identifies the
crashed executable as `ocloc`, SIGSEGV inside IGC TranslateImpl. All 229 compared
backend/header paths and normalized build settings/link commands matched P06;
recorded compiler-library paths match, but historical library content hashes
were not available. Temporary SPIR-V/output files were gone and interleaved
warnings cannot identify the failing kernel. No matching kernel-journal OOM or
GPU failure was observed; that does not exclude an unlogged host fault.

One unchanged incremental retry used the same build directory, four host/device
jobs and 7200-second bound, retaining attempt 1 in place and writing separate
`p07-sycl-build-retry` evidence. The retry passed in **549.125624 seconds**, with
exit status 0 and all twenty remaining build steps complete. Preservation checks
for all 27 source paths, HEAD, index and patches passed, as did current binary
identity and final resource checks. The generated UI remained empty;
the checked actual-template fixture accepts zero assets and rejects one asset.
The [retry report](/home/svnbjrn/wt/kmic68-evidence/p07-sycl-build-retry/report.md)
retains these results separately. The retry neither repairs the crash nor
establishes its cause. The preserved core supports investigation, not runtime
acceptance.

## Frozen real-model protocol and completed failure

The [measurement handoff](/home/svnbjrn/wt/kmic68-evidence/p07-measurement-prepared/handoff.md) requires exact
local Qwen4Exp target shards and matching MTP head already qualified for P04/P06.
No qwen35 substitute is permitted. One fixed raw request runs in each mode
0/1/2: seed 123, temperature 0.8, stateless temperature-only target chain,
4096 output budget and returned token IDs, without an ignore-EOS override.
Canonical payload SHA256:
`efbf5a75ab0b2acac2b73b452d6294c7fffd67d658e47923792b0446cf7e3d01`.

Every arm must return exactly 4096 tokens with `stop_type=limit`. Early EOG or
insufficient eligible commits fails the frozen protocol; no prompt/seed/budget
change, retry selection or cross-request aggregation is permitted after results.
At least 256 observed eligible committed blocks must belong to one acceptor;
a 4096-token budget alone does not prove that count.

Mode 0 means P06 enabled per-token distribution acceptance. All arms use host
non-chained MTP, draft temperature 0.8, top-p 1, top-k 10, widths 1..7, context
16384, one slot, q8_0 K/V, FA on, mmap and automatic fit target 1024 on SYCL0. Modes 0/1
must emit no `spec_block` line. Mode 2 requires complete 256-block buckets joined
to ordered eligible cycles with positive identity and disjoint proposal ranges;
partial buckets cannot be emitted or combined. Signed gain must be reported
regardless of sign, with actual drafted/accepted counts and matching denominator.

Modes 1/2 must have identical output tokens and normalized proposal-relative
sampling/acceptance decisions, q supports, settings and positions. Numeric
cross-process identities and timings may differ, but original identities remain
validated. Mode 0 token identity is not required. One trained sequence is not a
replacement for the frozen statistical corpus or a throughput comparison.

The outer campaign requires repository locking, bounded execution, owned-process
cleanup, readable before/after fault logs, sole tenancy, preservation of all 27
recorded source paths and binary identities, and actual loaded-image/device
provenance. Model hashes must be
observed and validated; a predeclared expected hash is not observed proof. If
interruption prevents the final model-hash report from being persisted, partial
logs remain unqualified. Floors at the end of synthetic native verification are
only pre-real evidence. In this execution the three planned post-real README
floors were not run because the wrapper stopped at the first failed gate.

The completed [campaign report](/home/svnbjrn/wt/kmic68-evidence/p07-real-campaign/campaign/report.json)
records `ok=false`, `worker_exit=1`, and error
`mode 1/2 output tokens differ`. The
[measurement summary](/home/svnbjrn/wt/kmic68-evidence/p07-real-campaign/measurement-summary.json)
records actual outer exit 1 and the first mismatch at output index 1, with
mode 1 token 103196 versus mode 2 token 248068. No inference retry ran and no
prompt, seed, budget or threshold was changed after the failure.
The [measurement publication](/home/svnbjrn/wt/kmic68-evidence/p07-real-campaign/report.md)
records actual outer wait status, retained raw worker bytes and the runtime
provenance separately from the failed plan gate.

| Mode | Completed requests | Output tokens / stop | Selected distribution cycles | Drafted | Accepted |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 1 | 4096 / limit | 868 | 6076 | 3227 |
| 1 | 1 | 4096 / limit | 1119 | 7830 | 2976 |
| 2 | 1 | 4096 / limit | 870 | 6086 | 3225 |

Modes 0/1 emitted no block diagnostic. Mode 2 emitted the following raw
observations for launch 1791708829225225511, sequence 0, acceptor 2:

| Proposal range | Blocks | Expected block draft tokens | Expected per-token draft tokens | Signed difference | Drafted-token denominator |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1..256 | 256 | 829.925880740381 | 792.8241894117941 | 37.10169132858687 | 1792 |
| 257..512 | 256 | 754.4003813357016 | 716.9087655849611 | 37.49161575074049 | 1792 |
| 513..768 | 256 | 1111.8569777261282 | 1089.1742876377705 | 22.68269008835773 | 1792 |

These complete arm observations are not accepted partial campaigns. The outer
failure deliberately set every nested arm `ok` flag false while preserving the
original worker-report bytes. Positive conditional gains do not waive token
identity, establish universal target-distribution correctness or authorize a
default change. Cross-mode proposal/distribution equality was not reached after
the output-token mismatch; it is not reported as passing. Differing trajectories
also bar a controlled timing or causal interpretation of incidental raw timings.

The [ownership-release record](/home/svnbjrn/wt/kmic68-evidence/p07-real-campaign/ownership-release.json)
has `released=true`, empty resource errors and final GPU-fault lists, preserved
source/binary identity, valid native mappings, observed full model hashes and
preserved identities for all three model files. Before/after and outer identity
records agree. All three native servers shut down cleanly after SIGINT without
escalation. No observed test PID survived, and services remained
inactive/inactive/failed. It explicitly records `campaign_pass=false` and
`all_four_evidence_audits_pass=false`: cleanup and preservation do not turn the
failed campaign into a pass. Final accepted-tree regression passed separately.

The [independent disposition review](/home/svnbjrn/wt/kmic68-evidence/p07-gate-disposition-review.md)
confirmed rejection and found no actionable evidence-handling flaw. It checked
the finalized raw outputs, mandatory gate, actual launcher exit, withdrawn nested
success flags, preserved identities and resource release. Different initial
proposals do not identify a diagnostic RNG error, nondeterminism, backend defect
or Algorithm 2 failure. The coordinator revoked the runtime lease; no retry is
part of this disposition.

## Interface and integration boundary

The [interface source review](/home/svnbjrn/wt/kmic68-evidence/p07-interface-source-review.md) found no P07
diff in public include/ggml headers, `src/llama-ext.h` or backend interface tables
relative to accepted P06. Public by-value `llama_batch`/context parameters and
backend vtables are unchanged. Old common C++ declarations and out-of-line
forwarding definitions remain, with new result/diagnostic overloads additive.

However, `common_speculative_cycle_record` inserts `int block_verify_mode`
before `std::string sampling_mode`. **Consumers accessing, constructing or copying
this exposed common record must rebuild with matching headers and llama-common.**
Do not mix P06-compiled record clients with P07 objects. Retained old signatures
are not proof of C++ binary-layout compatibility. The [compatibility report](/home/svnbjrn/wt/kmic68-evidence/apiabi-p07/report.md)
records 100% public-header source compatibility with no reported problems or
warnings in the source report. Three explicit abidiff calls returned 0 for
libllama, libggml and libggml-base. The unsuppressed llama-common audit found
zero removed and fifteen added exports; its SONAME remains libllama-common.so.0.
These are symbol/source checks, not complete typed ABI proof. Release lacks
complete debug types, the default checker suppresses C++ symbols, and the header
parser separately warned about GCC 16 support. Source inspection independently
confirms unchanged public C layouts. P05 was rejected, P10 is blocked, and excluded P09 interfaces
are not part of this candidate. No corresponding combination is claimed.

## Disposition and campaign follow-up

| Field | Completed evidence or remaining observation |
| --- | --- |
| AOT retry | Complete: exit 0 in 549.125624 seconds; 27 source paths, HEAD/index/patches, binary identities and resource release qualified in the retry report. |
| Compatibility | Complete within stated scope: public C/source-symbol and unsuppressed common C++ checks. Common-record rebuild requirement and full typed-ABI limitation remain. |
| Native gates | Complete: all eleven payloads/audits, fresh 437-comparison/1311-hypothesis corpus under the unchanged 240-second bound, actual mappings/device routes and ownership release. |
| Trained arms | Completed observations: 4096 limit-stopped outputs each; the enclosing identity failure invalidates every arm success flag. |
| Mode 2 diagnostics | Three raw complete buckets for acceptor 2, with operands and denominators above; no partial-campaign acceptance. |
| Mode 1/2 equivalence | Failed at output index 1; subsequent proposal/distribution equality check was not reached. No cause established. |
| Trained fault/resource/floor | Cleanup, mappings and source/binary/model preservation recorded; fault/resource error lists empty. All three post-real floor commands unrun after the first failed gate. |
| Review/integration | Independent review confirms P07 rejection and code exclusion. Outcome commit `eae53a37e` contains no candidate code. Final accepted-tree regression, combined review and service restoration are recorded in the campaign ledger. |

## Not claimed

No P07 acceptance, passing trained campaign, integrated feature,
universal distribution equivalence, trained replay/EOG/repair equivalence,
throughput improvement, default promotion, full typed ABI compatibility or
compiler repair. All reported test/build figures are observations in the linked retained records;
source review is distinguished from runtime verification.
