# Kmic68 twelve-plan campaign outcome ledger

Evidence date: 2026-10-11. **Campaign complete: all twelve plans are accounted
for, accepted changes are integrated, final verification passed, and the original
active service was restored.** Four plans are accepted, four rejected by their
gates, and four blocked by explicit prerequisites. Rejected and blocked candidate
code is excluded.

The verified source revision is `eae53a37e6f0cbe526da83dc7daa5bbcbaf59196` on
`codex/kmic68-integration`, starting from `2f623c6607c078d1069dac23e9d8af9cb146e771`.
The closing publication commit changes documentation only; its actual HEAD and
clean-state result are retained in the [closure record](/home/svnbjrn/wt/kmic68-evidence/final-closure.json).
Accepted feature commits and outcome-only documentation commits are distinguished
below. P07's trained failure is preserved without overriding it with synthetic
passes or selecting another seed, prompt or attempt.

The named publications contain requirement tables, commands and the limits of
each result. Build, test and measurement statements below derive from those
reports and their retained raw artifacts; source reviews remain distinct from
runtime evidence. Repository links are relative to this report. Absolute paths
identify retained host-local evidence and are not portable repository fixtures.

## Plan dispositions

| Plan | Outcome | Requirement-linked reason and retained evidence | Integration commits |
| --- | --- | --- | --- |
| P01: benchmark token count | Accepted | R01.1-R01.7: examples use 512 tokens; old 128-token observations retain their values and are marked superseded in measurement contract. One same-binary depth-16384 calibration used three repetitions per arm. The +0.107398% unequal-token comparison is calibration-only, not a performance gain. [P01/P02 report](kmic68-a770-rebaseline.md). | `5fdc96f62` (documentation and calibration acceptance) |
| P02: minimum free memory | Accepted | R02.1-R02.11: process-lifetime telemetry, terminal records, status-bearing query through procedure lookup, disabled behavior and failure handling. Real 8%-fill/full-fill records and same-process unload/reload observations are qualified by deterministic retention tests and source review. Sampled minima do not bound unsampled peaks; live exact Level Zero selection was not exercised. [P01/P02 report](kmic68-a770-rebaseline.md). | `189cbc499` (feature), `440da495d` (qualification) |
| P03: FA memory-path proxy | Rejected by readiness gate | R03.8 generated retained-path semantic fingerprint and runtime image binding remain incomplete. Offline device sections exist, but counts are not semantic equivalence; extraction also changed two parent containers. No paired proxy timing or measured memory floor exists. [P03/P11 report](kmic68-a770-fa-memory-floor.md). | `d14ec8009` (outcome only) |
| P04: proposal lifecycle/cycle log | Accepted | R04.1-R04.12: immutable identity, attempted/finalized accounting, shared phases, replay/cancellation and disabled instrumentation. Six exact-model responses reconcile 391 starts/completions and 2694 attempted tokens; profile attribution is not a speedup. [P04 report](kmic68-speculative-cycle-evidence.md). | `57ba6d4ac` (feature), `1e7f42a8c` (evidence), `e768c2e7c` (excluded-download disclosure) |
| P05: top-k prefilter | Rejected by real-model gate | R05.1-R05.12 scoped guards/counters/fixtures passed, including grammar-first fallback and positive optimized calls. All 36 requests completed; all 18 paired OFF/ON native-token sequences differed. That mandatory identity failure rejects the candidate regardless of unit passes. It does not prove distribution bias or identify the cause. [P05 report](kmic68-a770-sampler-prefilter.md). | `d7671c8e0` (outcome only) |
| P06: distribution acceptance | Accepted | R06.1-R06.16: single-use proposal-bound distributions, transactional acceptance, fallback, replay/repair and consumers. The normative synthetic family passed 77 cases/231 Pearson-Holm hypotheses; 19 executable consumer cases passed. The trained enabled arm recorded 287 proposals, 1984 normalized q records and 369 sampled tokens differing from top-1. These do not establish universal trained-output equivalence. [P06 report](speculative/kmic68-distribution-acceptance-2026-10-11.md). | `09b203c47` (feature; local candidate `c303c6c58`), `0cc012ca1` (integration evidence) |
| P07: block verification | Rejected by mandatory mode-identity gate; code excluded | R07.1-R07.14 have scoped source/CPU and normal AOT evidence, including 437 comparisons/1311 hypotheses and 60 separate CPU consumer cases. All three trained arms completed 4096 limit-stopped tokens, but modes 1/2 first differ at output index 1: 103196 versus 248068. Actual outer exit was 1; every nested arm success flag was invalidated. Three raw 256-block diagnostics do not override this failure. Cause unproven; post-real floors unrun. [P07 report](kmic68-a770-block-verify.md), [measurement summary](/home/svnbjrn/wt/kmic68-evidence/p07-real-campaign/measurement-summary.json), [plan](../plans/kmic68-07-block-verify.md). | `eae53a37e` (outcome only; no feature commit) |
| P08: draft microbatch cap | Blocked at acceptance | R08.1-R08.12 have generated/CPU evidence, but mandatory compatible trained EAGLE3/DFlash/DSpark target-draft pairs remain unresolved. Generated accepted counts were zero. Production MTP cap crossing, cap comparisons, joint placement/buffer savings and non-regression gates also remain incomplete. [P08/P12 report](kmic68-draft-capacity-blockers.md). | `d571adc65` (outcome only) |
| P09: padded verification | Rejected by parity gate | Dense Qwen3 self-drafting passed output/KV isolation and reduced graph rebuilds from 16 to 12, with actual dummy rows. R09.12 recurrent fallback and mandatory token/drafting parity did not qualify; a repeated OFF run also varied. An independent initialization reservation/fit side effect remains unfixed in the excluded candidate. Earlier R09 scoped passes remain valid; recurrent oversized coverage was not completed. [P09 report](kmic68-a770-padded-verify.md). | `459d95cf5` (outcome only) |
| P10: cumulative draft width | Blocked by calibration protocol | R10.4/R10.14 require an observed committed-position-zero knot. An ordinary request begins drafting at actual prompt length, so the required zero-position cycle is unreachable; position 1 cannot retain the three prescribed prompts. No protocol amendment was selected. Offline replay supports portions of R10.1-R10.3/R10.5-R10.8 and sampled/top-1 accounting, not runtime acceptance or calibration. [P10 report](kmic68-a770-pcum-ramp.md). | `6f9a4c67d` (outcome only) |
| P11: fp32 VEC tile fold | Blocked before implementation | P03 must first supply qualifying paired evidence with P11 eligibility true. It did not. R11.1-R11.7 and the complete correctness/performance matrix remain unexecuted; no P11 kernel change was made. [P03/P11 report](kmic68-a770-fa-memory-floor.md). | `d14ec8009` (shared outcome only) |
| P12: prefix mask | Blocked before implementation | P08's accepted effective-cap/reservation interface is a hard prerequisite and is unavailable. R12.1-R12.14, reservation-failure-before-cache-mutation and combined padding/prefix tests remain unexecuted. P03/P11 is not a P12 prerequisite. [P08/P12 report](kmic68-draft-capacity-blockers.md). | `d571adc65` (shared outcome only) |

## Model identities and unresolved fixtures

The original exact-Qwen4Exp blocker was resolved. The local two-shard target under
`/mnt/ssd1/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/IQ1_M/` and canonical Q8_0
MTP head under `/mnt/ssd2/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/mtp/`
both declare `qwen4exp`; P04 and P06 exercised the trained route. Their identities
and commands are in the linked P04/P06 reports. Available Ornith/qwen35 models
were not counted as exact Qwen4Exp substitutes.

The ordinary-transformer fixture was local `/mnt/mrgr/models/Qwen3-1.7B-Q8_0.gguf`;
P09 used it as both target and draft-simple drafter. Recurrent fallback is
separate coverage and does not count as successful padding. The supplied Mistral
path under `/mnt/mrgr/download-backup/mistral-7b-v0.1/` is acknowledged in the
P01 report; no new same-model Mistral depth sweep was measured.

P08 still needs compatible trained EAGLE3, DFlash and DSpark pairs with matching
architecture metadata, dimensions, vocabulary and required tensors/settings.
DSpark uses the DFlash architecture with its required Markov weights; an
ordinary model or generated fixture cannot substitute. Bounded search results
are not proof that compatible files are absent everywhere.

## Combination and compatibility boundaries

| Requested combination | Evidence/outcome |
| --- | --- |
| P05 with P06/P07 | Excluded: P05 failed its gate and its sampler changes were not integrated. No combined coverage claim. |
| P06 with P07 | Candidate CPU statistical, transaction, fallback, replay/RNG/EOG, consumer and eleven normal AOT/native gates passed. The normative sampler executes host logic; explicit SYCL MTP fixtures establish GPU execution. The mandatory trained mode-identity gate failed, so no integrated P06/P07 capability is claimed. |
| P06 with P10 | Unrun: no accepted P10 runtime change or calibrated artifact. |
| P08 with P12 | Unrun: P08 unaccepted and P12 not implemented. |
| P09 with P12 | Unrun: P09 excluded and P12 not implemented. |

P02's local API/ABI tools were provisioned; the original missing-PATH-tools gate
is no longer the whole explanation. Public C checks reported no removed/changed
symbols and an additive P02 memory-status symbol. Public by-value structures
and backend vtables were separately reviewed. Release objects lack complete
type debug information, so these checks do not prove the full typed ABI.

The accepted P04/P06 common C++ interfaces require consumers to rebuild.
P06's unsuppressed audit found five removed and 39 added exports and changed
common structure layouts while the common library SONAME stayed at 0.
**Rebuild common C++ consumers against the accepted tree's matching headers and libraries.** Public C
source/symbol compatibility must not be described as drop-in common C++ binary
compatibility. [P06 compatibility qualification](speculative/kmic68-distribution-acceptance-2026-10-11.md#build-and-compatibility-update).

The completed [P07 compatibility check](/home/svnbjrn/wt/kmic68-evidence/apiabi-p07/report.md)
applies only to the excluded P07 candidate. Against accepted P06 it reports
100% public-header source compatibility and no
changes in three explicit abidiff comparisons. Its unsuppressed common C++
audit found zero removed and fifteen added exports. That candidate changes
the exposed common cycle-record layout and would require matching rebuilt
consumers; this additional P07 layout change is not integrated. Release lacks
complete debug types, the public
checker suppresses C++ symbols, and the header parser warned about GCC 16
support. The final accepted-tree comparison below is separate and excludes P07.

## Evidence preservation and execution limits

Raw evidence remains under `/home/svnbjrn/wt/kmic68-evidence/`, including commands,
source/binary identities, actual mappings where captured, model identities,
before/after fault logs, requirement audits and failed attempts. Candidate source
archives remain in `p03-rejected-source`, `p05-rejected-source`,
`p08-blocked-source`, `p09-unaccepted-source` and `p10-blocked-source`; the linked
publications identify their manifests and qualifications. P07's 15 frozen
product/harness files are preserved in
[candidate.tar.gz](/home/svnbjrn/wt/kmic68-evidence/p07-rejected-source/candidate.tar.gz)
under `p07-rejected-source`, 154552 bytes, SHA256
`b4b7e8da2ecd1b2833b8020f3ea18af5ec9a889153915e8fde9b2ea949dedb88`.
The coordinator reports independent archive-member rehashing; manifests,
HEAD/status and candidate patch remain beside it. Do not delete failed evidence or pool
partial/rejected attempts into a successful campaign.

Historical i915 observations remain historical. The actual current campaigns
identify the A770's xe driver and their own compiler, runtime, device and binary
provenance. P01's bounded calibration does not replace every old depth row;
P10's historical 391-cycle replay is not final-binary calibration. P09's 25%
graph-rebuild-count reduction is not a throughput measurement.

Early P04 preparation downloaded prebuilt UI assets and Stories/BERT fixtures,
contrary to the local-only instruction. The incomplete server run was excluded,
and its inventory and removal are documented. Earlier P06 CPU attempts also
downloaded UI assets because the separate prebuilt-UI option remained enabled;
their functional observations are retained but are not compliant offline-build
evidence. Later accepted preparations disabled both UI options. **The campaign
must not be described as having performed zero downloads.** Neither exception
licenses further downloads. The P07 Stories downloader suite was not run and
its 60 synthetic consumers must not be labeled a pass of that unavailable suite.

The supporting integration commits are `1a13eef9a` (plan contracts), `ca41c077e`
(Release test assertions), `bf38e17bb` (ABI tool version parser), `20ec20622`
(P06 repair semantics clarification) and `456db50ba` (atomic test PID publication).
Accepted changes remain local on the integration branch; no push or PR is part
of this campaign. Candidate local commit IDs and tested source manifests can
differ from integrated commit IDs, so HEAD alone is not a build identity.

## Final accepted-tree verification and closure

The [combined source review](/home/svnbjrn/wt/kmic68-evidence/final-combined-source-review.md)
is CLEAR at `eae53a37e6f0cbe526da83dc7daa5bbcbaf59196` against the campaign base.
The last commit adds only the reviewed P07 rejection report and index entry.
The reviewer confirmed that unaccepted candidate controls remain absent and
carried forward the P02/P04/P06 source findings and compatibility limits.
It read the coordinator's 1214-file match manifest without independently
rehashing it. Source review is not a final build, runtime or ABI result.

P07's three post-real floor commands were unrun after the mandatory identity
failure. No inference retry or protocol change followed that result, and its
code remains excluded. The final checks below apply to the accepted integration
tree; successful checks cannot retroactively qualify a rejected candidate.

| Final gate | Result | Retained evidence |
| --- | --- | --- |
| Combined source review | CLEAR at verified source revision | [Combined review](/home/svnbjrn/wt/kmic68-evidence/final-combined-source-review.md); prior combined P02/P04/P06 review plus outcome-only P07 closure. |
| Fresh CPU build and regression | PASS | [CPU report](/home/svnbjrn/wt/kmic68-evidence/final-integration-verification/report.md): 234 build steps, 113.22 seconds, four jobs; focused 4; 77 comparisons/154 arms/231 hypotheses/N100000 in 49.72 seconds under 180 seconds; Python 40/8/29; README default/Q8 and Python floor before/after; zero skipped coverage. Actual launcher exit 0. |
| Final API and symbol qualification | PASS within stated scope | [API report](/home/svnbjrn/wt/kmic68-evidence/apiabi-final/report.md): public-header source compatibility 100%; explicit abidiff exits 0/0/4, only additive P02 C symbol; no removed/changed public C exports. Full typed ABI remains unproven; common C++ clients require rebuilding. |
| Final affected SYCL regression | PASS | [SYCL report](/home/svnbjrn/wt/kmic68-evidence/final-sycl-verification/report.md): all five gates, actual outer exit 0, README default/Q8/Python40 plus explicit SYCL0 default/Q8 routes; same-PID maps, exact source/binary bindings, clean faults and owned-process release. |
| Service restoration | Completed and rechecked | [Restoration report](/home/svnbjrn/wt/kmic68-evidence/service-restoration/report.md): Ornith active/running; named SYCL and Vulkan units inactive; AEON remains failed. Only Ornith was started. |
| Final documentation and branch closure | Outcome ledger and indexes published | [Closure record](/home/svnbjrn/wt/kmic68-evidence/final-closure.json) records the final documentation commit, exact tested-source correspondence, clean state and twelve-plan reconciliation. No push or PR. |

The fresh CPU build disables both UI options and consumer injection, retains
production Release/NDEBUG with test assertions, and invokes no trained model or
downloader. Final verification independently preserved all 1214 selected source
files and all fresh binaries. Nineteen prior synthetic consumer cases carry
through exact caller/fixture/generator identity; they were not rerun or labeled
trained-model coverage.

Final AOT checks reused the identified P06 build only after the root verified
its 29 artifact paths/15 resolved hashes, compiled source and build configuration
graph. The verifier independently checked those bindings before/after. Historical
build metadata remains historical; these were not compiled at the final Git HEAD.
The complete P06 AOT statistical/focused gates carry through unchanged identities;
fresh CPU verification reran their relevant corpus and focused checks.

The final common audit compares a source-qualified pre-P04/P02-enabled Intel/SYCL
artifact with final GCC/CPU binaries. It found 826 removed and 753 added common
exports; all removed entries are weak functions/objects. Compiler-generated and
configuration differences prevent attributing every export delta to this campaign.
The separate source-layout and incremental same-toolchain audits establish the
common-client rebuild requirement. Neither unchanged SONAME nor a compatible
public C check proves common C++ binary compatibility. The source parser's GCC16
warning and incomplete Release type information remain explicit limits.

Service attribution correction: earlier notes described Vulkan as failed from
`systemctl is-active` output containing a wildcard. That output did not identify
units positionally. Explicit named checks found the failed unit to be
`llama-gpu@AEON-NVFP4-MIXED.service`; Vulkan is inactive. The first restoration
precheck stopped before mutation, and its failed expectation is preserved.
The corrected attempt restored only `llama-gpu@Ornith-Uncensored-Q8-Test.service`
to active/running; later rechecks confirm all four named states and no new faults.
No HTTP health or inference-quality claim follows from service-state restoration.

## Not claimed

No universal distribution proof, full typed ABI compatibility, integrated
rejected feature or global speedup is established. The four blocked plans retain
their exact prerequisites; synthetic passes do not replace trained fixtures or
an executable calibration protocol. Missing Stories fixtures, blocked model pairs and the unamended calibration
protocol remain explicit unavailable gates; they were not replaced with synthetic
passes. The final checks qualify only the accepted source tree.
