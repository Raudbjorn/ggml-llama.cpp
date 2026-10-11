# Kmic68 proof follow-up: results and evidence index

The follow-up restores two identified C++ ABI breaks, repairs two distribution
implementation defects, establishes the ideal rejection-sampling theorem under
explicit assumptions, and implements the frozen stochastic benchmark harness.
**Exhaustive C++ ABI compatibility, universal finite-precision runtime equality,
and the overall speedup claim remain incomplete.** No positive performance
claim or blanket ABI pass follows from the bounded successful tests.

The pristine comparison base is
`2f623c6607c078d1069dac23e9d8af9cb146e771`. This follow-up starts at
`5acb895208d584c81779262fb5ec478e00d06035`, after the
[twelve-plan campaign](kmic68-campaign-outcomes-2026-10-11.md). That campaign's
four accepted, four rejected and four blocked dispositions remain unchanged.
No rejected kernel or blocked dependent feature was reintroduced.

## Claim dispositions

| Claim | Result | Established evidence and exact remaining boundary |
| --- | --- | --- |
| Backward C++ ABI on qualified GCC/CPU and oneAPI/SYCL profiles | Incomplete | Original 48-byte draft record restored; required ngram-vector destructor providers retained; pristine-header clients run unchanged with verified replacement mappings on both profiles. Exhaustive independent header-root/type association is missing; CPU actual-library ledger and separate SYCL tool results retain unresolved items. [ABI report](kmic68-cpp-abi-proof-2026-10-11.md). |
| Ideal distribution preservation | Established under stated assumptions | One-token mass identity extends to finite history-dependent sequences, bonus tokens and predictable stopping with valid normalized rows, exact arithmetic, correct histories and conditionally independent randomness. Exact rational checks cover 5,778 pairs plus bounded sequence/adversarial cases. [Theorem and implementation audit](speculative/kmic68-distribution-proof-2026-10-11.md). |
| Universal finite-precision production distribution equality | Incomplete | Canonical recorded-q draws and pre-draw minimum-length exclusion repair concrete defects. The 77-case/231-hypothesis family and 19 executable consumer cases pass. Finite-PRNG assumptions, proposal-dependent invalid suffixes, target-law rounding and trained batch/sequential numerical differences prevent an unconditional claim. The repaired exact-Qwen4Exp frozen runtime profile crashes before route qualification; separate cache-disabled off/on diagnostics pass bounded route checks. |
| Overall workload speedup above 2%, excluding cell regressions above 3% | Incomplete | Protocol, held-out prompts, seeds and statistical gates are frozen; harness and model hashes are available. The pristine downloads-disabled server cannot link because its cache-resolver stub is missing. There are zero qualifying paired performance samples. No optimization round was triggered. [Speed report](kmic68-speed-proof-2026-10-11.md). |

The historical unrestricted minimum-length distribution claim is disproved by
an exact `(5/8,3/8)` counterexample against target `(1/2,1/2)`; the repair excludes
that configuration before drawing. A separate invalid-future-suffix construction
remains outside the valid-row theorem. These mathematical counterexamples are
not represented as measured trained-model output frequencies.

## Integrated changes

All commits are local on `codex/kmic68-integration`, independently revertible,
and carry `Assisted-by: OpenAI Codex`. No push or pull request was created.

| Commit | Logical change |
| --- | --- |
| `384e1dc7a` | Complete reviewed C++ dump-coverage mode and fail-closed ABI checker. |
| `b512876f5` | Frozen stochastic comparison harness and paired simultaneous inference. |
| `eef488983` | Restore base draft-parameter ABI; opaque sequence-owned proposal metadata; update consumers. |
| `1a89077e5` | Draw proposals from recorded q with private RNG; reject unsafe minimum-length selection before draws; retain complete statistical counts. |
| `a7830e9a8` | Arm callback injection only for its target phase; flush asynchronous fixture logs; register release-only invalid-row coverage. |
| `ab4c1b5d4` | Complete candidate disabled-download resolver stub. Pristine baseline remains untouched. |
| `1044c50b3` | Restore exported ngram-vector destructor providers and base-built client references. |
| `56b69b831` | Exact rational oracle, masked numerical probe and CTest registration. |
| `502ba7eae` | Restrict ordinary-vector destructor alias probes to non-debug containers. |
| `c1de964f7` | Validate qualified effective suppression masks and explicitly evidenced nonserialized reasoning settings. |

No public by-value batch/context fields or backend vtable entries are appended.
Defaults, grammar/stateful/backend/chained fallback guards and the legacy
parameter reference/result_q output are preserved. The proposal token remains
separate from confidence's top-1 probability.

## Verification and provenance

The [machine-readable requirement ledger](/home/svnbjrn/wt/kmic68-proof-evidence/requirement-ledger.json)
indexes ABI-01..07, DIST-01..08 and SPEED-01..10.

Evidence root: `/home/svnbjrn/wt/kmic68-proof-evidence` (E). Historical evidence
under `/home/svnbjrn/wt/kmic68-evidence` is preserved. Absolute links below are
host-local artifacts, not portable repository fixtures.

| Evidence | Result and scope |
| --- | --- |
| [Approved plan](/home/svnbjrn/wt/kmic68-proof-evidence/approved-plan.md) | Exact plan retained in E; pristine comparison, required models and thresholds unchanged. |
| [Fresh integration CPU report](/home/svnbjrn/wt/kmic68-proof-evidence/integration-cpu/report.md) | Optimized production build; MTP default/q8/release, 77 statistical cases/231 Holm hypotheses, cycle/adaptive/client, registered oracle/probe and 77 Python tests pass. Full counts retained; zero Python skips. |
| [Final tooling tests](/home/svnbjrn/wt/kmic68-proof-evidence/final-python-tooling-after-response-repair.log) | 84 ABI-checker, stochastic/deterministic harness and exact-oracle tests pass; no model performance inferred. |
| [Fresh consumer report](/home/svnbjrn/wt/kmic68-proof-evidence/integration-consumer-cpu/report.md) | All 19 executable cases pass in a separate build with synthetic hooks; no ABI/performance claim from those binaries. |
| [Matched pristine SYCL build](/home/svnbjrn/wt/kmic68-proof-evidence/baseline-sycl-hostdwarf/report.md) and [candidate build](/home/svnbjrn/wt/kmic68-proof-evidence/candidate-sycl-hostdwarf/report.md) | Libraries build successfully with full host DWARF and device debug disabled; 455 common translation-unit command profiles match. This is compilation, not GPU execution. |
| [Final GCC client](/home/svnbjrn/wt/kmic68-proof-evidence/final-client-cpu/report.md) and [oneAPI client](/home/svnbjrn/wt/kmic68-proof-evidence/final-client-sycl/report.md) | Fresh pristine-header binaries pass baseline and unchanged replacement execution, plus rebuilt candidate clients. Providers and maps verified. oneAPI traces contain no GPU-device opens. |
| [CPU actual-library delta ledger](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/canonical/v2/all-library-deltas/report.md) | 168/169 changed-type rows adjudicated and all eleven changed functions individually dispositioned, with exact remaining definition gaps recorded; not an exhaustive ABI pass. |
| [SYCL command ledger](/home/svnbjrn/wt/kmic68-proof-evidence/abi-sycl-comparison/results.json) | Independent unsuppressed full-type dump/diff commands, actual exits and original ELF identities. Raw diff exits and tool failures remain failures to establish compatibility. |
| [Candidate trained numerics](/home/svnbjrn/wt/kmic68-proof-evidence/gpu-numerical-qualification/report.md) | 216 valid full-vocabulary comparisons across three dense models; observed nonidentity on sequential/restored-prefix axis, zero differences in 72 full-snapshot replay comparisons. One canonical Mistral response passes repaired validation; no speed claim. |
| [Exact-model diagnostic routes](/home/svnbjrn/wt/kmic68-proof-evidence/qwen4exp-current-qualification/cache-off-diagnostic-summary.json) | Cache-disabled P06 off/on report 512/512 completions; 85 exact-match versus 66 distribution attempts. Frozen cache-enabled profile crashed; no speed or numerical-equivalence claim. |
| [Final source correspondence](/home/svnbjrn/wt/kmic68-proof-evidence/final-source-correspondence.json) | Maps integrated sources to build snapshots; separates refreshed test fixtures and generated commit strings from preserved earlier ELF identities. |
| [Independent review closure](/home/svnbjrn/wt/kmic68-proof-evidence/review-closure.md) | Source findings fixed and re-reviewed; claim reports reviewed for evidence overreach. Review is not execution. |
| [Final closure](/home/svnbjrn/wt/kmic68-proof-evidence/final-closure.json) | Publication HEAD, clean state, source and protocol identities, evidence manifest and terminal resource state. |

The first full-device-DWARF SYCL build failed when earlyoom terminated the
post-link process at about 32 GiB RSS. Its failure is retained; matched host-DWARF
builds use the explicitly qualified mitigation on both arms. Earlier callback,
log-flush, parser and ABI-coverage failures likewise remain in the evidence.
No failed attempt was relabeled a successful run.

CPU production metadata refreshed in the final client rebuild; before/after
library hashes and prior common-library bytes are preserved. Earlier ABI corpora
remain associated with their original binaries. The final SYCL fixture refreshes
changed no production library bytes. The profile-specific builds exclude
downloads and both UI options, use at most four compiler jobs, and preserve
the storage reserve and all existing artifacts.

## Blockers and continuation conditions

- ABI acceptance needs exhaustive independent header/export root coverage and
  actual-library type definitions/signature associations for both compiler
  profiles. Bounded successful clients and matching header layouts cannot
  substitute for this contract.
- Trained-model distribution qualification has bounded dense observations and
  exact-Qwen4Exp placement evidence, but its frozen cache-enabled request
  crashes in the SYCL runtime. Current route and same-history numerical
  qualification of that profile remain incomplete. Finite randomness and invalid-row
  exclusions still need explicit assumptions; measured rows do not bound unseen histories.
- Speed measurement needs a working approved baseline server. The pristine base
  omits `common_download_resolve_path` from its offline stub; applying the candidate
  repair to that arm changes the baseline contract. No alternative arm or enabled
  download path was silently substituted. Every required model/cell remains.
- The A770 readiness checks found unrelated QSV encoding jobs. No unrelated job
  was killed. A later free window allowed candidate dense numerical runs; no
  service was changed.
  Ornith changed externally from active to inactive; the coordinator did not
  undo another actor's state change. Future GPU runs must freshly snapshot
  services, use `/tmp/a770.lock`, verify sole tenancy, apply timeouts and retain
  readable before/after fault logs.

The approved implementation and reporting work is complete, with terminal
closure indexed above; the incomplete claims remain incomplete. No full compatibility guarantee,
universal runtime equality or performance gain is asserted.
