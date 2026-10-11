# Kmic68 draft capacity: P08 and P12 prerequisite outcomes

Evidence audited on 2026-10-11. **P08 is blocked at acceptance; P12 is blocked
before implementation by unaccepted P08.** Neither is rejected by a measured
performance gate. Candidate P08 code is not integrated.

The [P08 plan](../plans/kmic68-08-draft-ubatch-cap.md) makes real EAGLE3,
DFlash and DSpark correctness/capacity fixtures mandatory in addition to the
production MTP campaign. It requires each family to execute a concrete workload
whose row count exceeds the effective draft cap. Generated graphs establish
bounded implementation behavior; they do not meet the real-model gate.

## Candidate and CPU evidence

Candidate worktree: `/home/svnbjrn/wt/kmic68-draft-capacity`, branch
`codex/kmic68-draft-capacity`. Its source snapshot, tracked patch, base revision,
per-file hashes and archive hash are preserved under
`/home/svnbjrn/wt/kmic68-evidence/p08-blocked-source/`.

Raw CPU results are in `p08-independent/verification-report.md`,
`p08-parser-fix-test.log`, and `p08-adaptive-assert-test.log`, relative to
`/home/svnbjrn/wt/kmic68-evidence/`. The later parser and assertion fixes address
the initial verifier's findings. The assertion infrastructure fix is recorded
separately in commit `446cd23d7`; it is not P08 feature acceptance.

| Requirements | Established evidence | Remaining limit |
| --- | --- | --- |
| R08.1-R08.3 | CPU defaults, aliases, environment parsing and cap resolution at 0/32/64/equal/above target; generated live cap cases | Real family capacity runs |
| R08.4 | Generated live-normalization test preserves caller target parameters | Real target-state equivalence |
| R08.5 | Generated diagnostics include provisional/live/effective dimensions, minimum, reconciliation and compute MiB | Production joint-fit and long-depth buffer savings |
| R08.6-R08.8 | Invalid/small cap rejection, zero inheritance and corrected value-bearing parser errors pass | Inheritance retains the documented minimum 32 |
| R08.9 | Generated EAGLE3/DFlash/DSpark/MTP chunking, independent blocks, oversized indivisible-block rejection and 65-token MTP fallback pass | All generated accepted counts were zero; no positive-acceptance or real-family proof |
| R08.10-R08.12 | Live width normalization, impossible-width rejection and injected failed fit measurement pass | Successful production joint placement and reload |

The independent CPU report also records MTP default/q8 and Python checks
(38 Python tests). The current candidate matches 17 of 19 entries in its
historical source manifest; the two changed files are the subsequently retested
parser correction. These are CPU results, not the exact SYCL README floor or
an A770 acceptance campaign.

## Exact missing fixtures

The candidate `scripts/perf/fixtures/draft-ubatch.json` still has null target and
draft paths for these three matching trained pairs:

| Family | Required local fixture identity |
| --- | --- |
| EAGLE3 | Compatible target and trained `general.architecture=eagle3` draft, exactly three `eagle3.target_layers`, matching `eagle3.target_hidden_size`, compatible projection/normalization/decoder tensors and target vocabulary |
| DFlash | Compatible target and trained `general.architecture=dflash` draft, nonempty `dflash.target_layers`, compatible feature dimensions, trained block configuration and usable mask token; an indivisible block must fit the cap |
| DSpark | Compatible target and DFlash-architecture draft with `markov_w1.weight` and `markov_w2.weight`, compatible trained block/backbone dimensions, and confidence settings matching the available head |

There is no separate DSpark GGUF architecture in this implementation. A declared
absent confidence head requires compatible settings including
`--spec-draft-p-min 0`. Arbitrary ordinary models are not trained substitutes.
The bounded local inventory and unresolved manifest do not prove these pairs
are absent everywhere.

**Exact Qwen4Exp target and MTP head are available.** The two-shard target under
`/mnt/ssd1/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/IQ1_M/` and canonical head
under `/mnt/ssd2/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/mtp/` declare
`qwen4exp`. A recovered smoke request completed 16 tokens with 14 drafted and
12 accepted; the raw diagnosis is `p04-stack-diagnosis/report.md` in the evidence
root. Process-local Intel-only OpenCL ICD isolation avoided the independently
reproduced Mesa/ROCm LLVM coexistence crash. This is one smoke observation,
not P08 capacity evidence. Available `qwen35` models are not substitutes.

Outstanding P08-specific A770 gates include the real 65-token MTP crossing,
cap-32/cap-64 comparisons, cap-64 long-depth buffer reduction with raw MiB, and
paired MTP performance without statistically detected regression. No P08
production memory saving or speed claim is made. Locating the three missing
pairs alone would not complete those gates.

## P12 dependency outcome

[P12](../plans/kmic68-12-non-materialised-mask.md) declares P08 as a hard
implementation dependency and uses P08's effective draft width in reservation
keys. The campaign requires accepted prerequisite commits before dependent
implementation. P08 has no accepted feature commit, so P12 kernel editing has
not begun. R12 requirements, reservation-failure-before-cache-mutation tests,
and combined P08/P12 and P09/P12 tests remain unexecuted.

The [P03/P11 outcome](kmic68-a770-fa-memory-floor.md) is recorded independently;
it does not block or satisfy P12's P08 prerequisite. Independent P09 work can
continue. When P12 becomes eligible, its corrected ABI, nullable-mask/sinks,
I32-prefix, reservation and padding-exclusion contracts still apply unchanged.

## Not claimed

No P08 or P12 acceptance, real-family absence everywhere, generated-fixture
substitution, production memory savings, throughput non-regression, complete
SYCL validation or prefix-mask implementation. Integration records only this
outcome and its index entry; candidate P08 source and raw evidence are preserved
outside the integration branch. No model download, package installation or new
runtime dependency is introduced by this report.
