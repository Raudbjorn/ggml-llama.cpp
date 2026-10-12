# P09 fixed-width verification: rejected by the parity gate

Date: 2026-10-11. Status: **Rejected by gate; candidate excluded from integration**.
Scope: [P09](../plans/kmic68-09-padded-verify.md).

The dense-transformer campaign passed its padding, output/KV isolation and
graph-rebuild criteria. The required exact-Qwen4Exp recurrent fallback campaign
failed token and drafted/accepted-count parity. A fresh OFF repeat also differed
from the original OFF result, so the evidence does not identify padding as the
cause. Independently, source inspection found that enabling padding changes
output reservation and model fitting before recurrent eligibility is checked.
That behavior remains unfixed in the excluded candidate.

Only this outcome report is integrated. The uncommitted implementation remains
in `/home/svnbjrn/wt/kmic68-kv-graph`, branch `codex/kmic68-kv-graph`, based on
`1a13eef9a38a8f0b232da13086ccc8b01cc0f425`. Its 41 source files and the two
measurement-harness files from `codex/kmic68-measurement` are preserved in
[candidate-and-harness.tar.gz](/home/svnbjrn/wt/kmic68-evidence/p09-unaccepted-source/candidate-and-harness.tar.gz).
The archive is 869049 bytes, SHA256
`cfcb6318dcdf66d479e104a79883fd998ace9e6cbb336c76b89ec02260dec0c7`.
The adjacent manifest records bases, per-file hashes and paths; tracked patch,
status and index records are retained. Archive contents were hash-verified.

## Requirement-linked observations

These scoped passes remain evidence about the candidate; they do not establish
overall P09 acceptance or an integration-branch capability.

| Requirements | Evidence and qualification |
| --- | --- |
| R09.1-R09.3, R09.5, R09.7-R09.10 | CPU policy and real-server tests cover parsing, disabled shape, automatic width, multislot atomicity and fallback. Dense A770 auto/8/oversize arms pass. Recurrent oversized mode did not run after the parity failure. |
| R09.4 | Twelve generated live-model configurations check real logits, padding invisibility and last-real-row indexing. These are synthetic model fixtures. |
| R09.6, R09.14 | Row bijection, sentinel-before-offset checks, invalid-row isolation and cache allocation/position checks pass. Dense real-model committed K/V bytes and cell positions match OFF. |
| R09.11, R09.13 | Actual callbacks, fixed physical output capacity and dynamic real-row validity are exercised; dense GPU counters reconcile. The recurrent initialization reservation side effect remains unresolved. |
| R09.12 and mandatory cross-mode parity | Recurrent runs execute zero padded rows and report unsupported-context fallback, but required output/drafting parity fails. Full fallback qualification is unmet. |

The [earlier independent source and saved-evidence review](/home/svnbjrn/wt/kmic68-evidence/p09-final-source-review.md)
cleared the scoped implementation while explicitly leaving recurrent fallback
and integration verification pending. The later gate review supports exclusion,
without relabelling every requirement or earlier passing experiment as failed.

## Passing dense-transformer campaign

The [real GPU campaign](/home/svnbjrn/wt/kmic68-evidence/p09-real-gpu-campaign/report.md)
uses the same local `Qwen3-1.7B-Q8_0.gguf` as target and `draft-simple` drafter.
Four fresh server processes each complete six requests of 256 output tokens.
Each mode has 196 actual verification steps and totals 1536 output tokens,
1360 drafted tokens and 1332 accepted tokens.

| Mode | Graph rebuilds | Actual dummy rows |
| --- | ---: | ---: |
| OFF | 16 | 0 |
| auto | 12 | 12 |
| width 8 | 12 | 12 |
| oversized 4096 | 16 | 0 |

Matching requests have identical output tokens, drafted/accepted counts,
canonical committed K/V byte hashes and cell positions. Required within-request
real-width variation is present. Four fewer rebuilds is a 25% reduction in this
count, not a throughput measurement. The [CPU HTTP campaign](/home/svnbjrn/wt/kmic68-evidence/p09-real-cpu-server/report.md)
passes eight tests across 18 owned server processes, with no skips.

The [CPU fixtures](/home/svnbjrn/wt/kmic68-evidence/p09-cpu-final-fixtures/report.md),
[fresh numeric matrix](/home/svnbjrn/wt/kmic68-evidence/p09-q8-fresh-verification/report.md)
and [XMX retry](/home/svnbjrn/wt/kmic68-evidence/p09-xmx-retry/report.md)
cover invalid rows, masked softmax/FA, SET_ROWS, output validity, native/turbo
routes, WG1/WG2, VEC/TILE/MKL, quants-first layout, nullable masks/sinks and
explicit XMX D128/D256. Known expected failures remain labelled. An initial
MKL-disabled XMX invocation failed mandatory MKL checks; its corrected retry
does not erase that failure.

The original arbitrary-positive Q8 fixture exceeds the unchanged 5e-3 tolerance
on both candidate and pre-P09 baseline. A reviewed exact-value semantic fixture
passes with the same tolerance and meaningful zero/nonzero rows. Original
failure, baseline reproduction and localization remain preserved; no general
Q8 precision repair is claimed.

## Failed recurrent parity and the OFF control

The exact local `qwen4exp` target and matching MTP head were validated and used.
Other architectures did not substitute for this requirement. Earlier attempts
under a different environment bundle produced null target log-probabilities;
the matching old P04 binary reproduced that failure. A historical minimal
environment permitted finite outputs. A one-variable FA override did not
reproduce the null failure. These attempts remain separate, and no single
environment variable was established as its cause.

The [candidate campaign](/home/svnbjrn/wt/kmic68-evidence/p09-recurrent-candidate-prepared/runtime-report.md)
then ran the frozen 253-token input in that working minimal environment:

| Arm | Output tokens | Drafted | Accepted | Verification steps |
| --- | ---: | ---: | ---: | ---: |
| Original OFF | 64 | 104 | 46 | 16 |
| Original auto | 64 | 104 | 46 | 16 |
| Original width 8 | 64 | 92 | 49 | 14 |
| Fresh OFF repeat | 64 | 100 | 47 | 16 |

All completed arms have finite target-argmax outputs and zero padded rows.
OFF and auto match; width 8 first differs at output index 32. Oversized 4096
was not run after this failure.

The separately authorized [single OFF repeat](/home/svnbjrn/wt/kmic68-evidence/p09-recurrent-off-repeat-prepared/runtime-report.md)
uses the same binary, arguments, request and loaded libraries. Its checked
environment differs only by the unique inert cycle-log path. Its
startup/reservation placement matches original OFF. It also
first differs from original OFF at index 32, changing token 11782 to 1328.
Its entire token array equals the earlier width-8 array, but drafting counts,
verification steps and probabilities differ. This is observed OFF variability,
not equivalence between the complete responses or evidence of padding causality.

Before/after full model hashes and preservation of 41 source files, 20 AOT
artifacts and 350 files across six earlier campaigns passed. The diagnostic
finished with `diagnostic_completed=true`, `accepted=false`, wrapper exit 0 and
released resources. No retry or alternative seed/prompt followed it. Earlier
failed campaigns remain failed.

## Independent reservation issue and limits

The [source audit](/home/svnbjrn/wt/kmic68-evidence/p09-recurrent-parity-source-audit.md)
found that enabled padding sets the output-capacity sentinel to zero before
model/context eligibility. Context initialization expands it to `n_batch=2048`,
versus ordinary capacity 8. Actual fitted target SYCL weights change from
9234.54 to 9104.28 MiB and initial compute reservation from 394.20 to 510.00 MiB.
Auto and width 8 share this placement, so that observation alone cannot explain
their different outputs. A future correction must preserve ordinary reservation
for unsupported targets during both fitting and final initialization. The
candidate is excluded with this issue recorded; no correction is claimed.

The [API/ABI check](/home/svnbjrn/wt/kmic68-evidence/apiabi-p09/report.md)
reports 100% source compatibility, no removed/changed C symbols and two additive
C symbols. Public by-value batch/context structures and backend vtables remain
unchanged. Release binaries lack complete type debug information and checker
C++ suppression applies; full typed ABI compatibility is unproven.

The reported A770 runs use xe, repository locking, bounded timeouts, tenancy
checks and readable before/after fault logs. The completed scoped runs report
no new faults. Services remained inactive/inactive/failed during verification;
the coordinator restores initial state at campaign completion.

No P09 product code is integrated, so no integrated P06/P09 or P09/P12 pass is
claimed. P12 also remains blocked on P08. Unexplained OFF variability, recurrent
oversized coverage, full typed ABI and integrated-tree verification remain
unresolved; synthetic and dense-transformer passes do not replace these gates.
