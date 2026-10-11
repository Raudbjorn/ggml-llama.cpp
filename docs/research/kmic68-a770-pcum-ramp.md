# Kmic68 cumulative draft-width clamp: calibration protocol blocker

Evidence date: 2026-10-11. Disposition: **P10 blocked; no runtime change or
calibrated ramp accepted.** The requested calibration protocol remains unchanged.
The calibration protocol requires an explicit amendment before this blocker
can be cleared.

The [P10 plan](../plans/kmic68-10-pcum-draft-width.md) requires measured MTP ramp
knots at committed target positions 0, 4096 and 16384, with P01's 512-token
workload. Source auditing found that an ordinary server request cannot produce
a first MTP cycle at committed position zero. A fresh request with L actual
prompt tokens begins drafting at pos0=L. An empty token array is rejected;
an empty text prompt either receives a token or produces no drafting. Prompt
caching and checkpoint replay preserve logical prompt length rather than
creating an observed zero-position cycle.

Moving the first knot to 1 alone does not resolve the workload mismatch: the
three prescribed representative chat prompts cannot fit at position 1. The
historical first cycle positions start at 82. Selecting a new one-token seed
workload or assigning a measured ordinary-prompt window to an explicitly
extrapolated zero endpoint changes the measurement protocol. Neither amendment
has been selected. No observed position was relabeled and no threshold was
invented. This is a protocol blocker, not a measured performance rejection or
a missing-Qwen4Exp claim; exact target and MTP weights are available.

## Completed attempt and preservation

The offline helper and tests remain isolated in
`/home/svnbjrn/wt/kmic68-pcum-offline`, branch `codex/kmic68-pcum-offline`, base
`1e7f42a8c583ec80c9f669e802c97afe656379e8`. Two new Python files implement
bounded replay using the accepted P04 validator. No runtime C++/server/backend
code, artifact loader, device-identity procedure or calibration launcher was
implemented or integrated for P10.

One historical replay processed 391 selected cycles from six requests, with
2694 finalized draft tokens and 1138 originally accepted draft tokens. All 101
predeclared thresholds were retained. The fixed shortlist is 0, 0.25, 0.5, 0.75
and 1. These are candidates for future measurement, not measured winners.
Observed positions span 82 through 506; exact counts at every required knot are
zero. The old requests generated 256 tokens, so their durations and acceptance
counts do not meet the 512-token calibration protocol.

The helper preserves attempted diagnostics and uses finalized emitted tokens
only. It distinguishes sampled token identity from top-1 probability, applies
a strict threshold crossing, preserves the first finalized token and never
extends a proposal. Its unit checks establish offline accounting behavior;
they do not establish runtime stopping, proposal/distribution truncation or
trained-model distribution preservation.

Recorded checks: unittest ran 24 items with 23 passes and one optional compiled
recorder skip; pytest reran 16 of those test methods and passed 68 subtests.
These are overlapping checks, not a second independent population. The saved-input replay
exited zero with all input bytes, sizes and mtimes unchanged. Independent source
review found no material defect within this explicitly historical scope. No
build, inference or calibration ran for the helper.

Candidate files, all retained offline evidence including ignored test logs,
source identities and the unchanged Git status are archived under
`/home/svnbjrn/wt/kmic68-evidence/p10-blocked-source/`. The tar archive SHA256 is
`de158fd1ed8f47eab0da9244588147322def196679cfd879e047922410299513`.
Raw replay and test records also remain in the worktree's
`evidence/p10-offline-replay/`. Source audits are retained as
`p10-offline-source-review.md`, `p10-knot-reachability-audit.md`,
`p10-protocol-review.md` and `p10-low-depth-workload-audit.md` in the campaign
evidence root.

## Requirement-linked disposition

| Requirements | Established evidence | Missing gate |
| --- | --- | --- |
| R10.1-R10.3, R10.5-R10.8 | Offline threshold, equality, first-token and no-extension accounting fixtures | Runtime enabled/disabled stop and batch-shape behavior, existing stops and adaptive/n_min interaction |
| R10.4, R10.14 | Source audit of committed position and identification of unreachable exact zero knot | Selected executable calibration protocol, final-identity paired measurements, valid artifact and independent unchanged-binary validation |
| R10.9 | No runtime setting parser implemented | Complete finite numeric parsing and initialization failures |
| R10.10-R10.13 | Helper restricts replay to qualified historical non-chained MTP; sampled token/top-1 distinction tested | Runtime probability domains, all eligible/excluded drafter fixtures and DFlash block-decode accounting |

P06/P10 runtime combination tests remain unrun because no P10 runtime change
was accepted. P09 is independent; its remaining work is not a P10 prerequisite.
If the protocol is amended, final calibration still follows every accepted
code change and must match all model shards/head, server and loaded backend
binaries, actual executing devices and assignments, route, kernel/driver/runtime,
acceptance mode and effective target/draft sampling settings. Preserve the
predeclared threshold grid, three paired repetitions, 512 outputs, highest
median-throughput objective, acceptance floor, token parity and separate
same-binary validation. If no candidate qualifies, emit no ramp.

## Not claimed

No P10 feature acceptance, runtime implementation, calibrated or built-in ramp,
performance improvement, live acceptance improvement, distribution proof or
exact-depth coverage. The helper's saved historical data cannot replace fresh
measurement at the final binary identities. No deployment or model download is
introduced by this outcome record.
