# Fork history review, 2026-10-11

This fork has grown from inherited TurboQuant experiments into an Intel Arc
inference and research distribution. Its biggest functional difference from
upstream is the combination of fork-private quantization formats and cache
policies with a substantially modified SYCL execution stack. Its biggest
platform difference is deliberate removal of thirteen backends. Its biggest
file-count difference is now the benchmark archive, not executable code.

The history also contains real correctness failures: broken rotation and
attention indexing, a serialized-type renumbering, unsafe cache lifetimes,
incomplete merge resolutions, and measurements later retracted or qualified.
Several were repaired carefully. Some defects remain in the reviewed tree;
they are recorded explicitly rather than hidden behind later successful tests.

## Reading order

Read [How this history was assembled](0-methodology.md) for the source families,
processing steps, review roles and reasoning used to turn the evidence into
these chapters. Chapter 06 supplies the exact inventories and reproduction details.

For the performance/functionality extension, start with the
[progression summary](10-progression-summary.md), then the early and late
measurement chapters. The [dead-end register](11-dead-ends-and-open-questions.md)
distinguishes questions answerable by static analysis, correctness tests,
benchmarks, artifact recovery, or a new implementation.

1. [Chronology and milestones](01-chronology.md): inherited origins, the SYCL
   fork, experiments, product changes, and changes that reached master.
2. [Integration history](02-integration-history.md): ancestry, upstream and
   TheTom imports, merge/rebase damage, and branch-only recovery incidents.
3. [Mistakes, reversals, and open findings](03-correctness-and-reversals.md):
   demonstrated historical problems and source-confirmed remaining defects.
4. [Current comparison with upstream](04-current-upstream-delta.md): what the
   fork adds, removes, changes, and has not yet incorporated.
5. [Complete master ledger](05-master-ledger.md): all 516 first-parent entries
   from the first divergent commit through the reviewed master, with actual
   first-parent change counts and links.
6. [Method, evidence, and reproduction](06-method-and-evidence.md): coverage,
   immutable revisions, inventories, validation, and limits.
7. [Performance and functionality, March-August](07-performance-early.md):
   quality/capacity, P5, occupancy, fusion, failures and default promotions.
8. [Performance and functionality, September-October](08-performance-late.md):
   xe, prefetch, GRF, MoE, MTP and the later multi-axis tuning campaigns.
9. [Hindsight crosswalk](09-hindsight-crosswalk.md): selected memory records,
   repository corroboration, superseded claims and contradictions.
10. [Progression summary](10-progression-summary.md): mapped development and
    measurement timeline, with explicitly separate comparison cohorts.
11. [Dead ends and open questions](11-dead-ends-and-open-questions.md): 28
    development paths and 12 reconstruction gaps, with concrete next evidence.

The narrative is the review. The generated inventories make its scope
inspectable and provide the complete commit/path coverage underneath it.
Neither a commit subject nor its claimed validation is treated as a new test
result from this review.

## Immutable scope

| Item | Revision or count |
| --- | --- |
| Fork master reviewed | `2f623c6607c078d1069dac23e9d8af9cb146e771` |
| Upstream master fetched on 2026-10-11 | `5c8854fd88319125125dc4b836c98b1b215b0f23` |
| Last imported full snapshot's upstream source | `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`, 2026-10-05 |
| Initial first-parent divergence base | `b3d758750a268bf93f084ccfa3060fb9a203192a` |
| First divergent first-parent commit | `0b37448748d57191c99b7d43969f08c66cec5798` |
| Latest shared Git ancestor with current upstream | `81bc6b83f827df746eb129235488d325c49cae52`, 2026-09-26 |
| All reachable commits after the initial base | 3,528 |
| First-parent entries after that base | 516 |
| Fork commits not ancestors of current upstream | 1,135 |
| Upstream commits not ancestors of fork master | 357 |
| Upstream commits after the October 5 snapshot | 125 |

These are different populations. The 3,528 include imported upstream work and
side histories. The 1,135 include inherited TheTom work, not just changes
authored by this fork's maintainer. The 357 are not 357 missing patches:
snapshot imports and cherry-picks carry content without original ancestry.
Likewise, the 125 include work already ported before upstream merged it.

The first TurboQuant commit was authored on March 24 and committed into the
present lineage on April 15. The first-parent SYCL integration milestone is
June 25. Calling any one of these dates simply "the fork date" loses useful
history. The ledger preserves both author and committer timestamps in its
machine-readable form.

## Conclusions that matter

- **The fork is more than TurboQuant.** Graph replay, q8 cache layout, MoE
  caching/prefetch, speculative decoding, scheduler changes, SDK packaging,
  server policy, and hardware-specific operating defaults all affect its
  behavior independently of selecting a turbo cache type.
- **Capacity gains are not universal throughput gains.** Historical A770
  campaigns support particular layout, occupancy, and fusion improvements;
  they do not establish that turbo KV beats f16/q8 KV generally, or that
  i915-era measurements describe the present xe stack.
- **Ancestry repeatedly overstated preservation.** A change can remain an
  ancestor while a later squash or merge removes its implementation. The
  surviving cache-key CLI with missing selection logic is a concrete example.
- **The July type-ID transition is a real format boundary.** Current rules
  rightly protect numeric IDs, but the history did change them. Old artifacts
  cannot be assumed compatible merely because both versions say TurboQuant.
- **Later fixes do not close every related path.** The SYCL MoE cache received
  budget and invalidation repairs while corresponding Vulkan paths still
  contain source-level defects in the reviewed revision.
- **Upstream tracking is selective and delayed.** The October 9 import came
  from October 5 upstream even though its subject says September 1. Two
  subsequently merged upstream correctness fixes are still absent from this
  fork: the ACC offset guard and busy pinned-slot protection.
- **Documentation is evidence with a date, not a timeless contract.** Several
  old "killed" avenues were later reopened; some current orientation prose
  still describes obsolete defaults or unconditional support. Read the source
  and the dated qualification together.

## What was done in this review

Created `docs/fork-history-2026-10-11` from the clean, current master; fetched
upstream into a dedicated local review ref; enumerated the complete reachable
post-divergence graph; reviewed the master sequence and major side histories;
inspected important introduction/fix/revert diffs and present consumers; and
generated reproducible commit and tree inventories. The documents link back
to the commits and source behind their conclusions.

The performance extension examined dated reports and selected raw tables,
all 16 primary GRF products, and Hindsight recall. It preserves 32 selected
memory records from 228 distinct records returned by five focused parent-session
queries, and checks remembered conclusions against source and later reports.
The archive's repeated copies are not counted as independent experiments.
Numerical observations stay attached to their workload and provenance; no
cumulative fork speedup is inferred.

This is a historical and source review. It does not claim a fresh build,
hardware run, benchmark, exhaustive line-by-line proof of every imported
upstream change, or confirmation that every historical test claim was true.
No implementation changes are included. The concrete remaining findings are
follow-up work, not silently fixed as part of writing history.
