# How this history was assembled

This directory is a source-based historical review, assembled on 2026-10-11
with Codex, project specialist agents, local Git objects, checked-in research
and benchmark artifacts, and read-only Hindsight recall. It combines generated
inventories with interpreted prose. The inventories establish coverage; the
prose explains selected changes, their consequences, and the strength of the
evidence. Neither component is a fresh correctness or performance certification.

This document explains the research and writing process. The companion
[method and evidence reference](06-method-and-evidence.md) records exact Git
populations, inventory schemas, reproduction commands and validation limits.
The [index](README.md) gives the reading order and principal conclusions.

## 1. Freeze the question before reconstructing the history

The original task was to review the fork after divergence from upstream:
commits, master changes, milestones, the present difference from upstream,
reversions, integration mistakes and ordinary development mistakes. Follow-up
requests added a performance/functionality progression using repository and
Hindsight evidence, and a register of dead ends and answerable open questions.

The review began from clean fork master and created
`docs/fork-history-2026-10-11`. Three fixed endpoints prevented moving branches
from changing the meaning of a comparison during the work:

| Endpoint | Revision | Purpose |
| --- | --- | --- |
| Reviewed fork | `2f623c6607c078d1069dac23e9d8af9cb146e771` | Source tree and reachable history being described |
| Upstream fetched October 11 | `5c8854fd88319125125dc4b836c98b1b215b0f23` | Contemporary upstream comparison |
| Last imported full snapshot source | `7049ff0cbeb1f5ead231de4522af6b75d8d773c0` | Separates retained fork differences from subsequent upstream movement |

The upstream fetch used a dedicated local review ref. Existing master and
other branches were not rebased or rewritten. Later documentation commits on
the review branch are publication changes, not new endpoints for the historical
measurements. The resulting PR targets this fork's `master`.

## 2. Source families and the questions each can answer

| Source family | Material used | Appropriate use and limit |
| --- | --- | --- |
| Git object database | Parent graphs, full messages, author/committer dates, historical blobs, patches and tree comparisons | Establishes retained history and exact contents. Does not recover expired/private branches or the time a rebased ancestor was pushed to master. |
| Pinned source | Implementation, dispatch, callers, layout/type definitions, build settings, tests and defaults | Establishes contracts and concrete failure scenarios. A source defect is not automatically a reproduced runtime failure. |
| Repository orientation | AGENTS.md, development runbooks, README and feature/backend guides | Supplies vocabulary and claimed invariants to check. Orientation prose can be stale and was not treated as authority over source. |
| Dated research reports | docs/research and docs/turboquant/ppl-results | Supplies experiment setup, decisions, retractions and references to results. An old report is not a current benchmark. |
| Preserved measurements | JSON/JSONL products, CSVs, logs, provenance and benchmark archive catalogs | Can substantiate a reported comparison or reveal invalidity. Sample counts, effective paths, global validity and matching binaries still matter. |
| Hindsight | Read-only recall from bank `claude-history` | Locates forgotten observations and intermediate beliefs. Memory summaries can be stale, contradictory or attached to a different checkout. |
| Local historical guidance | Existing memory and artifact-recovery notes | Used as search leads and methodological context; repository evidence or an explicit memory-only label is needed for report claims. |
| Remote identity checks | Fork ref lookup, upstream fetch and upstream commit-page check | Pins repository identity and current endpoints. No exhaustive GitHub event, review-thread or deleted-branch archive was reconstructed. |

The [benchmark archive guide](../benchmarks/README.md) and
[campaign catalog](../benchmarks/CAMPAIGNS.md) were used to navigate the large
collection before selecting reports and original result files. Duplicate trees
and copied results were not counted as extra experiments. Files that were absent,
represented only by metadata, or outside the preserved collection remained
explicit gaps. Git tree comparisons use stored blobs, including any LFS pointer
representation; they do not imply every large payload was hydrated and read.

## 3. Reconstruct the retained lineage, then interpret it

The initial divergence boundary was found by walking the fork's first-parent
chain and locating its first commit absent from the pinned upstream ancestry.
That commit's parent is the retained starting base. This answers a different
question from the latest merge base, especially after later upstream imports.

The review kept separate sets for the full post-base graph, the first-parent
master sequence, fork-only ancestry, upstream ancestry not reachable from the
fork, and upstream changes after the last imported snapshot. The resulting
counts are 3,528, 516, 1,135, 357 and 125 respectively. These are not competing
estimates of one quantity. Inherited donor work belongs to fork-only ancestry;
snapshot imports can carry a patch without its original commit being reachable.

[reproduce.py](reproduce.py), using Python's standard library and Git, then:

1. Enumerates the graph and records complete hashes, parents, both timestamps,
   subjects, first-parent membership and upstream-ancestry membership.
2. Preserves complete messages and first-parent changed-path/line inventories
   for all 1,135 fork-only commits in JSONL.
3. Emits the readable 516-entry master ledger from the same records.
4. Compares the fork tree with both upstream endpoints, storing paths, statuses,
   modes, blob identities and line counts, then groups the contemporary delta
   by subsystem and documentation category.
5. Checks set coverage, uniqueness and consistency between raw diff and numstat
   records before writing the summary.

Rename detection, external diff drivers and text conversion are disabled so
heuristics cannot silently change the counts. A move therefore appears as a
deletion/addition in these inventories. Merge counts use the first parent;
they describe what changed from the previous mainline tree, not the sum of all
work imported through every parent. Non-ASCII original text is escaped rather
than silently discarded.

The generator does not decide which commits are important, identify bugs, write
the narrative, adjudicate benchmarks, or regenerate Hindsight recall. Those
steps require interpretation and source checking.

## 4. Turn the commit inventory into development episodes

The first-parent sequence supplied the chronological backbone. Related changes
were grouped into episodes: inherited codec development, SYCL import and repair,
backend pruning, cache policies, performance campaigns, upstream reconciliation,
SDK/server work, MoE/replay/speculation and evidence preservation.

Within an episode, review followed an introduction -> correction -> integration
or reversal chain. Large merges and squashes were examined against their first
parent and relevant donor trees. Subjects were navigation aids, not proof of
what a change did: a commit titled "merge" can have one parent, and a long squash
message can repeat work that its actual diff does not introduce.

Potential mistakes received deeper inspection when they involved serialized
IDs, data layout, rotation, shape/routing predicates, cache lifetime, memory
budgeting, unsupported features, disappearing consumers or conflicting reports.
This is a targeted semantic audit, not a new line-by-line proof of all imported
upstream work. Representative introduction/fix/revert patches and current
consumers were read together; absence claims used source searches.

Exact affected-file comparisons distinguished a true restoration from a nominal
revert that left related changes behind. Ancestry alone was never accepted as
proof of preservation: an implementation can disappear while its introducing
commit remains an ancestor. Conversely, imported content need not retain its
original ancestry. The integration and correctness chapters expose those cases.

Dates were kept attached to their meaning. Author date, committer date,
experiment date, report date and archive landing date are distinct. Timezones
were considered before inferring order; different calendar dates can denote
the same instant. The ledger does not reconstruct a complete public-master
push history.

## 5. Digest benchmark evidence without inventing a continuous speed curve

Campaign reports were read with their methods, failure gates and corrections,
then selected structured products, raw rows and provenance were checked.
Commit-reported results, report-backed observations, retained raw artifacts,
and static projections are labeled separately in the performance chapters.

Each comparison was interpreted with its model/weight variant, effective K/V
policy, workload, depth, backend route, driver/runtime/compiler, placement,
sample protocol and validation status where recorded. Missing fields were not
filled from a neighboring campaign. In particular:

- Stock Llama and Heretic results are separate cohorts, as are CPU-FA/GPU-FA,
  AOT/JIT, i915/xe, ordinary/speculative generation and different MoE placements.
- Cold first-output latency, warm prompt processing, generation rate, quality,
  allocation size and successful initialization are different endpoints.
- Requested options do not prove effective types or kernel execution. Defaults,
  automatic substitutions, eligibility guards and path counters matter.
- Printed benchmark spread is not automatically a paired confidence interval.
  A missing interval is not invented, and post-hoc subsets do not replace the
  planned comparison.
- Global product validity is checked as well as cell validity. All 16 primary
  GRF products were inspected: six reject binary/repository provenance despite
  reporting zero invalid individual cells.

Selected arithmetic was independently checked, including August occupancy
paired medians and the dense Turbo4 paired confidence bounds. This was analysis
of existing records, not a new statistical campaign or rerun of the models.
The report does not claim independent reaggregation of every archived row.

Measurements were mapped to retained implementation/integration commits where
possible. A benchmark's development hash was not silently replaced with a
later squash hash. Missing objects or disagreeing report, sample and provenance
identities were retained as limitations. A later successful campaign can reopen
a proposition without retroactively repairing an invalid earlier campaign.

## 6. Use Hindsight as a cross-check, not an authority

Five focused parent-session recall queries searched early TurboQuant/SYCL work,
July/August optimization, late xe/MoE/speculation, rejected avenues, and the
October matrix. The exact queries, parameters and dates are recorded in
[the crosswalk](09-hindsight-crosswalk.md). They returned 260 result occurrences
and 228 distinct IDs. Auxiliary agent recall also helped locate reports; it is
not included in those five-query counts.

Results were deduplicated by memory ID for that census. Thirty-two records were
selected to preserve useful observations, important corroboration and concrete
contradictions. Unrelated projects, duplicate observations and unsupported
transfers from other hardware were excluded from that curated subset. Original
selected text, timestamps, query membership and memory/document/chunk IDs remain
in [hindsight-progression.jsonl](data/hindsight-progression.jsonl).

Each selected record received a repository cross-check or an explicit statement
that only memory testimony was recovered. Examples include a permanent-sounding
June turbo dead-end verdict superseded by July repairs; an MKL sentence that
reverses its own numeric comparison; and early October gaps answered by the
later final report. Contradictory text was preserved alongside its disposition,
not rewritten to look correct.

No memory was edited or written back. This is neither a full-bank export nor
an independently authenticated transcript archive. A repeated semantic query
can retrieve different records; the frozen excerpt makes the actual selection
inspectable without pretending the retrieval itself is deterministic.

## 7. Synthesize chapters, with cross-review and explicit uncertainty

The main Codex session coordinated the inventory, upstream comparison,
Hindsight crosswalk, synthesis and final integration. Project specialist agents
reviewed chronology, integration and correctness in separate scopes; documentation
agents developed the early and late performance chapters from assigned evidence.
They shared a checkout with distinct writing responsibilities. Additional
read-only consistency reviews checked selected conclusions across chapters.

This division provided separate review passes, not statistical independence or
independent hardware replication. Agent output was treated as review material,
checked against source and artifacts where needed, and revised when conflicts
appeared. Examples of corrections during assembly include timezone ordering,
overly narrow treatment of code-splitting results as cold-start-only, per-slot
capacity wording, and the strength of provenance behind memory-reported figures.

The final documents divide the material by the reader's question:

| Output | How it was produced |
| --- | --- |
| 01 chronology | First-parent milestones plus inspected donor and development episodes |
| 02 integration history | Parent/tree comparisons and preserved integration/repair records |
| 03 correctness and reversals | Introduction/fix/revert chains, present consumers and evidence retractions |
| 04 upstream delta | Full tree inventory interpreted by subsystem and retained behavior |
| 05 master ledger | Deterministically generated commit rows and first-parent change counts |
| 06 method reference and data | Definitions, pins, schemas, commands and check transcripts |
| 07/08 performance chapters | Campaign-level readings, selected raw checks, commit mapping and scope limits |
| 09 Hindsight crosswalk | Frozen recall excerpts matched to source, later evidence or unresolved gaps |
| 10 progression summary | Synthesis of those chapters into capability and comparable-measurement timelines |
| 11 dead-end register | Failed, reopened and unresolved propositions classified by the evidence needed next |

The synthesis deliberately does not multiply unrelated percentages or splice
different absolute throughput figures into a single speed curve. A useful
progression can instead be broken -> repaired -> measured -> promoted -> later
qualified. Capability, performance and confidence can advance at different rates.

Dead ends were classified by what can settle them: static inspection,
correctness tests, matched timing, artifact recovery or a new mechanism.
Some gaps are irrecoverable from the present record. New benchmarks cannot
recover missing historical logs; timing an inactive path cannot measure a
feature; a passing default test cannot qualify an opt-in implementation.

## 8. Validation, reproducibility, and publication

Validation covered generator self-checks and byte-identical regeneration,
inventory counts, linked Git objects, relative links and Markdown anchors,
ASCII/whitespace, selected revert/tree comparisons, the repository's existing
upstream-sync invariant check, and selected benchmark validity/arithmetic checks.
The dated transcripts are [initial validation](data/validation.txt) and
[progression validation](data/progression-validation.txt). Passing a grep-based
invariant gate is not semantic certification; the review found defects that
such a gate does not detect.

[SHA256SUMS](SHA256SUMS) covers the history files except itself. It detects
changes relative to this published collection; it does not authenticate the
original experiment or prove the prose's interpretation. Publication checks
refresh the manifest after documentation edits and verify it before committing.
Only this directory and its documentation-index link are part of the change.

Reproduction has three different meanings:

1. **Inventory reproduction:** use a non-shallow checkout with the required Git
   objects, following chapter 06. The generator rejects shallow ancestry before
   writing anything. This regenerates ten files, not the prose.
2. **Review reproduction:** follow a claim's linked patch, source, report or
   raw artifact and reconsider its interpretation. No automated summarizer is
   claimed to recreate the editorial judgments verbatim.
3. **Experiment reproduction:** rebuild/run with the recorded model, binary,
   stack, effective route and resource conditions. This was not performed in
   this review and may be impossible where original artifacts are missing.

For a future update, retain these pins and retrieval dates as a historical
snapshot. New endpoints or experiments require new comparisons and explicit
supersession, not silent substitution into an old table. The inventory generator
is specific to this review, not a general-purpose history-analysis framework.

No new CPU/GPU build, inference run, quality campaign, benchmark, service change
or runtime-stack modification was performed to write these documents. Coverage
is exhaustive for the stated retained commit/path inventories and selective for
deep semantic and experimental review. That boundary is the basis for using
the overview responsibly, not an exception hidden after the conclusions.
