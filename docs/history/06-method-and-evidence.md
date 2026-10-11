# Method, evidence, coverage, and reproduction

Review date: 2026-10-11. The report is intentionally fixed to immutable
revisions; future changes to either master do not silently change its claims.

For the end-to-end research and writing process, see
[How this history was assembled](0-methodology.md). This chapter is the technical
reference for populations, evidence files and reproduction commands.

## 1. Inputs and state changes

The checkout was clean on fork `master` at
`2f623c6607c078d1069dac23e9d8af9cb146e771`. Both the existing
`origin/master` and a live `git ls-remote origin refs/heads/master` returned
that revision. The review branch `docs/fork-history-2026-10-11` was created
from it. This report and the documentation-index link are its only file changes.

Upstream was fetched directly from `https://github.com/ggml-org/llama.cpp.git`
into `refs/history-review/upstream-master-2026-10-11`, without adding a remote
or moving any existing branch. The fetched commit is
`5c8854fd88319125125dc4b836c98b1b215b0f23`, committed on October 11 at
17:43:35 +03:00 (14:43:35 UTC). The upstream GitHub commit page was also
opened as an external identity check; local Git objects supply the actual
diff evidence. Git version during the review was 2.56.0.

No shallow boundary is present (`git rev-parse --is-shallow-repository`
returned `false`). That means reachable local history is available; it does
not mean every abandoned, force-pushed, expired, or unpublished experiment
can be recovered. A snapshot of 68 observed local/remote/review refs is kept
in [observed-refs.tsv](data/observed-refs.tsv). Those are locally observed
refs, not a fresh fetch or full enumeration of every remote branch.

## 2. Define the populations before counting

Let F be the reviewed fork tip and U the fetched upstream tip. Walking F's
first-parent chain from its root, the first commit not in U's ancestry is
`0b3744874`; its parent B is `b3d758750`. This is the retained initial
divergence boundary used throughout the report.

| Population | Git definition | Count |
| --- | --- | ---: |
| Reachable post-divergence graph | `git rev-list F ^B` | 3,528 |
| Master first-parent sequence after divergence | `git rev-list --first-parent F ^B` | 516 |
| Fork-only ancestry against current upstream | `git rev-list F ^U` | 1,135 |
| Of those, side ancestry outside master first-parent | Set difference from the preceding first-parent list | 619 |
| Upstream ancestry absent from fork ancestry | `git rev-list U ^F` | 357 |
| Upstream work after last full snapshot source S | `git rev-list U ^S` | 125 |

S is `7049ff0cbeb1f5ead231de4522af6b75d8d773c0`, dated October 5.
The latest merge base of F and U is `81bc6b83f`, dated September 26.
B, S, and that merge base answer different questions.

The 516 entries are all non-upstream objects in this retained first-parent
chain. That does not establish exclusive authorship: many implement donor
work, ports, or squashes. A squash can contain many internal corrections;
a merge can import many ancestor commits while changing very few files
relative to the preceding fork master.

Topological order is used for the all-commit inventory. It can place a later-
dated donor branch before earlier first-parent commits because parent-before-
child ordering constrains the output. Author date is preserved separately
from committer date. Neither date establishes when every rebased ancestor
was individually pushed to GitHub master. No master reflog or GitHub event
stream reconstruction is claimed.

## 3. What was reviewed

The audit has four layers:

1. **Exhaustive enumeration.** Every one of the 3,528 reachable post-base
   commits is in the DAG ledger. Every first-parent commit is in the readable
   master ledger. Every fork-only commit has its complete message and actual
   first-parent changed-file/line inventory retained in JSONL. Both upstream
   comparison trees have complete path/mode/blob inventories.
2. **Chronological review.** The first-parent sequence was reviewed across
   inherited codec work, SYCL import, pruning, recovery, experiments, upstream
   syncs, SDK/server work, cache/speculation work, and the evidence archive.
   Major side histories were examined at their integration boundaries.
3. **Focused semantic review.** Important introduction/fix/revert sequences
   were read as patches and compared with current functions and consumers.
   This covers the reported rotation, row-count, ABI, layout, replay, cache,
   prefetch, MTP, slot, and offset findings. A whole-repository source search
   was used when asserting absence of a known consumer.
4. **Evidence review.** Dated reports and commit messages were checked for
   retractions, scope changes, missing-path tests, actual routing, driver
   changes, and provenance limitations. Reported numbers remain historical.

This is comprehensive inventory and milestone review with targeted deep
inspection. It is not a claim that all 3,528 commits or all changed lines
received a fresh independent semantic proof. Routine imported upstream
patches, every intermediate donor experiment, every archived measurement,
and every supported model/backend combination were not individually rebuilt
and rerun. The ledgers expose that breadth rather than hiding it behind a
selected shortlist of commits.

## 4. Evidence grades used in the chapters

| Grade | Example | What it establishes |
| --- | --- | --- |
| Git topology/tree fact | Parent lists, exact tree equality, ID changes in a diff | What is stored and reachable in the inspected repository |
| Current source finding | Vulkan expert-offset invalidation omits the offset | A concrete code-path defect/scenario, without claiming hardware reproduction |
| Checked-in historical observation | Paired campaign or recorded failing model output | What the dated artifact reports under its recorded conditions |
| Commit-message testimony | Staged conflict-marker count during a past merge | Historical account; not independent recovery of transient index/runtime state |
| Hindsight recall | Dated memory with document/chunk ID | Secondary historical testimony and a source locator; may be stale, contradictory, or misdated |
| Inference or unresolved hypothesis | Prompt-cache contamination as a suspected cause | A plausible explanation that needs additional evidence |

The integration chapter explicitly labels two source/head abbreviations that
do not resolve locally (`cd3a214b7`, `bb4e5e44e`). They are retained as recorded
provenance, not silently treated as inspected commits. Historical pre-rebase
hashes can differ from the reachable versions; exact candidate/revert tree
comparisons are used where possible.

PR numbers are qualified by context: inherited TheTom PR 146 is not fork PR
146. A title containing "merge" is not proof of multiple parents. A feature
commit being an ancestor is not proof that a later tree preserves the feature.

## 5. Inventory files

All generated text is ASCII. Original non-ASCII commit messages/subjects are
preserved with escape sequences, not silently dropped. JSONL uses JSON Unicode
escapes; TSV/Markdown use backslash escapes where necessary. This affects
display only, not commit/blob identities.

| File | Contents |
| --- | --- |
| [summary.json](data/summary.json) | Immutable tips, divergence/merge-base IDs, counts and month distribution |
| [all-post-divergence-commits.tsv](data/all-post-divergence-commits.tsv) | All 3,528 commit IDs, parents, both dates, subject, first-parent membership and upstream ancestry |
| [master-first-parent.tsv](data/master-first-parent.tsv) | The 516-entry master sequence with full hashes and both dates |
| [fork-only-commits.jsonl](data/fork-only-commits.jsonl) | All 1,135 non-upstream commit messages and per-path numstat against each first parent |
| [upstream-not-ancestors.tsv](data/upstream-not-ancestors.tsv) | All 357 upstream commits absent from fork ancestry; not a missing-patch list |
| [upstream-after-snapshot.tsv](data/upstream-after-snapshot.tsv) | The 125 commits after October 5's snapshot source |
| [current-upstream-delta.tsv](data/current-upstream-delta.tsv) | All 11,629 changed paths against current upstream, modes, blob IDs and line counts |
| [last-snapshot-delta.tsv](data/last-snapshot-delta.tsv) | The 11,290 path differences against the October 5 source |
| [current-delta-groups.tsv](data/current-delta-groups.tsv) | Aggregate additions/modifications/deletions and text line counts by path group |
| [observed-refs.tsv](data/observed-refs.tsv) | One-time local reference census and ancestor-of-reviewed-master flag |
| [validation.txt](data/validation.txt) | Documentation/data checks actually performed in this review |
| [hindsight-progression.jsonl](data/hindsight-progression.jsonl) | 32 frozen selected Hindsight records with full IDs, timestamps, original text and query membership |
| [progression-validation.txt](data/progression-validation.txt) | Additional checks for the performance/functionality extension |
| [SHA256SUMS](SHA256SUMS) | Checksums of the completed history files, excluding the checksum file itself |

Numstat and tree comparisons explicitly disable rename detection, external
diff tools, and text conversion. This keeps the inventory deterministic and
independent of a user's diff driver. Relocated docs count as add/delete;
archived/LFS blobs are compared in their stored Git representation. A `-`
line count denotes a binary diff. No benchmark payload hydration was needed
or performed to establish Git tree differences.

Merge change counts are measured against their first parent. They represent
what that integration changed from its previous mainline tree; they do not
sum independent original work across both parents. Snapshot line counts also
include imported upstream changes, not just fork code.

## 6. Reproduce the generated evidence

The checked-in [reproduce.py](reproduce.py) uses Python's standard library and
Git. It has fixed full hashes, no network access, no build dependency, and
assertions for set coverage, uniqueness, and raw-diff/numstat consistency.
It overwrites only the generated inventories and `05-master-ledger.md` within
this directory. It does not regenerate the prose or the one-time reference
census, validation transcript, or final checksums.

In a clone containing the fork endpoint, fetch the pinned upstream object if
necessary, then run:

```bash
git fetch --no-tags https://github.com/ggml-org/llama.cpp.git 5c8854fd88319125125dc4b836c98b1b215b0f23
python3 docs/history/reproduce.py
git merge-base 2f623c6607c078d1069dac23e9d8af9cb146e771 5c8854fd88319125125dc4b836c98b1b215b0f23
git diff --no-renames --shortstat 5c8854fd88319125125dc4b836c98b1b215b0f23 2f623c6607c078d1069dac23e9d8af9cb146e771
```

The fetch adds objects/FETCH_HEAD; the generator leaves source and branch
history alone. The original run used a dedicated review ref to retain the
upstream objects. An existing `upstream` remote is not required.

For a specific claim, inspect the exact patch and consumers, for example:

```bash
git show 61eed0aac -- ggml/include/ggml.h
git show 4bb688b3d -- ggml/src/ggml-sycl/fattn-mkl.cpp
git diff 4ed65c1cd^ 81c941664 -- ggml/src/ggml-sycl
git show 005bf5c01 -- ggml/src/ggml-sycl/moe-cache.cpp
git show 8ae386707 -- ggml/src/ggml.c
git grep -n 'slot_cache_key\|get_slot_by_cache_key\|cache_key' 2f623c660 -- common tools/server
```

## 7. Validation and limits

The local validation transcript records generator self-checks and deterministic
regeneration, parse/count checks of inventories, ASCII/whitespace checks,
relative-link and cited-object checks, and the existing upstream-sync invariant
script. That script passed, even though F3/F4/F6 remain in the tree. This is
itself a useful boundary: an invariant grep gate protects named surfaces,
not every semantic contract.

No production source was changed and no historical error was repaired by this
task. No CPU/SYCL/Vulkan/OpenVINO build, CTest suite, model run, GPU benchmark,
browser/server execution, old-file compatibility run, or live vulnerability
reproducer was performed. No service was stopped, runtime stack changed,
`lib` branch regenerated, or history rewritten. Evidence collection did not
create commits or push remotes; the later user-requested documentation commit,
branch push and pull request are publication steps, separate from that review.
All current performance claims would require fresh matched binaries, models,
driver/runtime state, resource ownership, and execution-path evidence.

This report's residual risk is incompleteness of semantic inspection across a
large evolving fork. Its value is a reproducible historical account, a clear
current comparison, and specific source-backed follow-ups; it is not a
correctness certification of master.

## 8. Performance/functionality extension

Chapters 7-11 add campaign-level review of dated quality, capacity, throughput
and functionality evidence. They distinguish experiment dates, recorded build
identities, retained master integration anchors, and October archive landing.
The scope includes selected raw measurements and all 16 primary GRF products;
it does not claim every archived timing row was independently reaggregated.
Benchmark copies are not independent replications. The Hindsight chapter records
five parent-session recall queries, their coverage and 32 preserved records;
auxiliary agent recall helped locate reports but is not added to those counts.

Repository evidence takes precedence over contradicted memory summaries.
Where memory is the only recovered testimony, that limit is explicit. No new
Hindsight entries were created or existing entries changed. These frozen
excerpts are not regenerated by reproduce.py. The progression validation file
records documentation and artifact checks, not new hardware validation.
