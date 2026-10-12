# Integration history: ancestry, snapshots, merge damage, and repairs

Review endpoint: fork master `2f623c6607c078d1069dac23e9d8af9cb146e771`,
compared with fetched upstream master
`5c8854fd88319125125dc4b836c98b1b215b0f23` on 2026-10-11.
This chapter reviews Git objects and source changes; historical build claims are
identified as such and were not rerun for this documentation review.

The central finding is that integration has repeatedly changed the fork's
behavior without obvious ancestry changes. A feature's original commit being
reachable does not establish that the feature survived. Conversely, newer
upstream content can be present even when the corresponding upstream commits
are not ancestors. At least one concrete omission remains at the endpoint:
the cache-key slot-selection implementation was removed in PR 50 while its CLI
options survived.

## 1. Three different meanings of divergence

### First retained non-upstream lineage

The earliest upstream-unique commit on the endpoint's first-parent chain is
`0b37448748d57191c99b7d43969f08c66cec5798`, "WIP: add TurboQuant KV cache
types (turbo3, turbo4)." Its parent is upstream commit
`b3d758750a268bf93f084ccfa3060fb9a203192a` (Vulkan im2col optimization).

The dates expose an inherited history rewrite:

| Object | Author timestamp | Committer timestamp |
| --- | --- | --- |
| `0b3744874` | 2026-03-24 21:51:55 -05:00 | 2026-04-15 14:42:34 -05:00 |
| Parent `b3d758750` | 2026-04-15 19:04:51 +02:00 | Same |

The March authorship date predates the April upstream parent. It records the
age of the original work, not the date when this exact commit topology was
created. These are inherited TurboQuant changes, not evidence that this fork's
Arc A770 development started in March.

### The SYCL integration boundary

`3faf37b41cdb8a2938bbfa666f6f29c59e3bf54b`, dated 2026-06-25 23:44:24 UTC,
merged the SYCL implementation from the FellypeMelo-derived branch. Its parents
are `a33ef00b13476e9c609caecc3c1c015b8615011d` and
`7eb9efe7268cfb444d8b95562c8d29d77babf047`. Relative to its first parent,
37 files changed, with 2,073 insertions and 184 deletions. That is a useful
product boundary for this fork's SYCL specialization, separate from the older
TurboQuant ancestry.

### Latest shared upstream ancestor

The current merge base with fetched upstream is
`81bc6b83f827df746eb129235488d325c49cae52`, dated 2026-09-26,
"jinja : implement sameas test." It entered through PR 55. This is the newest
shared ancestor, not the fork's original divergence point.

The later October snapshot did not move this merge base. Its source commit
`7049ff0cbeb1f5ead231de4522af6b75d8d773c0` is not an ancestor of fork master.
The resulting upstream distance must therefore be interpreted together with
content diffs; it is not a count of all missing upstream changes.

## 2. Integration map

Dates below are commit dates. Change counts are measured against the first
parent with `git diff --no-renames --shortstat`; they measure integration
surface, not the number of independently designed fork features.

| Date | Master or inherited integration | Mechanism | Net first-parent surface |
| --- | --- | --- | --- |
| 2026-04-23 | `67559e580`, upstream b8871 | Two-parent merge in inherited TurboQuant lineage | 263 files; +12,245/-8,521 |
| 2026-05-18 | `2b61ea24e`, TheTom PR 146 | Merge of upstream/MTP integration branch | 1,727 files; +174,612/-128,707 |
| 2026-06-16 | `35ac80d55`, TheTom PR 182 | Merge of upstream catch-up branch | 778 files; +79,385/-30,902 |
| 2026-06-25 | `3faf37b41`, SYCL TurboQuant | Two-parent backend integration | 37 files; +2,073/-184 |
| 2026-07-25 | `61eed0aac`, fork PR 30 | One-parent squash of snapshot integration | 939 files; +110,731/-19,625 |
| 2026-09-03 | `54d02625c`, fork PR 50 | One-parent squash of sanitized snapshot and review fixes | 1,454 files; +126,191/-51,573 |
| 2026-09-05 | `b1e4170ec`, compatible TheTom integration | One-parent content import | 132 files; +18,914/-1,144 |
| 2026-09-26 | `4ce310f10`, fork PR 55 | Real merge of re-resync branch | 637 files; +60,403/-35,631 |
| 2026-09-26 | `d37dd50bb`, fork PR 56 | Real merge of TheTom-derived feature branch | 83 files; +1,804/-962 |
| 2026-10-09 | `08d0baaca`, fork PR 98 | One-parent squash despite its "Merge" title | 352 files; +25,803/-18,121 |

The inherited June catch-up contains merge
`d75ef87aee9b8993759edbbc49f9f458c9db74c1` with parents
`7985f6b90bf19881ab7c7a8444954e91cae36056` and
`1fd6dfe9f3d4b69cce101d832339fbda2d14b056`. The repository's older
[turboquant-upstream-merge-notes](../development/turboquant-upstream-merge-notes.md)
describe that older, multi-backend lineage. Their statements about Metal/CUDA,
the active `feature/turboquant-kv-cache` branch, and a stale `master` are
historical context, not current policy.

## 3. PR 30: a real serialized-type compatibility break

Master commit `61eed0aac843210f0002ed90d68f2e21e95ce8a0` has one parent,
`7be8ce58c88f783cd73b8ea64b77f5f14aa9fd06`. The integration branch had an
actual merge, `eea7aa5b9a0e9afd18c6a28dc48eaeca8eb780b9`, of a source
snapshot `f95801cec35b032a278fcf53343952892b899477`; that topology was
flattened at the master landing.

The source snapshot's message says it came from a non-Git upstream directory
and omitted dotfiles and top-level project metadata. The message describes
preserving those omissions from the base rather than interpreting them as
upstream deletions. There is consequently weaker source provenance than a
snapshot pinned to a fetched upstream commit.

The `ggml/include/ggml.h` diff directly proves the ABI change:

| Tensor type | Before PR 30 | After PR 30 and at review endpoint |
| --- | ---: | ---: |
| TURBO2_0 | 42 | 43 |
| TURBO3_0 | 43 | 44 |
| TURBO4_0 | 44 | 45 |
| TQ3_1S | 45 | 46 |
| TQ4_1S | 46 | 47 |

Upstream Q2_0 occupied 42. PR 30 moved every listed private type up one and
added a comment explicitly saying private types were renumbered on upstream
syncs. Because GGUF records numeric tensor types, this is a format
compatibility break, not merely a C++ naming change. An older tensor numbered
43 can now be interpreted as TURBO2_0 instead of TURBO3_0; old TQ weight files
have the same ambiguity. The history reviewed here does not establish a
complete migration or compatibility bridge for those old serialized files.

The current source comment instead says to preserve serialized fork IDs.
The [current merge runbook](../development/upstream-merge.md) fixes slots
43-50 and requires a separately designed migration before future renumbering.
That protects future integrations; it does not retroactively make the July
transition compatible.

The PR also repaired a SYCL helper-name mismatch (`get_sycl_env` versus
`ggml_sycl_get_env`) and a full logging-queue shutdown deadlock, according to
its squashed subcommit record. Those explanations should not be confused with
proof that every intermediate snapshot built or was tested.

## 4. The August integration attempt: a useful repair that did not all survive

The retained branch `origin/sync/upstream-llama.cpp-2026-08-16` contains
`6d8312b8e01df0b9d16492e975481351c6ce8788` and
`c1732e37f8a2c055c8cf3afa3f0bf492512559c5`. The latter is dated
2026-08-16 12:38:11 UTC and is not an ancestor of the reviewed master.

Its actual diff restores:

- Cache-key slot lookup, key-to-slot binding, request routing and associated
  server guards in `tools/server/server-context.cpp`.
- Speculative embedding requirements through `need_embd` and
  `need_embd_nextn`, including declarations and callers.
- Signal-exit handling in vendored `subprocess.h` and its regeneration patch in
  `scripts/sync_vendor.py`.
- Fork build documentation and missing CLI rows.

The commit explains that an earlier audit only looked for TurboQuant/SYCL
keywords in a recent commit window, missing older or unrelated fork behavior.
Its repair is concrete source evidence; its account of the earlier manual
`--theirs` operation is historical developer testimony. The worktree/index
state before that operation cannot be reconstructed from the final commit
alone.

This incident is the source of the runbook's line-survival lesson: inventory
all fork modifications relative to the real base, not only files that look
like the central feature. It should not be retold as though every listed
repair necessarily reached master or as though this were the September PR 55.

## 5. PR 50: broad upstream adoption, broad repair, one persistent omission

`54d02625c723d2482abe20e9856113e8a6f1afc5`, dated 2026-09-03, integrates
an upstream snapshot described as September 1 / `cd3a214b7`. It has one
parent, `ca41ef399ad791c53e17c5bccc92aa5663287bf4`. The retained
sanitized source object is `a8bdeb9a8`, created September 2. The abbreviated
`cd3a214b7` from the title does not resolve in this object database; its
identity here is recorded provenance, not an independently inspected object.

The squashed message records several review rounds. In addition to ordinary
upstream changes, the review had to recover missing graph cases, backend
dispatch, shape handling, API consistency, safety checks and test coverage.
The following source comparisons substantiate representative repairs:

| Surface | Direct tree evidence | Meaning |
| --- | --- | --- |
| Graph FFN cases | Sanitized source and final PR both have two `LLM_FFN_SITU` and two `LLM_FFN_SWIGLU_OAI_MOE` case occurrences in `src/llama-graph.cpp` | Final landing contains implementations that the intermediate integration reportedly lost |
| SYCL op dispatch | Final PR contains both `GGML_OP_LIGHTNING_INDEXER` case occurrences and the `mmap_support` property present in the sanitized source | Declaration-only preservation was insufficient; dispatch/properties had to survive too |
| OpenVINO REPEAT | Pre-PR source has ShapeOf/Divide handling; sanitized upstream has neither; final PR restores both | Fork dynamic-shape behavior survived the final PR 50 tree |
| Installed CMake package | Pre-PR uses `LLAMA_INSTALL_VERSION` and no `llama-config-version.cmake`; final PR uses the conventional version-file name | Repairs consumers that use versioned `find_package` |
| Vendor signal handling | `WIFSIGNALED` remains in final vendored source and regeneration script | Repair is present at landing and endpoint |
| Ngram hard-off latch | `n_dead`, `off`, threshold update and generation reset remain in final `common/speculative.cpp` | Fewer keyword hits after integration do not mean the latch was deleted |

PR 50's message also records a mistaken review fix: composite DSA cache
rollback was added and then reverted after investigators found sequential
leaf forwarding was the established behavior in other composite memories.
The stated reason was consistency and scope, not proof that composite restore
operations are transactional. This add/revert survives in the squash message;
it does not appear as two master commits.

The message admits a second verification mistake: initial checks of oneMKL FA
and fusion exercised internal helpers without proving dispatcher reachability.
The later review wired the dispatcher. This is a historical admission of an
inadequate test claim, not a new runtime measurement in this review.

### Confirmed surviving loss: cache-key slot selection

The diff `ca41ef399..54d02625c` deletes all of these from server-context:

- `cache_key_slots` and the loaded similarity/minimum-prefix settings.
- `get_slot_by_cache_key`, similarity checking, clearing and binding helpers.
- The slot-selection branch that uses the request key before ordinary slot
  selection.
- Request parsing of `cache_key`, with `session_id` as fallback.

At endpoint `2f623c660`, a search of `tools/server` and `common` finds only
CLI options, parameter storage, the task field and its clearing operation.
There is no replacement server consumer. The following stale public options
remain in [common/arg.cpp](../../common/arg.cpp):

- `--slot-cache-key-similarity` (stored default 0.5).
- `--slot-cache-key-min-prefix` (stored default 32).

Their setters update fields in `common_params`, but no server implementation
reads those fields. `server_task::cache_key` also remains without the former
request population or lookup path. This is a concrete residual mismatch
between advertised behavior and code, even if the broader prompt-similarity
cache still works. No runtime server test was needed to establish that these
specific fields lack consumers; user-visible timing effects were not measured.

The historical preservation note says cache-key lookup survived the older
June merge, and the August branch repaired it again. Neither statement makes
it present after PR 50. Follow-up should either restore a deliberate supported
contract or retire the inert options and field with clear compatibility notes.
This report changes no implementation.

## 6. September TheTom import: reconcile content, then repair interfaces

`b1e4170ec` was authored September 5 at
20:17:49 UTC and committed at 20:38:00 UTC. Its sole parent is PR 50.
The commit records TheTom source `80007e71526b2566bda62a8fc98f68c4d231139c`,
content base `15586e2d7165570fb3aa7c26e0d442e289ef69de`, and reviewed PR
head `bb4e5e44e`. The reviewed-head abbreviation does not resolve locally,
so that part is commit-message provenance. The source and content-base objects
do resolve. These are content provenance, not merge parents.

Three immediate repairs show why accepting a donor snapshot wholesale was
unsafe:

1. `0f3e21f5342c27c267106e80271daaeb15ad6c4e` corrects a Turbo4 FA fixture
   that assumed a 66-byte block. The fork's block is 68 bytes: two fp16 values
   precede packed indices. The test now places indices after both `norm` and
   reserved `rnorm`. A donor test can be wrong even when its name matches.
2. `5e490fe863d66a3cd58265a37e9a7ee91beca52c` moves new ConvRot
   `llama_ftype` values from 42/45/46 to 512/513/514 in C and Python and adds
   a serialization regression. These are `general.file_type` IDs, distinct
   from tensor IDs 48/49/50. The high range reduces near-term collision risk;
   it is not an upstream reservation.
3. `a551b19fea9ece7cbf70c434a927d5e08a2d6e0f` restores upstream-compatible
   `ggml_gated_delta_net` and recurrent-memory constructor signatures while
   retaining extended fork behavior in `_ext`/overloads. It adds
   `test-upstream-api-shims.cpp`. Author date is September 5; committer date
   is September 14, so a chronology by author date alone hides when this exact
   commit was recorded.

Other immediate commits preserve the Exaone MoE converter alias and remove a
foreign OSCAR2 benchmark script. The integration was explicitly selective.
The runbook warning about TheTom squash ancestry reflects this content-based
history; a later real merge does not retroactively turn earlier squashes into
normal merges.

## 7. PR 55: upstream ancestry restored, with reversed merge orientation

PR 55 master landing `4ce310f1096f29ef7e01d6408a2386f47244e648` has parents
`c83464b8d7b3971f9b06d67d8e46af134ae3bde4` (previous fork master) and
`c29dbdfd7d6742dc750522d541945948dcd42eae` (re-resync branch).

The underlying merge `181a4a8fb1af6142a7bf611dddd36fe34a3de3fe` has the
opposite orientation: first parent is upstream `81bc6b83f`; second parent is
fork `c83464b8d`. This is why assuming that "ours" always means fork is
unsafe. The PR wrapper still preserves the fork's master first-parent chain.

The underlying merge message reports 365 files with conflict markers already
staged as resolved, followed by subsystem-by-subsystem correction. It also
reports duplicate definitions/cases in otherwise automatically merged
regions. These are explicit historical incident records; the transient
365-file index is not stored in the completed merge object, so this audit
cannot independently count that intermediate state.

Final tree effects that are directly reviewable include:

- Imported upstream ancestry through `81bc6b83f`.
- Preserved TurboQuant tensor IDs 43-50 and count 51.
- Retained CPU, BLAS, SYCL, Vulkan and OpenVINO while pruning excluded
  upstream backends and their related build surfaces.
- Removed `.github/` and `ci/` in full. This was an owner policy change
  recorded by the merge, not an accidental loss to be automatically restored.
- Added an explicit merge-resolver runbook in `67a29372c`; follow-up
  `c29dbdfd7` aligns both Windows UTF-8 conversion calls' flags and
  null-checks each threadpool before freeing it.

The commit reports CPU build, UI source build and selected CPU gates. It
explicitly does not claim SYCL/Vulkan/A770 runtime coverage for that pass.
This review does not upgrade those historical limits. A repaired merge is not
proof that every preserved backend or imported model ran correctly.

## 8. PR 56: large donor ancestry, relatively small master delta

Master landing `d37dd50bb` merges `2082374be0b7d48ee78985967ba1f841d43f531f`
onto PR 55. The branch's integration merge
`e45a697f6` has first parent TheTom-derived `a3d5603d1` and second parent
fork `4ce310f10`. Its author and committer timestamps differ by 22 seconds
on September 26. Again, the donor side was HEAD during the inner merge.

The inner merge touches 3,500 files relative to its donor first parent,
whereas the master landing changes only 83 files relative to PR 55. Counting
all newly reachable commits as new fork changes would badly overstate what
users received in this PR.

The final master delta includes strided TurboQuant SYCL dequant/copy handling,
mixed FA template instances, Vulkan turbo2/turbo4 dequantization, KV/model
policy and speculative corrections. The actual inner-merge diff removes the
CUDA-only KV-stream feature from the donor branch, including its source,
flags and supporting artifacts; it is absent from master. This is deliberate
exclusion, not an abandoned SYCL streaming implementation.

Review follow-ups are visible as separate ancestors before the PR landing:

- `53191cc79`: Grok/turbo, adaptive layer validation, conversion mapping.
- `51ebb55b7`: gguf-py, SYCL FA and speculative corrections.
- `172eef11f`: fit predicate, DFlash injection, SG32 build gating, Vulkan
  rotation setup, and 126 removed duplicate backend-test lines.
- `2082374be`: final enum documentation, perplexity argument validation and
  ASCII corrections.

The explicit diff in `172eef11f` is strong evidence that merely resolving
markers had not produced a finished integration: CMake gating, Vulkan setup,
and duplicated test definitions still required corrections. The original
merge's CPU-only test statement cannot substitute for later hardware checks.

### An inherited rebase repair that removed a working Vulkan capability

TheTom commit `28c68fe740a8079f19f84114b8d82b535ccd49a7` is titled
"reconstruct supports_op after rebase merge damage." Its diff repairs a
missing rejection/brace in `SET_ROWS`, but also deletes:

```cpp
case GGML_OP_POOL_1D:
    return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
```

That commit is reachable in fork master through inherited history. PR 94,
`440e35259`, restores those exact two lines on October 5. This is both a
concrete integration omission and a caution that a commit described as a
repair can introduce another omission. The restored case is present at the
review endpoint; no Vulkan numerical execution was performed here.

## 9. PR 98: October content under a stale September title

The source snapshot `be6f46959` has sole parent `81bc6b83f`, consistent with
the runbook's synthetic-base method. It records upstream
`7049ff0cbeb1f5ead231de4522af6b75d8d773c0`, whose actual author/committer
date is October 5, 2026 (21:52:20 +02:00). The snapshot was created later
that day at 22:31:13 UTC.

The title of final master commit `08d0baaca` says "2026-09-01 sync."
That date is inaccurate for the pinned source. This is a provenance mistake
in the title, not evidence that September content replaced October content.

The final object has only parent
`a3af25005e12a96042ad690b63d810c5c136bb89`. Both the synthetic snapshot
and its recorded upstream source are absent from master ancestry. The
integration is a squash despite the word "Merge" in its title. The current
runbook describes an unsquashed two-parent final merge as its target outcome;
PR 98's master topology does not meet that particular history-preservation
property, even though its source content was integrated.

Review changes inside the squash include imatrix activation-type validation,
statistics/schema sizing, redundant FA launcher removal, model tool-support
inference, draft-buffer bounds and vocabulary validation, empty recurrent
memory rollback handling, and verification using saved batch indices. These
are not simply documentation changes despite the ambiguous sync title.
The final diffs show these corrections; the source-level observation does not
reproduce the reported CPU/browser checks.

## 10. PR 102 and disconnected docs history: branch-only incidents

These objects are retained locally but none is an ancestor of reviewed master:

| Object | UTC timestamp | Role |
| --- | --- | --- |
| `0686e0fe01a2a84a2b2fdd5e398b2f18bd4fb0b3` | 2026-10-10 02:49:28 | "research" snapshot on top of `e519a4833` |
| `3f3eccfa1` | 2026-10-10 03:00:41 | Restore changes overwritten by stale snapshot |
| `5545bb69f` | 2026-10-10 03:15:09 | Reconnect `docs/refresh` to current master |

### Exact stale-content evidence

`0686e0fe0` changes 335 paths relative to its parent `e519a4833`.
All 335 resulting path entries, including modes/blob identities or absence,
match the older PR 94 tree `440e35259`. Its older source branch object
`6a0c9766f` has exactly the same whole-tree ID as PR 94:

```text
440e35259 tree ee6eca33829486083089dfe46facf0d6eb9e687c
6a0c9766f tree ee6eca33829486083089dfe46facf0d6eb9e687c
```

This does not mean the *whole* PR 102 tree equals PR 94. Later-added files
remained, producing a hybrid of old shared files and newer additions. That
hybrid is potentially worse than a clean old checkout because declarations,
build lists and implementations can describe different revisions.

The restoration commit and ancestry repair both exactly match the then-current
master tree, including modes:

```text
e519a4833 tree 56db79fc112dba34c4ba4a330dfb8dd2b838d6dd
3f3eccfa1 tree 56db79fc112dba34c4ba4a330dfb8dd2b838d6dd
5545bb69f tree 56db79fc112dba34c4ba4a330dfb8dd2b838d6dd
```

The changed paths include speculative code, OpenVINO REPEAT and benchmark
harnesses/tests. The tree comparison proves a stale overwrite and exact
restoration; it does not require assuming that the commit subject "research"
accurately describes its contents. Master did not receive the overwrite.

### Exact ancestry-repair evidence

Before repair, `git merge-base 6a0c9766f e519a4833` returns no common ancestor,
even though PR 94 and `6a0c9766f` have identical trees. Afterwards,
`5545bb69f` has parents `6a0c9766f` and `e519a4833`, and its merge base with
`e519a4833` is `e519a4833` itself. The preserved disconnected lineage makes
it 12,335 commits ahead while the file diff is empty. This was a graph repair,
not 12,335 new code changes or an upstream catch-up.

## 11. What the history supports, and what it does not

The strongest recurring failure mode is preservation by appearance: matching
names, a clean index, reachable commits, or green helper tests were treated
as evidence that behavior survived. The history contains counterexamples to
each. The practical controls already present in the current runbook follow
from those failures:

- Compare both parent trees and the true base, including older non-TurboQuant
  fork work; every removed fork behavior needs an explicit disposition.
- Check enum values, byte layouts, callers and dispatch together.
- Compare `supports_op` case sets; the Vulkan POOL_1D incident shows why.
- Search files for unresolved markers even when the index has no unmerged
  entries; then check duplicated constructs and dispatch reachability.
- Treat backend removal and CI removal as deliberate policy, separate from
  accidental capability loss inside retained backends.
- Verify a snapshot's source date and parent topology from Git, not its title.
- Compare large replacement changes against known old trees before calling
  them new research.

This chapter does not assert that all omitted behavior has been found. The
cache-key loss demonstrates that the existing preservation narrative is not
complete. Historical claims of hardware passes, transient conflict counts,
and developer choices are useful provenance but remain weaker than a fresh
controlled execution.

## Reproduction commands

These commands are read-only. Use the pinned objects rather than moving
branch names when reproducing the findings.

```bash
git merge-base 2f623c660 refs/history-review/upstream-master-2026-10-11
git show -s --format='%H %P %aI %cI %s' 61eed0aac 54d02625c 4ce310f10 d37dd50bb 08d0baaca
git diff 61eed0aac^ 61eed0aac -- ggml/include/ggml.h
git diff ca41ef399 54d02625c -- tools/server/server-context.cpp
git grep -n 'slot_cache_key\|get_slot_by_cache_key\|cache_key' 2f623c660 -- common tools/server
git show c1732e37f -- common/speculative.cpp tools/server/server-context.cpp scripts/sync_vendor.py
git show 28c68fe74 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp
git diff 440e35259^ 440e35259 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp
git show -s --format='%H %T %P' 440e35259 6a0c9766f 0686e0fe0 3f3eccfa1 5545bb69f e519a4833
git diff --exit-code e519a4833 3f3eccfa1
git diff --exit-code e519a4833 5545bb69f
```

Not run: historical builds, UI/server execution, model inference, GPU gates or
benchmarks. No history rewrite, source fix, commit, push or service change was
performed for this chapter. Static evidence establishes topology and source
presence/absence; it does not prove numerical correctness, current performance
or absence of other integration regressions.
