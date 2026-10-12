---
name: llama-fork-sync
description: Sync the Raudbjorn llama.cpp fork with newer upstream code, or merge fork master into an upstream-based branch, while preserving removed backends, fork extensions, and verification evidence. Use for whole-tree resyncs and merge preservation audits, not individual PR ports.
---

# Sync the llama.cpp fork

Perform the requested merge through resolution, preservation audit, and validation. Keep one evidence record containing pinned inputs, per-path decisions, exact commands and exit statuses, final revisions, and validation limits. Do not substitute a new merge strategy or a different checkout for the user's requested one.

This is a repository-local Claude Code skill, invoked as `/llama-fork-sync`. The Codex `upstream-sync-lead` role reads this same procedure. Paths below are relative to the assigned repository root. Read `AGENTS.md`, `docs/development/agents.md`, and `skills/code-review/SKILL.md` before editing. Use current sources when dated notes disagree.

## Establish inputs and choose the correct mode

Inspect `git status --short`, branch, worktrees, remotes, and any existing `MERGE_HEAD`. Record full commit IDs for the starting HEAD, incoming tip, authoritative fork tip, true merge base, and relevant upstream tip. Verify the remote URL before fetching; a branch named `master` does not identify the fork. Pin fetched refs once so a moving remote cannot change the resolution inputs.

Start in a clean worktree. Preserve unrelated changes; do not stash, reset, clean, or replace them. The main session can prepare an isolated worktree when necessary. For a merge already in progress, inspect its index stages and recorded inputs before proceeding; never abort and recreate it merely to match this procedure.

| Requested operation | Procedure and expected parents |
| --- | --- |
| Merge `Raudbjorn/ggml-llama.cpp:master` into the current branch, including a newer upstream-based branch | Direct merge of the pinned fork tip. First parent is the original current HEAD; second parent is the fetched fork master. Do not synthesize a snapshot or reverse the direction. |
| Normal whole-tree upstream sync starting from the fork, with no contrary history requirement | Follow the sanitized snapshot flow in `docs/development/upstream-merge.md`. First parent is the fork tip; second is the sanitized snapshot whose sole parent is the real merge base. |
| TheTom sync | Establish the last imported upstream boundary first; prior squash imports make its raw merge base misleading. Count source work against that boundary, for example `git rev-list <TheTom-tip> ^<last-synced-upstream>`, then audit already-imported hunks before preparing a sanitized delta. Apparent unique commits are not proof of missing fork content. |

For a fresh direct merge in an exclusively owned worktree, use `GIT_LFS_SKIP_SMUDGE=1 git merge --no-ff --no-commit "$INCOMING_TIP"` after setting `INCOMING_TIP` to the verified full commit ID. An already-integrated tip needs no empty merge. Inspect failures: conflicts are a pending merge, but unrelated Git errors are not. Record the decision to leave historical LFS payloads unhydrated; preserve their pointer blobs and do not bypass push integrity checks.

Determine the fork side from the pinned repository/ref, never from `ours`/`theirs`. Index stage 1 is the base, stage 2 is the starting HEAD, and stage 3 is the incoming commit. In the direct mode above, a file deleted by the fork has no stage-3 entry; that absence is a deletion decision to preserve, not missing input to restore.

Read the current invariants, fork-owned surfaces, resolution matrix, and verification ladder in `docs/development/upstream-merge.md` for both modes. Its snapshot-construction commands and first-parent assertions apply only to snapshot mode. GPU locking and service rules in `docs/development/agents.md` also apply to its older command examples.

## Inventory preservation before resolving

Compare every fork-changed path from the true merge base to the pinned fork tip, including deletions and paths with no conflict. Use NUL-delimited Git output when scripting inventories; use explicit no-rename views for deletion accounting and rename-aware views to investigate relocation. Save the base/fork/upstream blob IDs and final disposition for each path.

Use these dispositions: exact fork content, preserved fork deletion, combined/adapted, upstream-selected, or deliberately retired/relocated. Explain every non-exact choice with the behavior retained and its replacement/callers where relevant. Review fork-added lines that disappear, including automatic resolutions. An ancestor commit or a surviving filename does not prove its behavior survived.

Preserve these invariants:

- Retained backends are CPU, BLAS, SYCL, Vulkan, and OpenVINO. Keep CANN, CUDA, ET, Hexagon, HIP, Metal, MUSA, OpenCL, RPC, VirtGPU, WebGPU, zDNN, and ZenDNN removed, including new upstream files under excluded directories and their CMake/registry/runtime dependencies. Historical prose, SYCL OpenCL target names, and OpenVINO dependencies are not backend registrations.
- Keep fork-deleted `.github/`, `ci/`, `.devops/`, `CONTRIBUTING.md`, and `flake.nix` absent. Derive other exclusions from the pinned fork tree and current policy; do not delete by guess or restore modify/delete conflicts in bulk.
- Preserve TurboQuant slots 43-45, TQ slots 46-47, CR slots 48-50, count 51, block layouts, and `GGML_OP_TURBO_WHT`. Check serialized IDs, type traits, quantization functions, graph wiring, KV policies, and backend dispatch together.
- Preserve fork research/archive content, plans, scripts, vendor patches, proxy policy, documentation layout, `.codex/agents/`, `.claude/agents/`, and `.claude/skills/`. Do not hydrate or reformat large historical archives just to make a check quiet.

Zero unexplained reversals of fork deletions is the completion criterion. Replacing a fork file with a newer upstream mechanism requires a named replacement and a behavior check; classify it as a deliberate retirement, not accidental loss.

## Resolve behavior across API families

Never use blanket `--ours`, `--theirs`, or a wholesale parent-tree replacement. Read each enclosing function/class and both parent versions. Resolve declarations, implementations, callers, CMake source lists, exports, tests, and fixtures as a unit. Preserve error handling and bounds checks while moving code.

The 2026-10-12 direct merge exposed these recurring integration risks; inspect current code instead of applying these past resolutions mechanically:

| Surface | What must remain consistent |
| --- | --- |
| Shared runtime | Hidden-state versus ordinary embedding inputs, context/batch structures, graph callbacks, model decisions, parser/session interfaces, and every caller. Public structure changes can require downstream rebuilds even when source compiles. |
| MoE caching | Upstream expert-copy caching and fork scheduler/provider prefetch must not silently disable each other. Check callback installation and cache activation conditions, not just whether both implementations remain in the tree. |
| Qwen4Exp/MTP | Compressed metadata, QSA sequential execution, optional head loading, dense chain support, and driver fallback. Unsupported compressed packed chains must reject before state mutation; test retry/rollback, hidden states, and quantized KV. |
| Checkpoints | Appendix writes, bounded/fingerprinted legacy sidecar reads, zero-token payloads, returned token counts, UTF-8 paths, and incompatible draft-KV rejection. Ensure legacy tests really reach the legacy reader. |
| Server/UI/parsers | Preserve proxy destination validation and native parser reasoning/whitespace regressions while adopting newer APIs. Check component replacements, imports, option registration, and startup initialization for duplication. |
| SYCL | Conversion signatures, explicit device kernel names, shared-library exports under hidden visibility, required CMake targets, and current scratch-buffer ownership. A compiler workaround must not change the public tensor enum. |

Audit automatic resolutions as carefully as marked conflicts: duplicated converter classes, metadata writes, attention scaling, test blocks, initialization, and CLI registrations can compile while changing behavior. Compare `supports_op` cases, type-trait/op-name tables, and CMake sources against both parents. Read full backend test blocks rather than fixing isolated braces. A duplicate-symbol scan is a heuristic, not a semantic proof.

When delegated, work only on assigned files. If parallel resolution is requested, the main session assigns disjoint file clusters with pinned parents, side mapping, invariants, and a resolution ledger. Workers must preserve others' edits and return out-of-scope dependencies; they do not stage the shared index, commit, or spawn more agents. The main session or exclusively assigned sync lead integrates and audits the entire result.

## Validate the final resolved tree

Use the existing scripts and the runbook's build recipes; inspect current options before running them. Choose an external build directory with adequate space and a bounded job count. Prefer a non-mergerfs build location; if unavailable, record the chosen filesystem and verify the link actually finishes. Disable stale compiler caching with `GGML_CCACHE=OFF` and empty compiler launchers. Run CPU binaries with `LD_LIBRARY_PATH` unset to avoid loading another build's SYCL libraries.

Record initial failures and final reruns separately. After a source change, rebuild affected targets and shared libraries before rerunning dependent checks. Do not reuse a green result from an earlier binary. Record the tested tree/revision and binary identity; pre-commit version metadata may name a parent commit.

1. **Static integrity:** no unmerged entries or literal conflict markers in touched active files; review `git diff --check` and `git diff --cached --check`; run `scripts/check-upstream-sync-invariants.sh`. Prove inherited whitespace with parent blobs rather than modifying unrelated archives. Recompute the final disposition ledger, deletion audit, and stable type IDs.
2. **CPU build and tests:** full Release link, `scripts/check-required-targets.sh "$CPU_BUILD" cpu`, CLI `--version`/`--help`, CPU Turbo/InnerQ and KV policy gates, and real `test-backend-ops` WHT/SET_ROWS/TQ MUL_MAT cases. Verify the CPU backend and nonzero case counts. Run the relevant CTest selection, including affected MTP/state/rollback, parser, grammar, quantization, and API-shim regressions. Record skipped and excluded tests explicitly.
3. **Server and UI:** checkpoint/proxy tests with bounded, checksum-verified fixtures and an ephemeral loopback server; UI type checks and relevant unit/client/component tests from `tools/ui/package.json`. Build clean UI source through `scripts/ui-assets.cmake` with `HF_ENABLED=OFF`; require npm success and embedded assets. A fallback download or pre-existing `dist` is not source-build evidence. Name any modified fixture bootstrap without silently bypassing test assertions.
4. **Python:** changed conversion/GGUF sources and relevant harness suites; parse changed files and check for duplicate same-scope definitions. Do not automatically download full production models to unblock a small regression test.
5. **SYCL JIT:** oneAPI `icx`/`icpx`, F16, no AOT for iteration; run `scripts/check-required-targets.sh "$SYCL_BUILD" sycl`, and build all four required targets: `test-sycl-turbo-correctness`, `test-sycl-fusion-eligibility`, `test-sycl-fattn-mkl-policy`, `test-sycl-status-propagation`. Record optional dependencies such as oneDNN that are unavailable or disabled.
6. **A770 runtime when authorized:** follow `docs/development/agents.md` GPU protocol before and after the four required tests. Every GPU command uses `flock -w 900 /tmp/a770.lock timeout 600 ...`. Inspect device users and save both `sudo -n dmesg` results; failed reads mean the fault check is unavailable. Require successful exits, oracle zero GATE-FAIL/XPASS, and no new matching GPU fault lines. Do not stop/start services or kill unrelated processes. Leave `LLAMA_TEST_TURBO_FA`, `LLAMA_TEST_FA256`, and `LLAMA_TEST_INNERQ` disabled unless separately authorized. An oracle SKIP counter does not describe explicitly disabled sections.
7. **Evidence limits:** source inspection, CPU tests, SYCL compilation, and A770 execution prove different things. Report missing Vulkan/OpenVINO, AOT, API/ABI tooling, multi-GPU, model-quality, and performance coverage explicitly. Run additional backend or production checks only when the change and task warrant them.

For every failure, distinguish a merge regression, inherited behavior, unavailable fixture/tool, and unsupported case using current evidence. Compare the implicated source and test blobs with both parents and reproduce on a parent when needed. Do not weaken assertions, delete features, or relabel failures as passes. The prior merge retained a grammar-bound failure and a quantization-snapshot mismatch only after tracing them to unchanged policy/test content; those names are not a permanent exclusion list. Fix regressions; report unresolved incompatibilities before claiming completion.

## Commit, publication, and handoff

Review the entire staged tree against both parents, including automatic resolutions, not only the initially conflicted paths. Require the expected branch, no unmerged or unrelated staged work, no unstaged merge edits, correct parent IDs/order for the selected mode, and recorded validation. In a shared worktree, stage only reviewed paths; full-index merge commits require exclusive ownership under the agent contract.

Commit only within the user's existing authorization and include the required `Assisted-by:` trailer. Do not amend, rebase, force-push, or fabricate ancestry to make the branch look synced. Verify final parents and working-tree status after committing.

A delegated sync agent returns publication to the main session. When the user requests publication, the main session can normally push the current branch and create/update a PR targeting only `Raudbjorn/ggml-llama.cpp:master`; reuse an existing PR. Verify the remote head, PR base/head, mergeability, and required-review/check state. Do not merge the PR unless requested. Never put upstream PR URLs or `owner/repo#N` references in published text; use "upstream llama.cpp PR N" to avoid backlinks.

The handoff must state the merge mode, full pinned inputs and final commit/tree, backend/deletion preservation counts, explained retirements, actual test results and inherited failures, remaining uncommitted work, and publication state. Link the per-path ledger and command logs. End the report and any PR body with **Not claimed** covering unrun checks, what the evidence does not prove, residual risks, and changed ABI/state/dependency requirements. Do not reuse prior pass counts as evidence for a new sync.
