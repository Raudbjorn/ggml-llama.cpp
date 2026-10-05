# Subagents: roster and shared contract

Project subagents live in `.claude/agents/`. Each one owns a domain of this fork and carries
that domain's invariants, gates and known failures. This file is the contract they all share:
every agent reads it before starting. On a branch that predates it, read
`git show origin/master:docs/development/agents.md`.

Subagents cannot start other subagents. When an agent needs work outside its domain, it
stops and returns a brief for the agent named in the roster, and the main session dispatches it.

## Roster and routing

| Agent | Use for | Not for |
| --- | --- | --- |
| `sycl-backend-engineer` | How the SYCL backend computes: `ggml/src/ggml-sycl/`, FA routing, MMVQ/MMQ, set_rows/cpy/WHT kernels, quants-first q8_0 KV, graph replay, fusion, xe KMD defaults, MoE cache and expert prefetch | Changing what a turbo type means (turboquant-engineer) |
| `turboquant-engineer` | What the codec is: type slots and ABI, block layouts, CPU reference quantizer, centroids and WHT signs, `GGML_OP_TURBO_WHT` and its graph wiring, KV-cache type policy, InnerQ, quality thresholds | Kernel speed (sycl-backend-engineer); Vulkan shaders (backend-parity-engineer) |
| `backend-parity-engineer` | Vulkan (turbo shaders included), OpenVINO, CPU and BLAS deltas, `supports_op` parity with upstream, op support tables | SYCL (sycl-backend-engineer) |
| `model-spec-engineer` | `src/models`, conversion and gguf-py, model loading, speculative decoding, MTP heads, the server's draft and accept loops | A brand-new architecture: the main session runs `skills/add-new-model/SKILL.md`, which is interactive |
| `server-engineer` | `tools/server` HTTP layer, CORS and MCP proxy policy, `tools/ui` build, server-only flags | Draft and accept loops (model-spec-engineer) |
| `upstream-porter` | One upstream ggml-org or TheTom PR or commit onto a named branch | Whole-tree syncs (upstream-sync-lead) |
| `upstream-sync-lead` | Whole-tree syncs per `upstream-merge.md`: inventory, audits, conflict clusters, verification ladder; lib branch and API/ABI checks | Hand-resolving markers (merge-conflict-resolver) |
| `merge-conflict-resolver` | Resolving conflict markers in one disjoint file cluster during a merge | Anything else |
| `a770-benchmarker` | Numbers: throughput, PPL/KLD, capacity, cold JIT; hardening the harnesses | Pass/fail checks (verification-runner) |
| `verification-runner` | Pass/fail: builds, the CPU-vs-SYCL oracle, `test-backend-ops`, ctest, bisects | Timing (a770-benchmarker) |
| `fork-code-reviewer` | Read-only review of a diff or branch before it has review threads | Open PR threads (pr-thread-triager) |
| `pr-thread-triager` | Open PR review threads: fix, push back or defer each one; draft replies | Fresh diffs (fork-code-reviewer) |
| `docs-research-writer` | `docs/` layout, indexes, link checks, dated research notes, doc accuracy against the code | Producing the numbers it writes down (the measuring agent); op tables (backend-parity-engineer) |
| `intel-stack-engineer` | The host GPU stack: compute-runtime, Level Zero, IGC, gmmlib, oneAPI, firmware, xe KMD, packaging of llama.cpp | Kernel code (sycl-backend-engineer) |

Routing by invariant, not by path: a change belongs to the agent whose invariant it could break.
A server flag in `common/arg.cpp` belongs to whoever owns the feature it controls. An engineer
who measured something writes the facts; `docs-research-writer` places and indexes them.

## Precedence

1. The dispatcher's brief.
2. The agent's own file (for example, `merge-conflict-resolver` never touches the index, whatever
   this contract says about commits).
3. This contract.
4. `AGENTS.md` and `CLAUDE.md`.
5. `skills/*/SKILL.md`.

Live source and live command output beat all of the above wherever they disagree on facts.

## The dispatcher's brief

A dispatch must give: the worktree path, the branch, the build directory, whether GPU use is
allowed, a `-j` cap, the commit trailer lines, and the scope. If any of these is missing, stop
and list what is missing instead of guessing.

Work happens in the worktree the dispatcher names, normally under `~/wt/<slug>`. The shared
checkout `/mnt/mrgr/strt/ggml-llama.cpp` is used by other sessions at the same time; never
commit there unless the brief names it explicitly.

## Evidence

- Order of trust: live output and current source, then dated `docs/research/` notes, then commit
  messages, prose docs and Hindsight recall.
- Never claim a build, test, benchmark or fix succeeded without tool output that shows it.
- Label every number as measured or estimated. Quote the shortest decisive line, not a log dump.
- Prove that the code path under test actually ran. Past passes were vacuous: a grouped MoE GEMM
  that never dispatched, debug lines hidden below `-lv 5`, `pgrep -f` matching its own shell.
- Grade partial fixes as partial. "Untestable" needs a five-minute probe behind it.
- Recall project history at the start with `mcp__hindsight__recall` (bank `claude-history`, tag
  `cwd:/mnt/mrgr/strt/ggml-llama.cpp`). Host memory notes live in
  `/home/svnbjrn/.claude/projects/-mnt-mrgr-strt-ggml-llama-cpp/memory/`.

## Commits

Agents may commit locally on the branch the brief names. They never push, open, merge or comment
on PRs, post anything to GitHub, or rewrite history.

1. `git -C <worktree> branch --show-current` must print the branch from the brief.
2. `git -C <worktree> diff --cached --stat` must be empty or hold only your own work.
3. `git -C <worktree> diff HEAD -- <paths>` must contain only your own hunks. If another session
   edited the same file, stop and report.
4. Stage new files with `git -C <worktree> add <new paths>` (a path unknown to git makes the next
   step fail), then `git -C <worktree> commit -- <paths>` with one logical change per commit, an
   ASCII message, and exactly the trailer lines from the brief. `Assisted-by:` is required; an AI
   `Co-Authored-By:` line is accepted on this fork.

Never run `checkout`, `switch`, `stash`, `reset`, `rebase`, `commit --amend`, `clean`, `push`, or
`gh pr checkout`. Commit only after the gates for the change have passed; list anything left
uncommitted in the report.

## GPU protocol

There is one Arc A770 (`/dev/dri/renderD128`) and one compute engine. A hung kernel resets the
card for every user, including production services.

- Run every GPU command as `flock -w <seconds> /tmp/a770.lock timeout <seconds> <command>`.
  Older sessions do not take this lock, so also check the card yourself.
- Before and after a GPU run, check `fuser -v /dev/dri/renderD128`, `pgrep -a llama-`, and
  `sudo -n dmesg | grep -iE 'xe .*(reset|hang|timeout|GuC)'`. `kernel.dmesg_restrict` is 1 on
  this host, so plain `dmesg` fails; an unreadable log means the fault gate is unavailable, not
  clean.
- Never kill a process you did not start, never stop or start a service
  (`llama-sycl.cpp.service`, `llama-gpu@*`, `llama-vulkan.cpp.service`, `plexmediaserver`), and
  never run `fuser -k`. AGENTS.md and some harness messages suggest these; this contract overrides
  them. Ask the dispatcher instead.
- Kill only the PIDs you started and confirm the card is released when you finish.
- The only sudo an agent may run is `sudo -n dmesg`.
- Vulkan0 on this host is the Ryzen iGPU. Use `GGML_VK_VISIBLE_DEVICES=1` to reach the A770.
- CPU-only binaries run under `env -u LD_LIBRARY_PATH`, because RUNPATH plus a oneAPI
  `LD_LIBRARY_PATH` loads SYCL libraries from other build dirs.

## Upstream

- Before fixing code the fork shares with upstream, search ggml-org/llama.cpp master and open
  PRs, and TheTom/llama-cpp-turboquant. Port an existing fix instead of writing a new one.
- Never write `owner/repo#N` or an upstream PR URL in a commit message, PR body or comment. It
  posts a visible backlink on the upstream PR. Write "upstream llama.cpp PR N".
- Never reintroduce a removed backend (CUDA, HIP, Metal, OpenCL, CANN, MUSA, WebGPU, RPC, Hexagon,
  zDNN, zenDNN and the rest), `.github/`, `ci/`, `CONTRIBUTING.md` or `flake.nix`.
- PRs, when the user asks for one, go from a branch to `master` of `Raudbjorn/ggml-llama.cpp`
  only.

## Code style

ASCII only in code, comments, commit messages and docs. Concise comments that explain why.
Reuse existing infrastructure. Read the surrounding code first. Read env knobs through
`ggml_sycl_get_env` in SYCL code.

## Known stale prose (as of 2026-10-05)

Trust the code over these lines until they are fixed:

- Fork type slots are TURBO2/3/4 = 43/44/45, TQ3_1S/TQ4_1S = 46/47 and Q8/Q5/Q6_CR = 48/49/50,
  with `GGML_TYPE_COUNT` = 51. Docs that say "43-47" or "new types after 47" are stale. Re-read
  `ggml/include/ggml.h` before relying on any number.
- AGENTS.md: the `sudo systemctl stop/start llama-sycl.cpp.service` steps, the CI workflow,
  `GGML_SYCL_DISABLE_GRAPHS`, "SYCL graph replay killed" (superseded by PR 62), MMQ "disabled",
  and its oracle section table.
- CLAUDE.md: "`ggml_sycl_fuse` fuses top-k MoE only" (FFN and RMS-norm fusions exist).
- `skills/code-review/SKILL.md`: the rule against helping write a reviewer reply cites an AGENTS.md
  rule that does not exist. It applies to review mode only; `pr-thread-triager` drafts replies
  per the user's PR workflow. Its "AI Co-authored-by is blocking" line does not match this fork.
- `skills/add-new-model/SKILL.md`: the trailer rule, and an interactive gate a subagent cannot
  satisfy.
- `/mnt/mrgr/strt/intel-stack`: stock AUR clones, a duplicate fork clone and reference documents.
  The installed GPU stack is built from `~/projects/*`.

## Report format

Every agent ends with:

- **Result** - the conclusion first.
- **Evidence** - the shortest decisive output lines, with the commands that produced them.
- **Commits** - hashes and subjects, or none.
- **Not run** - checks that were skipped and why.
- **Not claimed** - what the evidence does not prove, residual risk, new state or dependencies.
