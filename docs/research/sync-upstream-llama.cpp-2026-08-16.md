# Sync: upstream llama.cpp 2026-08-16 into the fork

Date: 2026-08-16. Branch `sync/upstream-llama.cpp-2026-08-16`, off fork `master` at `cd8193d25`.

## Base and commits

- Upstream base: `ggml-org/llama.cpp` at `10bf611e5` (2026-08-16), imported as a full-tree squash
  snapshot commit `86fb07e38` on top of the fork's last sync tip `f95801cec` (same pattern as the
  2026-07-25 sync, PR #30).
- Merge commit: `8d7096a63d` ("merge: upstream llama.cpp snapshot 2026-08-16 into the fork").
- Follow-up fix commits, in order:
  - `6d8312b8e` - repair post-merge build and correctness regressions (missing `ggml_ssm_scan`
    param, `ggml_backend_sycl_device_context` placement, dropped `gated_op_fused_*` kernels,
    and — the significant find — silent deletion of fork's TurboQuant CPU-oracle support in
    `ggml-cpu` and the Vulkan backend's turbo support, caught only because the correctness harness
    crashed on `GGML_OP_TURBO_WHT`).
  - `c1732e37f` - restore fork content missed by the same heuristic in files with no turbo/innerq
    keywords: `tools/server/server-context.cpp`'s `cache_key`-based slot-selection system,
    `common/speculative.cpp`'s `need_embd()`/`need_embd_nextn()` overrides, the
    `vendor/sheredom/subprocess.h` WIFSIGNALED patch, and `docs/build.md`'s backend-pruned structure.
  - `03ea567aa` - merge `origin/master` (`ca41ef399`, PR #48, SYCL graph-recording FA crash fix) in
    explicitly, before it could collide with this branch's heavy rewrite of `ggml-sycl.cpp` at
    PR-merge time.
  - `d1b95d550` - restore the remaining `need_embd`/`need_embd_nextn` call sites and the
    `prompt_checkpoint_restored = true` setter, dedupe a silently-duplicated `test-backend-ops.cpp`
    block, and fix two test files that inherited upstream's un-hardened default assumptions.

## Excluded backends

Unchanged from the fork's standing policy: `ggml-cann`, `ggml-cuda`, `ggml-et`, `ggml-hexagon`,
`ggml-hip`, `ggml-metal`, `ggml-musa`, `ggml-opencl`, `ggml-rpc`, `ggml-virtgpu`, `ggml-webgpu`,
`ggml-zdnn`, `ggml-zendnn`, plus `tools/rpc`. The merge reintroduced clean adds inside these dirs
(new upstream files with no fork-side conflict to flag them); a `git status --porcelain` sweep for
adds matching the pruned-path patterns caught ~152 files, removed via `git rm --cached` + disk
delete. Also stripped: a new `.github/workflows/build-wasm.yml` (entirely `ggml-webgpu`-based) and
3 reintroduced `-DGGML_RPC=ON` flags in `.github/workflows/build-cpu.yml` (confirmed via
`git show cd8193d25:...` that the fork's pre-merge version had zero RPC references).

## MCP server/proxy reconciliation

Upstream replaced the old inline `/cors-proxy` GET/POST pair in `server.cpp` with a dedicated
~820-line `server-mcp.{cpp,h}` plus a broader `server-tools`/`server-schema`/`server-stream`
subsystem. PR #36 (`5aa5c34f8`, 2026-07-26) had hardened the *old* inline surface: SSRF target
policy (`proxy_target_policy`/`proxy_host_is_or_has_suffix`/`proxy_ipv4_is_global`), CORS
credential/wildcard rules, proxied-header filtering, HF-token fallback, opt-in unknown-env
diagnostics.

Resolution: kept the fork's full `server-cors-proxy.h` (the actual SSRF-hardened proxy
implementation) wholesale over upstream's stripped version; took upstream's new
`server-tools/server-schema/server-stream` files wholesale (0 fork commits, confirmed via content
diff against the fork's real per-file delta since the last sync - not just the recent-commit
window, see "Evidence for the wider audit" below); re-threaded `params.ui_mcp_proxy_allow`
allowlisting through the new `mcp_mgr.start(params)` flow in `server.cpp`. Verified end-to-end via
`tools/server/tests/unit/test_proxy.py` (14/14 passing): SSRF rejection of numeric-loopback,
IPv4-mapped-IPv6, userinfo-bearing, and cloud-metadata targets; exact-match (not near-match)
allowlisting; malformed-URL handling; CLI-flag-overrides-environment precedence.

## Enum renumbering

Confirmed absent. `GGML_TYPE_Q2_0 = 42`, `GGML_TYPE_TURBO2_0 = 43` (unchanged from the last sync's
renumber), `GGML_TYPE_COUNT = 48`. Upstream's own type enum still tops out below 43 this round, so
no collision and no re-renumbering was forced.

## The recent-commits heuristic failure (why this sync took longer than the last one)

The dominant lesson of this sync: **"0 fork commits touching file X in the recent sync window" is
not sufficient evidence that X is safe to take wholesale via `git checkout --theirs`.** Fork content
can be baked into a file from *before* the window (an earlier merge, the original port) with no
commits landing on it since, and a keyword grep (turbo/innerq/GGML_SYCL_Q8_KV/Raudbjorn) misses any
fork feature that doesn't use those words - which is most of the security/server-side hardening.

This bit twice in this sync:

1. First pass: `ggml/src/ggml-cpu/{ggml-cpu.c,ops.cpp,ops.h}` and the Vulkan backend
   (`ggml-vulkan.cpp` + 5 shader files) lost the TurboQuant CPU-oracle and Vulkan turbo support
   entirely. Caught when `test-sycl-turbo-correctness` crashed with
   `ggml_get_n_tasks: op not implemented: TURBO_WHT`. Fixed via proper `git merge-file` 3-way
   reconstruction against the real merge-base (`1fd6dfe9f3d4b69cce101d832339fbda2d14b056`), plus a
   full-tree turbo/innerq/`GGML_SYCL_Q8_KV`/Raudbjorn keyword-count audit that caught 5 more
   silently-lost non-code files (`gguf-py/gguf/constants.py`'s enum entries,
   `tools/llama-bench/llama-bench.cpp`'s CLI parser branches, 3 doc files).
2. Second pass (this segment, prompted by an independent review before declaring the sync done):
   the keyword audit's blind spot was itself blind to anything outside the turbo/SYCL surface. Two
   commands closed the gap for good: `git diff --diff-filter=D --name-only master HEAD` (every file
   master had that HEAD lacks - all 32 hits were confirmed legitimate upstream renames, e.g. the
   webui's `constants/*.ts` -> `*.constants.ts` split, zero real loss) and, the important one,
   `git diff --name-only --diff-filter=M f95801cec master` - the fork's **complete** modification
   list since the actual last sync, independent of any recent-commit window. Cross-referencing that
   140-file list against a **line-survival check** (every line the fork added since the true
   merge-base, checked for literal presence in the post-merge file) surfaced:
   - `tools/server/server-context.cpp`: a `cache_key`-based slot-selection system whose CLI flags
     (`--slot-cache-key-similarity`/`--slot-cache-key-min-prefix`) had already survived in
     `common/common.h`/`arg.cpp` as **dead parameters** - a strong tell the consumer was gone -
     plus a `prompt_checkpoint_restored` CUDA-concurrency guard, a `check_slot_no_media()` security
     gate, and an `n_ctx_slot` rope-scaling override. Every `cache_key` method existed in the file
     but was never called anywhere - the entire feature was dead code.
   - `common/speculative.cpp`/`.h`: `need_embd()`/`need_embd_nextn()` virtual overrides across 6 of
     8 speculative-impl structs, dropped by git's **non-conflicting** auto-merge (no `<<<<<<<`
     marker at all - the recurring silent-duplication/silent-drop bug class this whole sync
     surfaced repeatedly), plus the public API and header declarations, plus
     `tools/server/server-context.cpp`'s own call sites into that API.
   - `vendor/sheredom/subprocess.h`: lost the WIFSIGNALED/WTERMSIG patch letting
     `server-models.cpp`'s `is_signaled()`/`exit_signal()` distinguish signal deaths from normal
     exits - the higher-level plumbing looked intact, so this was invisible without checking the
     vendored library itself.
   - `docs/build.md`: reverted to upstream's full document (documents CUDA/MUSA/HIP, backends this
     fork deletes) and had dropped the fork's CPU-flags/BLAS-AOCL content.
   - `tools/cli/README.md`/`tools/completion/README.md`: missing a couple of newer hand-maintained
     CLI-flag doc rows.

   `tools/server/server-context.cpp` and `common/speculative.cpp` were large enough (56 and 26
   real conflicts respectively once 3-way reconstructed against the true merge-base) that they were
   each handed to a dedicated review pass reading every hunk against `ours`/`base`/`theirs`, rather
   than resolved by pattern.

**Recommendation for the next sync**: skip the recent-commits heuristic for whole-directory
`--theirs` decisions entirely. Compute the fork's true per-file delta list
(`git diff --name-only --diff-filter=M <last-sync-tip> <pre-merge-head>`) up front and treat every
file on it as requiring a real diff read, not a commit-count check.

## Gate results

- `cmake --build` (llama-server, llama-completion, llama-bench, test-sycl-turbo-correctness): clean
  exit, zero warnings, across every fix commit in this sync (rebuilt and reverified after each).
- `test-sycl-turbo-correctness`: `== summary: 0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP ==`, both
  default and `LLAMA_TEST_TURBO_FA=1`, re-verified after the PR #48 merge.
- `test-backend-ops -b SYCL0` full sweep (previously never run this sync - added specifically
  because it was the only coverage for fork's `gated_op_fused_*` kernels, the P5 fused-FFN SwiGLU
  path, and upstream's new fusion ops): zero failures on any fork-relevant op
  (`TURBO_WHT`, `SET_ROWS_TURBO3`, `GEGLU`/`GEGLU_ERF`/`GEGLU_QUICK`/`SWIGLU`/`SWIGLU_OAI`,
  `RMS_NORM_MUL_ADD`/`RMS_NORM_MUL_ROPE`, `MUL_MAT_ID_FUSION`). Two pre-existing, out-of-scope bugs
  found and confirmed present identically in fork's pre-merge master (not sync regressions, not
  fixed here - see below).
- `tools/server/tests/unit/test_security.py`: 36/38 passing (2 pre-existing failures, see below).
  One genuine merge-caused CORS-test conflict found and fixed (see "Test fixes" below).
- `tools/server/tests/unit/test_proxy.py`: 14/14 passing after fixing one test that inherited an
  upstream assumption incompatible with the fork's SSRF hardening.
- `pre-commit` (`trailing-whitespace`, `end-of-file-fixer`, `check-added-large-files`, scoped to
  every file touched this sync): clean. `flake8` hook is pre-existing broken (dependency conflict
  between the pinned `flake8==7.0.0` and `flake8-no-print`'s pin of `flake8==4.0.1`), confirmed
  identical on `master`, not a regression.

### Test fixes (not code fixes)

Two test files carried assumptions that were valid for upstream's default posture but wrong for
this fork's intentionally hardened defaults:

- `test_security.py::test_cors_options` (pure upstream, unmodified) asserted that with no
  `--cors-origins`/`--api-key` set, the server should reflect the request `Origin` and allow
  credentials by default. The fork's own adjacent test,
  `test_cors_default_wildcard_preflight_is_non_credentialed`, asserts the opposite (wildcard
  origin, credentials off) as the intentional hardened default - and that test passes against the
  actual server. The two cannot both hold for the same config; removed the upstream test rather
  than changing server behavior, since doing the latter would reintroduce the exact risk PR #36
  closed. Its non-conflicting assertions (Allow-Methods/Allow-Headers) already duplicate the
  sibling test.
- `test_proxy.py::test_mcp_proxy_no_content` (pure upstream, references `ggml-org` issue #26598)
  proxies to `127.0.0.1` without allowlisting it, which the fork's SSRF target-policy correctly
  rejects (`400 non-global numeric target host is not allowed`) by default. Added
  `server.ui_mcp_proxy_allow = ["127.0.0.1"]`, the same pattern every other passing proxy test in
  this file already uses, so it exercises what it's meant to (204-response proxying) instead of
  tripping the SSRF gate.

### Confirmed pre-existing, out-of-scope (not caused by this sync, not fixed here)

Both found via the newly-added `test-backend-ops`/security-suite runs, both verified byte-for-byte
identical in fork's pre-merge master (`cd8193d25`) and absent from upstream, i.e. present before
this sync started and unrelated to it:

- `ggml/src/ggml-sycl/set_rows.cpp` / `ggml-sycl.cpp`: `GGML_OP_SET_ROWS`'s `supports_op()` claims
  support for `GGML_TYPE_TQ3_1S`/`TQ4_1S` (and `cpy.cpp`'s CPY path claims support for `TQ2_0`
  self-copy), but the dispatch switch never implements either - `GGML_ABORT("Unsupported tensor
  type!")`. There is even a dedicated fork test (`test_set_rows_tq4_1s`, already present at
  `cd8193d25`) exercising exactly this gap, so it reads as an incomplete feature, not a stale
  capability claim. Fixing it means writing real SYCL quantize-and-set_rows kernels for weight
  types that were never wired up - out of scope for a sync; worth a dedicated follow-up task.
- `tools/server/server-context.cpp`'s checkpoint-restore-isolation branch (`if
  (slot.prompt_checkpoint_restored || (!slot.prompt.checkpoints.empty() && near_prompt_end)) {
  add_ok = false; return; }`) can fire *after* the current slot's tokens were already added to the
  shared `batch` in the same call, skipping the `slot_batched = &slot;` assignment that follows -
  tripping `GGML_ASSERT(batch.slot_batched || batch.size() == 0)`. Reproduces on two of the
  `test_local_media_file` traversal-path parametrizations (multi-decode-iteration image+text
  prompts near the checkpoint threshold); does not reproduce on the plain single-iteration case.
  Confirmed structurally identical in fork's pre-merge master - not introduced by this merge.
  Fixing it requires a real design decision (should `slot_batched` still be set before the early
  return, or should the tokens not have been added yet) that's out of scope here.

## Not claimed

- No timing/benchmark numbers were gathered. `scripts/bench-a770-fork-unique.py` and friends were
  never run this sync; all verification here is build + correctness + functional-test, not
  performance. Do not read "0 GATE-FAIL" as a performance claim.
- `test-sycl-turbo-correctness` ran with a foreign process (VSCodium, UI compositing) holding
  `/dev/dri/renderD128` for at least part of this sync. Per CLAUDE.md this is acceptable for
  correctness (NMSE/cosine checks aren't contention-sensitive) but would not be for timing.
- `scripts/turbo-quality-gate.sh` ran in non-strict mode only; no wikitext-2 `test.raw` was
  available on this host, so the PPL and context-scaling stages SKIPPED. Only the correctness
  stage (stage 0) ran to completion.
- A full local Vulkan binary build was not achieved. `glslc`/shaderc 2026.3 fails to compile the
  MXFP4/NVFP4 OCP cooperative-matrix2 shaders (`mul_mm_cm2.comp`) with `internal error: ...
  Invalid capability operand: 4229`. Confirmed via a scratch worktree of pristine, unmerged
  upstream `master` that this reproduces identically without any change from this sync - a
  pre-existing local-toolchain limitation, not a regression. The Vulkan *source* changes in this
  sync (`ggml-vulkan.cpp`, 5 shader files, 21 conflicts hand-resolved) are therefore
  **textually reconstructed and compile-unverified** on this host; they were not reached by a C++
  compiler here. A `-DGGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT=OFF
  -DGGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT=OFF` workaround to skip the OCP FP4 shaders was not
  attempted.
- The two "confirmed pre-existing" bugs above were verified by structural comparison (identical
  code present in `cd8193d25`, fork's actual pre-merge master) rather than by building and running
  that exact commit's binary from scratch - the comparison is textual, not an independent
  before/after benchmark run.
- No attempt was made to exhaustively re-run the rest of `tools/server/tests/unit/` beyond
  `test_security.py` and `test_proxy.py` - those two were chosen because they're the security-
  hardening surface this sync's highest-risk reconciliation zone (PR #36 / the MCP proxy rewrite)
  actually touches. Other suites (router/model-swap, slot-save, speculative decoding, etc.) were
  read and reconciled by hand during the merge but not executed.
