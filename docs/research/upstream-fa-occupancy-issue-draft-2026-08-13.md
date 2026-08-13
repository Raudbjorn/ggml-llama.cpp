# Draft upstream issue: SYCL flash-attention occupancy governor undercounts by ~8x

Status: **draft only, not filed.** Written for a human to review, edit, and decide
whether/where to file. Per repo policy this fork never pushes or opens issues/PRs against
`ggml-org/llama.cpp`.

## Source and verification

The bug and its consumer were re-verified live against upstream `ggml-org/llama.cpp`
`master` at commit `1d2869c6e54d5003f3927a79efbca0fefa034a6d` (fetched 2026-08-13 via
`raw.githubusercontent.com`), not assumed from a stale local mirror. Line numbers below are
for that commit and will drift on subsequent upstream commits - re-check before filing.

Internal evidence trail: `docs/research/fa-occupancy-prereg-2026-07-25.md` (pre-registered
prediction, Phase 0/1/2b measurement, promotion gate). This fork's fix landed via PR #35
(squashed; original branch commit `53f390a91`, "fix(sycl): grow flash-attention split-K
from one, not from the occupancy cap").

---

## Suggested issue title

`SYCL: flash-attention decode occupancy governor divides by execution-unit count instead of computing true concurrent-block capacity, undercounting occupancy by up to 8x on Xe-HPG (Arc A-series)`

## Suggested issue body

### Summary

`ggml_sycl_init()` derives the maximum number of concurrently resident flash-attention
work-groups per compute unit by dividing the device's max work-group size by its compute
unit count:

```cpp
// ggml/src/ggml-sycl/ggml-sycl.cpp:159-160 (upstream master, 1d2869c6e)
info.max_work_group_sizes[i] = prop.get_max_work_group_size();
info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
```

On Intel Arc A-series (Xe-HPG, e.g. A770: `Max compute units = 512`, `Max work group size =
1024` per `clinfo`), this evaluates to `1024 / 512 = 2`. That is not a measurement of actual
concurrent-block capacity - it conflates "maximum work-items per work-group" with "maximum
work-groups resident per compute unit," two unrelated hardware limits - and is a large
undercount versus the real per-EU thread capacity.

`ggml_sycl_flash_attn_ext_vec_case_impl()` in `ggml/src/ggml-sycl/fattn-common.hpp` consumes
this value as `max_blocks_per_sm` (line numbers below are the same upstream commit):

```cpp
// fattn-common.hpp:1047-1049
int max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
int parallel_blocks = max_blocks_per_sm;
```

For non-`stream_k` (decode) launches, `parallel_blocks` is **initialized to** the
undercounted value and the subsequent efficiency search only ever grows it
(`fattn-common.hpp:1072-1090`, `for (parallel_blocks_test = parallel_blocks; ...
parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test)`), so the search floor is wrong,
not just its ceiling. On a 4:1 GQA model at D=128, single-token decode at depth >= 4096
launches only `nsm * max_blocks_per_sm = 32 * 2 = 64` work-groups of 128 work-items each -
8,192 of the device's ~65,536 resident work-item capacity, roughly **12.5% occupancy** -
and that ceiling does not rise with context depth, so the wasted fraction of the device
grows as the per-work-group serial KV walk lengthens.

### Impact (measured on a downstream fork, same code path)

A fork of this project (`Raudbjorn/ggml-llama.cpp`) carrying this exact upstream code
independently discovered the geometry, confirmed the launch-parameter values by
instrumentation before making any change (pre-registered prediction, matched to six decimal
fields exactly), and measured the fix under sole-tenancy paired A/B (six launches/arm,
sample-zero discarded, five retained pairs, `GGML_SYCL_MAX_WG_PER_CU` env override):

Mistral-7B-Instruct Q4_K_M, Arc A770, `tg128` paired median delta from raising the governor
from 2 to 16 (16 was chosen as the best of `{2,4,8,16,32}` measured on this device; see
"Suggested fix" below for why the fix is not simply "hardcode 16"):

| depth | KV | tg128 delta |
|---:|---|---:|
| 0 | f16 | +1.62% |
| 0 | q8_0 | +0.08% |
| 4096 | f16 | +45.76% |
| 4096 | q8_0 | +18.42% |
| 8192 | f16 | +84.84% |
| 8192 | q8_0 | +38.53% |
| 16384 | f16 | **+145.96%** |
| 16384 | q8_0 | **+68.50%** |

Independently confirmed on Meta-Llama-3.1-8B-Instruct (heretic) Q4_K_M with matching-sign,
similar-magnitude deltas at every measured depth. Prefill (`pp512`) is neutral within
+/-0.6% at every depth on both models - the grid this governor controls only affects
decode's search loop, and the fix leaves prefill invariant (see "Prefill note" below).
Correctness harness: `0 GATE-FAIL` on both the unpatched and patched builds.

**Caveat on these numbers:** the fork carries other changes unrelated to this bug (a
custom KV cache row layout, additional flash-attention routing knobs). The *launch
geometry* (`parallel_blocks`, `blocks_total`, `work_items_total`, and their arithmetic
dependence on `max_wg_per_cu`) is a pure function of device properties and tensor shapes,
identical on vanilla upstream. The *throughput* numbers above should be treated as strong
evidence of the mechanism and its order of magnitude, not as a plain-upstream A/B - a
vanilla-upstream measurement is straightforward to reproduce (see below) but was not
separately run for this draft.

### Minimal reproduction (architecture-only, no build required)

1. On any Xe-HPG device (Arc A-series discrete or the built-in Arc iGPU family), run
   `clinfo` and note `Max compute units` and `Max work group size`.
2. Compute `max_work_group_size / max_compute_units`. On A770 this is `1024 / 512 = 2`.
3. This value becomes `max_blocks_per_sm` in `fattn-common.hpp`'s non-`stream_k` branch,
   which is also its search floor. For a GQA model with `ntiles_total` in the low tens
   (typical for single-token decode at GQA ratios of 4-8 and D=128), the resulting grid is
   `nsm * max_blocks_per_sm` work-groups - independently computable from `clinfo` output
   alone, no GPU run needed to establish the ceiling exists.
4. To observe the throughput effect directly: build with `-DGGML_SYCL_DEVICE_ARCH` unset
   (JIT), run `llama-bench -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 16384` on a GQA model,
   then patch `max_wg_per_cu` to a larger power of two (e.g. 16) and rebuild; compare
   `tg128`.

### Suggested fix (illustrative, not a ready-to-merge patch)

Two independent problems, both need fixing:

1. **Wrong formula.** Replace the divide with either a hardware constant appropriate to
   the architecture family, or (more portably) a small runtime probe of actual concurrent
   occupancy, rather than deriving it from an unrelated work-group-size limit.
2. **Floor-not-cap bug**, independent of (1): initialize `parallel_blocks = 1` in the
   non-`stream_k` branch instead of `max_blocks_per_sm`, so the existing efficiency search
   loop (`fattn-common.hpp:1072-1090`) can find low values too, and only use the corrected
   `max_wg_per_cu` as the loop's upper bound. This fork's fix (branch commit `53f390a91`,
   squash-merged as part of a larger PR) did exactly this and is what "prefill invariant"
   above refers to - fixing (1) alone without (2) still leaves prefill's split-K
   multiplied unnecessarily at high `max_wg_per_cu` values, since prefill's `ntiles_total`
   already saturates the device and does not benefit from more splits.

### Prefill note

Because `stream_k`'s `nblocks_stream_k = max_blocks_per_sm * nsm` also depends on the same
governor, raising it without the floor fix (item 2 above) would multiply prefill's grid
unnecessarily (observed 4x in the fork's pre-fix measurement: 8,192 -> 32,768 blocks for
one cell). Fixing only the floor bug, independent of the formula, made prefill invariant to
the governor's value in the fork's measurement - identical grid at `max_wg_per_cu` 2 and
16 - which is why the promoted fix changed both.

### Environment (fork's measurement environment, for completeness)

- Device: Intel Arc A770, DG2/Xe-HPG (`acm-g10`), driver Level Zero 1.15.38646, i915 kernel driver.
- Toolchain: Intel oneAPI DPC++/C++ 2026.0.0, Release build, JIT (no `GGML_SYCL_DEVICE_ARCH`).
- `GGML_SYCL_TARGET=INTEL`.

---

## What to check before filing

- [ ] Re-fetch `ggml/src/ggml-sycl/ggml-sycl.cpp` and `fattn-common.hpp` at the upstream
      commit current at filing time; line numbers above will have drifted.
- [ ] Decide whether to include the fork's throughput table (attributed as
      fork-measured, not upstream-measured) or ask upstream maintainers to reproduce
      independently.
- [ ] Consider filing as two smaller issues (formula bug; floor-not-cap bug) if that suits
      upstream's triage process better than one combined report.
- [ ] Optionally attach the fork's `docs/research/fa-occupancy-prereg-2026-07-25.md` as
      supporting evidence (public repo, linkable).
