# FA occupancy investigation - pre-registered predictions

Recorded 2026-07-25, before any measurement, so the experiment can fail honestly.

Base: `perf/fa-occupancy` at `64f5307b7` (equal to `master`; the post-P5 campaign state).
Device: Intel Arc A770, DG2/acm-g10, i915, Level Zero 1.15.38646.
Build: Release JIT, icpx 2026.0, `GGML_SYCL=ON`, `GGML_SYCL_TARGET=INTEL`, `GGML_SYCL_DEVICE_ARCH` omitted.

## Claim under test

`ggml/src/ggml-sycl/ggml-sycl.cpp:158` derives the flash-attention occupancy
governor by dividing a work-item count by an execution-unit count:

```cpp
info.devices[gpu_index].max_wg_per_cu = info.max_work_group_sizes[gpu_index] / prop.get_max_compute_units();
```

`clinfo` on this host reports `Max compute units 512` and `Max work group size 1024`,
so the expression evaluates to `1024 / 512 = 2`. `fattn-common.hpp:1248` consumes it
as `max_blocks_per_sm`.

The same two lines are present verbatim in upstream `ggml-org/llama.cpp` master.

## Derived prediction for Phase 0

Both fleet models are 32 query heads / 8 KV heads (GQA 4:1), D=128, 32 layers.
`GGML_SYCL_TARGET=INTEL` compiles `GGML_SYCL_WARP_SIZE=16`, and the VEC path uses
`nthreads = 128`, so `block_dim = (16, 8, 1) = 128` work-items.

Tracing `launch_fattn` for a single-token decode at any depth >= 4096:

| Quantity | Predicted value | Source |
|---|---:|---|
| `nsm` | 32 | `512 / 16` (`ggml-sycl.cpp:152`) |
| `max_wg_per_cu` | 2 | `1024 / 512` (`ggml-sycl.cpp:158`) |
| `ntiles_total` | 32 | `ntiles_x(1) * ntiles_z_gqa(4) * K->ne[2](8) * Q->ne[3](1)` |
| `parallel_blocks` | 2 | clamped by `max_blocks_per_sm`; search loop breaks at 100% "efficiency" |
| `blocks_total` | 64 | `1 * 2 * 32` |
| `work_items_total` | 8192 | `64 * 128` |

A770 residency is 32 Xe-cores x 16 XVE x 8 hardware threads x SIMD16 = 65 536
work-items, so 8 192 work-items is approximately **12.5 % occupancy**.

The search loop at `fattn-common.hpp:1276` is predicted to exit after its first
iteration: at `parallel_blocks_test = 2`, `nblocks_total = 64` equals
`blocks_per_wave = nsm * max_blocks_per_sm = 64`, giving `nwaves = 1` and
`efficiency_percent = 100`. The next iteration then satisfies
`efficiency_percent_best >= 95 && nwaves > nwaves_best` and breaks.

## Decision rule

- Observed `parallel_blocks == 2` and `blocks_total == 64` -> diagnosis confirmed; proceed to Phase 1.
- Any other value -> the diagnosis is wrong. Halt, re-open the analysis, and re-rank the program.
  Phases 1 and 2b lose their justification; Phases 3-6 survive on independent evidence
  but their order changes.

## Phase 1 predictions (recorded now, measured later)

With `max_wg_per_cu` corrected to 8:

1. `parallel_blocks` rises to 8, `blocks_total` to 256, `work_items_total` to 32 768.
2. d16384 f16 tg128 improves; effective requested-KV bandwidth moves off the
   124 GB/s recorded in P5.4.
3. **The q8_0-vs-f16 gap narrows.** This is the falsifiable core. If the gap does not
   narrow, the dequant-ALU-wall hypothesis is correct and a genuine split-KV q8_0
   kernel rewrite becomes justified rather than premature.
4. Depth 0 is neutral, because `parallel_blocks` is clamped by `ntiles_KQ` which is
   small at shallow depth. Any depth-0 regression below -2 % fails the shallow guard.

## Phase 0 result - CONFIRMED

Measured 2026-07-25 on the build above. Mistral-7B-Instruct v0.1 Q4_K_M,
`llama-bench -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 32 -d 16384`, `GGML_SYCL_FA_PROFILE=1`:

```
GGML_SYCL_FA_PROFILE: route=VEC layout=canonical launches=1056
  conversion_us=0 conversion_bytes=0 stage1_us=3650250 combine_us=76821
  gqa=4 repeated_packed_kv_bytes=111286419456
  parallel_blocks=2 ntiles_total=32 blocks_total=64 work_items_total=8192
  max_wg_per_cu=2 nsm=32
```

Every one of the six predicted fields matched exactly. The diagnosis is confirmed:
8 192 of ~65 536 resident work-items, approximately **12.5 % occupancy**.

Two findings beyond the prediction:

**1. The 64-work-group ceiling is universal across decode routes, not specific to VEC.**

| Route | KV | `parallel_blocks` | `ntiles_total` | `blocks_total` | `work_items_total` |
|---|---|---:|---:|---:|---:|
| VEC | q8_0 | 2 | 32 | 64 | 8192 |
| TILE | f16 | 8 | 8 | 64 | 8192 |
| VEC | f16 (forced) | 2 | 32 | 64 | 8192 |
| TILE | q8_0 prefill | 2 | 4096 | 8192 | 1048576 |

All three decode configurations converge on exactly `blocks_per_wave = nsm * max_wg_per_cu = 64`.
TILE reaches it through `ncols2 = 4` GQA packing (`ntiles_total` 32 -> 8) combined with an
8-way KV split; VEC reaches it through 32 tiles at 2-way split. The efficiency-search loop
at `fattn-common.hpp:1276` demonstrably works - it grew TILE's `parallel_blocks` from 2 to 8
to reach the target - which confirms 64 is precisely the ceiling it is aiming at.

This **corrects** an earlier claim that TILE outperforms VEC at depth because it has a
less-starved grid. It does not; the grids are identical in size. TILE's deep-context
advantage comes from work distribution instead: each work-group walks one eighth of the
KV rather than one half, and four query heads share each K/V load.

Prefill is unaffected - it launches 8 192 work-groups and is not occupancy-limited.

**2. The combine kernel is cheap, which bounds the risk of raising `parallel_blocks`.**

Over 1 056 decode launches: `stage1_us = 3 650 250` versus `combine_us = 76 821`. Combine is
**2.1 %** of flash-attention time. Raising `parallel_blocks` from 2 to 8 scales combine by
about 4x, to roughly 8 % of FA time, while giving stage 1 four times the memory-level
parallelism. The trade is heavily favourable unless stage 1 fails to improve.

### Measurement caveat

The render node was not under sole tenancy during these runs (`codium-insiders` and
`qoder` held `/dev/dri/renderD128`). Launch geometry is a pure function of device
properties and tensor shapes and is unaffected by contention, so the Phase 0 conclusion
stands. The throughput figures observed alongside it (q8_0 VEC 6.88 t/s, f16 forced-VEC
13.99 t/s, f16 TILE 14.83 t/s at d16384) are **not** promotion evidence and must not be
compared against the campaign baseline below. Phase 1 requires sole tenancy.

## Baseline to beat

Post-P5 phase-7 held numbers, Mistral-7B-Instruct v0.1 Q4_K_M, q8_0/q8_0 KV:

| Depth | tg128 tok/s | Effective KV GB/s |
|---:|---:|---:|
| 0 | 24.60 | - |
| 4096 | 17.93 | 20.46 |
| 8192 | 14.12 | 32.22 |
| 16384 | 9.93 | 45.32 |
