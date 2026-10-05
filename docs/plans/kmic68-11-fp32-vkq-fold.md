# P11 - Per-tile fp32 fold for the F16 FA output

**Kind:** port
**Depends on:** P03, which decides whether the 2.4% is worth paying

## Purpose

Kmic-68 found that the attention output accumulated over the whole KV cache in a
`half2` register, and that the error grew as the square root of the context:

| KV length | `half2` accumulator | per-tile fp32 fold |
| --- | --- | --- |
| 4096 | 3.3e-06 | 2.9e-06 |
| 65536 | 2.8e-05 | 3.1e-06 |

Folding into fp32 once per tile kept their fast inner loop and cost 2.4%.

**This is the one Kmic-68 finding that applies to us unchanged.** Their warning
that `fattn-vec.cuh` looks like the same bug but is HIP-only does not apply: we
are the `half2` build. `fattn-vec.hpp:170` declares `sycl::half2 VKQ[...]` on the
F16 branch, which our build selects with `-DGGML_SYCL_F16=ON`, and the
accumulation is at `:480` and `:499-500`, inside the KV loop. The fp32 branch at
`:184` already uses `float2`.

## Source

- Kmic-68 `p100-docs/FINDINGS.md`, *Bound fp16 accumulation error*, plus the
  `fattn-tile.cuh` +390/-38 change.

## In this fork

- `ggml/src/ggml-sycl/fattn-vec.hpp:170` (half2), `:184` (float2), `:480` and
  `:499-500` (accumulation), `:445-451` and `:527-534` (the online rescale that
  must survive the fold).
- `ggml/src/ggml-sycl/fattn-tile.hpp`, the TILE family, same question.

## Requirements

- **R11.1** (event-driven) WHEN the FA F16 build accumulates `VKQ` across the KV loop, the <kernel> shall fold the running accumulator into fp32 once per KV tile rather than rounding the running sum to fp16 at every cell.
- **R11.2** (ubiquitous) The <fold> shall preserve the online rescaling the kernel applies when `KQ_max` changes mid-loop.
- **R11.3** (ubiquitous) The <kernel> shall remain unchanged on the fp32 path, which already accumulates in `float2`.
- **R11.4** (unwanted) IF the per-tile fold costs more than 5% of FA throughput, THEN the <change> shall be reverted and the measurement recorded in the research note.
- **R11.5** (ubiquitous) The <fold> shall not change the shared-memory footprint of the kernel.
- **R11.6** (event-driven) WHEN the fold is enabled for the VEC kernel, the <TILE kernel> shall be changed separately and measured on its own.

## Acceptance

Turbo oracle nmse and cosine at depths 4096 and 16384 with `GGML_SYCL_F16=ON`
before and after, plus FA throughput over three reps, A770, named driver. P03's
floor tells us whether the memory work or the arithmetic is the binding constraint
here.
