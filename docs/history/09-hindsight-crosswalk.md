# Hindsight: evidence, contradictions, and historical gaps

Hindsight adds observations and abandoned leads that Git cannot retain by itself.
It is also a lossy secondary record: dates can describe the conversation rather
than the experiment, observations from different checkouts can be combined, and
an old conclusion can survive its retraction. This chapter records the useful
matches and disagreements, rather than treating recall as a source of truth.

## Retrieval and preservation

The preserved corpus derives from five parent-session read-only `recall` calls
on 2026-10-11 against bank
`claude-history`, with `budget=high` and token limits 6,500 for queries 1-3 and
5,000 for queries 4-5. No tag filter was used: early OMP records often have no
tags, and later experiments span several working directories. The queries were:

1. `llama.cpp TurboQuant SYCL A770 March April June July 2026 benchmarks failures centroid LUT XMX InnerQ performance capacity`
2. `llama.cpp July August 2026 P5 P6 FA occupancy q8 quants first fusion benchmark speedup retracted exact replay f16 exactness`
3. `llama.cpp September October 2026 xe i915 pinned memory copy engine readonly retry GRF benchmarks graph MoE speculative MTP`
4. `Raudbjorn llama.cpp A770 SLM LUT minus 8 percent XMX SG8 SG16 InnerQ trivial scales scale folding unreachable replay reverted July August`
5. `llama.cpp matrix-1006 FINAL-REPORT October 6 7 2026 MKL work group 32 CCS ngram MTP benchmarks dead ends`

The calls returned 58, 59, 54, 41, and 48 records respectively: 260 result
occurrences and 228 distinct memory IDs. This is targeted retrieval, not a dump
or exhaustive search of the bank. Unrelated projects, third-party B70 numbers,
plans presented as plans, and duplicate observations were excluded from the
curated evidence set. [hindsight-progression.jsonl](data/hindsight-progression.jsonl)
preserves 32 selected records, including original text, full memory/document/chunk
IDs, timestamps, tags, and query membership. Non-ASCII text is JSON-escaped,
not translated. Contradictory text remains intact so the correction is auditable.

The abbreviated IDs below uniquely identify entries in that file. To retrieve
one again, use `get_memory` with its full `memory_id` and bank `claude-history`;
its `document_id` identifies the associated source container. No memory was
edited, deleted, or written back. The JSONL is a frozen retrieval excerpt, not
a backup of source conversations; source transcripts were not independently
replayed or authenticated in this extension. A future semantic recall can return
a different result set even with the same query.

## Crosswalk

| Event or record date | Memory ID(s) | What recall says | Repository check and disposition |
| --- | --- | --- | --- |
| June 25 | `7b3c7ed2` | PR144 turbo produced garbage; the valid run was f16 KV, 81.8 prompt / 32.5 generation t/s | Consistent with subsequent disable/recovery history. Keep these as a memory-only smoke observation: prompt length, full runtime and binary provenance are insufficient for comparison with later pp512 tables. |
| June 27 | `a84f6611`, `95a968fc` | SYCL turbo FA was a dead end; exclusive A770 runs stalled for 120/600 seconds | A historically bounded failure, not a permanent hardware conclusion. The [disable and July repair chain](01-chronology.md) establishes later recovery. |
| July 2-3 | `5cba5f10`, `e37a3670` | Private Q-slice indexing and SET_ROWS row-count fixes; coherent FA completion, non-FA garbage | Matches [dbf32f863](https://github.com/Raudbjorn/ggml-llama.cpp/commit/dbf32f863). Separate non-FA repair [0d518a2ec](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0d518a2ec) prevents carrying that failed-path verdict into later builds. July 2 observation and July 3 retained commit date are different events. |
| July 7 | `64537843` | Turbo4 concurrent capacity 3.66-4.27x; negligible N-scaling cost, described as performance | Overstated interpretation. [P1.8](../turboquant/ppl-results/p1.8-concurrent-capacity/RESULTS.md) measures configured context ceilings with startup/one-token probes. Historical context/slot sizing does not justify N independent full-sized caches or sustained concurrent throughput. See the [early chapter](07-performance-early.md). |
| July 8-11 | `9c7011cb`, `5efd005c` | Per-cache InnerQ state added; later probe has five producer publications but zero consumer/update discriminators | Fits the [partial implementation and lifetime repairs](01-chronology.md). Producer activity and a finite PPL do not prove non-identity scaling reached a reused graph. No demonstrated quality improvement is inferred. |
| July 11 | `8e590ffe`, `9d75b5b6` | First asserts a fork-owned OPT/MUL_MAT_ID regression; then retracts it after 19 quiet-device cases have identical verdicts | Record the retraction. The specifically named historical findings file was not recovered in the current docs tree during this extension. This is memory-reported resolution, not a fresh reproduction or proof all quant types pass. The later [September evaluation](../research/sycl/ngxson-branch-evaluation-2026-09-27.md) documents its own distinct failures. |
| July 12 | `4e961ee8` | Full 564-chunk PPL: Mistral f16 7.6666, q8 7.6673, q4 7.7215, turbo4 7.6878; Llama f16 7.5433, q8 7.5456, q4 7.7722, turbo4 7.6649 | Useful locator, not permission to collapse runs. Retained per-model tables include nearby but different turbo4 values; the early chapter preserves each report's identity and warns against conflating the July runs. |
| July 14 record, referring to July 7 | `fd4d68bb` | LUT -8% and reverted; SG8 XMX runs; SG16 ICE | Corroborated by [incremental results](../research/sycl/2026-07-09-a770-benchmark-results-incremental.md). The later research report argues SG16 DPAS is a hardware capability mismatch; the observed ICE and that source-based explanation are separate grades of evidence. |
| July 19 | `5ff20bb3` | Exact graph reuse preserves output but loses 80.86-86.55% q8 decode; reverted | Matches the [P5 campaign](../research/sycl/sycl-a770-p5-performance-campaign-2026-07-19.md). Memory uses development hashes; the retained [master ledger](05-master-ledger.md) identifies `b5e6e4058` and `9b1324500`. September's redesigned replay is a separate implementation. |
| July 25 | `96bb99e3` | Llama pp512 f16 1190, q8 1178, turbo3 854; turbo3 slower at depth despite smaller cache | Consistent with early performance summaries, but this is a recollection of an older build, not a July 25 same-day baseline for every branch. It supports the capacity/speed distinction, not a cumulative speed curve. |
| July 25 / August 27 | `06c017ba`, `1fd56a32`, `8f35eb20` | Narrow turbo4 centroids, absent XMX, and LUT as the best next fix | Stale against retained master: [f14f0ef10](https://github.com/Raudbjorn/ggml-llama.cpp/commit/f14f0ef10) widened centroids on July 5; [9e43bd2a5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9e43bd2a5) introduced experimental XMX; earlier records already rejected the LUT experiment. Could describe an old checkout, but cannot establish current absence or justify repeating identical work. |
| September 26 | `9330bea9` | MoE cache 19/19 applicable tests; Qwen completion with 47.1% hit rate and no dispatch/collect failures | Functionality evidence consistent with the [cache integration](01-chronology.md). A cache hit rate is not a tokens/s improvement. This record explicitly supplies correctness/functionality rather than a matched speed comparison. |
| September 29 | `12629412`, `981b60f2` | Dense 8B i915 1408/849/60.2 vs xe 1495.56/990.40/60.58 for pp512/pp2048/tg128 | Memory-reported dense comparison; a directly matching raw dense A/B was not identified in this extension. Rounded and precise records must not be treated as independent repetitions; this is a reported stack transition comparison, not a fork-commit effect. |
| September 29 / October 6 | `83670ec0`, `8b308cc9` | Earlier correction restricts xe's -20 to -29% decode result to MoE auto placement; later memory generalizes it to xe decode | Retain the earlier qualification. `llama-bench` auto placement and production off placement are different workloads. No i915 production-placement comparator was identified for that claim. |
| September 30 | `1c44d256` | Pinned CPU experts: pp512 362.8 with copy engine on, 366.0 off, versus mmap 325; decode much closer | [Software-stack report](../research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md) and archived X4 rows support a configuration-specific prefill benefit. Pinned host storage removes the file-backed trigger; it is not evidence every copy-engine defect is solved. |
| October 4 | `02166587` | MTP code 17.2-18.2 t/s vs none 13.1-13.9, prose weaker; chain/adaptive not consistently better | Consistent with [MTP research](../research/speculative/qwen4exp-mtp-correctness-2026-10-04.md) and later campaign. Different heads, prompts and builds remain separate cohorts. No B70 result is attributed to this A770. |
| October 6 to 7 | `9c307dde`, `b0dcdc7a` | First: Ornith environment axes not run. Later: Q8 Ornith three-round results, MKL off -21.7% pp@8k and WG32 +5.9% tg@8k | The later [FINAL-REPORT](../benchmarks/matrix-1006/FINAL-REPORT.md) supersedes the early gap. It does not establish every unknown shape or correctness route. |
| October 6 to 7 | `b7a8141f`, `27d044b9` | Two-prompt ngram-simple +42%; then ten-prompt aggregate loses to no speculation | Broader clean campaign explicitly withdraws the earlier recommendation. Repetition prompts still win; a universal ngram preference does not follow. Values 45.353/45.301, 45.325/42.512 and 43.854/44.045 are independently reported aggregate rates for none/mod/simple at temperatures 0/0.6. |
| October 7 | `fa779a2c`, `3b219c9f` | Both give Llama MKL-on 63.5 versus off 103.7 and Ornith on 208 versus off 118; the first says MKL harms Ornith and helps Llama | Internal sign inversion in memory `fa779a2c`. Arithmetic and FINAL-REPORT show the opposite: disabling MKL helps this Llama cell and hurts this Ornith cell. The erroneous sentence is preserved, explicitly rejected, and never used as a recommendation. |
| October 7 | `8d0fada4`, `9a2df3ba` | Shape-aware routing is a hypothesis; transfer, concurrent serving, p95/TTFT and resource effects remain open | Still a bounded follow-up. Later FINAL-REPORT includes a WG32 output check, so the memory's blanket statement that WG32 output was unchecked is too broad. Route-independent FA oracle gaps remain explicitly identified by that report. |

## What memory changes about the history

The useful progression is often a progression in confidence: disabled ->
repaired -> measured -> default promoted -> broader workload invalidates an
overgeneralization. Memory exposes the intermediate beliefs between commits.
It cannot by itself establish which binary ran, that a claimed test actually
exercised its intended branch, or that a report's experiment date equals its
later archive-import date.

Three recurring errors become visible only when these records are compared:

- **Snapshot drift:** describing an old branch after master had fixed the issue.
- **Metric drift:** turning allocation capacity, cache hit rate, or a synthetic
  throughput result into a stronger end-to-end product claim.
- **Scope drift:** turning one shape, prompt class, placement, or driver result
  into a backend-wide statement.

These are findings about the retained evidence, not allegations that a particular
person fabricated experiments. Missing logs remain missing. The [dead-end
register](11-dead-ends-and-open-questions.md) states what could settle each gap.
