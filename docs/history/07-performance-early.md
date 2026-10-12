# Performance and functionality progression: March-August 2026

This chapter follows the evidence behind the early fork, from the inherited
Metal/CUDA TurboQuant experiments to the A770 occupancy, layout, fusion and
quality campaigns. The principal development was not a continuous multiplication
of speedups. It was a succession of functional repairs, narrower measurements,
rejected optimizations and changes to what the measurements could establish.

The inherited Metal work made compressed attention usable and substantially
faster on the machines named in its commits. The SYCL port then needed correctness
repairs before its throughput could be trusted. July established a reasonably
controlled conventional-KV baseline and retained a small subset of a much larger
experiment matrix. July/August occupancy and quants-first work delivered large
improvements at depth; August also invalidated an earlier fusion campaign and
reopened the turbo4 quality/capacity gate with substantially better evidence.
None of these results establishes that turbo3 universally beats conventional KV,
or that the current October tree reproduces a historical number.

## Reading the dates and evidence

The audit endpoints and complete commit ledgers are in the [history index](README.md)
and [chronology](01-chronology.md). Unless explicitly described as a master date,
a date below means the experiment/report date. Author dates can precede entry onto
master by weeks: much of the March work was committed on April 15, and several
July 22 experiments reached master on July 24. A commit message is evidence of a
historical claim, not an independently repeated measurement.

Evidence is separated here as follows:

- **Commit-reported:** numbers or functionality recorded in a commit. The relevant
  commit was inspected; original raw measurements were not necessarily preserved.
- **Report-backed:** a dated in-tree report documents the setup and result, but the
  original external evidence directory may now be missing.
- **Artifact-backed:** raw or structured results remain in this repository. Some
  arithmetic was recomputed during this audit; no model or kernel was executed.
- **Static or projected:** source inspection, generated-code inspection or a model
  of performance. These do not become measured throughput by appearing in a table.

`pp` means prompt processing; `tg` means token generation; rates are tokens/second.
A depth is existing KV-cache occupancy, not necessarily the configured maximum
context. PPL is perplexity, where lower is better on the same model, corpus and
settings. An initialization ceiling is not a tested useful context length.

The July/August A770 measurements below belong to the **i915** era, generally
oneAPI 2026.0 and compute-runtime 26.22. They must not be compared directly with
September/October xe measurements. Different Llama models occur in these reports:
Llama-3.1-8B **Heretic** is not interchangeable with stock Llama-3.1-8B-Instruct.
AOT and JIT figures, short smoke tests and full campaigns, and CPU-FA and GPU-FA
quality results are also kept separate.

## 1. March-April: inherited functionality and device-specific gains

These are TheTom-lineage developments inherited by this fork, not measurements of
Raudbjorn's later SYCL backend. Most entries in this section are commit-reported.
The [chronology](01-chronology.md) covers their integration and eventual backend
pruning; this section records their performance and correctness implications.

| Development | Historical result | Interpretation |
| --- | --- | --- |
| [Metal functional path, ac157eeaf](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ac157eeaf) | A reported journey from 2.4 to 51.4 tg | Includes removal of CPU fallback. It is not a clean isolated attention-kernel A/B. |
| [Rotation correction, 7300255c8](https://github.com/Raudbjorn/ggml-llama.cpp/commit/7300255c8) | Metal prefill 739 -> 1074; PPL 6.195 | Removed an incorrect pre-RoPE Q rotation: WHT and RoPE do not commute. Also repaired rotation-state clearing. Correctness and speed changed together. |
| [Half4 work, 557b03374](https://github.com/Raudbjorn/ggml-llama.cpp/commit/557b03374) | Prefill 1074 -> 1411 | Device-specific vectorization result. |
| [Prepacked signs, cc31405c5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/cc31405c5) | Prefill 1411 -> 1424 | A smaller incremental gain within that campaign. |
| [Graph-level rotation, 4de7c1c88](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4de7c1c88) | Prefill 2095 versus 1424 | Moving rotation out of attention changed the data contract that later ports had to preserve. |
| [32-element block experiment, e5b7470af](https://github.com/Raudbjorn/ggml-llama.cpp/commit/e5b7470af) | Prefill 2747, reported 1.02x q8; PPL 6.193 over 8 chunks and 5.460 over 32; compression 4.6x versus 4.9x | Different chunk counts are not quality improvement over time. The block experiment was later reversed to 128 elements by [b9df4c3d8](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b9df4c3d8). |
| [32-element WHT group experiment, af0cb0ae5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/af0cb0ae5) | PPL 7.06 versus 6.19 and slower execution | A smaller rotation group is a different experiment from a smaller storage block. Both must not be conflated as one successful optimization. |
| [Four-magnitude Metal decode, f19c98c50](https://github.com/Raudbjorn/ggml-llama.cpp/commit/f19c98c50) | M2 Pro: 8K depth 10.95 -> 15.1, 16K 8.0 -> 11.6; M5 essentially unchanged | +38%/+45% are M2 Pro results, not universal TurboQuant gains. |
| [Sparse-V Metal, de44bfe60](https://github.com/Raudbjorn/ggml-llama.cpp/commit/de44bfe60) | M5: short 77.6, 16K 66.5 (+12.9%), 32K 57.7 (+22.8%); PPL 6.1756 unchanged, NIAH 9/9 | Quality evidence is limited to those checks. The later CUDA sparse-V path was disabled in [11a241d0d](https://github.com/Raudbjorn/ggml-llama.cpp/commit/11a241d0d); this is not a demonstrated SYCL optimization. |
| [Native CUDA TQ4 weights, 51481c3b5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/51481c3b5) | RTX 5090, Qwen2.5-7B: native 240 tg versus old path 67 in the body (68 in title), converted q8 176; VRAM 4.5 versus 7.5 GiB | Weight quantization, not KV compression. Reported PPL 7.54 versus f16 7.18 and converted-q8 7.55; KLD 0.056/0.057 versus q4 0.078; NIAH 5/5 and coherence 4/4. Raw evidence was not independently reconstructed here. |
| [CUDA multi-token TQ routing, 941d4567f](https://github.com/Raudbjorn/ggml-llama.cpp/commit/941d4567f) | dp4a path through batch 8, cuBLAS prefill above it, allocation/linkage fixes | A functionality and routing milestone; the existence of the route is not a new cross-device speedup claim. |

The graph-level WHT contract and centroid formats outlived many of the devices on
which they were first developed. CUDA and Metal numbers therefore explain design
history, but they are not measurements of the retained SYCL/Vulkan product.

### The inherited Metal dead ends were already informative

[9637c1c1f](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9637c1c1f)
recorded an unusually useful failed-experiment table. Against a 15.1 tg
four-magnitude implementation, batched extraction gave 13.7, inline block handling
13.5, deferred norm 12.9, two-pair half2 12.0, select chains 11.9, bit arithmetic
11.6, branchless FMA 11.4, named-register ternaries 10.3, and a non-vector form
10.2. The eight-entry LUT baseline was 10.95. A 24.5 no-dequantization ceiling
removed required work and was not a valid codec implementation.

Additional checks prevented a simple "CUDA trick transfers to every GPU" story:

- [6c9cfb1be](https://github.com/Raudbjorn/ggml-llama.cpp/commit/6c9cfb1be):
  split two-by-four LUTs gave M5 74.0 versus 77.4, a 4.4% regression, and exposed
  a sign-mapping mistake.
- [884da3f00](https://github.com/Raudbjorn/ggml-llama.cpp/commit/884da3f00):
  a register LUT gave 75.2 versus 77.4, with register-pressure concerns.
- [927e68a1c](https://github.com/Raudbjorn/ggml-llama.cpp/commit/927e68a1c):
  SIMD shuffle gave 14.7 versus 15.1.
- [d9ba9bf2c](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d9ba9bf2c):
  a fused block dot gave only 8.1 at 8K depth.
- [b3cd6a769](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b3cd6a769):
  M5 four-magnitude results changed sign with depth: short 76.2 versus 76.7,
  16K 60.3 versus 58.9 (+2.4%), but 32K 44.1 versus 47.6 (-7.3%).

The corresponding norm/register/inline reversals include
[35034f142](https://github.com/Raudbjorn/ggml-llama.cpp/commit/35034f142),
[73d512c37](https://github.com/Raudbjorn/ggml-llama.cpp/commit/73d512c37), and
[b596a5c6c](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b596a5c6c).
Reconsidering these ideas on a retained backend requires a new implementation and
new device-specific measurements; merely repeating their original headline gains
would omit the negative evidence.

## 2. June-early July: make the SYCL result meaningful first

The June 25 SYCL integration
[3faf37b41](https://github.com/Raudbjorn/ggml-llama.cpp/commit/3faf37b41)
was followed by the June 27 disablement
[904738505](https://github.com/Raudbjorn/ggml-llama.cpp/commit/904738505).
The usable path was a sequence of repairs, not a completed port on its first day:

- [dbf32f863](https://github.com/Raudbjorn/ggml-llama.cpp/commit/dbf32f863),
  authored July 2 and committed July 3, repaired Q-private-storage behavior and
  row indexing involving `ne01`.
- [0d518a2ec](https://github.com/Raudbjorn/ggml-llama.cpp/commit/0d518a2ec)
  repaired the non-FA path. Later prose that calls that path permanently broken
  is a stale description of a prior state.
- [f14f0ef10](https://github.com/Raudbjorn/ggml-llama.cpp/commit/f14f0ef10)
  corrected centroids on July 5. Recommending a first-time centroid correction
  after this point without checking the source repeats already completed work.
- [9e43bd2a5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9e43bd2a5)
  and follow-ups established an opt-in XMX experiment. Its existence and its
  production suitability are separate questions.

The [July 9 incremental benchmark report](../research/sycl/2026-07-09-a770-benchmark-results-incremental.md)
and its [60-record raw JSONL](../research/sycl/a770-fork-unique-2026-07-09/results.jsonl)
provide an early functional checkpoint. These were short `p64/n16/r1` smoke runs,
not a stable throughput campaign. Representative Llama-Heretic rows were:

| Path | pp | tg |
| --- | ---: | ---: |
| Upstream f16 | 327.67 | 24.83 |
| Fork f16 | 326.65 | 24.85 |
| Upstream q8_0 | 318.51 | 23.45 |
| Fork q8_0 | 324.31 | 25.29 |
| Fork pure turbo2 | 313.06 | 24.33 |
| Fork pure turbo3 | 311.75 | 23.98 |
| Fork pure turbo4 | 315.10 | 24.25 |
| XMX f16 | 157.97 | 6.72 |
| XMX q8_0 | 222.95 | 10.90 |
| XMX turbo3 | 156.49 | 6.64 |

All 60 processes completed and 15 simple Paris-answer checks were coherent. That
is useful smoke coverage, not a language-quality proof. The explicit turbo-FA
oracle recorded zero gate failures, zero unexpected passes and six expected
failures; the default invocation skipped that optional section. Turbo3/4 D=128
GQA 4:1 and 8:1 passed, turbo3 D=256 passed, and turbo2 retained expected failures.
A default oracle success alone therefore did not prove the turbo-FA path ran.

Mistral turbo3 measured 314.45 pp / 24.83 tg in the same short test. Non-FA turbo3
was functional but slower: Mistral 224.60 / 9.27 and Llama 222.96 / 9.16.
The report also recorded InnerQ producer activity, up to 17 publishes, but zero
consumer-finalized events. High-GQA Qwen auto-asymmetric K became q8_0 and did not
exercise the intended calibration path. This is not evidence of an effective
end-to-end InnerQ quality or performance improvement.

## 3. July quality and capacity: useful findings with narrower meanings

The early model reports are retained under
[Mistral](../turboquant/ppl-results/mistral-7b-q4km/RESULTS.md),
[Llama-Heretic](../turboquant/ppl-results/llama31-8b-heretic/RESULTS.md), and
[Qwen3-Coder](../turboquant/ppl-results/qwen3-coder-30b-a3b/RESULTS.md).
They include CSVs and later corrections, and must be read with the backend and
chunk count attached to each row.

| Model and measurement | f16 | q8_0 | q4_0 | turbo2 | turbo3 | turbo4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral, 564 chunks, c512 | 7.6328 | 7.6332 | 7.6913 | 8.1216 | 7.7298 | 7.6534 |
| Llama-Heretic, retained PPL rows | 7.5433 | 7.5456 | 7.7722 | 10.6345 | 8.0200 | 7.6625 |

For Mistral, the report's f16, q4 and turbo quality rows used CPU FA, while q8 used
GPU FA. Llama's conventional baselines used GPU and its turbo rows CPU FA.
These are not matched GPU-kernel quality A/Bs. Mistral's reported turbo penalties
were approximately +6.4%, +1.27% and +0.27% against f16. The small differences
between CPU/GPU reference rows support near agreement, not byte-identical output.
Earlier GPU runs with FA disabled were withdrawn from the primary comparison;
the report's explanation of the then-broken non-FA route should not be carried
forward after its repair.

Qwen3-Coder's full 564-chunk conventional results were f16 9.7022, q8_0 9.7030 and
q4_0 9.8740. The asymmetric q8-K/turbo4-V figure 8.9105 used only 50 CPU-FA chunks;
it cannot establish a quality improvement over those full-corpus GPU baselines.
Turbo2 diverged by chunk 5 and turbo3 produced NaN by chunk 8. Those failures were
material constraints on supported policy, not just slow configurations.

### Allocation ceilings are not context-quality or throughput records

The retained capacity reports recorded initialization ceilings, including:

| Model | f16 | q8_0 | q4_0 | turbo2 | turbo3 | turbo4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral | 82,304 | - | 293,888 | 524,416 | 423,680 | 311,552 |
| Llama-Heretic | 79,764 | 150,528 | 285,440 | 508,928 | 411,392 | 302,336 |

Sources: [Mistral capacity](../turboquant/ppl-results/mistral-7b-q4km/capacity-RESULTS.md)
and [Llama capacity](../turboquant/ppl-results/llama31-8b-heretic/capacity-RESULTS.md).
Some turbo2 policies automatically used boundary-V changes. These ceilings exceed
training context in some cases and do not demonstrate accurate generation across
that context, sustained filling of every slot, or a speedup. The tables' very
small "KV bytes/token" entries are not credible literal byte units and are not
reused here. Original initialization logs were not recovered in this audit;
the checked-in summaries/CSVs are the surviving evidence.

The [P1.8 concurrent-capacity report](../turboquant/ppl-results/p1.8-concurrent-capacity/RESULTS.md)
has a more consequential labeling error. It presents configured context as a
per-slot capacity while its procedure establishes startup, health and a one-token
response, not simultaneous long-context requests. For example, Mistral turbo4
configured context was 338,816 with one slot and 346,880 with 2/4/8 slots;
f16 was 92,672, 81,152, 92,672 and 88,832 respectively. These numbers must not be
multiplied by the slot count to assert delivered concurrent capacity.

The historical implementation at
[3af033e0b, llama-context.cpp](https://github.com/Raudbjorn/ggml-llama.cpp/blob/3af033e0b/src/llama-context.cpp)
computed `n_ctx_seq = n_ctx / n_seq_max` when KV was not unified, and
[server-context.cpp](https://github.com/Raudbjorn/ggml-llama.cpp/blob/3af033e0b/tools/server/server-context.cpp)
assigned slots from `llama_n_ctx_seq`. With unified KV, sharing still does not
prove that every slot simultaneously owns the full allocation. Llama's f16
configured ceiling of 84,992 across slot counts and Qwen's 27,392/19,712 for
one/two slots have the same qualification; Qwen four/eight-slot cases were not
run. Static inspection resolves the labeling error. A simultaneous-load
benchmark is required to answer the originally implied service-capacity question.

## 4. July 11: conventional SYCL parity, with an attribution correction

The [standard baseline](../research/sycl/standard-sycl-baseline-2026-07-11.md)
pinned AOT source `10a70cde8161b5f2d3a50ed6e95ea74ec862ce25`, loaded-library
provenance, sole tenancy, pp512, tg256 at depth 2048, three repetitions and
8-chunk c512 PPL. It recorded:

| Model | f16 pp / tg / PPL | q8_0 pp / tg / PPL |
| --- | --- | --- |
| Llama-Heretic | 652.368 / 25.611 / 8.0287 | 648.821 / 23.083 / 8.0334 |
| Mistral | 651.653 / 26.475 / 8.2270 | 648.891 / 23.681 / 8.2318 |
| Qwen30B-A3B Q3_K_XL | 190.324 / 14.438 / 9.5895 | 189.702 / 12.950 / 9.5700 |

A [separate matched JIT upstream A/B](../research/sycl/standard-sycl-upstream-ab-2026-07-11.md)
used fork `9a43bf4de97551bb4f30016e0f620ded18b9c795` against upstream
`1d1d9a9ed7a4f09c4225ea4cc8fd3bd1cf2c940f`, separate verified libraries,
three repetitions and 24 empty occupancy-guard files. Its tg results were:

| Model | f16 fork / upstream | q8_0 fork / upstream | q8_0 difference |
| --- | --- | --- | ---: |
| Llama-Heretic | 22.579 / 22.582 | 20.787 / 19.744 | +5.28% |
| Mistral | 23.507 / 23.460 | 21.754 / 20.388 | +6.70% |
| Qwen30B-A3B | 13.977 / 13.916 | 12.792 / 12.047 | +6.19% |

The absolute AOT and JIT numbers should not be spliced into a regression story.
The useful conclusion is the internally matched parity of f16 and a measured
q8_0 difference on that setup. The original explanation that Q-register scale
folding caused the q8 gain was corrected: q8 uses the `Q_q8_1` compile branch;
the fold was in the f16-K branch. Measurement survived, attribution did not.
The report corrections entered with
[42b304e9c](https://github.com/Raudbjorn/ggml-llama.cpp/commit/42b304e9c),
[3c6c5d73f](https://github.com/Raudbjorn/ggml-llama.cpp/commit/3c6c5d73f), and
[8b11e9f93](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8b11e9f93).

The referenced `/tmp/p41-standard-baseline-10a70cde8/` and
`/tmp/p43-upstream-ab/` directories were absent during this audit. These remain
report-backed results; no reconstruction of the original binaries was attempted.

## 5. July 19 P5: many experiments, few promotions

The [full P5 campaign](../research/sycl/sycl-a770-p5-performance-campaign-2026-07-19.md)
used base `2b5149a98`, final branch state `f8f311aa4`, i915 A770, oneAPI 2026.0
and JIT builds on ZFS. Its primary paired protocol used six launches per arm,
dropped launch zero and retained five alternating pairs. A typical keep gate
required a +3% median improvement, positive lower 95% bound and no protected
regression worse than -2%. Individual subprojects had their own thresholds.

The report is thorough but the cited original evidence root
`/home/svnbjrn/llama-p5-evidence/2026-07-19` was absent when checked in this audit.
Its statement that 8,011 files/1.873 GiB had been rescued records a prior rescue;
it does not establish their current availability. Historical branch hashes in
that report sometimes differ from the commits that ultimately reached master.
The links below use surviving master objects where available.

| P5 item | Result and disposition |
| --- | --- |
| 1: backend truth | Corrected the misleading oneDNN availability story: installed build used `DNNL=0`, and prefill GEMM was oneMKL. Retained reporting correction, not a speedup. |
| 2: benchmark harness | Fail-closed checks and 12 tests improved the ability to reject invalid experiments. Harness correctness is not model correctness. |
| 3: global 256-GRF build | All 60 full-model launches failed because the kernel work-group ceiling became 512 while RMS_NORM requested more. A narrower oracle had passed, exposing a test-coverage gap. Rejected. |
| 4: forced-VEC attribution | At depth 16,384, f16 tg 14.4451 versus q8 9.8929; requested-KV bandwidth 124.0825 versus 45.1452 GB/s. Diagnostic forced-route comparison, not default f16 versus q8 product performance. |
| 5: q8 TILE | Mistral/Llama gains grew from -5.47%/-5.46% at depth 0 to +20.40%/+20.03% at 16K. Shallow regression blocked a blanket routing default. |
| 6: graph and submission settings | Graph replay +44.55% tg, lower bound +44.13%, but pp -3.05%. Batch sizes 1/4/16/64 gave -12.31%/+20.74%/+38.45%/+47.26%. These are environment-specific arms, not all retained defaults. |
| 7: reorder/DMMV toggles | Mistral about -24%; Qwen about -3.3 to -3.6%. Existing 0/0 defaults retained. |
| 8: MMVQ geometry | Twelve builds, 24 model campaigns, 288 launches; no geometry met +3% on both models. Best common cell was approximately flat. Later generic q4 coverage showed 4x32 would request 4096 threads versus a 1024 maximum; the runner was restricted to nine valid cells. |
| 9: per-kernel device split | Cold-time reductions 19.94%/17.01% for Mistral/Llama; warm pp +92.69%/+85.22%, warm tg +1.74%/+1.05%. Retained. Compile-fast's +1.52%/-0.93% cold effect was not useful. |
| 10: direct upload | Median load 2.278841 -> 2.784785 seconds, +22.20% time; a one-token exact Blue answer did not offset the load regression. Reverted. |
| 11: quants-first q8 KV | Deep tg improvements about +8% to +21%, protected pp worst -0.39%; retained opt-in first, default only in August. |
| 12: oneDNN GPU GEMM | An isolated DNNL 3.11.3 GPU build and DPAS dumps proved execution, but pp -0.42%/-0.20%, lower bounds -1.21%/-1.37%, missed the +5% gate. No retained replacement. |
| 13: MoE reorder proxy | Combined proxy -2.43%; Q3 +2.01%, Q4 -6.61%. Q4 n1 +55.49% became n8 -60.13%. No MoE implementation justified by these dense proxies. |
| 14: exact graph replay | Four exact tokens and replay hits coexisted with tg regressions around -81% to -87%. Reverted; a separate scratch pre-grow correctness fix survived. |

The per-kernel split result deserves precision. The report claims a **warm
prefill** improvement as well as a cold-start improvement; describing it as
"cold-start only" loses recorded evidence. The warm tg increase did not meet
the +3% decode gate. Conversely, the report's cold speedup ratios +24.91%/+20.50%
and elapsed-time reductions 19.94%/17.01% use different denominators and should
not be treated as contradictory percentages. The master default is
[255a24140](https://github.com/Raudbjorn/ggml-llama.cpp/commit/255a24140).

The global-GRF result also does not prove every kernel should avoid larger GRFs.
The dump inspection found q8 D128 VEC at GRF128 with zero spill, but f16 GQA TILE
at GRF128 with 7,744-12,480 spill bytes. A targeted change has a different risk
profile from forcing the setting on every kernel.

### Quants-first was a layout contract, not just a load instruction

The q8 D128 group retained its 136-byte total: 128 quant bytes followed by four
fp16 scales. The change required agreement among stores, attention, conversion,
copying and saved KV state. P5 reported cosine at least 0.999957 and a 12-token
state round trip reproducing 836,576 canonical bytes.

| Depth | Mistral paired tg gain | Llama paired tg gain |
| --- | ---: | ---: |
| 4096 | +8.25% | +8.01% |
| 8192 | +13.81% | +13.37% |
| 16384 | +20.62% | +20.43% |

The minimum lower bounds were +7.90%/+7.71%. These are distinct from the
contemporaneous TILE gains. The implementation entered master as opt-in
[77d356251](https://github.com/Raudbjorn/ggml-llama.cpp/commit/77d356251), despite
that commit's "scale-first" wording; the actual contract places quants first.
The default flip was
[754dc99e2](https://github.com/Raudbjorn/ggml-llama.cpp/commit/754dc99e2) on August 12.
The [round-two report](../research/sycl/sycl-a770-round2-report-2026-07-26.md)
explicitly noted missing shallow-depth promotion measurements at that stage.
A deep positive result is not automatically a complete shallow/deep default gate.

Direct-upload reversal was
[8aff564de](https://github.com/Raudbjorn/ggml-llama.cpp/commit/8aff564de) ->
[1c261d32b](https://github.com/Raudbjorn/ggml-llama.cpp/commit/1c261d32b).
Graph replay reversal was
[b5e6e4058](https://github.com/Raudbjorn/ggml-llama.cpp/commit/b5e6e4058) ->
[9b1324500](https://github.com/Raudbjorn/ggml-llama.cpp/commit/9b1324500).
The separate scratch fix
[e8d0466a1](https://github.com/Raudbjorn/ggml-llama.cpp/commit/e8d0466a1)
avoided a queue wait during recording and recorded 215.79 pp / 17.02 tg on the
formerly aborting shape. Fixing the abort did not rescue the rejected replay
performance result.

### Post-P5: improvements did not compose automatically

The P5 report's post-campaign material and
[retained post-campaign results](../../scripts/perf/results/p5-post-campaign)
record July 22 experiments, several committed July 24. Held versus final q8 tg
stayed essentially unchanged: Mistral 24.601/24.597 at depth 0 and 9.931/9.933 at
16K; Llama 23.605/23.561 and 9.767/9.760. This was no new blanket default win.

Quants-first plus TILE lost the apparent benefit of simply combining winners.
Packed GQA gave +7.59% at 16K but missed its +20% gate, and gave -1.84% at 4K;
[4ed65c1cd](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4ed65c1cd)
was reversed by [81c941664](https://github.com/Raudbjorn/ggml-llama.cpp/commit/81c941664).
Regular-q8 prefetch's four arms missed the +3% target and were about -1% at depth;
[6e32d0cb5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/6e32d0cb5)
was reversed by [a1456f1f5](https://github.com/Raudbjorn/ggml-llama.cpp/commit/a1456f1f5).
Mutable graph arguments lacked the needed driver capability, and the proposed
ROPE production fusion predicate reached 0% coverage with q8 SET_ROWS. Neither
was a measured production optimization waiting merely to be switched on.

## 6. July 25-August 13: occupancy changed the depth story

The [occupancy preregistration and results](../research/sycl/fa-occupancy-prereg-2026-07-25.md)
identified a real sizing error: dividing maximum work-group size 1024 by kernel
size 512 produced a cap of two, not a useful occupancy estimate. On 32 compute
units, this gave only 64 decode work groups. Raising the cap to 16 gave 512.

The first attempt had another mistake: initializing split count from the cap
created a prefill floor. A partial sole-tenancy check measured f16 pp -15.08% and
q8 pp -3.48%. An initial contended +19.5% observation reduced to +3.93% when
rechecked, and stream-k attribution was refuted by zero counters. The corrected
implementation started with one split; decode could expand from two to sixteen,
while the checked prefill geometry remained 4096 work groups.

The accepted report used a same-binary environment A/B, cap 2 versus 16, six
launches per arm with the first discarded. Representative Mistral medians:

| Depth | f16 old -> new tg | Paired gain | q8_0 old -> new tg | Paired gain |
| --- | --- | ---: | --- | ---: |
| 0 | 23.78 -> 24.16 | +1.62% | 24.51 -> 24.53 | +0.08% |
| 4096 | 14.82 -> 21.59 | +45.76% | 17.87 -> 21.16 | +18.42% |
| 8192 | 10.76 -> 19.89 | +84.84% | 13.94 -> 19.31 | +38.53% |
| 16384 | 6.95 -> 17.10 | +145.96% | 9.84 -> 16.58 | +68.50% |

These are paired median ratios; a ratio formed from rounded displayed medians can
differ slightly. Initial Llama-Heretic cells showed +1.66% f16 and -0.11% q8 at
zero depth, and +44.13% f16 at 4K. Protected pp in the table reached -0.75%,
slightly outside the prose's claim that every cell was within +/-0.6%.

The [July raw campaign](../research/sycl/fa-occupancy-campaign-2026-07-25)
has a provenance discrepancy worth preserving. The narrative names `53f390a91`;
268 inspected selected-row observations in raw sample files identify build
`a7834a213`; Mistral and Llama provenance identify repository HEAD
`fb85b43c68b20b5a7b43195331b0c94ba8ad87a4` and matching hashes for the bench
executable and SYCL library. The first two short commit objects were unavailable
locally. This does not erase the within-binary environment comparison, but exact
source attribution cannot be made by silently choosing the narrative hash.
Recovering the actual build tree or a saved source diff would resolve it.

[August follow-up](../research/sycl/round2-decode-probes-2026-08-13.md) completed
five Llama-Heretic cells at source `80d52e708`. Recomputing the five retained-pair
medians from structured products agreed with the report:

| Cell | Paired tg gain | pp change |
| --- | ---: | ---: |
| q8_0, 4096 | +21.13% | +0.13% |
| f16, 8192 | +77.43% | -0.02% |
| q8_0, 8192 | +40.11% | +0.20% |
| f16, 16384 | +135.78% | +0.13% |
| q8_0, 16384 | +72.92% | +0.01% |

The raw products marked cells valid, but the specific environment-log assertion
was false because that key was not emitted by the backend. Active-arm environment
and effects support the experiment; it is not a positive backend-log assertion.

These gains explain why an old forced-VEC result, a default f16 TILE result and a
post-occupancy result can disagree without any of them being interchangeable.
They weakened the case for a wholesale split-KV rewrite motivated by the old
occupancy premise. They did not establish that centroid dequantization has zero
cost or that all compressed formats have the same bottleneck.

A packed-GQA retest was contaminated by another GPU workload, with roughly 2x
absolute swings. Within-pass ratios still lost. The source-derived explanation
was plausible: packing reduced query tiles, turning 512 work groups back into
128 at the same split cap; increasing splits restored work but raised combination
cost. That explanation remains a hypothesis, not a clean isolated causal proof.

## 7. August: fusion correction, quality reopening and submission context

### FFN fusion: discard the invalid July campaign

The [July 26 fusion product](../research/sycl/ffn-fusion-campaign-2026-07-26/product.md)
now explicitly says **invalid for promotion**. It used known-bad binary
`b5ef0a84b`, whose interleaved activation quantizer was corrected by `02f848c83`.
The [structured product](../research/sycl/ffn-fusion-campaign-2026-07-26/product.json)
sets `all_cells_valid` false even though individual row-valid flags remain true.
Its apparent throughput improvement is not evidence for the corrected kernel.

The August 13 promotion
[968b451e9](https://github.com/Raudbjorn/ggml-llama.cpp/commit/968b451e9)
records a new August 12 paired campaign on the fixed implementation:

| Depth | Mistral tg gain | Llama tg gain |
| --- | ---: | ---: |
| 0 | +7.81% | +7.49% |
| 2048 | +7.37% | +6.86% |
| 4096 | +6.66% | +6.95% |
| 8192 | +6.31% | +5.88% |
| 16384 | +5.78% | +5.68% |

The commit reports worst lower bound +5.27%, worst pp -1.59%, 30/30
explicit environment bindings for each model, and zero oracle failures with both
true default and explicit-off configurations. It also made the harness request
verbose output: earlier environment assertions could be vacuous. These August
numbers are commit-reported here; the invalid July JSON must not be substituted
as their raw evidence.

### Turbo4: stronger evidence for a narrower claim

The [August 13 turbo4 reopening](../research/turbo/p4.5a-turbo4-capacity-reopen-2026-08-13.md)
and [artifact directory](../research/turbo/p4.5a-turbo4-capacity-reopen-2026-08-13)
used source `ef1c35835`, i915, a pinned JIT build, Mistral Q4_K_M and stock
Llama-3.1-8B-Instruct Q4_K_M. The original binding directory was lost by July 14;
this was a new gate, not a recovered rerun of the exact Heretic campaign.
Full PPL used 564 chunks at c512:

| Model | f16 | q8_0 | q4_0 | turbo4 | turbo4 versus f16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mistral | 7.6669 | 7.6674 | 7.7229 | 7.6879 | +0.27% |
| Llama-Instruct | 7.4852 | 7.4850 | 7.7231 | 7.6051 | +1.60% |

At fixed configured context 16,384, q8 K/V each allocated 544 MiB, total 1088;
turbo4 K/V each allocated 272 MiB, total 544. This proves a **2x KV allocation
reduction for that fixed context**, not a tested 2x maximum useful context.

Seven paired deep-decode measurements compared turbo4 with **q4_0**, not q8:
Mistral mean +8.4766%, lower 95% bound +8.1765%; Llama mean +8.6780%, lower
bound +6.6785%. Arithmetic was recomputed from the retained pairs and agreed
within numerical rounding. A slow Llama q4 pair at 16.818652 tg, versus about
17.7 for most other pairs, widened the reported variance; it was not dropped
from this audit. The oracle recorded zero failures, zero unexpected passes,
six expected failures and zero skips. The GO verdict combines provenance,
short/full quality, allocation and decode checks, each with its own scope.

### Qwen low-GQA counterexample repaired the policy premise

The [August probes](../research/sycl/round2-decode-probes-2026-08-13.md)
found Qwen3-1.7B Q8 weights with GQA 2:1 was not protected by a high-GQA-only
asymmetric policy. Symmetric turbo3 K/V produced PPL 659.6625 +/- 58.08 over
eight chunks versus q8 16.6391 +/- 1.33. The family-aware predicate fix
[d74cace9a](https://github.com/Raudbjorn/ggml-llama.cpp/commit/d74cace9a)
entered August 15. The retained
[fixed turbo3 log](../research/turbo/p6.6a-turbo-k-predicate-fix-2026-08-13/ppl-turbo3-fixed.log)
ends at PPL 16.9173 +/- 1.35452, with q8 K substituted by policy.
This was a functional quality repair. The result does not vindicate pure
symmetric turbo3 on Qwen or prove every Qwen/turbo combination safe.

### Submission gains depended on the immediate-command-list regime

The August probes rechecked batch size 64 under explicit immediate-list modes:

| Regime | Depth | pp change | tg change |
| --- | ---: | ---: | ---: |
| Immediate mode 0, batch64 versus unset | 0 | +2.71% | +32.18% |
| Immediate mode 0, batch64 versus unset | 16384 | -0.12% | +21.56% |
| Immediate mode 1, batch64 versus unset | 0 | -1.95% | +0.04% |
| Immediate mode 1, batch64 versus unset | 16384 | -0.01% | -0.13% |

With both mode and batch unset, changing only to batch64 gave about +0.30% at
16K, consistent with effective immediate-mode behavior in that setup. The old
+47.26% batch result was therefore not a universal deployment recommendation.

A profile measured 20.24 tg, approximately 49.41 ms/token. FA stage 1 consumed
13.68 ms (27.7%), combine 1.59 ms (3.2%), conversion about 0.001 ms: about
15.28 ms or 30.9% together. The 9.64 ms weight bandwidth floor and 24.49 ms
remaining budget mixed measurements with a theoretical estimate. The report's
modeled repeated-KV byte counts were not hardware traffic counters. A profiling
attempt also hit a wait-during-graph-capture failure, later repaired in
[ca41ef399](https://github.com/Raudbjorn/ggml-llama.cpp/commit/ca41ef399).

## 8. Speculation: useful throughput and a retracted exactness claim

The [speculation findings](../../scripts/perf/FINDINGS.md) preserve the early
results and the later retraction. June 27 work entered master in early July,
including [4599a2b03](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4599a2b03)
and the hard-off logic
[4350558c4](https://github.com/Raudbjorn/ggml-llama.cpp/commit/4350558c4).
On Llama-Heretic q8, n256, temperature zero, c16384 and one slot, the initial
medians were code 37.96 -> 124.76 (3.29x), multi-turn 38.98 -> 89.66 (2.30x),
and prose 38.47 -> 54.25 (1.41x). The report acknowledges concurrent AOT CPU
compilation; this is not a fully idle host baseline.

The hostile-match stress used about 2% acceptance, triggered three latch trips,
and measured 22.49 -> 39.22 in that setup. It demonstrates hard-off behavior in
that test, not a guarantee that drafting always improves an arbitrary workload.

July 22 five-repeat q8 results were code 21.61 -> 89.81 (4.15x), multi-turn
22.01 -> 55.12 (2.50x), and prose 21.78 -> 35.98 (1.65x). Code and multi-turn
were target-exact; prose produced four distinct hashes and was not exact.

The more dramatic f16 prose claim, 21.53 -> 246.65 (11.46x), was retracted by
the [July 26 exactness evidence](../research/speculative/spec-f16-exactness-2026-07-26).
The rerun gave code 21.42 -> 78.15 and multi-turn 21.58 -> 51.74 with exact
outputs, but prose 21.42 -> 23.12 with three hashes. Both occupancy caps 2 and
16 retained the exactness failure, so the occupancy change did not explain it.
A warm ngram cache was a suspected explanation of the extraordinary prior speed,
not a demonstrated cause. Restoring the withdrawn claim requires the original
cache/request state and a controlled reproduction, not another citation of the
old table.

Source inspection suggested different attention routing with draft batch size
could contribute to arithmetic differences. That is not a complete causal proof
of all output nondeterminism. Finally, `llama-bench` tg does not run a drafter;
its kernel gains must not be multiplied by these application-level speedups.

## 9. Dead-end register and what could actually resolve each one

This register covers distinct early avenues found in the inspected commits and
campaign reports. It does not assert that every uncommitted private experiment
was recovered. "Closed" means the tested proposition failed in its historical
scope, not that all future algorithms bearing the same name are impossible.

| Avenue | Historical disposition | What would resolve or justify reopening it |
| --- | --- | --- |
| Pre-RoPE Q WHT | Incorrect because transformations do not commute; repaired | Static algebra and an oracle resolve correctness. Timing cannot make an incorrect transform valid. |
| Smaller WHT group | Worse quality and speed in inherited tests | A different codec/rotation design plus matched quality and throughput, not a benchmark-only rerun. |
| 32-element storage block | Initially faster, later returned to 128 | Recover original constraints and compare full quality/space/backend support before a new ABI proposal. |
| Metal register/split LUT, shuffle, branchless forms | Measured losses, sometimes sign errors | New device/compiler or materially different implementation; regenerate code/spill evidence and repeat quality/performance. |
| Metal four-magnitude universal win | Benefit reversed by M5 context depth | Model/device/depth matrix; source existence alone cannot settle portability. |
| No-dequantization ceiling | Removed required computation | Only an upper-bound diagnostic; needs a valid alternative codec to become a candidate. |
| Sparse-V transfer to SYCL | No demonstrated equivalent port | New online-softmax-compatible error analysis, late-maximum oracle, implementation and full quality/performance gate. |
| SYCL SLM centroid LUT | Reported -8% at depth and reverted; bank-conflict/spill explanation | New layout/register evidence before new timing. Do not re-propose the same CUDA translation as untried. |
| `nthreads_KQ=1` bundled with LUT | Reverted together; isolated causal contribution unclear | Recover or construct an isolated safe A/B. The bundled loss does not independently quantify each component. |
| SG16 XMX FA on DG2 | Initial repeated ICE, later hardware-capability explanation | The round-two report identifies exec-size-16 DPAS as Xe-HPC+ via IGC source. A compiler bump alone is not a justified remedy; a legal SG8/new hardware implementation is required. |
| SG8 XMX FA | Functional, much slower than VEC in July smoke | New tiling/occupancy/data-movement implementation and route-specific gates. Existing path is not absent. Decode peak-ALU projections are not a substitute for measurement. |
| InnerQ calibration | Producer evidence without finalized consumer; applicable path not demonstrated | Static producer/consumer reachability and nontrivial calibration first, then quality and speed with path counters. |
| Pure low-bit turbo on MoE/Qwen | Divergence/NaN or enormous PPL | Policy restriction already changed behavior. A new calibration/codec needs full matched PPL, not only coherent short text. |
| q8 parity caused by Q scale folding | Compile-branch inspection refuted cause | Static analysis settled that specific explanation; controlled code A/B required for the actual cause. |
| Global 256-GRF mode | Full-model work-group failures | Per-kernel allocation/geometry analysis and targeted implementation; a narrow oracle alone is insufficient. |
| Blanket q8 TILE | Deep gains, shallow losses | Depth/shape-aware routing plus correctness and shallow/deep measurements; distinguish later occupancy changes. |
| Reorder/DMMV toggles | Material losses | New kernel or genuinely different workload; repeating the old losing defaults without a mechanism is not progress. |
| MMVQ geometry sweep | No common winning cell; invalid generic geometry discovered | Legal geometry proof first, broader shapes and paired timing next. Selected-model success did not prove generic launch legality. |
| Compile-fast | Negligible/mixed cold effect | Different compiler/runtime evidence; no historical steady-state claim to recover. |
| Direct upload | Load time +22.20% | Different transfer/allocation mechanism and load-time A/B. Token equality only answers output correctness. |
| GPU oneDNN GEMM | Real DPAS, missed speed gate | New packing/caching/integration work, not merely enabling the same measured path. Prefill SDPA is a separate question. |
| Dense MoE reorder proxy | Opposite effects by type and batch | Actual MoE implementation and representative routing/expert-transfer measurements. Proxy gains alone cannot promote it. |
| Exact graph replay experiment | Correct short output but extreme slowdown | New replay/capture strategy and driver capability, with real replay counters. Separate the retained abort fix from rejected performance. |
| Quants-first plus TILE | Gains failed to compose | Isolated route/layout interaction profiling and matched paired A/B. |
| Packed GQA | Missed gate; later contended retest | Clean sole-tenancy test and work-group/combine accounting; contamination limits precise magnitudes. |
| Regular-q8 prefetch | No +3% win, slightly worse at depth | Different prefetch distance/mechanism backed by profiling; old four arms already tested. |
| Mutable graph arguments | Required capability false | Capability query on a changed stack before implementation or GPU timing. |
| ROPE fusion under tested q8 path | Predicate coverage zero | Implement a route actually reached, then require nonzero coverage counters. |
| Occupancy cap change without prefill fix | Prefill regression from split-count floor | Already corrected by starting at one split; do not attribute old broken-arm numbers to final implementation. |
| Load-width fix as speed optimization | Generated load counts identical | Compile-only evidence closed the claimed mechanism; rerun only with changed compiler/code. Keep alignment correctness distinct. |
| 136 -> 192-byte q8 padding | Adds about 41% storage; demoted proposal | New cache/transaction evidence and a memory-capacity budget; no measured win existed. |
| Persistent mega-kernel | Forward-progress/residency concerns, no implementation | Capability/residency proof and bounded design before benchmarks. |
| Per-channel K / TriAttention / true QJL | Research possibilities or naming errors, not working speedups | New calibration, cache/eviction or mathematical/ABI implementation as applicable, then full quality and performance. |
| July FFN fusion campaign | Known-bad quantizer; globally invalid product | Corrected August gate supersedes it. Do not salvage July rates by relabeling the binary. |
| f16 11.46x prose speculation | Exactness and throughput claim withdrawn | Original cache/request artifacts and controlled reproduction; fresh successful prose would not validate the old claim. |
| P1.8 per-slot concurrent capacity | Source/procedure do not support label | Static labeling correction plus simultaneous long-context load test for actual service capacity. |

The advanced-idea distinctions are supported by the
[canonical advanced-ideas review](../research/turbo/p4.6-canonical-advanced-ideas-2026-07-11.md)
and [round-two candidate review](../research/sycl/sycl-a770-round2-decode-candidates-2026-07-25.md).
In particular, an existing centroid/nibble codec is not automatically an active
QJL residual estimator, and inherited asymmetric/boundary-layer policies are not
proof of paper-faithful TriAttention. The later SG16 explanation is source-derived
historical research; this audit did not fetch or rebuild the cited IGC tree.

The load-width probe is an example of a useful dead end without GPU timing.
[Retained compile summaries](../../scripts/perf/results/round2-load-width-probe)
showed quants-first baseline and widened builds both emitted 512 `d16` and
622 `d32` loads, despite different IR and roughly 2% smaller generated text.
IGC had already merged the adjacent halfword copies. A separate destination
alignment arm also had unchanged counts. Those two arms used different source
states and different instance counts, so their absolute totals must not be
compared with one another. Canonical q8 AoS still required two-byte alignment:
its 34-byte block and two-byte header make a blanket four-byte alignment claim
incorrect. Full ISA dumps were not committed; summary files are the surviving
compile evidence.

## 10. What this evidence does and does not support

The strongest early progression is functional SYCL turbo attention, a conventional
KV parity check, guarded retention of a few P5 improvements, a large depth-sensitive
occupancy correction, the quants-first default, corrected FFN fusion, and a
reopened turbo4 gate with full-corpus PPL and allocation evidence. A parallel
history of rejected paths prevented incorrect arithmetic, invalid benchmarks and
unrepresentative proxies from becoming accepted product claims.

The figures cannot be assembled into one March-to-August speedup ratio. They span
Metal M2/M5, CUDA RTX 5090 and SYCL A770; CPU/GPU quality paths; different models;
AOT/JIT; short smoke and paired campaigns; default and forced routes; immediate
and batched command lists; and changing cache policies. Nor does a historical
regression establish that the pinned October implementation still has that bug.

This audit read reports, commit bodies, selected diffs, historical capacity source
and structured artifacts; it recomputed the August occupancy paired medians and
turbo4 confidence-bound arithmetic. It did not rerun any build, GPU kernel,
perplexity job, service, or benchmark. Missing external directories, unavailable
branch objects, invalidated July fusion data and the July occupancy hash mismatch
are explicit evidence limits. No assertion is made that every private or deleted
artifact has been recovered. Further recovery should prioritize those concrete
holes before spending GPU time re-testing propositions that source inspection
has already disproved.
