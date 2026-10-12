# P01/P02 A770 token-count and memory measurements

Campaign start: 2026-10-10. Source baseline:
`2f623c6607c078d1069dac23e9d8af9cb146e771`.

Status (2026-10-11): the bounded depth-16384 AOT calibration and the P02
8%-fill/full-fill memory pair completed with clean run gates. P01 is accepted
after independent review reproduced the statistics and checked the raw gates,
source/library manifests, historical records and R01.1-R01.7. P02 is accepted
after independent lifecycle verification and source/evidence review. Historical
values have not been replaced.
This is one depth, one Llama model and one Qwen model, not a complete depth sweep.

## Long-depth comparison

These preserved standard-KV rows illustrate the old token-count contract. They
are original single repetitions, not current TurboQuant measurements. Driver
provenance is the [dated P5 report](sycl/sycl-a770-p5-performance-campaign-2026-07-19.md).
The root AGENTS performance summary also contains historical turbo ratios, but
lacks complete per-row command/build provenance; do not manufacture rows from it.

| ID | Model | K/V | Depth | n | Tokens/s | Build | Driver | State |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| H1 | Mistral-7B | q8_0/q8_0 | 4096 | 128 | 17.920418 | `d7a2bcf1d` | i915 | historical; token-count contract superseded; not directly comparable |
| H2 | Mistral-7B | q8_0/q8_0 | 8192 | 128 | 14.120583 | `d7a2bcf1d` | i915 | historical; token-count contract superseded; not directly comparable |
| H3 | Mistral-7B | q8_0/q8_0 | 16384 | 128 | 9.931606 | `d7a2bcf1d` | i915 | historical; token-count contract superseded; not directly comparable |
| C128 | Llama-3.1-8B heretic Q4_K_M | turbo3/turbo3 | 16384 | 128 | 18.538440 | `1a13eef9a` + dirty manifest | xe | calibration reference; token-count contract superseded; not directly comparable |
| C512 | Llama-3.1-8B heretic Q4_K_M | turbo3/turbo3 | 16384 | 512 | 18.558350 | same AOT artifacts as C128 | xe | corrected token-count observation; independently reviewed |

The retained n=128 rows are superseded as the long-depth token-count contract,
not erased or numerically replaced by another model/driver/build. H1-H3 remain
original historical observations. C128 versus C512 has status
`token-count-mismatch` and **no ordinary paired percentage delta**. H1-H3 versus
C512 additionally differ in model, KV type, driver and build.

No paired delta is defined between these i915-era rows and a new xe run.
Unequal token counts additionally produce `token-count-mismatch`, with no
ordinary performance percentage delta. No same-model, same-KV, same-driver
numeric replacement for H1-H3 was measured.

## Token-count calibration

Use the same AOT binary, model, driver/boot, environment, and depth 16384 for
both arms, with three repetitions each. Record both token counts and raw rows.
Only this table may report `100*(tps_512/tps_128-1)` for unequal token counts,
explicitly labeled `calibration-only`. It is not a performance A/B.

| n=128 tokens/s | n=512 tokens/s | Calibration delta | State |
| --- | --- | --- | --- |
| 18.538440 (n=128) | 18.558350 (n=512) | +0.107398% | measured; calibration-only; no performance claim |

The delta is `100*(18.558350/18.538440-1)`, using the mean-throughput fields
from the two raw benchmark rows. The external settings match except `-n`.
The benchmark sizes context from depth plus generation length, so the resulting
KV allocations have 16640 and 16896 cells respectively. This inherent setup
consequence is part of this calibration, not a separately controlled effect.

| n | Raw tokens/s samples | Median tokens/s | Mean tokens/s | Descriptive mean 95% interval | Repetitions retained/discarded |
| ---: | --- | ---: | ---: | --- | --- |
| 128 | 18.5365, 18.5270, 18.5518 | 18.536516 | 18.538440 | [18.507265, 18.569616] | 3/0 |
| 512 | 18.5555, 18.5608, 18.5588 | 18.558813 | 18.558350 | [18.551703, 18.564997] | 3/0 |

The intervals use Student t with two degrees of freedom on the three
within-process repetitions. They are descriptive, not independent-launch or
paired performance confidence intervals. Each process retained its default
one-token generation warmup outside timing. The first repetition filled depth
16384; repetitions two and three restored the cached depth state. No measured
sample was discarded. The raw nanosecond samples and token counts are retained
in `p01-n128/stdout` and `p01-n512/stdout` under the evidence directory below.

### Historical inventory limits

H1-H3 were checked against their preserved JSON records, including original
argv, n=128, build ID, and throughput. `historical-inventory.json` records their
file hashes and the environment fields actually present in those artifacts;
unrecorded inherited environment is explicitly unknown.

`AGENTS.md` also preserves historical Llama turbo3 tg128 ratios of -11% at depth
4096 and -17% at depth 16384 versus q8_0. Those aggregate ratios lack complete
per-row commands, raw absolute timings, build hashes and environment identities.
They are superseded as the n=128 long-depth measurement contract and not directly
comparable to C512. Their values are preserved; no missing raw record or build
identity has been reconstructed. The historical pp512 and shallow depth-zero
observations are outside this bounded long-depth calibration.

## Preserved original commands

The paths below are historical argv, not instructions to use missing paths.
The readable replacement Mistral is under
`/mnt/mrgr/download-backup/mistral-7b-v0.1/`; its identity must be recorded anew.

### H1

[Original raw record](../../scripts/perf/results/p5-post-campaign/phase7-final-mistral/samples/cell02_q8_0_q8_0_d4096_candidate_rep1.json) (candidate repetition 1, not a paired median).

```bash
/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-final/bin/llama-bench -m /home/svnbjrn/Downloads/mistral-7b-v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 512 -ub 512 --no-warmup -r 1 -o json -d 4096
```

### H2

[Original raw record](../../scripts/perf/results/p5-post-campaign/phase7-final-mistral/samples/cell03_q8_0_q8_0_d8192_candidate_rep1.json) (candidate repetition 1, not a paired median).

```bash
/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-final/bin/llama-bench -m /home/svnbjrn/Downloads/mistral-7b-v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 512 -ub 512 --no-warmup -r 1 -o json -d 8192
```

### H3

[Original raw record](../../scripts/perf/results/p5-post-campaign/phase7-final-mistral/samples/cell04_q8_0_q8_0_d16384_candidate_rep1.json) (candidate repetition 1, not a paired median).

```bash
/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-final/bin/llama-bench -m /home/svnbjrn/Downloads/mistral-7b-v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 512 -ub 512 --no-warmup -r 1 -o json -d 16384
```

## Current campaign provenance and commands

Raw evidence directory:
`/home/svnbjrn/wt/kmic68-evidence/p01-p02-aot/20261011T002830Z/`.
The complete stdout/stderr, commands, selected environment, process mappings,
readable before/after kernel logs, tenancy/process/load checks and gate outcomes
are retained in each `p01-n*` and `p02-*-run` directory. `p02-8pct/` and
`p02-full/` additionally contain each original fixture, summary, launch record,
server log, terminal memory JSONL, and reconstructed server command.

- Worktree: `/home/svnbjrn/wt/kmic68-measurement`.
- Branch: `codex/kmic68-measurement`.
- HEAD: `1a13eef9a38a8f0b232da13086ccc8b01cc0f425`; uncommitted P02 source is
  represented by `initial.diff` and the source SHA-256 manifests, not HEAD alone.
- Build: `build-aot`, Release, F16 ON, `acm-g10`, empty compiler launchers;
  the supplied successful build log used `-j4`. This campaign performed no build.
  `CMakeCache.txt`, `build.ninja`, `.ninja_log`, and the supplied build log are
  copied into the evidence directory. The original log is
  `/home/svnbjrn/wt/kmic68-evidence/measurement-aot-build.log`.
- Kernel: `7.3.0-rc5-273-linux73-tkg-bore-rc5-xe-aocc`; driver: `xe`, A770 at
  `0000:03:00.0`, render node `/dev/dri/renderD128`.
- Boot ID: `72d6ada6-a9d4-477e-b9b7-5e076ff5ee07`, unchanged across all arms.
- Installed compute-runtime package:
  `intel-compute-runtime-git 22.43.24558.r13063.gbe85a8d685-1`;
  IGC `1:2.41.10-1`; Level Zero loader `1.34.0.r2.gae3db48-1`.
  These are the recorded package identities, not inferred release versions.
- All 12595 source-file hashes were unchanged through the final measurement.
  All selected executable/library hashes were unchanged. All 53 actually mapped
  shared libraries were covered by pre-timing hashes and post-run comparison.
  Five late-loaded runtime libraries were hashed during the first untimed depth
  prefill, after its one-token warmup and before timed generation. No hashing
  ran during measured generation. See `identity-check.json`,
  `prepared-binary-library-sha256.json`,
  `before-timed-generation-extra-libraries.json`, and
  `after-binary-library-sha256.json`. Report edits occurred after that check.
- Both local models were SHA-256 checked before and after; neither changed.

| Artifact | SHA-256 |
| --- | --- |
| `build-aot/bin/llama-bench` | `61b58a030a1b51609002943a2aa0828ce456af80199ad5a0473091bfc8641330` |
| `build-aot/bin/llama-server` | `e07536e83ad61730c7d4099c9ac673f2d8fa86041dc993b27d969dfa56cd17d2` |
| actual `libggml-sycl.so` | `6cc719fc9b645084da4c9ab88d1575f706ea647a6ec85c822aea25d8808bf121` |
| Llama model | `7eddce1074a77d70fb77e3ff87bc49f16e7988dab7c0a20caee21b3c1b63bec7` |
| Qwen model | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |

Both P01 commands used this environment after sourcing
`/opt/intel/oneapi/setvars.sh`:

```bash
export LD_LIBRARY_PATH=/home/svnbjrn/wt/kmic68-measurement/build-aot/bin:$LD_LIBRARY_PATH
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export ZES_ENABLE_SYSMAN=1
```

The exact measured payloads were:

```bash
/home/svnbjrn/wt/kmic68-measurement/build-aot/bin/llama-bench --offline -dev SYCL0 -m /mnt/ssd1/models/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf -ngl 99 -fa on -ctk turbo3 -ctv turbo3 -p 0 -n 128 -d 16384 -r 3 -t 4 -o json --progress -v
/home/svnbjrn/wt/kmic68-measurement/build-aot/bin/llama-bench --offline -dev SYCL0 -m /mnt/ssd1/models/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf -ngl 99 -fa on -ctk turbo3 -ctv turbo3 -p 0 -n 512 -d 16384 -r 3 -t 4 -o json --progress -v
```

These payloads ran through `flock -w 5 /tmp/a770.lock`, an outer 1020-second
`timeout --kill-after=45s`, and `guard.py`, which held the lock across the
before/after checks and applied the 900-second payload timeout. The guard reused
`bench_spec.py:new_gpu_faults`, including its exact two-driver failure filter.
Both returned zero with an empty new-fault list, no foreign holders or compiler
processes, low host load, and no surviving llama process. Copy-engine overrides
were left unset; startup logged the fork's DG2/xe defaults. No services were
changed. An initial help preflight refused a pre-existing failed service state;
the local guard was corrected to allow non-running `failed` as well as `inactive`
states. The subsequent bounded help command passed and supplied the actual flags.

## P02 process-lifetime sampled memory

The same AOT server and local
`/mnt/mrgr/models/Qwen3-1.7B-Q8_0.gguf` served both fixtures. Each launch used
`MODE=baseline ONLY=none-q8_0 CTX=16384 REPEATS=1`, `THREADS=4`, `PORT=8784`,
`ZES_ENABLE_SYSMAN=1`, `LLAMA_BENCH_MEM_INTERVAL_MS=50`, `SERVER_EXTRA=--offline`,
`HEALTH_TIMEOUT=600`, and `REQ_TIMEOUT=900`. Full placement selected SYCL0,
999 GPU layers, no mmap, FA on, one slot and q8_0 K/V. The harness additionally
sets `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` in both server environments.
`SETVARS` was empty because the outer shell had already sourced oneAPI; the
AOT library directory was first in `LD_LIBRARY_PATH`.

`run-p02.sh 8pct` and then `run-p02.sh full` in the evidence directory preserve
the exact reproduction commands. Each used a unique absolute `PROMPTS`,
`LLAMA_BENCH_MEM_LOG` and `OUT_TAG`. Baseline mode does not apply `OUT_TAG` to its
filenames, so the small `baseline_launcher.py` imports the unchanged harness,
sets its `RESULTS` to that absolute directory, and calls its existing `main()`.
The shared source harness was not changed. Its session/process-group ownership,
bounded shutdown and cancellation handling remained in use. Each launch ran
under the same flock guard, a 2400-second outer timeout and 1800-second payload
timeout with 45 seconds of kill grace. The first arm passed before the second
was launched.

| Fixture | Server prompt tokens | Requested/generated tokens | Verified argmax rows | Exact samples | Minimum free bytes | Final free bytes | Provider | Query failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 8% | 1308 | 16/16 | 16/16 | 59 | 14033788928 | 14033788928 | exact_sycl | 0 |
| full | 16352 | 16/16 | 16/16 | 2017 | 14008356864 | 14008356864 | exact_sycl | 0 |

Both records identify SYCL0, interval 50 ms, total bytes 17079205888,
`valid=true`, and `completed=true`. Each file contains exactly one terminal
process record with the launch identity assigned by the harness:

- 8%: `d9f0de7588e64f08a9b0ff39b44ce447`.
- Full: `34bfa89b75074e658ade97e6238126a4`.

The prepared tokenizer counts were confirmed by the actual measured server
requests: 1308/16352. Both had positive verified decode and no truncation. The
32-token reserve was retained, with 16 generated tokens. The full minimum is
25432064 bytes (24.25390625 MiB) below the 8% minimum, so `full_min <= 8pct_min`
passes. `min <= final` passes in both arms as equality; no transient dip below
the final sample was observed. These are sampled minima, not guaranteed bounds
on unsampled short allocation peaks.

Both summaries have `memory_only=true`, `memory_valid=true` and
`telemetry_valid=true`. Throughput and elapsed-time fields in their measured
request records are null. Incidental timing text in raw server/console logs is
preserved but excluded from every performance claim. Both processes returned
zero, new GPU-fault lists were empty, and no owned process or render-node holder
remained afterward. Pair data is in `p02-pair.json`; the original fixtures and
preparation provenance remain unchanged, with observed counts recorded here
and in the launch summaries.

## Independent P02 lifecycle and compatibility verification

The verification-runner used the same frozen AOT artifacts and local Qwen model.
Raw evidence, commands, source/binary hashes and requirement outcomes are under
`/home/svnbjrn/wt/kmic68-evidence/p02-lifecycle-verification/`. Both normal SYCL
oracles reported zero gate failures and XPASS. Focused sampler tests and seven
Python telemetry/cancellation tests passed. An unsampled control completed with
no sampler record, and interval `-50` failed before model allocation.

One actual server process completed a 16352-token prompt and 16 verified decoded
tokens, slept/unloaded, then woke for an 11-token prompt and 16 verified decoded
tokens, and slept again. PID and process start time were unchanged. A separate
observation library forwarded the real memory query unchanged and recorded all
1136 exact SYCL responses. Exactly one valid terminal record matched their
count, minimum 13714350080 bytes, and final 16975257600 bytes, with zero query
failures. Both sleep transitions freed model memory. All fault gates were clean
and no owned process or render-node holder remained.

The original scratch driver additionally required the first epoch's minimum to
be lower than every later sample. That extra assertion failed: first minimum
14008356864 bytes exceeded the later minimum 13714350080 bytes. Its wrapper
exit 1 and traceback are retained, not relabeled as a passing wrapper. The
server exited zero; separate validation reproduced the required aggregate from
the unchanged query log. This hardware run cannot independently distinguish
retention from a hypothetical minimum reset at wake. Retention is supported by
the source's process-owned sampler without an epoch reset and the deterministic
real-class sequence 100,60,100,80, which emits one minimum/final record 60/80.
Independent review accepted this combined evidence against R02.1-R02.11; the
plan does not require the earliest epoch to be the global minimum.

The observation library and concurrent P04 compilation make this lifecycle run
unsuitable for performance claims. Live exact Level Zero selection was not
exercised; provider/failure tests and source inspection cover that branch.
The API/ABI comparison in `kmic68-evidence/apiabi-p02/compare.log` found no
source compatibility problems or removed/changed symbols, and one additive
`ggml_backend_dev_memory_status` symbol. Tools were provisioned locally under
`kmic68-evidence/apiabi-tools`. Release binaries lack rich debug type information,
so the binary comparison primarily checks symbols; it is not an exhaustive
layout proof. The unchanged public by-value structures and backend vtables were
also reviewed in source.

## Requirement and gate outcomes

The original measurement task produced the pair data. The subsequent independent
verification and review above completed acceptance; their evidence categories
remain separate in the table.

| Requirement | Outcome in this bounded campaign |
| --- | --- |
| R01.1 | Integrated AGENTS product/depth examples both use n=512; remaining options preserved. |
| R01.2 | Both current depth-16384 turbo rows and every raw repetition retain n. |
| R01.3 | Retained long-depth n=128 records are labeled superseded in token-count contract and not directly comparable. No full multi-depth rerun is claimed. |
| R01.4 | Ordinary C128/C512 paired delta is null; historical cross-driver deltas are excluded. |
| R01.5 | Original H1-H3 commands, raw links and numeric values are preserved; incomplete historical turbo ratios remain identified as incomplete. |
| R01.6 | Ordinary comparison explicitly records token-count-mismatch. |
| R01.7 | Same-setup calibration completed; both n values and +0.107398% calibration-only delta are recorded. |
| R02.1 | Real pair and lifecycle records satisfy minimum <= final. Source lifetime, deterministic retention and actual query-log agreement support retention with the limitation above. |
| R02.2 | Exactly one matching completed terminal row observed after each request loop and graceful shutdown. |
| R02.3 | Actual expected prompt counts and positive verified decode were required and observed. Negative pre-decode-only rejection was not injected here. |
| R02.4 | Source confirms pre-model start and pre-cleanup finish; actual same-process unload/reload and continuous query observation passed. |
| R02.5 | One process record per fixture with identity, SYCL0, interval, sample count, minimum/final, exact provider and failure count is preserved. |
| R02.6 | Real launches had zero query failures; independent fake-query and write-failure tests passed. |
| R02.7 | Independent unsampled AOT control completed with no memory record or observation preload. |
| R02.8 | exact_sycl was observed. Exact Level Zero, fallback and failure outcomes were not hardware-exercised here. |
| R02.9 | Actual valid records pass the parser; independent invalid-record and sticky-failure fixtures passed. |
| R02.10 | Actual interval 50 ms observed; real -50 rejected before allocation. Deterministic malformed/overflow parser cases passed. |
| R02.11 | Real shutdown left no owned process; independent cancellation, orphan and forced-escalation fixtures passed. |

## Not claimed

No throughput improvement, cross-driver speed comparison, independent-launch
performance CI, complete depth sweep, P10 calibration acceptance, or P02 safe
instantaneous headroom floor is established. Hardware-only discrimination of
earlier-epoch retention remains unproved as described above. Correctness or quality
of generated model text is not established by the benchmark; the P02 response
check establishes the requested token/argmax invariants only.

The measurement and lifecycle agents made no production source changes, service
changes, model downloads, pushes, PRs or deployments. The coordinator separately
built, reviewed and integrated the implementation and compatibility checks.
The coordinator retains service restoration responsibility. The new state is
this report and the external raw evidence directory with its bounded launchers;
no dependency was added. Periodic 50 ms sampling can miss shorter memory peaks.
