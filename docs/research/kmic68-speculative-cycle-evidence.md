# Kmic68 P04: speculative proposal identity and cycle evidence

Evidence collected on 2026-10-11. Accepted implementation `57ba6d4ac` (candidate `f6a09bfaa`) adds immutable proposal
identity, lifecycle accounting and opt-in cycle logging/profiling. The real-model
campaign, focused checks and P04-absent SYCL regression floor passed independent
verification and review. The feature is accepted; logging and profiling remain
runtime opt-in.

## Real-model evidence

The clean verification branch `7021e1aa8` contains P04 plus the separately
accepted Release assertion correction. Its frozen AOT build used oneAPI 2026.1.1,
F16 and `acm-g10`. The target is the local two-shard, 58,408,584,928-byte
Qwen4Exp model under `/mnt/ssd1/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/IQ1_M/`.
The matching 4,137,429,280-byte Q8_0 MTP head is under the corresponding
`/mnt/ssd2/gguf/.../mtp/` directory. Both declare `qwen4exp`.

The campaign used context 16384, q8_0 K/V, fit on, fit-target 1024, mmap,
`--n-gpu-layers auto`, non-chained host MTP sampling, draft minimum 1/maximum 7,
one slot, three prompts repeated twice and 256 output tokens per response.
The separate uninstrumented warmup completed 256 tokens without cycle records.

| Request | Cycles | Drafted | Accepted | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| 0 | 70 | 482 | 185 | 256 |
| 74 | 55 | 379 | 200 | 256 |
| 133 | 53 | 364 | 202 | 256 |
| 191 | 51 | 357 | 204 | 256 |
| 247 | 82 | 565 | 172 | 256 |
| 334 | 80 | 547 | 175 | 256 |

All 391 unique starts matched exactly 391 completions, with 391 distinct draft
phases, 391 verify phases and 2,694 retained attempted tokens. Every selected
prefix and response total reconciled. Six profile summaries (64 through 384
samples) matched independently recomputed denominators and means within printed
rounding. Every single-slot additive residual passed its synchronization plus
scheduling bound. These are attribution checks, not throughput improvements.

## Requirement mapping

| Requirements | Evidence |
| --- | --- |
| R04.1-R04.2 | Unique lifecycle identities, schema and six response reconciliations |
| R04.3-R04.4 | Synchronized campaign, source-audited boundaries, phase and cycle profile denominators |
| R04.5-R04.6 | Uninstrumented warmup plus source-audited disabled recording and synchronization guards |
| R04.7-R04.8 | Unchanged focused fixtures: stopped attempts through clamp/clear/unselection and log write failures |
| R04.9-R04.10 | Identity rejection and distinct shared draft/verification participants in focused fixtures; real request identities |
| R04.11-R04.12 | Immutable finalized prefixes, replay, unselected and cancelled fixture cases; selected/completed real rows |

The independent focused verifier passed eleven groups, including cycle and
sampling fixtures, parser/adaptive checks, MTP default/q8 on CPU and explicitly
selected SYCL, the normal CPU/SYCL oracle and Python checks. The real campaign
repeated the required MTP default/q8 and Python floor afterward. The default
MTP fixture chooses CPU even when linked to the SYCL-enabled build; explicit
SYCL variants are separate evidence.

## Reproduction and preservation

Raw evidence is under `/home/svnbjrn/wt/kmic68-evidence/`:
`p04-sycl-verification/20261011T010515Z/` retains focused checks;
`p04-icd-retry/` retains exact commands, responses, JSONL, profile reconciliation,
requirement outcomes, launch environment, hashes and before/after kernel logs.
Its preservation record checked 12,620 unchanged source/build paths. Model
identities use paths, sizes and mtimes, not full-content hashes.

All real executions held `/tmp/a770.lock`, used bounded timeouts and verified
idle tenancy on the xe A770. The successful campaign observed no concurrent
builds or new GPU faults. Earlier failed placement and LLVM initialization
attempts remain preserved, including their nonzero statuses.

Earlier CPU preparation in `p04-verification/` was not fully offline. Its build
downloaded prebuilt UI assets, and the server test preset loader downloaded
Stories and BERT fixtures contrary to the local-only constraint. The coordinator
stopped that suite, inventoried and removed the downloaded model cache, and
excluded its incomplete server run from acceptance. The retained
`progress.txt`, `excluded-download-inventory.txt` and `build.log` record this
exception. The later accepted AOT verification explicitly disabled both
`LLAMA_BUILD_UI` and `LLAMA_USE_PREBUILT_UI`; its real-model campaign used the
local Qwen4Exp target and head described above. This report does not claim that
every preparation attempt avoided downloads.

The process-local `OCL_ICD_VENDORS` directory in
`p04-stack-diagnosis/icd-intel` contains only Intel's ICD. Together with
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, this avoided the independently reproduced
Mesa/ROCm LLVM registration collision. No system package or ICD configuration
was changed. Subsequent invocations using this recovery require that directory.

## Before-change regression floor

`p04-prechange-aot-floor-verification/` retains the independent pre-P04 check.
Its source HEAD `3a4855a44` includes accepted P02, with memory profiling disabled,
and no P04 code. All 968 audited source paths matched integration revision
`d571adc65`. The Release SYCL/F16 AOT build used `acm-g10` and IntelLLVM 2026.1.1.
The exact Python unittest command passed 40 tests; MTP default and `--q8-kv`
both ended with the regression-suite PASS marker. All three guarded commands
exited zero with clean fault, tenancy and preservation checks. Loaded project
libraries came from the recorded baseline directory.

The first incremental device link failed with compiler exit 254. It exposed an
older internal link concurrency setting of 24 despite Ninja `-j4`. The retained
retry explicitly set four internal link jobs and succeeded; the compiler
failure's underlying cause was not independently established. Reconfiguration
changed only `libggml-base` and `libllama-common` among the recorded preexisting
binary identities (six versioned/unversioned paths). The replacement hashes
are stored separately in `p04-prechange-aot-floor-retry-result.json`; original
P01/P02 raw campaign identities have not been rewritten.

## Integration checks

After cherry-picking P04, the only conflict was adjacent CMake test entries;
both P02's `test-server-memory` and P04's `test-speculative-cycle` were retained.
All P04 implementation files match the accepted candidate. Independent checks
in `p04-postintegration-verification/` passed the ordinary cycle fixture and
48 Python tests with zero skips, including the compiled profile, cancellation,
identity and write-failure fixtures. Source, binary, tenancy and fault checks
passed. The cycle executable matched the earlier focused-verification hash;
an initial lookup in a manifest that omitted that executable was corrected
using the retained focused inventory before execution.

These checks reused the frozen P04 AOT executable plus the assertion correction;
they are not a new compilation of the combined integration branch. The final
combined build and runtime verification remain separate campaign obligations.

## Not claimed

No throughput improvement, broad output-quality proof, full model hashes,
instruction-level tracing or system-wide LLVM repair. All six real responses
finished by length: real EOG stops, probability stops, shared phases,
cancellations and multiple-verification replay did not occur. Focused fixtures
and source audit provide those bounded requirement checks.
