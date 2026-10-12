# What differs from upstream at the review cutoff

Compare fork `2f623c6607c078d1069dac23e9d8af9cb146e771` with upstream
`5c8854fd88319125125dc4b836c98b1b215b0f23`, fetched on 2026-10-11. The last
full snapshot imported into the fork came from upstream
`7049ff0cbeb1f5ead231de4522af6b75d8d773c0` on October 5. Comparing against both
revisions separates persistent fork customization from subsequent upstream
development; neither comparison alone proves a change's intent.

## Size and composition

The deterministic, rename-disabled comparison finds **11,629 changed paths**:
9,968 additions, 517 modifications, and 1,144 deletions, with 1,753,795 added
and 531,880 deleted text lines. Binary entries are excluded from line totals.
These counts describe Git blobs, including LFS pointer text, not the unpacked
size of benchmark data or the size of executable code.

| Surface | Changed paths | Added lines | Deleted lines | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `docs/benchmarks/` | 8,695 | 1,300,093 | 0 | Historical archive dominates the count |
| `docs/research/` | 571 | 61,395 | 0 | Dated experiments and supporting evidence |
| Other `docs/` | 326 | 212,084 | 200,947 | Includes moved upstream documentation |
| `ggml/src/ggml-sycl/` | 89 | 8,942 | 2,275 | Kernel, routing, graph, cache, and operational changes |
| `ggml/src/ggml-vulkan/` | 40 | 5,691 | 389 | Turbo/TQ support and expert-cache integration |
| `ggml/src/ggml-cpu/` | 11 | 2,374 | 101 | Codec, IQ panel GEMM, and cache dispatch |
| `ggml/src/ggml-openvino/` | 18 | 72 | 296 | Fork fixes plus upstream changes not yet imported |
| `src/` | 81 | 9,626 | 4,443 | KV, model graphs, rollback, and MTP integration |
| `common/` | 49 | 3,514 | 1,340 | Speculation, fitting, configuration, SDK support |
| `tests/` | 65 | 13,566 | 2,405 | Extra gates and changes to upstream tests |
| `scripts/` | 211 | 98,627 | 9,251 | Harnesses, preservation checks, and archived tooling |

There are 9,592 changed paths under `docs/` and 2,037 outside it. Counts are
not additive across overlapping scopes other than the rows explicitly shown.
The full [path inventory](data/current-upstream-delta.tsv) includes status,
line counts, file modes, and both blob IDs. The
[group inventory](data/current-delta-groups.tsv) has every group, including
removed backends and the server/UI. Against October 5 upstream the fork has
11,290 differing paths. Rename detection would produce smaller headline
counts; it does not change either tree.

## The largest functional distinction: codec plus execution stack

Upstream's `ggml_type` ends at `Q2_0 = 42`, `COUNT = 43` at this cutoff. This
fork extends it with eight formats and ends at `COUNT = 51`:

| Tensor type | ID | Stored block | Purpose |
| --- | ---: | --- | --- |
| TURBO2_0 | 43 | 34 bytes / 128 values | Rotated 2-bit KV |
| TURBO3_0 | 44 | 50 bytes / 128 values | Rotated 3-bit KV |
| TURBO4_0 | 45 | 68 bytes / 128 values | Rotated 4-bit KV |
| TQ3_1S | 46 | 16 bytes / 32 values | WHT weight format |
| TQ4_1S | 47 | 20 bytes / 32 values | WHT weight format |
| Q8_CR | 48 | 272 bytes / 256 values | ConvRot weights |
| Q5_CR | 49 | 176 bytes / 256 values | ConvRot weights |
| Q6_CR | 50 | 210 bytes / 256 values | ConvRot weights |

Source: [ggml.h](../../ggml/include/ggml.h),
[block layouts](../../ggml/src/ggml-common.h),
[CPU codec](../../ggml/src/ggml-turbo-quant.c), and
[Python constants](../../gguf-py/gguf/constants.py). Tensor IDs are distinct
from `llama_ftype` / `general.file_type`: CR file-type values are 512-514,
while TQ file-type values remain 43/44. The July transition and September
collision avoidance are explained in the integration and correctness chapters.

Turbo attention requires changes across the graph, not just a dequantizer.
When K is turbo, Q is rotated into the matching basis before attention;
when V is turbo, the attention output is inverse-rotated. Mixed q8 K/turbo V
therefore requires the output transform without the turbo Q transform.
The fork adds `GGML_OP_TURBO_WHT`, CPU and
GPU implementations, cache padding, state save/load handling, and graph
wiring. Non-128-aligned heads can incur padding overhead. Block-size ratios
therefore do not by themselves establish whole-model memory savings.

The [KV policy](../../src/llama-kv-cache.cpp) also changes effective requested
types. Same-type turbo K/V requests can become q8 K plus turbo V for high-GQA
or Qwen-family models, with model exceptions; layer-adaptive Boundary V can
protect first/last layers, and turbo2 V auto-selects mode 7 for sufficiently
deep models. These are fork behaviors, not upstream flags with identical
semantics. The documented knobs allow experiments, but requested types and
effective per-layer types must both be recorded.

InnerQ has host policy, a runtime state machine, recovery paths, a profile
kernel, and graph hooks. Presence of these surfaces is not evidence of
useful end-to-end calibration. Its opt-in and historical runtime limitations
remain material. Similarly, inherited Sparse V work on removed CUDA/Metal
paths is not proof that present SYCL/Vulkan turbo kernels implement it.

## SYCL is now an independently maintained execution path

The useful distinction is concrete behavior, not a blanket claim that the
fork is faster than upstream:

| Surface | Fork behavior at the cutoff | Boundary |
| --- | --- | --- |
| Turbo FA | VEC by default; experimental same-type XMX at D=128/256 | No turbo TILE route; mixed types remain VEC; feature predicates restrict XMX |
| q8 KV | Quants-first layout is on by default for eligible 128-wide SYCL cache heads | Save/load and all consumers must understand it; MKL integration previously broke this |
| Standard FA | VEC, TILE, and oneMKL routes, with fork occupancy/diagnostic changes | Selection depends on shape/type/features, not just a global toggle |
| FFN SwiGLU | Fusion flag defaults on | Only eligible graph patterns and types fuse |
| SYCL graph | Record-once/replay support, `GGML_SYCL_ENABLE_GRAPH` defaults off | Stable shape and compatibility gates; July's failed prototype does not describe this later implementation |
| MoE cache | SYCL provider and fork cache policy; common mode defaults off | Placement, budget, lifetime, and CPU fallback paths matter |
| Expert prefetch | Lookahead copies and a private copy queue | Adapter/driver prerequisites; cannot assume overlap or speedup |
| xe/DG2 policy | Copy-engine disabling defaults selected before runtime initialization | Host runtime retry patch is separately maintained; not installed by this repository |
| Large GRF | Optional FA kernel instantiations and runtime selection | Shape-specific historical benefit, not a universal decode improvement |
| XMX gather build | AUTO/ON/OFF target policy and separate-build integration | AOT target/compiler capabilities govern what gets built |

Read [SYCL dispatch](../../ggml/src/ggml-sycl/ggml-sycl.cpp),
[FA routing](../../ggml/src/ggml-sycl/fattn.cpp),
[xe policy](../../ggml/src/ggml-sycl/xe-kmd.cpp), and
[backend guide](../backend/SYCL.md). The source also exposes a remaining
configuration defect: XMX enablement tests `getenv` presence, so setting it
to `0` still enables eligible XMX routes. See the findings chapter.

Some originally fork-carried optimizations now exist in current upstream.
For example, grouped MoE XMX GEMM and Q5_K reorder/fused GLU landed in this
fork before their upstream merges. Their original port commits are
`e985ccb9e` and `f045c2e50`; upstream subsequently merged `847f447c3` and
`3d65c90d0`. The fork retains integration differences, but calling those
feature ideas exclusively fork-only as of October 11 would be wrong.

## CPU, Vulkan, OpenVINO, and scheduler

CPU carries codec/reference and dot-product support, IQ panel GEMM, and a
backend-independent expert-cache interface with a CPU dispatch bridge.
Vulkan carries TurboQuant/TQ kernels and its own cache provider. Backend
capabilities are not identical: SYCL TQ single-token matvec support is not
general TQ matrix/MoE support; Vulkan implements a wider TQ surface. CR enum
registration likewise does not establish accelerated support on every backend.

The shared scheduler has fork expert-prefetch logic, in-place destination
reachability checks, and experimental stream-ordered input copies. The latter
was merged as `bba3994a9`, remains opt-in through the exact value
`GGML_SCHED_COPY_SYNC=0`, and adds upload lifetime/event constraints. It is not
a demonstrated default speedup. Source:
[ggml-backend.cpp](../../ggml/src/ggml-backend.cpp) and the
[scheduler guide](../backend/SYCL.md#scheduler-input-copy-synchronization).

OpenVINO retains the fork's dynamic-shape REPEAT work, including October's
static validation and captured multipliers. Upstream changed OpenVINO again
after the snapshot, so a smaller line count on the fork side is not evidence
of deliberate removal of every newer upstream implementation. Operator support
tables are dated observations for particular devices, not exhaustive runtime
proof for all devices or models.

## Models, speculation, and state

The fork carries additional Qwen4Exp MTP graph/loading/conversion integration,
adaptive draft control, validation bookkeeping, and recurrent/KV rollback
work. It also retains ngram-mod's hard-off latch for poor acceptance. These
touch [common/speculative.cpp](../../common/speculative.cpp),
[llama-context.cpp](../../src/llama-context.cpp),
[llama-ext.h](../../src/llama-ext.h), model builders, and server acceptance
loops. They are architectural maintenance commitments, not a kernel patch.

October's MTP corrections include shared-cache geometry, acceptance/rejection
bookkeeping, output indexing, and adaptive measurement fixes. `d09fb6ba4` has
a long message repeating earlier work; its actual six-file delta is chiefly
acceptance-measurement harness work. The chronology uses diffs to avoid
counting that narrative as another implementation landing.

Current upstream has model and multimodal work after the snapshot. For
example, `src/models/gemma-embedding2.cpp` and `src/models/k2-horizon.cpp` are
present upstream and absent here. Existing common models can also have
different graph implementations. Consequently the README's broad claim that
all existing upstream families remain fully supported cannot serve as a
verified parity statement against October 11 upstream.

## Server, UI, SDK, packaging, and development scope

Server differences include MCP proxy destination policy, credential/CORS
handling, tool process restrictions, cache-memory policy, speculative APIs,
and environment diagnostics. The policy is consumed at the HTTP handler
boundary; it is not merely a parser option. Read
[server-cors-proxy.h](../../tools/server/server-cors-proxy.h),
[server.cpp](../../tools/server/server.cpp), and
[proxy tests](../../tools/server/tests/unit/test_proxy.py). The security
hardening milestone is `5aa5c34f8`; later merges needed preservation checks.

There is also negative functionality: cache-key slot selection was lost
during integration while its CLI survived. This review documents that current
regression instead of listing the surviving flags as working features.

The fork's UI source, embedding/build path, and provider/model configuration
have local changes. Current upstream's newer model-manager work makes the
October 11 UI diff a mixture of local customization and lag. The archive does
not establish browser/runtime parity for the current pair of tips.

The SDK work (`5beb6c3f3`) provides installed CMake/pkg-config surfaces,
optional `llama-common`, and a downloader-disabled stub. `lib` is generated
by [prune-to-lib.sh](../../scripts/prune-to-lib.sh), not an independent source
development branch. This review did not regenerate it or verify its remote
tip against master. Arch packaging and a systemd unit are shipped under
[packaging/arch/](../../packaging/arch/).

The retained backend directories are exactly CPU, BLAS, SYCL, Vulkan, and
OpenVINO. CANN, CUDA, ET, Hexagon, HIP, Metal, MUSA, OpenCL, RPC, VirtGPU,
WebGPU, zDNN, and ZenDNN are deliberately absent. `.github/`, `ci/`, and
other automation/metadata were removed as well. This reduces the maintenance
surface but also means historical GitHub workflow claims do not describe
present repository automation. Documentation or old commits mentioning a
removed backend do not make it supported today.

## Upstream lag is real, but raw ahead/behind counts mislead

The [125-commit post-snapshot ledger](data/upstream-after-snapshot.tsv)
includes new kernels, models, refactors, and fixes from October 5-11. Some
equivalents were already ported here. Two recent fixes were checked against
the actual current fork code and remain absent:

- [`8ae386707`](https://github.com/ggml-org/llama.cpp/commit/8ae386707bed05f02354dbfc6fa5ee0cf58e5f0f)
  guards ACC offsets before narrowing them into signed op parameters.
- [`23b0202a1`](https://github.com/ggml-org/llama.cpp/commit/23b0202a189c44a54625aadcb37a946dd1d6278d)
  returns a busy explicitly pinned server slot before cache selection can
  mutate it.

The findings chapter gives trigger scenarios and limits. These examples
demonstrate a present correctness cost of lag, not that all 125 commits should
be applied blindly. Upstream also refactored expert copying and introduced
its own cache mechanisms in this window; integration must reconcile them with
the fork's existing lifetime/provider contracts.

No throughput comparison was performed for these two current tips. The
answer to "biggest difference" is therefore the codec/platform/execution
and maintenance scope above, not a newly measured percentage speed advantage.
