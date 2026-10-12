# Adaptive KV fork: four independent extractions

Date: 2026-10-11

Target base: `2f623c6607c078d1069dac23e9d8af9cb146e771`, branch
`adaptive-kv-streaming-fork`. Source: RaymondHuang210129 adaptive KV fork,
`1240f34c77441d8afc1afecf34f60c702d436c61`, from the sibling checkout.
The initial evaluation was source-only; the checks below concern this port.

## Provenance and adaptations

- Server: `da2678f76`, repair intent from `aaf5739f5`. Use this branch's
  generated-text string; do not import `common_chat_input`, parser mappings,
  vision replay, or benchmark assets. Retain ordinary streaming token IDs,
  track all unsent IDs, reset the cursor per request, and finalize incomplete
  UTF-8 and pending stop prefixes. Repair overlong/non-scalar encodings too.
- Memory primitives: first eight V2 commits ending at `0dd2af676`, omitting
  removed backend implementations; Vulkan views from `551f863d7`; allocator
  clear and scheduler clear only from `4387ac04f`; view capability query from
  `d04b056ac`. No memory lease or arena types are required.
- SYCL VMM: `0f3667a1b`, adapted to the target's `gpu_index`. Unmap, destroy
  physical mappings, then release virtual memory; disable VMM for positively
  identified integrated Level Zero GPUs.
- Meta: `6a63b5b2e`, `dfa3f94a0`, and applicable `e2a0c7821` test coverage.
  Preserve stable owned metadata, translate source pointers before child
  initialization, and propagate initialization errors. Tests use this branch's
  buffer-type API and omit arena-only assertions.

CPU/SYCL/Vulkan views retain their immediate parent; nested offsets compose.
SYCL views own their tensor metadata but not the parent's USM allocation.
Vulkan views own neither the device allocation nor its accounting. Vulkan
requires four-byte-aligned view sizes for bounded `vkCmdFillBuffer` clears.
The clear pattern repeats the requested byte in all four bytes of the word.

Host buffer writes invalidate MoE cache entries by address range regardless of
an alias's usage tag. A weight parent can have an ANY child and vice versa;
usage-local invalidation would leave stale cached rows. Registered providers
retain their own address-overlap filters. Final release retains the fork's
existing host-buffer invalidation hook.

## API and lifetime contract

New APIs: `ggml_backend_buffer_retain`, `ggml_backend_buffer_view`,
`ggml_backend_buffer_supports_views`, `ggml_backend_buffer_is_view`,
`ggml_tallocr_new_range`, `ggml_gallocr_set_buffer_range`,
`ggml_gallocr_clear_buffer_range`, `ggml_backend_sched_set_buffer_range`, and
`ggml_backend_sched_clear_buffer_range`.

A caller retaining a buffer must already own a live reference. Tensor aliases
do not independently retain buffers. Callers must finish backend work before
releasing the final buffer reference. Stateful graph workspaces require a
complete independently resettable buffer view. Attach after measurement and
before reservation; clear synchronizes through the scheduler, releases the
range, and invalidates graph placements. Direct allocator callers provide
their own synchronization. No production caller is automatically attached. Buffer allocation now carries
reference-count metadata, and host mutations may notify cache providers even
when their local usage tag is not WEIGHTS. That extra work is not benchmarked.

The public `ggml_tallocr` layout changes. The internal backend API version is
4 so dynamically loaded modules built against the old layout are rejected.
Rebuild libraries, backend modules, and consumers together; this is not a
binary-compatible drop-in for previously compiled consumers.

## Validation

Observed on the A770 with the xe driver, kernel 7.3.0-rc5, oneAPI 2026.1,
IGC 2.41.10, compute-runtime `be85a8d685`, and Level Zero loader 1.34.0.
Builds use empty compiler launchers and `GGML_CCACHE=OFF`.

| Gate | Result |
| --- | --- |
| CPU + OpenBLAS build | Pass, OpenBLAS 0.3.34 |
| Focused CPU CTests | 8/8 pass: allocator, buffers, Meta, MoE invalidation, three scheduler cases, UTF-8 repair |
| Server integration | 21 pass, including 16 new UTF-8/token combinations |
| ngram-mod path | Twelve streamed results accepted 10 draft tokens each; four short results accepted zero |
| ASan + UBSan | 7/7 pass with ODR detection disabled; leak detection and address/UB halt checks enabled |
| Vulkan build and A770 buffer/allocator checks | Pass, forced `Vulkan0` identifies the Intel A770 |
| OpenVINO backend build | Pass against system OpenVINO 2026.5.0; runtime not exercised |
| Repository upstream invariant check | Pass |
| SYCL JIT build and A770 checks | Pass: forced SYCL0 buffer/allocator checks; oracle reports zero GATE-FAIL and XPASS |

The full `test-chat` suite aborts late in PEG-native template
parsing. Baseline test source from `2f623c660`, compiled against the same
current libraries, reproduces that exception. This isolates the new test
source; it is not a full baseline-tree build. The dedicated
`test-chat --utf8-repair-only` gate passes. Server tests use an isolated Python
environment and the existing tiny model fixture with a recorded model hash.

Unsuppressed ASan stops before main on duplicate `turbo3_cpu_wht_group_size`
globals in unchanged `ggml-cpu/ops.cpp` and `ggml-turbo-quant.c`. The passing
run uses `ASAN_OPTIONS=detect_odr_violation=0` with leak detection retained.
It is not an unsuppressed sanitizer pass. Automated ABI tooling could not run:
`abi-compliance-checker` and `abidw` are absent.

The SYCL oracle additionally recreates backend pool owners three times in one
process. Turbo3 matrix-vector probes each commit a fresh 2 MiB VMM range and
pass the CPU comparison before backend destruction. A first Q8 variant passed
without allocating pool scratch; it was replaced by the traced Turbo3 path.
VMM was explicitly enabled and its allocation log confirms actual execution.
The default oracle leaves turbo flash attention, d=256 flash attention, and
InnerQ opt-in sections disabled. No new xe/i915 fault-filter matches appeared
between readable before/after kernel logs for the focused GPU runs.

The initial Vulkan workspace test failed because it budgeted one backend
alignment unit per 16-byte tensor; Vulkan alignment is four bytes. The test
now rounds the actual tensor size to backend alignment. Its final run passes
12 workspace reuse steps while checking persistent bytes and output values.
The earlier CPU invalid-alignment failure used stale objects during the first
build; the final rebuilt normal and sanitizer tests pass.

The `/opt/intel/openvino-genai` package initially referenced a missing
frontend library; the installed `/usr/lib/cmake/openvino` package built
successfully without package changes. GPU checks use the shared lock and
explicit timeouts. Other ffmpeg GPU users were observed; no performance
measurement or sole-tenancy claim is made. The production service was already
inactive and was not changed.
Raw commands, logs, and environment records are retained locally under
`/home/svnbjrn/.docs/adaptive-kv-extraction-2026-10-11/`.

## Not claimed

No KV streaming, automatic compute arenas, execution buffers, projector
lifecycle, MTP vision drafting, host-spilled rollback, or host-USM experiment
is included. No performance improvement or production service equivalence is
claimed. Synthetic Meta tests do not establish physical multi-device behavior.
A770 execution does not validate the integrated-GPU VMM workaround. The
source's mapping-aware UTF-8 overload has no consumer in this branch and is
intentionally omitted. Retain/release is atomic; tensor metadata mutation and
buffer usage changes do not become generally thread-safe.
