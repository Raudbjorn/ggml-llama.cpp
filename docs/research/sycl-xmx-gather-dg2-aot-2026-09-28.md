# XMX gather GEMMs vs DG2 AOT builds - 2026-09-28

Why `GGML_SYCL_XMX_GATHER` exists, what it does, and the evidence behind it. The user-facing
description is in `docs/backend/SYCL.md`, "XMX gather GEMMs and DG2 AOT builds". This note
records the investigation.

Evidence labels: **measured** means tool output on this host; **source** means read from code;
**reported** means the Arch packager's run, not repeated here.

## The problem

- Fork PR #67 (`e985ccb9e`, port of upstream #29245) added `ggml/src/ggml-sycl/fused-gemm.cpp`.
  It holds grouped `MUL_MAT_ID` and plain `MUL_MAT` dequant-in-GEMM kernels for nine IQ formats.
  They use a sub-group-16 8x16x16 fp16/fp16/fp32 `joint_matrix` shape (source).
- The A770 (`acm-g10`) reports no such matrix combination, so
  `ggml_sycl_fused_dequant_gemm_f16_device_ok()` returns false there and the kernels never run
  (measured earlier on this host, when PR #67's verification claim was corrected).
- JIT builds compile a kernel for the device only on first launch, so the kernels do no harm.
  AOT builds (`GGML_SYCL_DEVICE_ARCH=acm-g10`) compile every kernel at link time, and IGC crashes
  on them. The Arch packager hit this on the `llama.cpp-sycl-f16-git` build of `4e7400c3a`
  (reported: IGC 2.41.5, all 18 IQ instantiations, both the ALHP and the generic IGC build).
  The packager worked around it with a local patch that stubs the two entry points under
  `GGML_SYCL_NO_XMX_GATHER`, plus `-DCMAKE_CXX_FLAGS=-DGGML_SYCL_NO_XMX_GATHER` (packaging repo
  commit `28e9839`, not pushed).

## Reproduction (measured, this host)

IGC `intel-graphics-compiler 1:2.41.5-1.1`, compute-runtime 26.35.39758, oneAPI 2026.1. The
`fused-gemm.cpp` compile command was taken from `compile_commands.json` of an
`-DGGML_SYCL_DEVICE_ARCH=acm-g10` configure. The file was compiled on its own, then linked into
a shared object with `-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
"-device acm-g10" -shared`; that device-link step runs ocloc on the file's kernels.

| variant | compile | AOT device link |
|---|---|---|
| without `-DGGML_SYCL_NO_XMX_GATHER` | rc 0 | **rc 1**: `[acm-g10] IGC: Internal Compiler Error: Floating point exception`, `gen compiler command failed with exit code 245` |
| with `-DGGML_SYCL_NO_XMX_GATHER` | rc 0 | rc 0 |

## The fix

- CMake option `GGML_SYCL_XMX_GATHER` (default ON), in `ggml/src/ggml-sycl/CMakeLists.txt`.
  When it is ON and the lower-cased `GGML_SYCL_DEVICE_ARCH` has a list entry starting with
  `acm`, `dg2`, `xe-hpg` or `12.55`-`12.57`, the kernels are compiled out and CMake prints why.
  OFF always compiles them out. Compiling out defines `GGML_SYCL_NO_XMX_GATHER` privately for
  `ggml-sycl`.
- `fused-gemm.cpp`: under `GGML_SYCL_NO_XMX_GATHER` the whole kernel body is excluded, and
  `ggml_sycl_fused_dequant_gemm_f16_device_ok()`, `ggml_sycl_fused_dequant_gemm_f16()` and
  `ggml_sycl_grouped_dequant_gemm_f16()` return false.
- `ggml-sycl.cpp`: the startup info block prints
  `GGML_SYCL_XMX_GATHER_TYPES: XMX gather GEMMs disabled by compile flag` in that build.

Configure-only check of the detection (measured, `fused-gemm.cpp` entry in `compile_commands.json`):

| `GGML_SYCL_DEVICE_ARCH` | result |
|---|---|
| (empty, JIT) | kernels built |
| `acm-g10` | compiled out, message printed |
| `bmg-g21` | kernels built |
| `acm-g10,bmg-g21` | compiled out, message printed |
| `12.55.8` | compiled out, message printed |
| `ACM-G11` | compiled out, message printed |
| `xe2-hpg` | kernels built |

A default JIT build (`~/build-pr-28414`, targets `ggml-sycl llama-server
test-sycl-turbo-correctness`) completed with rc 0, and its `compile_commands.json` has no
`GGML_SYCL_NO_XMX_GATHER` (measured).

## Choices

1. **Detect in CMake, not by preprocessor or compiler architecture macros.** One set of sources
   is compiled once for all AOT targets, so per-target specialization inside the file
   (`sycl_ext_oneapi_device_architecture`) would be the only finer-grained option. It was not
   used: it adds device-code branches to kernels that DG2 can never run, and it has not been
   tried with this compiler.
2. **The whole target list loses the kernels when any entry is DG2.** This is simple and safe.
   The cost falls only on mixed lists such as `acm-g10,bmg-g21`, which are documented as such.
3. **Keep the packager's macro name** (`GGML_SYCL_NO_XMX_GATHER`), so the existing PKGBUILD flag
   stays harmless until it is removed, and the local patch simply stops applying.
4. **Also stub `..._device_ok()`**, which the packager's patch left live. A build without the
   kernels should not report the device as able to run them.
5. **Match only verified DG2 spellings.** `acm*`, `dg2*` and `xe-hpg` are the ocloc names for
   DG2. The IP versions are `12.55`-`12.57`; `12.55.8` is the A770 as `sycl-ls` reports it.
   Flex/ATS-M product names were deliberately left out, because it was not checked which names
   ocloc uses for them. Anyone targeting those uses `GGML_SYCL_XMX_GATHER=OFF`.
6. **No sub-group-8 variant.** The only sub-group-8 `joint_matrix` data on the A770 is the fork's
   XMX flash-attention kernel, which measured 4-7x slower than the vector FA kernel. A GEMM
   variant would be new kernel work with no evidence it pays off. Not pursued.

## Not verified here

- A full AOT build of the whole tree with this branch. The packager's full `acm-g10` AOT build of
  `4e7400c3a` with the equivalent stub patch passed the correctness gate, printed the expected
  version, and ran Ornith correctly (reported). This branch compiles out a superset of what that
  patch did (the pack kernels and helpers too). Only the single-file AOT link above was run here.
- The runtime behavior of a JIT build on the A770 after this change. The kernels are still
  present in JIT builds, and the only source change there is the `#ifndef` wrapper and the log
  line, so the correctness gate was not rerun.
- Other DG2 variants (`acm-g11`, `acm-g12`): matched by the detection, not compiled here.
