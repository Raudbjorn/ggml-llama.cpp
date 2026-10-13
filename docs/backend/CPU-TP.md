# CPU as a tensor-parallel member (`--cpu-tp`)

With `--split-mode tensor`, llama.cpp splits every split-capable weight matrix by rows across the
members of a meta device and runs the members' graphs one after the other in launch order,
reducing their partial results. Its default device selection skips the CPU. `--cpu-tp` appends the
CPU as the last member, so a model slightly larger than the accelerators' memory can keep every
layer row-split instead of falling back to layer offload. Ported from thecodacus/llama.cpp
`fable5/cpu-tensor-parallel` (e406c8c2a, 10f8aff0c) with the corrections below. EXPERIMENTAL.

```sh
# explicit split: one value per accelerator, then the CPU
llama-server -m model.gguf -sm tensor --cpu-tp -ts 3,1

# sized from free memory: each accelerator keeps MiB free, the CPU takes the rest
llama-server -m model.gguf -sm tensor --cpu-tp --cpu-tp-margin 1536
```

| Flag | Environment | Meaning |
| --- | --- | --- |
| `--cpu-tp` | `LLAMA_ARG_CPU_TP` | append the CPU to the tensor-parallel members |
| `--cpu-tp-margin MiB` | `LLAMA_ARG_CPU_TP_MARGIN` | size the split from free memory instead of `-ts` |

## Rules

- `--split-mode tensor` is required; an architecture that does not support it
  (`llm_arch_supports_sm_tensor`: DeepSeek2, Nemotron-H, GLM5-Next and others) still fails to
  load, the CPU flag does not change that.
- The accelerators are the explicit `-dev` list or, without one, the devices the default tensor
  split takes. `-dev none` with `--cpu-tp` is an error: the option never turns a CPU-only or
  constrained device choice into GPU use.
- The split comes from exactly one place. `-ts` must hold one value per accelerator and then a
  non-zero value for the CPU; `--cpu-tp-margin` sizes it. Both, neither, a zero CPU share or extra
  values are errors.
- Automatic sizing sums the tensor data of every file of a split GGUF (the source read the size of
  the first file only), gives each accelerator its reported free memory minus the margin, and the
  CPU the rest. It is an estimate of weights only: the margin must hold the KV cache, the compute
  buffers and any draft model on each accelerator. It also treats every tensor as split: tensors
  the meta backend mirrors (norms, router weights, some biases) take their full size on every
  accelerator on top of its share, while the input embedding is counted but stays on the CPU.
  When memory is tight, give `-ts` instead. An accelerator with no more free memory than the
  margin, or a model whose tensors already fit, is an error rather than a silent change of devices.
- `--fit` is not implemented for split mode tensor; it reports that and leaves the parameters as
  given. The source disabled `--fit` itself.

## Status

Checked only functionally: on the cloud test host, Vulkan on Mesa llvmpipe plus the CPU, Qwen3-0.6B
Q8_0 gives perplexity 50.94 (split 1:1) and 50.85 (3:1) against 51.03 with Vulkan alone and 51.36
on the CPU alone (4 chunks of 128 tokens), and automatic sizing produced a working 22.6 / 77.4 %
split. llvmpipe runs on the CPU, so these runs say nothing about speed.

Before relying on it on an A770 host:

- SYCL has no `cpy_tensor_async` (`ggml-sycl.cpp`), so every reduction between the GPU and the CPU
  member goes through the meta device's synchronous generic fallback; the GPU waits for the CPU at
  each split point. Measure decode and prefill against plain layer offload (`-ngl`) and against
  `--moe-cache` / `--moe-pack-*` for MoE models.
- Without Level Zero Sysman (`ZES_ENABLE_SYSMAN=1`) the A770 reports free memory equal to total,
  which makes `--cpu-tp-margin` sizing over-commit VRAM; prefer `-ts` there.
- Single-member tensor splits (`-sm tensor` with one accelerator, without `--cpu-tp`) abort in
  `llama-completion` in this tree ("multi buffers are not supported by the meta backend"),
  independent of this option.
- OpenVINO reports no async support and was not tried as a member.
