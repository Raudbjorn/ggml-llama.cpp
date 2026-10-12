#!/usr/bin/env bash
# End-to-end check of llama-moe-trace on random fixture models: prompt rows
# cover every prompt token across several ubatches, decode rows cover every
# step, provenance lines are present, and a dense model is rejected. When a
# Python interpreter is given, the simulator must accept the trace.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    printf 'usage: %s TEST_LLAMA_ARCHS LLAMA_MOE_TRACE [PYTHON3 SIMULATE_PY]\n' "$0" >&2
    exit 2
fi

archs_bin=$1
trace_bin=$2
python_bin=${3:-}
simulate_py=${4:-}

for bin in "$archs_bin" "$trace_bin"; do
    if [ ! -x "$bin" ]; then
        printf 'required binary is not executable: %s\n' "$bin" >&2
        exit 2
    fi
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-moe-trace.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

"$archs_bin" -o "$work_dir" -a '^(qwen3moe|qwen35moe|llama)$' > "$work_dir/archs.log" 2>&1

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

n_predict=5
for arch in qwen3moe qwen35moe; do
    model="$work_dir/$arch-moe.gguf"
    out="$work_dir/$arch.csv"
    [ -f "$model" ] || fail "fixture model $model was not generated"

    "$trace_bin" -m "$model" -p "one two three four five six seven eight nine ten" \
        -n "$n_predict" -c 128 -b 4 -ub 4 --seed 7 -o "$out" > "$work_dir/$arch.log" 2>&1 \
        || fail "$arch: llama-moe-trace exited with $? (log: $(tail -n 3 "$work_dir/$arch.log"))"

    grep -q '^# llama-moe-trace v1$' "$out" || fail "$arch: missing version line"
    grep -q '^# session=[0-9a-f]\{16\}$' "$out" || fail "$arch: missing session line"
    grep -q '^# n_expert=[0-9][0-9]*$' "$out" || fail "$arch: missing n_expert line"
    grep -q '^# sampling seed=7 ' "$out" || fail "$arch: seed not recorded"

    n_prompt=$(sed -n 's/^# n_prompt=\([0-9]*\) .*/\1/p' "$out")
    [ -n "$n_prompt" ] && [ "$n_prompt" -gt 4 ] || fail "$arch: prompt must span several ubatches (n_prompt=$n_prompt)"

    # layer 0 is never output-pruned: exactly one row per prompt token, labelled -1..-n_prompt
    prompt_pos=$(grep -v '^#' "$out" | awk -F, '$2 == 0 && $1 < 0 { print -$1 }' | sort -n | uniq | tr '\n' ' ')
    expected=$(seq 1 "$n_prompt" | tr '\n' ' ')
    [ "$prompt_pos" = "$expected" ] || fail "$arch: layer 0 prompt rows '$prompt_pos' != '$expected'"
    n_rows=$(grep -v '^#' "$out" | awk -F, '$2 == 0 && $1 < 0' | wc -l)
    [ "$n_rows" -eq "$n_prompt" ] || fail "$arch: $n_rows layer 0 prompt rows for $n_prompt tokens"

    n_decode=$(grep -v '^#' "$out" | awk -F, '$2 == 0 && $1 >= 0' | wc -l)
    [ "$n_decode" -ge 1 ] && [ "$n_decode" -le "$n_predict" ] || fail "$arch: $n_decode decode rows for -n $n_predict"

    if [ -n "$python_bin" ]; then
        "$python_bin" "$simulate_py" "$out" --budgets 0.5 > "$work_dir/$arch.sim" 2>&1 \
            || fail "$arch: simulate.py rejected the trace: $(tail -n 2 "$work_dir/$arch.sim")"
    fi
    printf 'ok: %s (%s prompt, %s decode rows in layer 0)\n' "$arch" "$n_prompt" "$n_decode"
done

if "$trace_bin" -m "$work_dir/llama-dense.gguf" -p "hello world" -n 2 -c 64 \
        -o "$work_dir/dense.csv" > "$work_dir/dense.log" 2>&1; then
    fail "a dense model must not produce a successful trace"
fi
grep -q "no 'ffn_moe_topk-\*' tensors were observed" "$work_dir/dense.log" || fail "dense model: missing diagnostic"
printf 'ok: dense model rejected\n'

if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 2 -c 64 \
        -o "$work_dir/no-such-dir/x.csv" > "$work_dir/badpath.log" 2>&1; then
    fail "an unwritable output path must fail"
fi
printf 'ok: unwritable output rejected\n'
