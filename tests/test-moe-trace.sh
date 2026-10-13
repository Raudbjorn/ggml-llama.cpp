#!/usr/bin/env bash
# End-to-end check of llama-moe-trace on random fixture models: prompt rows
# cover every prompt token across several ubatches, decode rows cover every
# step, provenance lines are present, and a dense model is rejected. When a
# Python interpreter is given, the simulator must accept the trace. The
# --spec-router and --hidden-out diagnostics must leave the routing unchanged,
# and with numpy the host recomputation (moe_trace_oracle.py) must match them.
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

"$archs_bin" -o "$work_dir" -a '^(qwen3moe|qwen35moe|grovemoe|llama)$' > "$work_dir/archs.log" 2>&1

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

# data rows with each row's ids sorted: the order within a top-k row is not part of the
# routing, and OpenVINO orders a row differently when the graph is partitioned differently
routing_sets() {
    grep -v '^#' "$1" | awk -F, 'NR > 1 {
        n = 0
        for (i = 3; i <= NF; i++) { v[++n] = $i + 0 }
        for (i = 2; i <= n; i++) { x = v[i]; j = i - 1; while (j > 0 && v[j] > x) { v[j + 1] = v[j]; j-- } v[j + 1] = x }
        row = $1 "," $2
        for (i = 1; i <= n; i++) { row = row "," v[i] }
        print row
    }'
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

    # one line per routed layer with its expert tensors (type, bytes, inputs and outputs per expert):
    # separate gate, up and down in qwen3moe, a merged gate_up and down in qwen35moe
    case "$arch" in qwen3moe) n_tensors=3 ;; *) n_tensors=2 ;; esac
    routed=$(grep -v '^#' "$out" | awk -F, 'NR > 1 { print $2 }' | sort -un | tr '\n' ' ')
    entry='[A-Za-z0-9_]+/[0-9]+/[0-9]+/[0-9]+'
    listed=$(grep -E "^# expert_tensors=[0-9]+:$entry(,$entry){$((n_tensors - 1))}\$" "$out" \
        | sed 's/^# expert_tensors=\([0-9]*\):.*/\1/' | sort -un | tr '\n' ' ')
    [ "$listed" = "$routed" ] || fail "$arch: expert_tensors lines for layers '$listed', routed layers '$routed'"

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

    # diagnostics: the same run with the pre-attention router and the hidden-state dump
    diag="$work_dir/$arch-diag.csv"
    spec="$work_dir/$arch-spec.csv"
    hid="$work_dir/$arch.hid"
    "$trace_bin" -m "$model" -p "one two three four five six seven eight nine ten" \
        -n "$n_predict" -c 128 -b 4 -ub 4 --seed 7 -o "$diag" \
        --spec-router 2 --spec-out "$spec" --hidden-out "$hid" --hidden-n 3 > "$work_dir/$arch-diag.log" 2>&1 \
        || fail "$arch: diagnostic trace exited with $? (log: $(tail -n 3 "$work_dir/$arch-diag.log"))"
    cmp -s <(routing_sets "$out") <(routing_sets "$diag") || fail "$arch: the diagnostics changed the routing"
    grep -q '^# kind=spec-router k=2$' "$spec" || fail "$arch: missing spec kind line"
    n_spec0=$(grep -v '^#' "$spec" | awk -F, '$2 == 0 && $1 < 0' | wc -l)
    [ "$n_spec0" -eq "$n_prompt" ] || fail "$arch: $n_spec0 layer 0 spec prompt rows for $n_prompt tokens"
    grep -v '^#' "$spec" | awk -F, 'NR > 1 && NF != 4 { bad = 1 } END { exit bad }' || fail "$arch: spec rows need 2 ids"
    grep -q 'spec router K=2: recall [01]\.[0-9]* of ' "$work_dir/$arch-diag.log" || fail "$arch: no recall line"
    grep -q '^llama-moe-trace hidden v1$' "$hid" || fail "$arch: missing hidden header"

    # predicting every expert must cover every routed one
    n_expert=$(sed -n 's/^# n_expert=\([0-9]*\)$/\1/p' "$out")
    "$trace_bin" -m "$model" -p "one two three" -n 3 -c 64 --seed 7 -o "$work_dir/$arch-all.csv" \
        --spec-router "$n_expert" --spec-out "$work_dir/$arch-all-spec.csv" > "$work_dir/$arch-all.log" 2>&1 \
        || fail "$arch: spec router with K = n_expert failed"
    grep -q "spec router K=$n_expert: recall 1\.0000 " "$work_dir/$arch-all.log" || fail "$arch: recall with K = n_expert is not 1"
    if "$trace_bin" -m "$model" -p "one" -n 1 -c 64 -o "$work_dir/x.csv" --spec-router $((n_expert + 1)) \
            --spec-out "$work_dir/x-spec.csv" > "$work_dir/$arch-big.log" 2>&1; then
        fail "$arch: --spec-router above n_expert must fail"
    fi

    if [ -n "$python_bin" ]; then
        "$python_bin" "$(dirname "$simulate_py")/hidden.py" "$hid" > "$work_dir/$arch.hidsum" 2>&1 \
            || fail "$arch: hidden.py rejected the dump: $(tail -n 2 "$work_dir/$arch.hidsum")"
        n_layer=$(sed -n 's/^# n_layer=\([0-9]*\)$/\1/p' "$out")
        for tag in layer_out residual_after_attention router_input; do
            grep -q "^records $tag: $((3 * n_layer))$" "$work_dir/$arch.hidsum" \
                || fail "$arch: expected $((3 * n_layer)) $tag records: $(cat "$work_dir/$arch.hidsum")"
        done
        if "$python_bin" -c 'import numpy' > /dev/null 2>&1; then
            "$python_bin" "$(dirname "$0")/moe_trace_oracle.py" "$model" "$diag" "$spec" "$hid" 2 \
                || fail "$arch: the host recomputation disagrees"
        else
            printf 'skip: %s host recomputation (no numpy)\n' "$arch"
        fi
    fi
    printf 'ok: %s diagnostics (%s spec rows in layer 0)\n' "$arch" "$n_spec0"
done

# a LoRA adapter on the router wraps its matmul: --hidden-out must stop instead of dropping router_input
if [ -n "$python_bin" ] && "$python_bin" -c 'import numpy' > /dev/null 2>&1; then
    "$python_bin" - "$(dirname "$0")/../gguf-py" "$work_dir/qwen3moe-moe.gguf" "$work_dir/router-lora.gguf" <<'PY' \
        || fail "could not write the router LoRA fixture"
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import gguf
model = gguf.GGUFReader(sys.argv[2])
gate = next(t for t in model.tensors if t.name == "blk.0.ffn_gate_inp.weight")
n_embd, n_expert = int(gate.shape[0]), int(gate.shape[1])
w = gguf.GGUFWriter(sys.argv[3], "qwen3moe")
w.add_string("general.type", "adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", 1.0)
w.add_tensor("blk.0.ffn_gate_inp.weight.lora_a", np.full((2, n_embd), 0.01, dtype=np.float32))
w.add_tensor("blk.0.ffn_gate_inp.weight.lora_b", np.full((n_expert, 2), 0.01, dtype=np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
PY
    "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" --lora "$work_dir/router-lora.gguf" -p "hello" -n 2 -c 64 \
        -o "$work_dir/lora.csv" > "$work_dir/lora.log" 2>&1 || fail "trace with a router LoRA failed: $(tail -n 2 "$work_dir/lora.log")"
    if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" --lora "$work_dir/router-lora.gguf" -p "hello" -n 2 -c 64 \
            -o "$work_dir/lora.csv" --hidden-out "$work_dir/lora.hid" > "$work_dir/lora-hid.log" 2>&1; then
        fail "--hidden-out with a LoRA on the router must fail"
    fi
    grep -q "cannot record the router input of layer 0" "$work_dir/lora-hid.log" || fail "router LoRA: missing diagnostic"
    printf 'ok: --hidden-out with a router LoRA rejected\n'
else
    printf 'skip: router LoRA (no numpy)\n'
fi

# on the CPU: the random dense fixture fails a shape check in OpenVINO (also on master)
if "$trace_bin" -m "$work_dir/llama-dense.gguf" -p "hello world" -n 2 -c 64 -dev none \
        -o "$work_dir/dense.csv" > "$work_dir/dense.log" 2>&1; then
    fail "a dense model must not produce a successful trace"
fi
grep -q "no 'ffn_moe_topk-\*' tensors were observed" "$work_dir/dense.log" || fail "dense model: missing diagnostic"
printf 'ok: dense model rejected\n'

# GroveMoE routes each layer twice, for its main and its adjugate experts, under one tensor name:
# the trace keeps one row per (pos, layer), the main router's
"$trace_bin" -m "$work_dir/grovemoe-moe.gguf" -p "one two three" -n 2 -c 64 --seed 7 -o "$work_dir/grove.csv" \
    > "$work_dir/grove.log" 2>&1 || fail "grovemoe: llama-moe-trace failed: $(tail -n 2 "$work_dir/grove.log")"
dups=$(grep -v '^#' "$work_dir/grove.csv" | awk -F, 'NR > 1 { print $1 "," $2 }' | sort | uniq -d | wc -l)
[ "$dups" -eq 0 ] || fail "grovemoe: $dups (pos, layer) pairs have more than one row"
grep -q "second router in the same layer" "$work_dir/grove.log" || fail "grovemoe: no note about the second router"
# and lists only the main experts' gate, up and down tensors
if grep '^# expert_tensors=' "$work_dir/grove.csv" | grep -vqE '^# expert_tensors=[0-9]+:[^,]+,[^,]+,[^,]+$'; then
    fail "grovemoe: expert_tensors lines must hold the 3 main expert tensors: $(grep '^# expert_tensors=' "$work_dir/grove.csv" | head -n 1)"
fi
if [ -n "$python_bin" ]; then
    "$python_bin" "$simulate_py" "$work_dir/grove.csv" --budgets 0.5 > "$work_dir/grove.sim" 2>&1 \
        || fail "grovemoe: simulate.py rejected the trace: $(tail -n 2 "$work_dir/grove.sim")"
fi
printf 'ok: grovemoe keeps the main router\n'

# the header records the context's sizes: -c 0 takes the model's, n_batch and n_ubatch are capped
"$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 0 -b 4096 -ub 4096 -o "$work_dir/ctx.csv" \
    > "$work_dir/ctx.log" 2>&1 || fail "trace with -c 0 failed: $(tail -n 2 "$work_dir/ctx.log")"
sizes=$(sed -n 's/^# n_prompt=[0-9]* n_predict=[0-9]* n_ctx=\([0-9]*\) n_batch=\([0-9]*\) n_ubatch=\([0-9]*\)$/\1 \2 \3/p' \
    "$work_dir/ctx.csv")
read -r h_ctx h_batch h_ubatch <<< "$sizes"
{ [ -n "$h_ctx" ] && [ "$h_ctx" -gt 0 ] && [ "$h_batch" -le "$h_ctx" ] && [ "$h_ubatch" -le "$h_batch" ]; } \
    || fail "header sizes '$sizes' are not the context's"
printf 'ok: header records the context sizes (%s)\n' "$sizes"

# every sampling control that changes the continuation is in the header: the chain, the
# parameters, and the grammar as an identity
"$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 2 -c 64 --seed 7 -o "$work_dir/smp0.csv" \
    > "$work_dir/smp.log" 2>&1 || fail "default sampling trace failed: $(tail -n 2 "$work_dir/smp.log")"
"$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 2 -c 64 --seed 7 --repeat-penalty 1.25 \
    --samplers 'top_k;temperature' --grammar 'root ::= [^\x01]*' -o "$work_dir/smp1.csv" \
    > "$work_dir/smp.log" 2>&1 || fail "sampling trace failed: $(tail -n 2 "$work_dir/smp.log")"
grep -q '^# sampling_chain=logits -> top-k -> temp-ext -> dist$' "$work_dir/smp1.csv" || fail "--samplers not recorded"
grep -q '^# sampling_params=.* repeat_penalty=1.25 ' "$work_dir/smp1.csv" || fail "--repeat-penalty not recorded"
# the values are exact: 0.9996 and 1.0004 both round to 1.000 at three decimals
for p in 0.9996 1.0004; do
    "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 --seed 7 --repeat-penalty "$p" \
        -o "$work_dir/smp-$p.csv" > "$work_dir/smp.log" 2>&1 || fail "sampling trace failed: $(tail -n 2 "$work_dir/smp.log")"
    grep -q "^# sampling_params=.* repeat_penalty=$p " "$work_dir/smp-$p.csv" || fail "--repeat-penalty $p not recorded exactly"
done
grep -q '^# sampling_inputs=.* grammar=none ' "$work_dir/smp0.csv" || fail "default trace must record no grammar"
grep -q '^# sampling_inputs=.* grammar=17:[0-9a-f]\{16\} ' "$work_dir/smp1.csv" || fail "--grammar not recorded"
[ "$(grep -c '^# sampling' "$work_dir/smp0.csv")" -eq 4 ] || fail "expected 4 sampling lines"
[ "$(grep '^# sampling' "$work_dir/smp0.csv")" != "$(grep '^# sampling' "$work_dir/smp1.csv")" ] \
    || fail "two sampling policies must not share their provenance"
printf 'ok: header records the sampling chain, parameters and grammar\n'

# without --seed, every stochastic sampler must take the recorded seed: XTC (p = 0.5, so its own
# draws decide) replayed with --seed gives the same routing
xtc="--xtc-probability 0.5 --xtc-threshold 0"
# shellcheck disable=SC2086
"$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "one two three four five" -n 8 -c 64 $xtc -o "$work_dir/xtc0.csv" \
    > "$work_dir/xtc.log" 2>&1 || fail "XTC trace failed: $(tail -n 2 "$work_dir/xtc.log")"
seed=$(sed -n 's/^# sampling seed=\([0-9]*\) .*/\1/p' "$work_dir/xtc0.csv")
[ -n "$seed" ] && [ "$seed" != 4294967295 ] || fail "XTC trace recorded no resolved seed ('$seed')"
# shellcheck disable=SC2086
"$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "one two three four five" -n 8 -c 64 $xtc --seed "$seed" \
    -o "$work_dir/xtc1.csv" > "$work_dir/xtc.log" 2>&1 || fail "XTC replay failed: $(tail -n 2 "$work_dir/xtc.log")"
cmp -s <(grep -v '^#' "$work_dir/xtc0.csv") <(grep -v '^#' "$work_dir/xtc1.csv") \
    || fail "an XTC trace replayed with its recorded seed $seed routed differently"
printf 'ok: the recorded seed reproduces every sampler\n'

if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 2 -c 64 \
        -o "$work_dir/no-such-dir/x.csv" > "$work_dir/badpath.log" 2>&1; then
    fail "an unwritable output path must fail"
fi
printf 'ok: unwritable output rejected\n'

for bad in '--spec-router 12abc' '--spec-router -1' '--hidden-n 0'; do
    # shellcheck disable=SC2086
    if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 -o "$work_dir/y.csv" $bad \
            > "$work_dir/badarg.log" 2>&1; then
        fail "'$bad' must be rejected"
    fi
done
printf 'ok: malformed diagnostic options rejected\n'

if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 -o "$work_dir/same.csv" \
        --spec-router 1 --spec-out "$work_dir/./same.csv" > "$work_dir/same.log" 2>&1; then
    fail "--spec-out naming the -o file must be rejected"
fi
grep -q "must name different files" "$work_dir/same.log" || fail "colliding outputs: missing diagnostic"
: > "$work_dir/link-a.csv"
ln -f "$work_dir/link-a.csv" "$work_dir/link-b.csv"
if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 -o "$work_dir/link-a.csv" \
        --spec-router 1 --spec-out "$work_dir/link-b.csv" > "$work_dir/link.log" 2>&1; then
    fail "--spec-out hard-linked to the -o file must be rejected"
fi
grep -q "must name different files" "$work_dir/link.log" || fail "hard-linked outputs: missing diagnostic"
# an output replaces only a file llama-moe-trace wrote: never the mapped model (on a copy)
cp "$work_dir/qwen3moe-moe.gguf" "$work_dir/victim.gguf"
victim_size=$(wc -c < "$work_dir/victim.gguf")
for opt in "-o" "--spec-out"; do
    if "$trace_bin" -m "$work_dir/victim.gguf" -p "hello" -n 1 -c 64 -o "$work_dir/v.csv" --spec-router 1 \
            --spec-out "$work_dir/v-spec.csv" "$opt" "$work_dir/victim.gguf" > "$work_dir/victim.log" 2>&1; then
        fail "$opt naming the model must be rejected"
    fi
    grep -q "was not written by llama-moe-trace" "$work_dir/victim.log" || fail "$opt naming the model: missing diagnostic"
    [ "$(wc -c < "$work_dir/victim.gguf")" = "$victim_size" ] || fail "$opt naming the model truncated it"
done
# nor any other input: the prompt file, a later shard of a split model, an adapter, a control
# vector or a grammar file (stand-ins: the check runs before anything is loaded)
printf 'hello\n' > "$work_dir/prompt.txt"
if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -f "$work_dir/prompt.txt" -n 1 -c 64 \
        -o "$work_dir/prompt.txt" > "$work_dir/pf.log" 2>&1; then
    fail "-o naming the prompt file must be rejected"
fi
grep -q "was not written by llama-moe-trace" "$work_dir/pf.log" || fail "-o naming the prompt file: missing diagnostic"
[ "$(cat "$work_dir/prompt.txt")" = "hello" ] || fail "-o naming the prompt file truncated it"
split_bin="$(dirname "$trace_bin")/llama-gguf-split"
if [ -x "$split_bin" ]; then
    "$split_bin" --split --split-max-tensors 8 "$work_dir/qwen3moe-moe.gguf" "$work_dir/shard" \
        > "$work_dir/split.log" 2>&1 || fail "llama-gguf-split failed: $(tail -n 2 "$work_dir/split.log")"
    first=$(find "$work_dir" -name 'shard-*-of-*.gguf' | sort | head -n 1)
    last=$(find "$work_dir" -name 'shard-*-of-*.gguf' | sort | tail -n 1)
    [ "$first" != "$last" ] || fail "the fixture split into a single shard"
    "$trace_bin" -m "$first" -p "hello" -n 1 -c 64 -o "$work_dir/shard.csv" > "$work_dir/shard-ok.log" 2>&1 \
        || fail "trace of the split model failed: $(tail -n 2 "$work_dir/shard-ok.log")"
    last_size=$(wc -c < "$last")
    if "$trace_bin" -m "$first" -p "hello" -n 1 -c 64 -o "$last" > "$work_dir/shard.log" 2>&1; then
        fail "-o naming a later model shard must be rejected"
    fi
    grep -q "was not written by llama-moe-trace" "$work_dir/shard.log" || fail "-o naming a shard: missing diagnostic"
    [ "$(wc -c < "$last")" = "$last_size" ] || fail "-o naming a shard truncated it"
else
    printf 'skip: shard aliasing (no %s)\n' "$split_bin"
fi
cp "$work_dir/qwen3moe-moe.gguf" "$work_dir/adapter.gguf"
printf 'root ::= "a"\n' > "$work_dir/g.gbnf"
for opt in "--lora $work_dir/adapter.gguf" "--control-vector $work_dir/adapter.gguf" "--grammar-file $work_dir/g.gbnf"; do
    victim=${opt#* }
    before=$(cksum < "$victim")
    # shellcheck disable=SC2086
    if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 $opt -o "$victim" > "$work_dir/in.log" 2>&1; then
        fail "-o naming the $opt file must be rejected"
    fi
    grep -q "was not written by llama-moe-trace" "$work_dir/in.log" || fail "-o naming the $opt file: missing diagnostic"
    [ "$(cksum < "$victim")" = "$before" ] || fail "-o naming the $opt file changed it"
done
# a header is no proof: a grammar may open with the trace's first line as a comment, and a
# pack profile is a trace; both are inputs and must survive
printf '# llama-moe-trace v1\nroot ::= "a"\n' > "$work_dir/spoof.gbnf"
"$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 -o "$work_dir/profile.csv" > "$work_dir/prof.log" 2>&1 \
    || fail "writing a profile trace failed: $(tail -n 2 "$work_dir/prof.log")"
for opt in "--grammar-file $work_dir/spoof.gbnf" "--moe-pack-profile $work_dir/profile.csv --moe-pack-slots 1"; do
    victim=$(printf '%s\n' "$opt" | cut -d' ' -f2)
    before=$(cksum < "$victim")
    # shellcheck disable=SC2086
    if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 $opt -o "$victim" > "$work_dir/in.log" 2>&1; then
        fail "-o naming the input of '$opt' must be rejected"
    fi
    grep -q "must not name the input file" "$work_dir/in.log" || fail "-o naming the input of '$opt': missing diagnostic"
    [ "$(cksum < "$victim")" = "$before" ] || fail "-o naming the input of '$opt' changed it"
done
# a trace may be replaced, unless it is also the prompt file
for _ in 1 2; do
    "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -p "hello" -n 1 -c 64 -o "$work_dir/again.csv" > "$work_dir/again.log" 2>&1 \
        || fail "replacing an earlier trace failed: $(tail -n 2 "$work_dir/again.log")"
done
before=$(cksum < "$work_dir/again.csv")
if "$trace_bin" -m "$work_dir/qwen3moe-moe.gguf" -f "$work_dir/again.csv" -n 1 -c 2048 -o "$work_dir/again.csv" \
        > "$work_dir/again-pf.log" 2>&1; then
    fail "-o naming a trace that is also the prompt file must be rejected"
fi
grep -q "must not name the input file" "$work_dir/again-pf.log" || fail "trace as prompt file: missing diagnostic"
[ "$(cksum < "$work_dir/again.csv")" = "$before" ] || fail "-o naming the prompt file changed it"
printf 'ok: colliding output paths rejected\n'

if "$trace_bin" -m "$work_dir/llama-moe.gguf" -p "hello world" -n 2 -c 64 -o "$work_dir/lm.csv" \
        --spec-router 1 --spec-out "$work_dir/lm-spec.csv" > "$work_dir/lm.log" 2>&1; then
    fail "--spec-router on an architecture without it must fail"
fi
grep -q "no 'ffn_moe_spec_topk-\*' tensors were observed" "$work_dir/lm.log" || fail "llama moe: missing spec router diagnostic"
printf 'ok: spec router on an unsupported architecture rejected\n'
