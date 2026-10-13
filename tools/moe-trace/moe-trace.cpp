// llama-moe-trace: record the routed expert ids of every MoE layer for a
// prompt plus a sampled continuation.
//
// The ids are read from the "ffn_moe_topk-<il>" tensors through the scheduler
// eval callback, so the trace does not depend on the backend that computed the
// router. Tracing synchronizes on every captured tensor: a traced run is a
// routing sample, not a throughput measurement.
//
// Output (CSV, version 1):
//   # llama-moe-trace v1        provenance lines start with '#'
//   # session=<hex>             one per run; concatenated traces keep sessions apart
//   pos,layer,id0,...,idK-1     header row, then one row per (token, layer)
// Prompt rows use pos = -(p + 1) for prompt token p; decode rows use the
// decode step 0, 1, ... so that pos >= 0 selects decode routing.
//
// Research diagnostics, off by default (README.md):
//   --spec-router K   the top K of each layer's router evaluated on the layer input
//                     before attention ("ffn_moe_spec_topk-<il>"), to --spec-out in
//                     the same format, and the recall of the routed experts
//   --hidden-out F    decode-step hidden states, binary records after a text header

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "../../src/llama-ext.h" // llama_model_get_moe_tensor_info, llama_model_router_fingerprint

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

static const char * TOPK_PREFIX = "ffn_moe_topk-";
static const char * SPEC_PREFIX = "ffn_moe_spec_topk-";

// hidden-state taps; the residual after attention has two names among the builders
enum hidden_tag : int32_t { HID_LAYER_OUT = 0, HID_RESIDUAL = 1, HID_ROUTER_IN = 2 };
static const struct { const char * prefix; hidden_tag tag; } HIDDEN_TAPS[] = {
    { "l_out-",          HID_LAYER_OUT },
    { "ffn_inp-",        HID_RESIDUAL  },
    { "attn_residual-",  HID_RESIDUAL  },
    { "ffn_moe_logits-", HID_ROUTER_IN }, // the router matmul; its src[1] is the router input
};

struct trace_state {
    FILE *  out          = nullptr;
    bool    prefill      = true;
    int64_t batch_start  = 0; // prompt index of the first token of the current decode call
    int32_t n_eval       = 0; // tokens in the current decode call (one ubatch)
    int64_t decode_step  = 0;
    bool    write_failed = false;

    uint64_t rows_written  = 0;
    uint64_t rows_unmapped = 0;

    // a layer may route twice (GroveMoE's adjugate experts reuse ffn_moe_topk-<il>): only the first
    // router of a layer in each decode call, the main experts that packs and simulate.py model, is kept
    int64_t              epoch = 0; // the current decode call
    std::vector<int64_t> topk_epoch;
    uint64_t             rows_second_group = 0;

    // --spec-router: per layer, the latest decode-step ids of both routers, matched by step
    FILE *   spec_out  = nullptr;
    uint64_t spec_rows = 0;
    std::vector<std::vector<int32_t>> spec_ids, act_ids;
    std::vector<int64_t>  spec_step, act_step;
    std::vector<uint64_t> spec_hits, spec_total;

    // --hidden-out
    FILE *   hid_out     = nullptr;
    int32_t  hid_n       = 0;
    uint64_t hid_records = 0;
    int      router_wrapped = -1; // a layer whose router is not a plain matmul (a LoRA adapter wraps it)

    std::vector<int32_t> buf;
    std::vector<float>   fbuf;
};

// layer index after prefix, or -1
static int parse_layer(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    char * end = nullptr;
    errno = 0;
    const long v = strtol(name + n, &end, 10);
    if (errno != 0 || end == name + n || *end != '\0' || v < 0 || v > INT32_MAX) {
        return -1;
    }
    return (int) v;
}

static int hidden_tap(const char * name, hidden_tag & tag) {
    for (const auto & tap : HIDDEN_TAPS) {
        const int layer = parse_layer(name, tap.prefix);
        if (layer >= 0) {
            tag = tap.tag;
            return layer;
        }
    }
    return -1;
}

static void ensure_layer(trace_state * ts, int layer) {
    if ((size_t) layer < ts->spec_step.size()) {
        return;
    }
    const size_t n = (size_t) layer + 1;
    ts->spec_ids.resize(n);
    ts->act_ids.resize(n);
    ts->spec_step.resize(n, -1);
    ts->act_step.resize(n, -1);
    ts->spec_hits.resize(n, 0);
    ts->spec_total.resize(n, 0);
}

// recall of the routed experts among the predicted ones, once both are known for a step
static void match_spec(trace_state * ts, int layer) {
    if (ts->spec_step[layer] < 0 || ts->spec_step[layer] != ts->act_step[layer]) {
        return;
    }
    const std::vector<int32_t> & pred = ts->spec_ids[layer];
    for (int32_t id : ts->act_ids[layer]) {
        ts->spec_hits[layer] += std::find(pred.begin(), pred.end(), id) != pred.end();
    }
    ts->spec_total[layer] += ts->act_ids[layer].size();
    ts->spec_step[layer] = ts->act_step[layer] = -1;
}

// one row per token of an id tensor; the top-k tensors are strided views over the
// argsort output, so the full byte range is read and indexed with nb[]
static void write_ids(trace_state * ts, const ggml_tensor * t, int layer, FILE * dst, uint64_t & rows,
                      std::vector<int64_t> * step, std::vector<std::vector<int32_t>> * keep) {
    const int64_t n_used = t->ne[0];
    const int64_t n_rows = t->ne[1];

    const size_t nbytes = ggml_nbytes(t);
    ts->buf.resize((nbytes + sizeof(int32_t) - 1) / sizeof(int32_t));
    ggml_backend_tensor_get(t, ts->buf.data(), 0, nbytes);
    const char * base = (const char *) ts->buf.data();

    // the last layer of many models only keeps the output rows; outputs are
    // requested for the last token of each decode call only
    if (n_rows > ts->n_eval) {
        ts->rows_unmapped += (uint64_t) n_rows;
        return;
    }
    const int64_t first = ts->batch_start + (ts->n_eval - n_rows);

    for (int64_t j = 0; j < n_rows; j++) {
        const int64_t pos = ts->prefill ? -(first + j + 1) : ts->decode_step;
        if (fprintf(dst, "%" PRId64 ",%d", pos, layer) < 0) {
            ts->write_failed = true;
            return;
        }
        for (int64_t i = 0; i < n_used; i++) {
            const int32_t id = *(const int32_t *)(base + j*t->nb[1] + i*t->nb[0]);
            if (fprintf(dst, ",%d", id) < 0) {
                ts->write_failed = true;
                return;
            }
        }
        if (fputc('\n', dst) == EOF) {
            ts->write_failed = true;
            return;
        }
        rows++;
    }

    if (step && keep && !ts->prefill && n_rows == 1) {
        ensure_layer(ts, layer);
        std::vector<int32_t> & ids = (*keep)[layer];
        ids.resize((size_t) n_used);
        for (int64_t i = 0; i < n_used; i++) {
            ids[i] = *(const int32_t *)(base + i*t->nb[0]);
        }
        (*step)[layer] = ts->decode_step;
        match_spec(ts, layer);
    }
}

static void write_hidden(trace_state * ts, const ggml_tensor * t, hidden_tag tag, int layer) {
    if (tag == HID_ROUTER_IN) {
        t = t->op == GGML_OP_MUL_MAT ? t->src[1] : nullptr; // a wrapped router stops the run (trace_cb)
    }
    if (t == nullptr || t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[2] != 1 || t->ne[3] != 1) {
        return;
    }
    const int64_t n_embd = t->ne[0];
    const int64_t n_rows = t->ne[1];
    const size_t nbytes = ggml_nbytes(t);
    ts->fbuf.resize((nbytes + sizeof(float) - 1) / sizeof(float));
    ggml_backend_tensor_get(t, ts->fbuf.data(), 0, nbytes);
    const char * base = (const char *) ts->fbuf.data();
    for (int64_t j = 0; j < n_rows; j++) {
        const int32_t hdr[4] = { (int32_t) ts->decode_step, layer, (int32_t) tag, (int32_t) n_embd };
        if (fwrite(hdr, sizeof(hdr), 1, ts->hid_out) != 1 ||
            fwrite(base + j*t->nb[1], sizeof(float), (size_t) n_embd, ts->hid_out) != (size_t) n_embd) {
            ts->write_failed = true;
            return;
        }
        ts->hid_records++;
    }
}

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    trace_state * ts = (trace_state *) user_data;

    const int  layer_topk = parse_layer(t->name, TOPK_PREFIX);
    const int  layer_spec = ts->spec_out ? parse_layer(t->name, SPEC_PREFIX) : -1;
    hidden_tag tag        = HID_LAYER_OUT;
    const bool hidden_now = ts->hid_out && !ts->prefill && ts->decode_step < ts->hid_n;
    const int  layer_hid  = hidden_now ? hidden_tap(t->name, tag) : -1;
    if (ask) {
        // build_lora_mm() adds an adapter's product to the router matmul. The router input is then
        // no source of this tensor, and its buffer may be reused by the time the sum is computed, so
        // the run stops instead of dropping or misreading the record (seen on the prompt already)
        hidden_tag tag_any = HID_LAYER_OUT;
        const int  layer_any = ts->hid_out && t->op != GGML_OP_MUL_MAT ? hidden_tap(t->name, tag_any) : -1;
        if (layer_any >= 0 && tag_any == HID_ROUTER_IN && ts->router_wrapped < 0) {
            ts->router_wrapped = layer_any;
        }
        return layer_topk >= 0 || layer_spec >= 0 || layer_hid >= 0;
    }
    if (ts->write_failed) {
        return true;
    }
    if (layer_hid >= 0) {
        write_hidden(ts, t, tag, layer_hid);
    }
    if (t->type != GGML_TYPE_I32) {
        return true;
    }
    if (layer_topk >= 0 && (size_t) layer_topk < ts->topk_epoch.size() && ts->topk_epoch[layer_topk] == ts->epoch) {
        ts->rows_second_group += (uint64_t) t->ne[1];
    } else if (layer_topk >= 0) {
        if ((size_t) layer_topk >= ts->topk_epoch.size()) {
            ts->topk_epoch.resize((size_t) layer_topk + 1, -1);
        }
        ts->topk_epoch[layer_topk] = ts->epoch;
        write_ids(ts, t, layer_topk, ts->out, ts->rows_written,
                  ts->spec_out ? &ts->act_step : nullptr, ts->spec_out ? &ts->act_ids : nullptr);
    } else if (layer_spec >= 0) {
        write_ids(ts, t, layer_spec, ts->spec_out, ts->spec_rows, &ts->spec_step, &ts->spec_ids);
    }
    return true;
}

static std::string meta_str(const llama_model * model, const std::string & key) {
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf);
}

// the routed expert tensors of each MoE layer as TYPE/BYTES/N_IN/N_OUT per expert, in the order
// build_moe_ffn multiplies them: the --moe-cache provider keeps one pool per weight type and expert
// size, so simulate.py replays them per pool, and the SYCL provider holds back scratch sized by
// N_IN and N_OUT. Only the main experts: a second router of the layer (GroveMoE's
// _chexps) is not traced
static bool write_expert_tensors(FILE * out, const llama_model * model) {
    static const char * const order[] = {
        "ffn_gate_up_exps.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight",
    };
    std::vector<llama_moe_tensor_info> info(llama_model_get_moe_tensor_info(model, nullptr, 0));
    info.resize(std::min(info.size(), llama_model_get_moe_tensor_info(model, info.data(), info.size())));
    std::map<int64_t, std::map<size_t, std::string>> by_layer; // layer -> rank in order -> entry
    for (const llama_moe_tensor_info & t : info) {
        const char * dot = t.name && t.layer >= 0 ? strchr(t.name + 4, '.') : nullptr; // after "blk.N"
        for (size_t rank = 0; dot && rank < std::size(order); rank++) {
            if (strcmp(dot + 1, order[rank]) == 0) {
                by_layer[t.layer][rank] = std::string(ggml_type_name(t.type)) + "/" + std::to_string(t.expert_size) +
                                          "/" + std::to_string(t.n_input) + "/" + std::to_string(t.n_output);
            }
        }
    }
    bool failed = false;
    for (const auto & [layer, tensors] : by_layer) {
        std::string line;
        for (const auto & [rank, entry] : tensors) {
            line += (line.empty() ? "" : ",") + entry;
        }
        failed |= fprintf(out, "# expert_tensors=%lld:%s\n", (long long) layer, line.c_str()) < 0;
    }
    return !failed;
}

// FNV-1a 64, a stable identity for the sampling inputs a header line cannot spell out
static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

// "none", or COUNT:HASH
static std::string fingerprint(size_t count, uint64_t hash) {
    char buf[48];
    snprintf(buf, sizeof(buf), "%zu:%016" PRIx64, count, hash);
    return count ? std::string(buf) : std::string("none");
}

// a float as the shortest text that reads back as the same value
static std::string exact(float v) {
    char buf[32];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return res.ec == std::errc() ? std::string(buf, res.ptr) : std::string("?");
}

// every sampling control that can change the continuation, and with it the routing: the sampler
// chain in order, the parameters common_params_sampling::print() shows (its names, but every float
// exact: print() rounds to three decimals, and repeat penalties 0.9996 and 1.0004 differ), the ones
// it leaves out, and identities of the grammar, the logit biases and the DRY breakers
static bool write_sampling(FILE * out, const common_params_sampling & s, const std::string & chain) {
    const std::pair<const char *, std::string> values[] = {
        { "repeat_last_n",      std::to_string(s.penalty_last_n) },
        { "repeat_penalty",     exact(s.penalty_repeat) },
        { "frequency_penalty",  exact(s.penalty_freq) },
        { "presence_penalty",   exact(s.penalty_present) },
        { "dry_multiplier",     exact(s.dry_multiplier) },
        { "dry_base",           exact(s.dry_base) },
        { "dry_allowed_length", std::to_string(s.dry_allowed_length) },
        { "dry_penalty_last_n", std::to_string(s.dry_penalty_last_n) },
        { "top_k",              std::to_string(s.top_k) },
        { "top_p",              exact(s.top_p) },
        { "min_p",              exact(s.min_p) },
        { "xtc_probability",    exact(s.xtc_probability) },
        { "xtc_threshold",      exact(s.xtc_threshold) },
        { "typical_p",          exact(s.typ_p) },
        { "top_n_sigma",        exact(s.top_n_sigma) },
        { "temp",               exact(s.temp) },
        { "mirostat",           std::to_string(s.mirostat) },
        { "mirostat_lr",        exact(s.mirostat_eta) },
        { "mirostat_ent",       exact(s.mirostat_tau) },
        { "adaptive_target",    exact(s.adaptive_target) },
        { "adaptive_decay",     exact(s.adaptive_decay) },
    };
    std::string params;
    for (const auto & [name, value] : values) {
        params += (params.empty() ? "" : " ") + std::string(name) + "=" + value;
    }
    uint64_t bias = fnv1a(nullptr, 0);
    for (const llama_logit_bias & b : s.logit_bias) {
        bias = fnv1a(&b.bias, sizeof(b.bias), fnv1a(&b.token, sizeof(b.token), bias));
    }
    uint64_t breakers = fnv1a(nullptr, 0);
    for (const std::string & b : s.dry_sequence_breakers) {
        breakers = fnv1a("", 1, fnv1a(b.data(), b.size(), breakers)); // NUL-separated
    }
    uint64_t triggers = fnv1a(nullptr, 0);
    for (const common_grammar_trigger & t : s.grammar_triggers) {
        const int32_t type = (int32_t) t.type;
        triggers = fnv1a(&type, sizeof(type), triggers);
        triggers = fnv1a("", 1, fnv1a(t.value.data(), t.value.size(), triggers));
        triggers = fnv1a(&t.token, sizeof(t.token), triggers);
    }
    const std::string & grammar = s.grammar.grammar;
    bool failed = fprintf(out, "# sampling_chain=%s\n", chain.c_str()) < 0;
    failed |= fprintf(out, "# sampling_params=%s\n", params.c_str()) < 0;
    failed |= fprintf(out, "# sampling_inputs=n_prev=%d min_keep=%d dynatemp_range=%s dynatemp_exponent=%s "
                      "ignore_eos=%d backend_sampling=%d logit_bias=%s dry_breakers=%s grammar=%s grammar_lazy=%d "
                      "grammar_triggers=%s\n",
                      s.n_prev, s.min_keep, exact(s.dynatemp_range).c_str(), exact(s.dynatemp_exponent).c_str(), (int) s.ignore_eos,
                      (int) s.backend_sampling, fingerprint(s.logit_bias.size(), bias).c_str(),
                      fingerprint(s.dry_sequence_breakers.size(), breakers).c_str(),
                      fingerprint(grammar.size(), fnv1a(grammar.data(), grammar.size())).c_str(), (int) s.grammar_lazy,
                      fingerprint(s.grammar_triggers.size(), triggers).c_str()) < 0;
    return !failed;
}

// the prompt's token ids, each as 4 little-endian bytes, as COUNT:FNV-1a-64
static std::string prompt_identity(const std::vector<llama_token> & tokens) {
    uint64_t h = fnv1a(nullptr, 0);
    for (const llama_token t : tokens) {
        const uint32_t u = (uint32_t) t;
        const unsigned char b[4] = { (unsigned char) u, (unsigned char) (u >> 8), (unsigned char) (u >> 16),
                                     (unsigned char) (u >> 24) };
        h = fnv1a(b, sizeof(b), h);
    }
    return fingerprint(tokens.size(), h);
}

static bool write_header(FILE * out, const llama_model * model, const llama_context * lctx, const common_params & params,
                         uint32_t seed, const std::string & chain, const std::vector<llama_token> & prompt,
                         uint64_t session, uint64_t router, const char * kind) {
    const size_t n_prompt = prompt.size();
    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    const std::string arch     = meta_str(model, "general.architecture");
    const std::string n_expert = arch.empty() ? std::string() : meta_str(model, arch + ".expert_count");
    const std::string n_used   = arch.empty() ? std::string() : meta_str(model, arch + ".expert_used_count");

    const auto & s = params.sampling;
    bool failed = false;
    failed |= fprintf(out, "# llama-moe-trace v1\n") < 0;
    failed |= fprintf(out, "# session=%016" PRIx64 "\n", session) < 0;
    if (kind) {
        failed |= fprintf(out, "# %s\n", kind) < 0;
    }
    failed |= fprintf(out, "# model=%s\n", params.model.path.c_str()) < 0;
    failed |= fprintf(out, "# model_desc=%s\n", desc) < 0;
    failed |= fprintf(out, "# model_size=%" PRIu64 "\n", llama_model_size(model)) < 0;
    failed |= fprintf(out, "# arch=%s\n", arch.c_str()) < 0;
    failed |= fprintf(out, "# n_layer=%d\n", llama_model_n_layer(model)) < 0;
    // which routers: a fine-tune of the same shapes numbers its experts differently
    if (router != 0) {
        failed |= fprintf(out, "# router_fingerprint=%016" PRIx64 "\n", router) < 0;
    }
    if (!n_expert.empty()) {
        failed |= fprintf(out, "# n_expert=%s\n", n_expert.c_str()) < 0;
    }
    if (!n_used.empty()) {
        failed |= fprintf(out, "# n_expert_used=%s\n", n_used.c_str()) < 0;
    }
    failed |= !write_expert_tensors(out, model);
    // the context's sizes, not the requested ones: -c 0 takes the model's, and n_batch/n_ubatch are capped
    failed |= fprintf(out, "# n_prompt=%zu n_predict=%d n_ctx=%u n_batch=%u n_ubatch=%u\n", n_prompt, params.n_predict,
                      llama_n_ctx(lctx), llama_n_batch(lctx), llama_n_ubatch(lctx)) < 0;
    // which prompt: runs of equal length but different text route differently
    failed |= fprintf(out, "# prompt_tokens=%s\n", prompt_identity(prompt).c_str()) < 0;
    failed |= fprintf(out, "# sampling seed=%u temp=%.4f top_k=%d top_p=%.4f min_p=%.4f\n",
                  seed, s.temp, s.top_k, s.top_p, s.min_p) < 0;
    failed |= !write_sampling(out, s, chain);
    failed |= fprintf(out, "pos,layer,ids...\n") < 0;
    return !failed;
}

// text lines up to "end\n", then the records; the records are in the byte order the header names
static bool write_hidden_header(FILE * out, const llama_model * model, const common_params & params, uint64_t session) {
    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    bool failed = false;
    failed |= fprintf(out, "llama-moe-trace hidden v1\n") < 0;
    failed |= fprintf(out, "session=%016" PRIx64 "\n", session) < 0;
    failed |= fprintf(out, "byte_order=little\n") < 0;
    failed |= fprintf(out, "model=%s\n", params.model.path.c_str()) < 0;
    failed |= fprintf(out, "model_desc=%s\n", desc) < 0;
    failed |= fprintf(out, "model_size=%" PRIu64 "\n", llama_model_size(model)) < 0;
    failed |= fprintf(out, "arch=%s\n", meta_str(model, "general.architecture").c_str()) < 0;
    failed |= fprintf(out, "n_layer=%d\n", llama_model_n_layer(model)) < 0;
    failed |= fprintf(out, "n_embd=%d\n", llama_model_n_embd(model)) < 0;
    failed |= fprintf(out, "max_decode=%d\n", params.moe_trace_hidden_n) < 0; // fewer at end of generation
    failed |= fprintf(out, "tags=0:layer_out,1:residual_after_attention,2:router_input\n") < 0;
    failed |= fprintf(out, "record=pos:i32,layer:i32,tag:i32,n:i32,values:f32[n]\n") < 0;
    failed |= fprintf(out, "end\n") < 0;
    return !failed;
}

// two output paths that name the same file, also through another spelling or a hard link
static bool same_file(const std::string & a, const std::string & b) {
    std::error_code ec;
    if (std::filesystem::equivalent(a, b, ec) && !ec) {
        return true;
    }
    const auto ca = std::filesystem::weakly_canonical(a, ec);
    if (ec) {
        return a == b;
    }
    const auto cb = std::filesystem::weakly_canonical(b, ec);
    return ec ? a == b : ca == cb;
}

#ifndef _WIN32
extern char ** environ;
#endif

// every existing file named on the command line or in a LLAMA_ARG_* variable, except the values
// of the output options: the parser reads adapters, grammar, schema and profile files itself, and
// common_params keeps only their contents. A file named twice, as an output and as an input, stays.
static std::vector<std::string> named_files(int argc, char ** argv) {
    static const char * const outputs[] = { "-o", "--output", "--output-file", "--spec-out", "--hidden-out" };
    const auto is_output = [](const std::string & flag) {
        return std::any_of(std::begin(outputs), std::end(outputs), [&](const char * o) { return flag == o; });
    };
    std::vector<std::string> files;
    const auto add = [&](const std::string & value) {
        std::error_code ec;
        if (!value.empty() && std::filesystem::is_regular_file(value, ec) && !ec) {
            files.push_back(value);
        }
    };
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if (arg.rfind("-", 0) == 0 && eq != std::string::npos) {
            if (!is_output(arg.substr(0, eq))) {
                add(arg.substr(eq + 1));
            }
        } else if (is_output(arg)) {
            i++; // its value is the output
        } else {
            add(arg);
        }
    }
#ifdef _WIN32
    char ** env = _environ;
#else
    char ** env = environ;
#endif
    for (; env != nullptr && *env != nullptr; env++) {
        const std::string kv = *env;
        const size_t eq = kv.find('=');
        if (kv.rfind("LLAMA_ARG_", 0) == 0 && eq != std::string::npos) {
            add(kv.substr(eq + 1));
        }
    }
    return files;
}

// the files the trace reads: every model shard (mapped while the trace runs), the prompt file and
// the files named on the command line
static std::vector<std::string> trace_inputs(const llama_model * model, const common_params & params,
                                             const std::vector<std::string> & named) {
    std::vector<std::string> inputs = { params.model.path };
    inputs.insert(inputs.end(), named.begin(), named.end());
    const std::string count = meta_str(model, "split.count");
    const int n_split = count.empty() ? 0 : atoi(count.c_str());
    // the loader derives the other shard names the same way from the first shard
    std::vector<char> prefix(params.model.path.size() + 1, 0);
    if (n_split > 1 && llama_split_prefix(prefix.data(), prefix.size(), params.model.path.c_str(), 0, n_split) > 0) {
        std::vector<char> path(params.model.path.size() + 32, 0);
        for (int i = 1; i < n_split; i++) {
            if (llama_split_path(path.data(), path.size(), prefix.data(), i, n_split) > 0) {
                inputs.emplace_back(path.data());
            }
        }
    }
    if (!params.prompt_file.empty()) {
        inputs.push_back(params.prompt_file);
    }
    return inputs;
}

static uint64_t new_session_id() {
    std::random_device rd;
    const uint64_t t = (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
    return t ^ ((uint64_t) rd() << 32) ^ rd();
}

// closes f and reports whether everything written to it reached the file
static bool close_file(FILE *& f) {
    if (f == nullptr) {
        return true;
    }
    const bool flush_failed = fflush(f) != 0 || ferror(f) != 0;
    const bool close_failed = fclose(f) != 0;
    f = nullptr;
    return !flush_failed && !close_failed;
}

static void log_spec_recall(const trace_state & ts, int32_t k) {
    uint64_t hits = 0, total = 0;
    double lo = 1.0, hi = 0.0;
    for (size_t il = 0; il < ts.spec_total.size(); il++) {
        if (ts.spec_total[il] == 0) {
            continue;
        }
        hits  += ts.spec_hits[il];
        total += ts.spec_total[il];
        const double r = (double) ts.spec_hits[il] / (double) ts.spec_total[il];
        lo = std::min(lo, r);
        hi = std::max(hi, r);
    }
    if (total == 0) {
        LOG_WRN("%s: no decode step had both routers; recall unknown\n", __func__);
        return;
    }
    LOG_INF("%s: spec router K=%d: recall %.4f of %" PRIu64 " routed decode experts (layers %.4f..%.4f)\n",
            __func__, k, (double) hits / (double) total, total, lo, hi);
}

// returns 0 on a complete trace; owns the model and context so that they are
// freed before the backend
// an output may replace only a file this tool wrote: a non-empty regular file must start with
// one of its headers. A header proves little (a GBNF comment can spell it, and a pack profile is
// a trace), so the files named as inputs are also compared by path (named_files).
static bool written_by_trace(const std::string & path) {
    std::error_code ec;
    const auto st = std::filesystem::status(path, ec);
    if (ec || !std::filesystem::is_regular_file(st) || std::filesystem::file_size(path, ec) == 0 || ec) {
        return true; // absent, a device such as /dev/null, or empty; fopen reports the rest
    }
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return true;
    }
    char head[32] = {};
    const size_t n = fread(head, 1, sizeof(head), f);
    fclose(f);
    const std::string h(head, n);
    return h.rfind("# llama-moe-trace v1\n", 0) == 0 || h.rfind("llama-moe-trace hidden v1\n", 0) == 0;
}

static int run_trace(common_params & params, trace_state & ts, const std::vector<std::string> & named) {
    std::vector<std::string> outs = { params.out_file };
    if (params.moe_spec_router_k > 0) {
        outs.push_back(params.moe_trace_spec_out);
    }
    if (!params.moe_trace_hidden_out.empty()) {
        outs.push_back(params.moe_trace_hidden_out);
    }
    for (const std::string & out : outs) {
        if (!written_by_trace(out)) {
            LOG_ERR("%s: '%s' exists and was not written by llama-moe-trace; remove it or choose another output\n",
                    __func__, out.c_str());
            return 1;
        }
    }

    // each stochastic sampler (dist, XTC, mirostat) resolves a default seed on its own, so resolve
    // one here: the recorded seed then reproduces all of them, as --seed does. Before the context
    // is made: it creates the sampler this tool uses, which --backend-sampling attaches to it
    if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
        std::random_device rd;
        const uint32_t t = (uint32_t) std::chrono::system_clock::now().time_since_epoch().count();
        do {
            params.sampling.seed = rd() ^ t;
        } while (params.sampling.seed == LLAMA_DEFAULT_SEED);
    }

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * lctx  = llama_init ? llama_init->context() : nullptr;
    if (model == nullptr || lctx == nullptr) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(lctx, params.prompt, true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt; pass -p or -f\n", __func__);
        return 1;
    }
    const uint32_t n_ctx = llama_n_ctx(lctx);
    if (tokens.size() + (size_t) params.n_predict > n_ctx) {
        LOG_ERR("%s: prompt (%zu tokens) + -n %d exceeds the context size %u\n",
                __func__, tokens.size(), params.n_predict, n_ctx);
        return 1;
    }

    // the context's own sampler: with --backend-sampling it is the chain the context runs, so the
    // accepted tokens (penalty and DRY history) must reach it, as in llama-completion
    common_sampler * smpl = llama_init->sampler(0);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize the sampler\n", __func__);
        return 1;
    }
    const uint32_t seed = params.sampling.seed;
    std::string chain = common_sampler_print(smpl); // "logits -> a -> b "
    chain.erase(chain.find_last_not_of(' ') + 1);
    const uint64_t session = new_session_id();

    const bool spec_on = params.moe_spec_router_k > 0;
    const bool hid_on  = !params.moe_trace_hidden_out.empty();
    if ((spec_on && same_file(params.moe_trace_spec_out, params.out_file)) ||
        (hid_on && same_file(params.moe_trace_hidden_out, params.out_file)) ||
        (spec_on && hid_on && same_file(params.moe_trace_spec_out, params.moe_trace_hidden_out))) {
        LOG_ERR("%s: -o, --spec-out and --hidden-out must name different files\n", __func__);
        return 1;
    }
    // opening an output truncates it: none may be an input, least of all a mapped model shard
    for (const std::string & in : trace_inputs(model, params, named)) {
        if (same_file(params.out_file, in) || (spec_on && same_file(params.moe_trace_spec_out, in)) ||
            (hid_on && same_file(params.moe_trace_hidden_out, in))) {
            LOG_ERR("%s: -o, --spec-out and --hidden-out must not name the input file '%s'\n", __func__, in.c_str());
            return 1;
        }
    }

    if (params.moe_spec_router_k > 0) {
        const std::string arch     = meta_str(model, "general.architecture");
        const std::string n_expert = arch.empty() ? std::string() : meta_str(model, arch + ".expert_count");
        if (!n_expert.empty() && params.moe_spec_router_k > atoi(n_expert.c_str())) {
            LOG_ERR("%s: --spec-router %d exceeds the model's %s experts\n", __func__, params.moe_spec_router_k, n_expert.c_str());
            return 1;
        }
    }

    const uint64_t router = llama_model_router_fingerprint(model); // reads every router: once
    ts.out = fopen(params.out_file.c_str(), "w");
    if (ts.out == nullptr) {
        LOG_ERR("%s: cannot open '%s' for writing: %s\n", __func__, params.out_file.c_str(), strerror(errno));
        return 1;
    }
    if (!write_header(ts.out, model, lctx, params, seed, chain, tokens, session, router, nullptr)) {
        ts.write_failed = true;
    }
    if (params.moe_spec_router_k > 0) {
        ts.spec_out = fopen(params.moe_trace_spec_out.c_str(), "w");
        if (ts.spec_out == nullptr) {
            LOG_ERR("%s: cannot open '%s' for writing: %s\n", __func__, params.moe_trace_spec_out.c_str(), strerror(errno));
            close_file(ts.out);
            return 1;
        }
        const std::string kind = "kind=spec-router k=" + std::to_string(params.moe_spec_router_k);
        if (!write_header(ts.spec_out, model, lctx, params, seed, chain, tokens, session, router, kind.c_str())) {
            ts.write_failed = true;
        }
    }
    if (!params.moe_trace_hidden_out.empty()) {
        ts.hid_out = fopen(params.moe_trace_hidden_out.c_str(), "wb");
        if (ts.hid_out == nullptr) {
            LOG_ERR("%s: cannot open '%s' for writing: %s\n", __func__, params.moe_trace_hidden_out.c_str(), strerror(errno));
            close_file(ts.out);
            close_file(ts.spec_out);
            return 1;
        }
        ts.hid_n = params.moe_trace_hidden_n;
        if (!write_hidden_header(ts.hid_out, model, params, session)) {
            ts.write_failed = true;
        }
        const double max_mib = (double) std::min(params.n_predict, ts.hid_n) * llama_model_n_layer(model) * 3 *
                               (16.0 + 4.0 * llama_model_n_embd(model)) / 1048576.0;
        LOG_INF("%s: hidden states of up to %d decode steps, at most %.1f MiB, to '%s'\n",
                __func__, ts.hid_n, max_mib, params.moe_trace_hidden_out.c_str());
    }

    // one ubatch per decode call keeps the position of every captured row exact
    const int32_t n_chunk = std::max(1, (int32_t) std::min(llama_n_batch(lctx), llama_n_ubatch(lctx)));
    bool decode_ok = true;

    ts.prefill = true;
    for (size_t i = 0; i < tokens.size() && decode_ok && !ts.write_failed && ts.router_wrapped < 0; i += n_chunk) {
        const int32_t n_eval = (int32_t) std::min<size_t>(tokens.size() - i, n_chunk);
        ts.batch_start = (int64_t) i;
        ts.n_eval      = n_eval;
        ts.epoch++;
        if (llama_decode(lctx, llama_batch_get_one(tokens.data() + i, n_eval)) != 0) {
            LOG_ERR("%s: prompt decode failed at token %zu of %zu\n", __func__, i, tokens.size());
            decode_ok = false;
        }
    }

    // the continuation is sampled with the command-line sampling flags so that
    // the routing follows the generated tokens a server would see; like a server, the
    // sampler's history (repeat, presence, frequency and DRY penalties) starts with the prompt
    for (llama_token t : tokens) {
        common_sampler_accept(smpl, t, false);
    }
    ts.prefill = false;
    ts.n_eval  = 1;
    int n_decoded = 0;
    for (int i = 0; i < params.n_predict && decode_ok && !ts.write_failed && ts.router_wrapped < 0; i++) {
        llama_token tok = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, tok, true);
        if (llama_vocab_is_eog(vocab, tok)) {
            LOG_INF("%s: end of generation after %d decode steps\n", __func__, i);
            break;
        }
        ts.decode_step = i;
        ts.epoch++;
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1)) != 0) {
            LOG_ERR("%s: decode failed at step %d\n", __func__, i);
            decode_ok = false;
            break;
        }
        n_decoded++;
    }

    const bool out_ok  = close_file(ts.out);
    const bool spec_ok = close_file(ts.spec_out);
    const bool hid_ok  = close_file(ts.hid_out);

    if (ts.router_wrapped >= 0) {
        LOG_ERR("%s: --hidden-out cannot record the router input of layer %d: its router is not a plain matmul "
                "(a LoRA adapter on ffn_gate_inp?); run without the adapter or without --hidden-out\n",
                __func__, ts.router_wrapped);
        return 1;
    }

    if (ts.rows_second_group > 0) {
        LOG_INF("%s: %" PRIu64 " rows of a second router in the same layer (GroveMoE adjugate experts) were not "
                "traced; the trace holds the main experts\n", __func__, ts.rows_second_group);
    }
    if (ts.rows_unmapped > 0) {
        LOG_WRN("%s: %" PRIu64 " rows had more tokens than the decode call and were skipped\n", __func__, ts.rows_unmapped);
    }
    if (ts.write_failed || !out_ok || !spec_ok || !hid_ok) {
        LOG_ERR("%s: writing the trace failed: %s\n", __func__, strerror(errno));
        return 1;
    }
    if (!decode_ok) {
        LOG_ERR("%s: trace in '%s' is incomplete (%" PRIu64 " rows)\n", __func__, params.out_file.c_str(), ts.rows_written);
        return 1;
    }
    if (ts.rows_written == 0) {
        LOG_ERR("%s: no '%s*' tensors were observed: the model has no MoE layers or a backend fused them away\n",
                __func__, TOPK_PREFIX);
        return 1;
    }
    if (params.moe_spec_router_k > 0) {
        if (ts.spec_rows == 0) {
            LOG_ERR("%s: no '%s*' tensors were observed: the architecture has no pre-attention router (qwen3moe and "
                    "qwen35moe do)\n", __func__, SPEC_PREFIX);
            return 1;
        }
        log_spec_recall(ts, params.moe_spec_router_k);
    }
    if (!params.moe_trace_hidden_out.empty()) {
        LOG_INF("%s: wrote %" PRIu64 " hidden-state records to '%s'\n", __func__, ts.hid_records, params.moe_trace_hidden_out.c_str());
    }
    LOG_INF("%s: wrote %" PRIu64 " rows (%zu prompt tokens, %d decode steps) to '%s'\n",
            __func__, ts.rows_written, tokens.size(), n_decoded, params.out_file.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MOE_TRACE)) {
        return 1;
    }
    if (params.n_predict <= 0) {
        LOG_ERR("%s: -n must be positive (got %d); the trace is bounded by design\n", __func__, params.n_predict);
        common_log_flush(common_log_main());
        return 1;
    }
    {
        const uint16_t probe = 1;
        uint8_t first = 0;
        memcpy(&first, &probe, 1);
        if (!params.moe_trace_hidden_out.empty() && first != 1) {
            LOG_ERR("%s: --hidden-out writes little-endian records and this host is big-endian\n", __func__);
            common_log_flush(common_log_main());
            return 1;
        }
    }

    common_init();

    // the callback copies every top-k tensor to the host; a warmup run would
    // write rows for a dummy batch
    params.warmup = false;

    trace_state ts;
    params.cb_eval           = trace_cb;
    params.cb_eval_user_data = &ts;

    llama_backend_init();
    llama_numa_init(params.numa);

    const int ret = run_trace(params, ts, named_files(argc, argv));

    llama_backend_free();
    common_log_flush(common_log_main());
    return ret;
}
