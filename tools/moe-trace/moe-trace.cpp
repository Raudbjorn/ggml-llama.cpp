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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
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
        t = t->op == GGML_OP_MUL_MAT ? t->src[1] : nullptr; // a LoRA router hides its input
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
    if (layer_topk >= 0) {
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

static bool write_header(FILE * out, const llama_model * model, const common_params & params,
                         uint32_t seed, size_t n_prompt, uint64_t session, const char * kind) {
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
    if (!n_expert.empty()) {
        failed |= fprintf(out, "# n_expert=%s\n", n_expert.c_str()) < 0;
    }
    if (!n_used.empty()) {
        failed |= fprintf(out, "# n_expert_used=%s\n", n_used.c_str()) < 0;
    }
    failed |= fprintf(out, "# n_prompt=%zu n_predict=%d n_ctx=%d n_batch=%d n_ubatch=%d\n",
                  n_prompt, params.n_predict, params.n_ctx, params.n_batch, params.n_ubatch) < 0;
    failed |= fprintf(out, "# sampling seed=%u temp=%.4f top_k=%d top_p=%.4f min_p=%.4f\n",
                  seed, s.temp, s.top_k, s.top_p, s.min_p) < 0;
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
static int run_trace(common_params & params, trace_state & ts) {
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

    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    if (!smpl) {
        LOG_ERR("%s: failed to initialize the sampler\n", __func__);
        return 1;
    }
    const uint32_t seed = llama_sampler_get_seed(common_sampler_get(smpl.get()));
    const uint64_t session = new_session_id();

    const bool spec_on = params.moe_spec_router_k > 0;
    const bool hid_on  = !params.moe_trace_hidden_out.empty();
    if ((spec_on && same_file(params.moe_trace_spec_out, params.out_file)) ||
        (hid_on && same_file(params.moe_trace_hidden_out, params.out_file)) ||
        (spec_on && hid_on && same_file(params.moe_trace_spec_out, params.moe_trace_hidden_out))) {
        LOG_ERR("%s: -o, --spec-out and --hidden-out must name different files\n", __func__);
        return 1;
    }
    // opening an output truncates it: none may be the model file, which is mapped and read
    if (same_file(params.out_file, params.model.path) || (spec_on && same_file(params.moe_trace_spec_out, params.model.path)) ||
        (hid_on && same_file(params.moe_trace_hidden_out, params.model.path))) {
        LOG_ERR("%s: -o, --spec-out and --hidden-out must not name the model file '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    if (params.moe_spec_router_k > 0) {
        const std::string arch     = meta_str(model, "general.architecture");
        const std::string n_expert = arch.empty() ? std::string() : meta_str(model, arch + ".expert_count");
        if (!n_expert.empty() && params.moe_spec_router_k > atoi(n_expert.c_str())) {
            LOG_ERR("%s: --spec-router %d exceeds the model's %s experts\n", __func__, params.moe_spec_router_k, n_expert.c_str());
            return 1;
        }
    }

    ts.out = fopen(params.out_file.c_str(), "w");
    if (ts.out == nullptr) {
        LOG_ERR("%s: cannot open '%s' for writing: %s\n", __func__, params.out_file.c_str(), strerror(errno));
        return 1;
    }
    if (!write_header(ts.out, model, params, seed, tokens.size(), session, nullptr)) {
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
        if (!write_header(ts.spec_out, model, params, seed, tokens.size(), session, kind.c_str())) {
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
    for (size_t i = 0; i < tokens.size() && decode_ok && !ts.write_failed; i += n_chunk) {
        const int32_t n_eval = (int32_t) std::min<size_t>(tokens.size() - i, n_chunk);
        ts.batch_start = (int64_t) i;
        ts.n_eval      = n_eval;
        if (llama_decode(lctx, llama_batch_get_one(tokens.data() + i, n_eval)) != 0) {
            LOG_ERR("%s: prompt decode failed at token %zu of %zu\n", __func__, i, tokens.size());
            decode_ok = false;
        }
    }

    // the continuation is sampled with the command-line sampling flags so that
    // the routing follows the generated tokens a server would see
    ts.prefill = false;
    ts.n_eval  = 1;
    int n_decoded = 0;
    for (int i = 0; i < params.n_predict && decode_ok && !ts.write_failed; i++) {
        llama_token tok = common_sampler_sample(smpl.get(), lctx, -1);
        common_sampler_accept(smpl.get(), tok, true);
        if (llama_vocab_is_eog(vocab, tok)) {
            LOG_INF("%s: end of generation after %d decode steps\n", __func__, i);
            break;
        }
        ts.decode_step = i;
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

    const int ret = run_trace(params, ts);

    llama_backend_free();
    common_log_flush(common_log_main());
    return ret;
}
