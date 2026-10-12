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
#include <random>
#include <string>
#include <vector>

static const char * TOPK_PREFIX = "ffn_moe_topk-";

struct trace_state {
    FILE *  out          = nullptr;
    bool    prefill      = true;
    int64_t batch_start  = 0; // prompt index of the first token of the current decode call
    int32_t n_eval       = 0; // tokens in the current decode call (one ubatch)
    int64_t decode_step  = 0;
    bool    write_failed = false;

    uint64_t rows_written  = 0;
    uint64_t rows_unmapped = 0;

    std::vector<int32_t> buf;
};

static bool parse_layer(const char * name, int & layer) {
    const size_t n = strlen(TOPK_PREFIX);
    if (strncmp(name, TOPK_PREFIX, n) != 0) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long v = strtol(name + n, &end, 10);
    if (errno != 0 || end == name + n || *end != '\0' || v < 0 || v > INT32_MAX) {
        return false;
    }
    layer = (int) v;
    return true;
}

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    trace_state * ts = (trace_state *) user_data;

    int layer = -1;
    const bool is_topk = parse_layer(t->name, layer);
    if (ask) {
        return is_topk;
    }
    if (!is_topk || t->type != GGML_TYPE_I32 || ts->write_failed) {
        return true;
    }

    const int64_t n_used = t->ne[0];
    const int64_t n_rows = t->ne[1];

    // the topk tensor is a strided view over the argsort output: read the full
    // byte range and index it with nb[]
    const size_t nbytes = ggml_nbytes(t);
    ts->buf.resize((nbytes + sizeof(int32_t) - 1) / sizeof(int32_t));
    ggml_backend_tensor_get(t, ts->buf.data(), 0, nbytes);
    const char * base = (const char *) ts->buf.data();

    // the last layer of many models only keeps the output rows; outputs are
    // requested for the last token of each decode call only
    if (n_rows > ts->n_eval) {
        ts->rows_unmapped += (uint64_t) n_rows;
        return true;
    }
    const int64_t first = ts->batch_start + (ts->n_eval - n_rows);

    for (int64_t j = 0; j < n_rows; j++) {
        const int64_t pos = ts->prefill ? -(first + j + 1) : ts->decode_step;
        if (fprintf(ts->out, "%" PRId64 ",%d", pos, layer) < 0) {
            ts->write_failed = true;
            return true;
        }
        for (int64_t i = 0; i < n_used; i++) {
            const int32_t id = *(const int32_t *)(base + j*t->nb[1] + i*t->nb[0]);
            if (fprintf(ts->out, ",%d", id) < 0) {
                ts->write_failed = true;
                return true;
            }
        }
        if (fputc('\n', ts->out) == EOF) {
            ts->write_failed = true;
            return true;
        }
        ts->rows_written++;
    }
    return true;
}

static std::string meta_str(const llama_model * model, const std::string & key) {
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf);
}

static bool write_header(FILE * out, const llama_model * model, const common_params & params,
                         uint32_t seed, size_t n_prompt, uint64_t session) {
    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    const std::string arch     = meta_str(model, "general.architecture");
    const std::string n_expert = arch.empty() ? std::string() : meta_str(model, arch + ".expert_count");
    const std::string n_used   = arch.empty() ? std::string() : meta_str(model, arch + ".expert_used_count");

    const auto & s = params.sampling;
    bool failed = false;
    failed |= fprintf(out, "# llama-moe-trace v1\n") < 0;
    failed |= fprintf(out, "# session=%016" PRIx64 "\n", session) < 0;
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

static uint64_t new_session_id() {
    std::random_device rd;
    const uint64_t t = (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
    return t ^ ((uint64_t) rd() << 32) ^ rd();
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

    ts.out = fopen(params.out_file.c_str(), "w");
    if (ts.out == nullptr) {
        LOG_ERR("%s: cannot open '%s' for writing: %s\n", __func__, params.out_file.c_str(), strerror(errno));
        return 1;
    }
    if (!write_header(ts.out, model, params, seed, tokens.size(), new_session_id())) {
        ts.write_failed = true;
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

    const bool flush_failed = fflush(ts.out) != 0 || ferror(ts.out) != 0;
    const bool close_failed = fclose(ts.out) != 0;
    ts.out = nullptr;

    if (ts.rows_unmapped > 0) {
        LOG_WRN("%s: %" PRIu64 " rows had more tokens than the decode call and were skipped\n", __func__, ts.rows_unmapped);
    }
    if (ts.write_failed || flush_failed || close_failed) {
        LOG_ERR("%s: writing '%s' failed: %s\n", __func__, params.out_file.c_str(), strerror(errno));
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
