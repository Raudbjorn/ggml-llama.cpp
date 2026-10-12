// Static MoE expert pack equivalence.
//
// LLAMA_MOE_PACK_TEST_CPU makes the loader build packs for CPU-placed layers,
// which exercises profile parsing, expert selection, the file-based slab
// upload and the hot/cold graph without a GPU. The logits with a pack must
// equal the logits without one: bit-identical for F32 weights, and within a
// small NMSE when the CPU experts are repacked Q4_0 and the pack is plain Q4_0.
// When a GPU device is present the pack is also placed on it, with every layer
// offloaded and the routed experts pinned to the CPU. Those logits must match
// either the device without a pack or the CPU: a device's own deviation from the
// CPU can change with the graph splits (OpenVINO), and a pack adds splits.
// The scheduler's CPU worker (sched_async_cpu) must not change any logit: the
// worker computes the same splits, and in decode the hot chain overlaps the
// cold one with the merge moved to the CPU, an exact f32 add either way. An
// abort raised by the CPU split on the worker fails the decode, and the context
// decodes the same logits afterwards. A layer with Hadamard-folded experts gets no
// pack, and the device MoE cache (moe_cache_size) leaves a packed layer to its pack.
//
// usage: test-moe-pack MODEL.gguf WORK_DIR
//   MODEL is a MoE fixture with separate gate/up/down experts and at least
//   eight experts per layer (test-llama-archs --n-expert 8 --n-expert-used 2).

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

static std::mutex  g_log_mutex;
static std::string g_log;

static void capture_log(ggml_log_level level, const char * text, void * /*user_data*/) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += text;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static int g_failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fputc('\n', stderr); \
        g_failures++; \
    } \
} while (0)

static bool write_text(const std::string & path, const std::string & text) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return false;
    }
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    return fclose(f) == 0 && ok;
}

// copy a GGUF without per-expert scale tensors (the fixture creates every
// optional tensor; real Q4_K MoE files carry no expert scales)
static bool strip_expert_scales(const std::string & src, const std::string & dst) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { /*no_alloc =*/ false, /*ctx =*/ &meta };
    gguf_context * in = gguf_init_from_file(src.c_str(), params);
    if (!in) {
        return false;
    }
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    for (int64_t i = 0; i < gguf_get_n_tensors(in); i++) {
        const char * name = gguf_get_tensor_name(in, i);
        if (strstr(name, "_exps.scale") || strstr(name, "_exps.input_scale")) {
            continue;
        }
        gguf_add_tensor(out, ggml_get_tensor(meta, name));
    }
    const bool ok = gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(in);
    ggml_free(meta);
    return ok;
}

// copy a GGUF that declares every routed expert weight Hadamard-folded; the weights stay
// random, the graph applies the transform to their inputs all the same
static bool fold_expert_hadamard(const std::string & src, const std::string & dst) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { /*no_alloc =*/ false, /*ctx =*/ &meta };
    gguf_context * in = gguf_init_from_file(src.c_str(), params);
    if (!in) {
        return false;
    }
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    std::vector<const char *> names;
    for (int64_t i = 0; i < gguf_get_n_tensors(in); i++) {
        const char * name = gguf_get_tensor_name(in, i);
        if (strstr(name, "_exps.weight")) {
            names.push_back(name);
        }
        gguf_add_tensor(out, ggml_get_tensor(meta, name));
    }
    gguf_set_val_u32(out, "prism.hadamard.version", 1);
    gguf_set_val_u32(out, "prism.hadamard.block_size", 64);
    gguf_set_val_str(out, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    gguf_set_val_str(out, "prism.hadamard.axis", "input-last-dimension");
    gguf_set_val_str(out, "prism.hadamard.sign_mode", "identity");
    gguf_set_arr_str(out, "prism.hadamard.weight_names", names.data(), names.size());
    const bool ok = !names.empty() && gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(in);
    ggml_free(meta);
    return ok;
}

struct run_result {
    bool ok = false;
    int  abort_rc = 0;         // llama_decode result of the aborted first decode
    std::vector<float> logits; // [n_tokens, n_vocab]
    std::string log;
};

static bool g_abort = false;

static bool abort_requested(void * /*data*/) {
    return g_abort;
}

// CPU: every layer on the CPU, packs through LLAMA_MOE_PACK_TEST_CPU.
// DEVICE: every layer on the first GPU, routed experts pinned to the CPU, so the
// pack lands on the GPU exactly as with -ngl 99 --n-cpu-moe 99.
enum placement { PLACE_CPU, PLACE_DEVICE };

static ggml_backend_dev_t first_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return dev;
        }
    }
    return nullptr;
}

static llama_model_params model_params(const char * profile, int32_t slots, placement place) {
    static llama_model_tensor_buft_override experts_on_cpu[2] = {};
    static ggml_backend_dev_t cpu_only[2] = {};
    llama_model_params mp = llama_model_default_params();
    mp.moe_pack_profile = profile;
    mp.moe_pack_slots   = slots;
    if (place == PLACE_CPU) {
        // no accelerator takes part, not even for offloaded ops
        cpu_only[0]     = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        cpu_only[1]     = nullptr;
        mp.devices      = cpu_only;
        mp.n_gpu_layers = 0;
    } else {
        experts_on_cpu[0] = { "\\.ffn_(gate|up|down|gate_up)_exps", ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) };
        experts_on_cpu[1] = { nullptr, nullptr };
        mp.n_gpu_layers          = 999;
        mp.tensor_buft_overrides = experts_on_cpu;
    }
    return mp;
}

// a prompt of 12 tokens in one batch, then 4 single-token steps; logits for every token.
// abort_first: an aborted decode of the prompt and a cleared memory come first;
// moe_cache_size: the device MoE cache for the experts in host memory, 0 = none
static run_result run_model(const std::string & path, const char * profile, int32_t slots, placement place = PLACE_CPU,
                            bool async = false, bool abort_first = false, size_t moe_cache_size = 0) {
    run_result r;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log.clear();
    }
    llama_model_params mp = model_params(profile, slots, place);
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        return r;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 64;
    cp.n_batch         = 16;
    cp.n_ubatch        = 16;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    cp.sched_async_cpu = async;
    cp.moe_cache_size  = moe_cache_size;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return r;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const std::vector<llama_token> prompt = { 1, 5, 9, 2, 33, 17, 4, 4, 8, 60, 21, 3 };
    const std::vector<llama_token> steps  = { 7, 11, 13, 19 };

    llama_batch batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
    bool ok = true;
    batch.n_tokens = (int32_t) prompt.size();
    for (size_t i = 0; i < prompt.size(); i++) {
        batch.token[i]     = prompt[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    if (abort_first) {
        g_abort = true;
        llama_set_abort_callback(ctx, abort_requested, nullptr);
        r.abort_rc = llama_decode(ctx, batch);
        g_abort = false;
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    ok = llama_decode(ctx, batch) == 0;
    for (size_t i = 0; ok && i < prompt.size(); i++) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        r.logits.insert(r.logits.end(), l, l + n_vocab);
    }
    for (size_t s = 0; ok && s < steps.size(); s++) {
        batch.n_tokens     = 1;
        batch.token[0]     = steps[s];
        batch.pos[0]       = (llama_pos) (prompt.size() + s);
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits_ith(ctx, 0);
            r.logits.insert(r.logits.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    r.ok = ok;
    std::lock_guard<std::mutex> lock(g_log_mutex);
    r.log = g_log;
    return r;
}

// model bytes over all buffer types, as the --fit planner sees them
static bool model_bytes(const std::string & path, const char * profile, int32_t slots, bool no_alloc, size_t & bytes) {
    llama_model_params mp = model_params(profile, slots, PLACE_CPU);
    mp.no_alloc         = no_alloc;
    mp.load_mode        = no_alloc ? LLAMA_LOAD_MODE_NONE : mp.load_mode; // as common/fit.cpp
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 64;
    cp.n_batch   = 16;
    cp.n_ubatch  = 16;
    cp.n_threads = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return false;
    }
    bytes = 0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        bytes += mb.model;
    }
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) a[i] * a[i];
    }
    return den > 0.0 ? num / den : num;
}

static bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}

// the scheduler's counters, logged when an async scheduler is freed (summed over schedulers)
static bool async_counts(const std::string & log, unsigned long long & splits, unsigned long long & overlaps) {
    static const char key[] = "async CPU splits: ";
    splits = overlaps = 0;
    bool found = false;
    for (size_t at = log.find(key); at != std::string::npos; at = log.find(key, at + 1)) {
        unsigned long long s = 0, o = 0;
        if (sscanf(log.c_str() + at + sizeof(key) - 1, "%llu, splits launched beside one: %llu", &s, &o) == 2) {
            splits += s;
            overlaps += o;
            found = true;
        }
    }
    return found;
}

// the layers the MoE cache serves, summed over its groups ("N layers, TYPE: S slots")
static int cache_layers(const std::string & log) {
    static const char key[] = " layers, ";
    int n = 0;
    for (size_t at = log.find(key); at != std::string::npos; at = log.find(key, at + 1)) {
        size_t b = at;
        while (b > 0 && isdigit((unsigned char) log[b - 1])) {
            b--;
        }
        if (b < at && log.find(" slots (", at) < log.find('\n', at)) {
            n += atoi(log.c_str() + b);
        }
    }
    return n;
}

static size_t bitwise_diff(const std::vector<float> & a, const std::vector<float> & b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        n += memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return n + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

// the worker must give the synchronous logits bit for bit; need_overlap: a split was
// launched while a CPU split ran
static void expect_async_same(const char * name, const run_result & sync, const run_result & async, bool need_overlap) {
    const int failures_before = g_failures;
    CHECK(async.ok, "%s: async run failed", name);
    if (!async.ok) {
        return;
    }
    unsigned long long splits = 0, overlaps = 0;
    CHECK(async_counts(async.log, splits, overlaps), "%s: no async scheduler counters in the log", name);
    CHECK(splits > 0, "%s: the worker computed no CPU split", name);
    CHECK(!need_overlap || overlaps > 0, "%s: no split was launched beside a CPU split", name);
    const size_t n_diff = bitwise_diff(sync.logits, async.logits);
    CHECK(n_diff == 0, "%s: %zu of %zu logits differ bitwise", name, n_diff, sync.logits.size());
    if (g_failures == failures_before) {
        printf("ok: %s (%llu CPU splits on the worker, %llu splits beside them)\n", name, splits, overlaps);
    }
}

static void expect_abort_recovers(const char * name, const run_result & ref, const run_result & r) {
    const int failures_before = g_failures;
    CHECK(r.abort_rc == 2, "%s: the aborted decode returned %d, expected 2", name, r.abort_rc);
    CHECK(r.ok, "%s: decoding after the abort failed", name);
    const size_t n_diff = bitwise_diff(ref.logits, r.logits);
    CHECK(n_diff == 0, "%s: %zu of %zu logits differ bitwise after the abort", name, n_diff, ref.logits.size());
    if (g_failures == failures_before) {
        printf("ok: %s\n", name);
    }
}

// oracle: the CPU result, for device runs whose own deviation from it can change with the
// graph splits, which a pack changes
static void expect_equivalent(const char * name, const run_result & base, const run_result & packed,
                              const char * placed, bool exact, const run_result * oracle = nullptr) {
    const int failures_before = g_failures;
    CHECK(packed.ok, "%s: packed run failed", name);
    if (!packed.ok) {
        return;
    }
    CHECK(contains(packed.log, placed), "%s: missing '%s' in the load log", name, placed);
    CHECK(contains(packed.log, "the graph uses all"), "%s: the graph did not use every pack", name);
    CHECK(base.logits.size() == packed.logits.size(), "%s: logit counts differ", name);
    if (base.logits.size() != packed.logits.size()) {
        return;
    }
    if (exact) {
        size_t n_diff = 0;
        for (size_t i = 0; i < base.logits.size(); i++) {
            n_diff += memcmp(&base.logits[i], &packed.logits[i], sizeof(float)) != 0;
        }
        CHECK(n_diff == 0, "%s: %zu of %zu logits differ bitwise", name, n_diff, base.logits.size());
    } else {
        // repacked and plain Q4_0 kernels accumulate in different orders; wrong
        // uploaded bytes would give an nmse near 1
        const double e = nmse(base.logits, packed.logits);
        if (oracle && oracle->logits.size() == packed.logits.size()) {
            // agreeing with either reference still rules out a wrong remap or upload
            const double eo = nmse(oracle->logits, packed.logits);
            CHECK(e < 1e-4 || eo < 1e-4, "%s: logits nmse %.3g, %.3g vs CPU", name, e, eo);
            printf("    nmse %.3g, %.3g vs CPU\n", e, eo);
        } else {
            CHECK(e < 1e-4, "%s: logits nmse %.3g", name, e);
            printf("    nmse %.3g\n", e);
        }
    }
    if (g_failures == failures_before) {
        printf("ok: %s\n", name);
    }
}

static void expect_disabled(const char * name, const run_result & base, const run_result & r, const char * reason) {
    const int failures_before = g_failures;
    CHECK(r.ok, "%s: the model must still load", name);
    CHECK(contains(r.log, reason), "%s: missing '%s'", name, reason);
    CHECK(!contains(r.log, "static expert packs on"), "%s: a pack was built", name);
    CHECK(r.logits == base.logits, "%s: logits changed without a pack", name);
    if (g_failures == failures_before) {
        printf("ok: %s\n", name);
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf WORK_DIR\n", argv[0]);
        return 2;
    }
    const std::string work = argv[2];
    setenv("LLAMA_MOE_PACK_TEST_CPU", "1", 1);

    llama_log_set(capture_log, nullptr);
    llama_backend_init();

    const std::string f32 = work + "/f32.gguf";
    const std::string q40 = work + "/q4_0.gguf";
    if (!strip_expert_scales(argv[1], f32)) {
        fprintf(stderr, "cannot write %s\n", f32.c_str());
        return 2;
    }

    // layer 0: experts 3 > 5 > 1 by count, 6 and 2 tie (6 precedes 2 only by count order);
    // layer 1: one expert
    const char * arch_env = getenv("LLAMA_MOE_PACK_TEST_ARCH");
    const std::string arch = arch_env ? arch_env : "qwen3moe";
    const std::string hdr = "# llama-moe-trace v1\n# session=00000000000000aa\n# arch=" + arch + "\n";
    const std::string rows =
        "pos,layer,ids...\n-1,0,7,0\n"
        "0,0,3,5\n1,0,3,5\n2,0,3,1\n3,0,6,2\n4,0,3,5\n"
        "0,1,4,4\n1,1,4,0\n";
    const std::string p_mixed   = work + "/mixed.csv";
    const std::string p_all     = work + "/all.csv";
    const std::string p_layer1  = work + "/layer1.csv";
    const std::string p_bad     = work + "/bad.csv";
    const std::string p_arch    = work + "/arch.csv";
    const std::string p_range   = work + "/range.csv";
    const std::string p_prefill = work + "/prefill.csv";
    const std::string p_spec    = work + "/spec.csv";
    const std::string p_two     = work + "/two-archs.csv";
    std::string all_rows = "pos,layer,ids...\n";
    for (int e = 0; e < 8; e++) {
        all_rows += std::to_string(e) + ",0," + std::to_string(e) + "," + std::to_string((e + 1) % 8) + "\n";
    }
    bool written =
        write_text(p_mixed,   hdr + rows) &&
        write_text(p_all,     hdr + all_rows) &&
        write_text(p_layer1,  hdr + "0,1,4,2\n1,1,4,4\n") &&
        write_text(p_bad,     hdr + "0,0,3,x\n") &&
        write_text(p_arch,    "# arch=llama\n0,0,1,2\n") &&
        write_text(p_range,   hdr + "0,0,3,99\n") &&
        write_text(p_prefill, hdr + "-1,0,1,2\n-2,0,3,4\n") &&
        write_text(p_spec,    hdr + "# kind=spec-router k=2\n" + rows) &&
        // a session from another model concatenated before one that matches
        write_text(p_two,     "# llama-moe-trace v1\n# session=00000000000000ab\n# arch=llama\n0,0,1,2\n" + hdr + rows);
    if (!written) {
        fprintf(stderr, "cannot write profiles under %s\n", work.c_str());
        return 2;
    }

    const run_result base = run_model(f32, nullptr, 0);
    CHECK(base.ok && !base.logits.empty(), "baseline run failed");
    if (!base.ok) {
        return 1;
    }
    CHECK(!contains(base.log, "static expert packs on"), "a pack was built without a profile");

    expect_equivalent("f32 mixed hot/cold, 2 slots", base, run_model(f32, p_mixed.c_str(), 2),
            "static expert packs on CPU: 2 layer(s), up to 2 slot(s)", true);
    expect_equivalent("f32 slots capped by profiled experts", base, run_model(f32, p_mixed.c_str(), 8),
            "static expert packs on CPU: 2 layer(s), up to 5 slot(s)", true);
    expect_equivalent("f32 every expert hot in layer 0", base, run_model(f32, p_all.c_str(), 8),
            "static expert packs on CPU: 1 layer(s), up to 8 slot(s)", true);
    expect_equivalent("f32 pack on layer 1 only", base, run_model(f32, p_layer1.c_str(), 1),
            "static expert packs on CPU: 1 layer(s), up to 1 slot(s)", true);

    expect_async_same("async CPU splits, no pack", base, run_model(f32, nullptr, 0, PLACE_CPU, true), false);
    expect_async_same("async CPU splits, f32 mixed hot/cold", base, run_model(f32, p_mixed.c_str(), 2, PLACE_CPU, true), false);
    expect_abort_recovers("abort on the CPU worker", base, run_model(f32, p_mixed.c_str(), 2, PLACE_CPU, true, true));
    expect_abort_recovers("abort without the worker", base, run_model(f32, p_mixed.c_str(), 2, PLACE_CPU, false, true));

    expect_disabled("malformed profile", base, run_model(f32, p_bad.c_str(), 2), "malformed row");
    expect_disabled("profile from another arch", base, run_model(f32, p_arch.c_str(), 2), "was traced on arch 'llama'");
    expect_disabled("expert id out of range", base, run_model(f32, p_range.c_str(), 2), "routes expert 99");
    expect_disabled("profile without decode rows", base, run_model(f32, p_prefill.c_str(), 2), "has no decode rows");
    expect_disabled("missing profile", base, run_model(f32, (work + "/none.csv").c_str(), 2), "cannot open");
    expect_disabled("spec-router trace as a profile", base, run_model(f32, p_spec.c_str(), 2), "holds no routing");
    expect_disabled("sessions from two models", base, run_model(f32, p_two.c_str(), 2), "conflicts with 'llama'");

    // a pack copy of a Hadamard-folded expert would get unrotated activations
    {
        const std::string hdm = work + "/hadamard.gguf";
        CHECK(fold_expert_hadamard(f32, hdm), "cannot write %s", hdm.c_str());
        const run_result hbase = run_model(hdm, nullptr, 0);
        CHECK(hbase.ok, "Hadamard baseline run failed");
        if (hbase.ok) {
            CHECK(bitwise_diff(base.logits, hbase.logits) > 0, "the Hadamard metadata did not change the logits");
            expect_disabled("Hadamard-folded experts", hbase, run_model(hdm, p_mixed.c_str(), 2), "Hadamard-folded experts");
        }
    }

    // the memory-fit pass must account for exactly the bytes a real load allocates
    {
        size_t dry_base = 0, dry_pack = 0, real_base = 0, real_pack = 0;
        const bool ok =
            model_bytes(f32, nullptr,         0, true,  dry_base)  &&
            model_bytes(f32, p_mixed.c_str(), 2, true,  dry_pack)  &&
            model_bytes(f32, nullptr,         0, false, real_base) &&
            model_bytes(f32, p_mixed.c_str(), 2, false, real_pack);
        CHECK(ok, "memory breakdown runs failed");
        if (ok) {
            CHECK(dry_pack > dry_base, "the memory-fit pass does not see the pack (%zu <= %zu)", dry_pack, dry_base);
            CHECK(dry_pack - dry_base == real_pack - real_base,
                    "fit pass pack bytes %zu != allocated pack bytes %zu", dry_pack - dry_base, real_pack - real_base);
            printf("ok: memory-fit pass accounts %zu pack bytes\n", dry_pack - dry_base);
        }
    }

    // Q4_0: the CPU-side experts are repacked on this host when the CPU supports it,
    // while the pack is uploaded from the file in the stored layout
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype   = LLAMA_FTYPE_MOSTLY_Q4_0;
    qp.nthread = 2;
    run_result qbase;
    if (llama_model_quantize(f32.c_str(), q40.c_str(), &qp) != 0) {
        fprintf(stderr, "FAIL: quantizing %s failed\n", f32.c_str());
        g_failures++;
    } else {
        qbase = run_model(q40, nullptr, 0);
        CHECK(qbase.ok, "q4_0 baseline run failed");
        if (qbase.ok) {
            expect_equivalent("q4_0 mixed hot/cold, 2 slots", qbase, run_model(q40, p_mixed.c_str(), 2),
                    "static expert packs on CPU: 2 layer(s), up to 2 slot(s)", false);
        }
    }

    if (ggml_backend_dev_t gpu = first_gpu()) {
        const std::string placed = std::string("static expert packs on ") + ggml_backend_dev_name(gpu) + ": 2 layer(s), up to 2 slot(s)";
        const run_result dbase = run_model(f32, nullptr, 0, PLACE_DEVICE);
        CHECK(dbase.ok, "device baseline run failed");
        if (dbase.ok && dbase.logits.size() == base.logits.size()) {
            // the scale of the device's own deviation, for reading the device case results
            printf("    %s without a pack vs CPU: nmse %.3g\n", ggml_backend_dev_name(gpu), nmse(base.logits, dbase.logits));
        }
        if (dbase.ok) {
            // the hot experts now run on the GPU and the cold ones on the CPU
            const run_result dpack = run_model(f32, p_mixed.c_str(), 2, PLACE_DEVICE);
            expect_equivalent("device f32 mixed hot/cold, 2 slots", dbase, dpack, placed.c_str(), false, &base);
            // OpenVINO keeps the hot remap on the CPU, so its pack split waits for the CPU split
            const bool dev_map = strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu)), "OPENVINO") != 0;
            expect_async_same("device async CPU splits, f32 mixed hot/cold", dpack,
                    run_model(f32, p_mixed.c_str(), 2, PLACE_DEVICE, true), dev_map);
            expect_async_same("device async CPU splits, no pack", dbase, run_model(f32, nullptr, 0, PLACE_DEVICE, true), false);
            if (dev_map) {
                expect_abort_recovers("device abort on the CPU worker", dpack, run_model(f32, p_mixed.c_str(), 2, PLACE_DEVICE, true, true));
            } else {
                // the OpenVINO splits before the aborted CPU split have run; the next decode then
                // fails in OpenVINO's ScatterUpdate, with or without the worker
                printf("skip: device abort recovery (fails on OpenVINO without the worker too)\n");
            }
            const std::string all = std::string("static expert packs on ") + ggml_backend_dev_name(gpu) + ": 1 layer(s), up to 8 slot(s)";
            expect_equivalent("device f32 every expert hot in layer 0", dbase, run_model(f32, p_all.c_str(), 8, PLACE_DEVICE),
                    all.c_str(), false, &base);
        }
        const run_result dqbase = run_model(q40, nullptr, 0, PLACE_DEVICE);
        if (dqbase.ok) {
            expect_equivalent("device q4_0 mixed hot/cold, 2 slots", dqbase, run_model(q40, p_mixed.c_str(), 2, PLACE_DEVICE),
                    placed.c_str(), false, qbase.ok ? &qbase : nullptr);
        }
        // the MoE cache leaves a packed layer to its pack
        const size_t cache_bytes = 64u << 20;
        const run_result cbase = run_model(f32, nullptr, 0, PLACE_DEVICE, false, false, cache_bytes);
        const int n_cached = cbase.ok ? cache_layers(cbase.log) : 0;
        if (n_cached > 0) {
            const std::string one = std::string("static expert packs on ") + ggml_backend_dev_name(gpu) + ": 1 layer(s), up to 1 slot(s)";
            const run_result cpack = run_model(f32, p_layer1.c_str(), 1, PLACE_DEVICE, false, false, cache_bytes);
            printf("    MoE cache: %d layer(s), %d beside a pack\n", n_cached, cache_layers(cpack.log));
            expect_equivalent("device pack beside the MoE cache", cbase, cpack, one.c_str(), false, &base);
            CHECK(cache_layers(cpack.log) == n_cached - 1, "the MoE cache serves %d layer(s) beside a pack, %d without",
                    cache_layers(cpack.log), n_cached);
        } else {
            printf("skip: device pack beside the MoE cache (no MoE cache on %s)\n", ggml_backend_dev_name(gpu));
        }
    } else {
        printf("skip: device cases (no GPU device)\n");
    }

    llama_backend_free();
    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all static expert pack checks passed\n");
    return 0;
}
