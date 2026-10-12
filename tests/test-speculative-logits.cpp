// Full-vocabulary observations at identical histories; no universal tolerance is inferred.
#include "arg.h"
#include "common.h"
#include "llama-cpp.h"
#include "log.h"
#include "../src/llama-ext.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using tokens = std::vector<llama_token>;
using row = std::vector<float>;
using rows = std::vector<row>;
namespace fs = std::filesystem;

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void write_json(const fs::path & path, const json & value) {
    std::ofstream out(path);
    out << value.dump(2) << '\n';
    out.close();
    require(bool(out), "write " + path.string());
}

static void maps(const fs::path & path) {
#ifdef __linux__
    std::ifstream in("/proc/self/maps");
    std::ofstream out(path);
    require(bool(in) && bool(out), "open maps snapshot");
    out << in.rdbuf();
    out.close();
    require(bool(out), "write maps snapshot");
#else
    (void) path;
#endif
}

struct distribution {
    std::vector<double> p;
    double float_mass = 0;
    bool valid = false;
};

static distribution probabilities(const row & logits, const tokens & suppressed = {}) {
    distribution result;
    if (logits.empty() || std::any_of(logits.begin(), logits.end(), [](float x) { return !std::isfinite(x); })) {
        return result;
    }
    std::vector<llama_token_data> data;
    for (size_t i = 0; i < logits.size(); ++i) { data.push_back({llama_token(i), logits[i], 0}); }
    llama_token_data_array candidates = {data.data(), data.size(), -1, false};
    std::vector<llama_logit_bias> biases;
    for (const auto id : suppressed) {
        require(id >= 0 && size_t(id) < logits.size(), "suppressed token outside vocabulary");
        biases.push_back({id, -INFINITY});
    }
    if (!biases.empty()) {
        llama_sampler_ptr mask(llama_sampler_init_logit_bias(logits.size(), biases.size(), biases.data()));
        llama_sampler_apply(mask.get(), &candidates);
    }
    llama_sampler_ptr temp(llama_sampler_init_temp(0.8f));
    llama_sampler_ptr dist(llama_sampler_init_dist(0));
    llama_sampler_apply(temp.get(), &candidates);
    for (const auto & x : data) {
        if (!std::isfinite(x.logit) && std::find(suppressed.begin(), suppressed.end(), x.id) == suppressed.end()) {
            return result;
        }
    }
    if (std::none_of(data.begin(), data.end(), [](const llama_token_data & x) { return std::isfinite(x.logit); })) {
        return result;
    }
    // Use the production float exp/normalization; its scratch sampled token is irrelevant here.
    llama_sampler_apply(dist.get(), &candidates);
    for (const auto & x : data) {
        if (!std::isfinite(x.p) || x.p < 0) { return result; }
        result.float_mass += x.p;
    }
    if (!(result.float_mass > 0) || !std::isfinite(result.float_mass)) { return result; }
    for (const auto & x : data) { result.p.push_back(double(x.p) / result.float_mass); }
    result.valid = true;
    return result;
}

static json differences(const std::vector<double> & a, const std::vector<double> & b) {
    double max_abs = 0, max_relative = 0, l1 = 0;
    size_t support_changes = 0, zero_reference_nonzero = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double delta = std::abs(a[i] - b[i]);
        max_abs = std::max(max_abs, delta);
        l1 += delta;
        support_changes += (a[i] == 0) != (b[i] == 0);
        if (a[i] != 0) { max_relative = std::max(max_relative, delta / std::abs(a[i])); }
        else { zero_reference_nonzero += b[i] != 0; }
    }
    return {{"max_absolute", max_abs}, {"max_relative_nonzero_reference", max_relative},
            {"zero_reference_nonzero", zero_reference_nonzero}, {"support_changes", support_changes}, {"l1", l1}};
}

static json compare(const row & a, const row & b, const tokens & suppressed = {}) {
    require(a.size() == b.size() && !a.empty(), "matched full vocabulary");
    auto count_bad = [](const row & x) {
        return std::count_if(x.begin(), x.end(), [](float v) { return !std::isfinite(v); });
    };
    const auto pa = probabilities(a, suppressed), pb = probabilities(b, suppressed);
    json result = {{"n_vocab", a.size()}, {"reference_nonfinite", count_bad(a)},
                   {"candidate_nonfinite", count_bad(b)}, {"valid_distributions", pa.valid && pb.valid}};
    if (count_bad(a) == 0 && count_bad(b) == 0) {
        result["logits"] = differences({a.begin(), a.end()}, {b.begin(), b.end()});
    }
    if (pa.valid && pb.valid) {
        result["probabilities"] = differences(pa.p, pb.p);
        result["tv"] = result["probabilities"]["l1"].get<double>() * 0.5;
        result["reference_float_normalization_residual"] = std::abs(pa.float_mass - 1);
        result["candidate_float_normalization_residual"] = std::abs(pb.float_mass - 1);
        result["reference_canonical_normalization_residual"] = std::abs(std::accumulate(pa.p.begin(), pa.p.end(), 0.0) - 1);
        result["candidate_canonical_normalization_residual"] = std::abs(std::accumulate(pb.p.begin(), pb.p.end(), 0.0) - 1);
    }
    return result;
}

static bool valid_target_context(const json & settings) {
    for (const auto * key : {"n_rs_seq", "n_outputs_max", "n_outputs_max_per_seq"}) {
        if (!settings.contains(key) || !settings[key].is_number_integer()) { return false; }
    }
    for (const auto * key : {"gdn_replay", "embeddings_nextn", "embeddings_nextn_masked"}) {
        if (!settings.contains(key) || !settings[key].is_boolean()) { return false; }
    }
    if (settings["embeddings_nextn_masked"] != false) { return false; }
    const bool mtp = settings["n_rs_seq"] == 7 && settings["embeddings_nextn"] == true &&
        settings["n_outputs_max"] == 8 && settings["n_outputs_max_per_seq"] == 8;
    const bool dense = settings["n_rs_seq"] == 0 && settings["embeddings_nextn"] == false &&
        settings["n_outputs_max"] == 1 && settings["n_outputs_max_per_seq"] == 1;
    return mtp || dense;
}

static void self_test() {
    require(compare({0, 0}, {0, 0})["tv"] == 0, "identical uniform distributions");
    const auto swapped = compare({0, -1000}, {-1000, 0});
    require(swapped["tv"] == 1 && swapped["probabilities"]["support_changes"] == 2, "disjoint rounded support");
    require(swapped["logits"]["zero_reference_nonzero"] == 1, "relative zero denominator explicit");
    const auto shifted = compare({0, 0.8f}, {0.8f, 1.6f});
    require(shifted["tv"].get<double>() < 1e-7, "constant logit shift");
    require(!compare({0, std::numeric_limits<float>::infinity()}, {0, 0})["valid_distributions"].get<bool>(),
            "invalid input retained as invalid");
    require(compare({0, 0.001f}, {0, 0})["tv"].get<double>() > 0, "small difference visible");
    const auto masked = probabilities({10, 0, 0}, {0});
    require(masked.valid && masked.p == std::vector<double>({0, 0.5, 0.5}), "masked support and renormalization");
    require(!probabilities({0, 0}, {0, 1}).valid, "fully masked row is invalid");
    require(compare({10, 0, -1}, {10, -1, 0}, {0})["tv"].get<double>() >
            compare({10, 0, -1}, {10, -1, 0})["tv"].get<double>(), "mask can amplify target-law difference");
    json settings = {{"n_rs_seq", 0}, {"n_outputs_max", 1}, {"n_outputs_max_per_seq", 1},
        {"gdn_replay", false}, {"embeddings_nextn", false}, {"embeddings_nextn_masked", false}};
    require(valid_target_context(settings), "dense target context");
    settings["n_rs_seq"] = 7;
    require(!valid_target_context(settings), "partial MTP configuration rejected");
    settings["embeddings_nextn"] = true;
    settings["n_outputs_max"] = settings["n_outputs_max_per_seq"] = 8;
    require(valid_target_context(settings), "MTP target context");
    settings["n_rs_seq"] = 7.0;
    require(!valid_target_context(settings), "noninteger recurrent count rejected");
    settings["n_rs_seq"] = 7;
    settings.erase("gdn_replay");
    require(!valid_target_context(settings), "missing qualification setting rejected");
    std::puts("PASS speculative logits comparison/context self-test (no model/backend initialization)");
}

struct probe {
    fs::path output;
    json report;
    int n_vocab;

    llama_context_ptr context(llama_model * model, const llama_context_params & params, const std::string & phase) {
        llama_context_ptr ctx(llama_init_from_model(model, params));
        require(bool(ctx), "create " + phase + " context");
        const auto settings = report.at("target_context");
        llama_set_embeddings_nextn(ctx.get(), settings["embeddings_nextn"].get<bool>(),
                                  settings["embeddings_nextn_masked"].get<bool>());
        const json actual = {{"n_ctx", llama_n_ctx(ctx.get())}, {"n_ctx_seq", llama_n_ctx_seq(ctx.get())},
            {"n_batch", llama_n_batch(ctx.get())}, {"n_ubatch", llama_n_ubatch(ctx.get())},
            {"n_seq_max", llama_n_seq_max(ctx.get())}, {"n_rs_seq", llama_n_rs_seq(ctx.get())}};
        report["contexts"].push_back({{"phase", phase}, {"actual", actual}});
        require(actual["n_ctx"] == 8192 && actual["n_seq_max"] == 1 &&
                actual["n_ctx_seq"] == 8192 && actual["n_batch"] == params.n_batch &&
                actual["n_ubatch"] == params.n_ubatch, "actual context differs from requested dimensions");
        require(actual["n_rs_seq"] == settings["n_rs_seq"], "actual recurrent rollback count differs from qualification");
        require(report["contexts"][0]["actual"] == actual, "actual context configuration changed");
        return ctx;
    }

    rows decode(llama_context * ctx, const tokens & input, size_t begin, size_t count, int pos,
                bool all, const std::string & phase) {
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (size_t i = 0; i < count; ++i) {
            common_batch_add(batch, input[begin + i], pos + int(i), {0}, all || i + 1 == count);
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        report["decodes"].push_back({{"phase", phase}, {"position", pos}, {"count", count}, {"return_code", rc},
                                     {"sequence", 0}, {"all_logits", all}});
        write_json(output / "report.json", report);
        require(rc == 0, "decode " + phase + " returned " + std::to_string(rc));
        if (report["target_context"]["embeddings_nextn"].get<bool>()) {
            require(llama_get_embeddings_nextn(ctx) != nullptr, "qualified target hidden-state output absent");
            report["decodes"].back()["target_hidden_output_observed"] = true;
        }
        rows copied;
        for (size_t i = all ? 0 : count - 1; i < count; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            require(logits != nullptr, "missing full logits row");
            copied.emplace_back(logits, logits + n_vocab);
        }
        require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == pos + int(count) - 1, "decoded position mismatch");
        return copied;
    }

    row prefix(llama_context * ctx, const tokens & input, const std::string & phase) {
        row last;
        for (size_t pos = 0; pos < input.size();) {
            const size_t count = std::min<size_t>(llama_n_batch(ctx), input.size() - pos);
            last = decode(ctx, input, pos, count, pos, false, phase)[0];
            pos += count;
        }
        return last;
    }

    void restore(llama_context * ctx, const std::vector<uint8_t> & state, size_t prefix_length, const std::string & phase) {
        const auto restored = llama_state_seq_set_data(ctx, state.data(), state.size(), 0);
        const auto pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        report["restores"].push_back({{"phase", phase}, {"expected_bytes", state.size()}, {"restored_bytes", restored},
                                     {"checkpoint_history_length", prefix_length}, {"actual_max_position", pos}});
        write_json(output / "report.json", report);
        require(restored == state.size(), "restore full sequence state");
        require(pos == int(prefix_length) - 1, "restored prefix position");
    }

    void dump(const std::string & name, const rows & data, const tokens & history, const tokens & suffix, bool first) {
        std::ofstream out(output / (name + ".f32"), std::ios::binary);
        for (const auto & r : data) { out.write(reinterpret_cast<const char *>(r.data()), r.size() * sizeof(float)); }
        out.close();
        require(bool(out), "write raw logits " + name);
        json positions = json::array();
        for (size_t i = 0; i < data.size(); ++i) { positions.push_back(history.size() - (first ? 1 : 0) + i); }
        report["artifacts"].push_back({{"file", name + ".f32"}, {"shape", {data.size(), n_vocab}},
            {"dtype", "IEEE754-f32"}, {"endianness", "little"}, {"order", "row-major-token-id"},
            {"prefix_tokens", history}, {"suffix_tokens", suffix}, {"prediction_input_positions", positions},
            {"includes_prefix_last_row", first}, {"sequence", 0}, {"hash_status", "external runner required"}});
    }
};

static tokens fixture_tokens(const json & cell, int n_vocab, size_t length) {
    require(cell.at("tokens").is_array() && cell.at("tokens").size() == length, "fixture token count");
    tokens result;
    for (const auto & value : cell.at("tokens")) {
        require(value.is_number_integer(), "fixture token is not integer");
        const auto id = value.get<int64_t>();
        require(id >= 0 && id < n_vocab, "fixture token outside vocabulary");
        result.push_back(llama_token(id));
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        try { self_test(); return 0; } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    }
    if (argc < 6 || std::string(argv[1]) != "--fixture" || std::string(argv[3]) != "--output") {
        std::fprintf(stderr, "usage: %s --self-test | --fixture six-cells.json --output NEW-DIR <common model/context args>\n", argv[0]);
        return 1;
    }
    probe p{argv[4], {{"schema_version", 1}, {"complete", false}, {"decodes", json::array()},
                    {"artifacts", json::array()}, {"comparisons", json::array()}}, 0};
    bool owns_output = false, backend_initialized = false;
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE754 float required");
        const uint16_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "little endian required");
        require(fs::create_directory(p.output), "output must be a new directory");
        owns_output = true;
        require(fs::file_size(argv[2]) <= 1024 * 1024, "fixture exceeds 1 MiB");
        std::ifstream in(argv[2]);
        require(bool(in), "open fixture");
        json fixture;
        in >> fixture;
        require(fixture.at("schema_version") == 1 && fixture.at("cells").size() == 6, "fixture schema/six cells");
        require(valid_target_context(fixture.at("target_context")), "missing/invalid qualified target_context");
        p.report["fixture"] = fixture;
        p.report["target_context"] = fixture.at("target_context");
        p.report["context_observation_limits"] = "n_rs_seq and dimensions queried; output caps/gdn_replay/nextn masking applied explicitly, no public getters; retain initialization logs";
        p.report["argv"] = std::vector<std::string>(argv, argv + argc);
        p.report["scope"] = "observed histories only; external provenance/route qualification and tolerance adjudication required";
        p.report["temperature"] = 0.8;
        p.report["state_flags"] = "LLAMA_STATE_SEQ_FLAGS_NONE (full host sequence snapshot, not PARTIAL_ONLY)";
        p.report["continuation_rule"] = "last seven prefix IDs; accept first three, correction=(fourth+1)%vocab";
        write_json(p.output / "report.json", p.report);
        common_params params;
        params.n_ctx = 8192;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_parallel = 1;
        params.cache_type_k = params.cache_type_v = GGML_TYPE_Q8_0;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.fit_params = false;
        params.warmup = false;
        std::vector<char *> arguments{argv[0]};
        arguments.insert(arguments.end(), argv + 5, argv + argc);
        common_init();
        require(common_params_parse(arguments.size(), arguments.data(), params, LLAMA_EXAMPLE_COMMON), "parse common args");
        require(!params.fit_params && params.n_parallel == 1 && params.n_ctx == 8192 && params.n_batch >= 8 && params.n_ubatch >= 8 &&
                params.cache_type_k == GGML_TYPE_Q8_0 && params.cache_type_v == GGML_TYPE_Q8_0 &&
                params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED, "requires frozen placement, ctx8192, q8 KV, FA, one sequence, batch >=8");
        require(!params.load_mtp && !params.speculative.has_mtp() && !params.speculative.has_dft(),
                "load target only; MTP context settings must come from fixture target_context");
        require(fs::is_regular_file(params.model.path), "local model file required");
        llama_backend_init();
        backend_initialized = true;
        {
            auto loaded = common_init_from_params(params, true);
            require(loaded && loaded->model(), "load local model");
            auto * model = loaded->model();
            const auto * vocab = llama_model_get_vocab(model);
            p.n_vocab = llama_vocab_n_tokens(vocab);
            require(p.n_vocab > 1 && p.n_vocab <= 1024 * 1024, "bounded vocabulary");
            // Match the frozen workload's EOG and model masks, including the missing-EOS guard.
            const bool ignore_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
            tokens eog, suppressed;
            if (ignore_eos) {
                for (llama_token id = 0; id < p.n_vocab; ++id) {
                    if (llama_vocab_is_eog(vocab, id)) { eog.push_back(id); }
                }
            }
            int32_t n_suppress = 0;
            const auto * model_suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
            require(n_suppress >= 0 && (n_suppress == 0 || model_suppress), "invalid model suppression list");
            for (int32_t i = 0; i < n_suppress; ++i) { suppressed.push_back(model_suppress[i]); }
            p.report["probability_transform"] = {{"temperature", 0.8}, {"top_p", 1}, {"top_k", 0},
                {"ignore_eos_requested", true}, {"ignore_eos_effective", ignore_eos},
                {"eog_ids", eog}, {"model_suppress_ids", suppressed},
                {"scope", "fixed workload transform; common CLI sampling options do not configure this probe"}};
            suppressed.insert(suppressed.end(), eog.begin(), eog.end());
            std::sort(suppressed.begin(), suppressed.end());
            suppressed.erase(std::unique(suppressed.begin(), suppressed.end()), suppressed.end());
            p.report["probability_transform"]["suppressed_ids"] = suppressed;
            auto cp = common_context_params_to_llama(params);
            const auto & settings = p.report["target_context"];
            cp.n_rs_seq = settings["n_rs_seq"].get<uint32_t>();
            cp.n_outputs_max = settings["n_outputs_max"].get<uint32_t>();
            cp.n_outputs_max_per_seq = settings["n_outputs_max_per_seq"].get<uint32_t>();
            cp.gdn_replay = settings["gdn_replay"].get<bool>();
            const bool block_outputs = cp.n_outputs_max == 8;
            char architecture[128] = {};
            require(llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) > 0,
                    "model architecture metadata missing");
            require((std::string(architecture) == "qwen4exp") == block_outputs, "model architecture and qualified context mode differ");
            const std::string batch_axis = block_outputs ? "sequential-vs-eight-output-block" :
                                                          "sequential-vs-restored-prefix-last-output";
            p.report["architecture"] = architecture;
            p.report["batch_axis"] = batch_axis;
            p.report["context"] = {{"n_ctx", cp.n_ctx}, {"n_batch", cp.n_batch}, {"n_ubatch", cp.n_ubatch},
                {"n_threads", cp.n_threads}, {"n_threads_batch", cp.n_threads_batch}, {"n_seq_max", cp.n_seq_max},
                {"type_k", int(cp.type_k)}, {"type_v", int(cp.type_v)}, {"flash_attn_type", int(cp.flash_attn_type)},
                {"n_rs_seq", cp.n_rs_seq}, {"n_outputs_max", cp.n_outputs_max},
                {"n_outputs_max_per_seq", cp.n_outputs_max_per_seq}, {"gdn_replay", cp.gdn_replay}};
            p.report["model_path"] = fs::canonical(params.model.path).string();
            p.report["model_load_scope"] = "target only, embedded MTP disabled, no draft context";
            p.report["system_info"] = llama_print_system_info();
            maps(p.output / "maps-model.txt");
            const std::vector<std::string> categories{"code_edit", "multi_turn", "free_prose"};
            bool all_valid = true;
            for (size_t c = 0; c < 6; ++c) {
                const size_t length = c % 2 ? 4096 : 512;
                const std::string id = categories[c / 2] + "-" + std::to_string(length);
                const auto & cell = fixture.at("cells").at(c);
                require(cell.at("id") == id, "fixture cell order mismatch");
                const auto history = fixture_tokens(cell, p.n_vocab, length);
                const tokens prefix(history.begin(), history.end() - 1);
                const tokens suffix(history.end() - 7, history.end());
                tokens verification{history.back()};
                verification.insert(verification.end(), suffix.begin(), suffix.end());
                tokens repair(suffix.begin(), suffix.begin() + 3);
                repair.push_back((suffix[3] + 1) % p.n_vocab);
                rows sequential, batched, replay, fresh;
                {
                    auto ctx = p.context(model, cp, id + "/sequential");
                    p.prefix(ctx.get(), prefix, id + "/sequential/prefix");
                    for (size_t i = 0; i < verification.size(); ++i) {
                        sequential.push_back(p.decode(ctx.get(), verification, i, 1, length - 1 + i, true, id + "/sequential/verification")[0]);
                    }
                    maps(p.output / (id + "-sequential.maps"));
                }
                p.dump(id + "-sequential", sequential, history, suffix, true);
                {
                    auto ctx = p.context(model, cp, id + "/batch");
                    p.prefix(ctx.get(), prefix, id + "/batch/prefix");
                    const size_t size = llama_state_seq_get_size(ctx.get(), 0);
                    require(size > 0 && size <= size_t(2) * 1024 * 1024 * 1024, "full snapshot must fit 2 GiB bound");
                    std::vector<uint8_t> state(size);
                    const auto saved = llama_state_seq_get_data(ctx.get(), state.data(), size, 0);
                    require(saved == size, "save full sequence state");
                    p.report["snapshots"].push_back({{"cell", id}, {"bytes", size}, {"saved", saved},
                        {"checkpoint_history_length", prefix.size()}, {"replayed_committed_token", history.back()}});
                    if (block_outputs) {
                        batched = p.decode(ctx.get(), verification, 0, verification.size(), length - 1, true, id + "/batch/verification");
                    } else {
                        // Preserve the dense server's one-output reservation at every sampled history.
                        for (size_t count = 1; count <= verification.size(); ++count) {
                            if (count > 1) { p.restore(ctx.get(), state, prefix.size(), id + "/batch/prefix-restore"); }
                            batched.push_back(p.decode(ctx.get(), verification, 0, count, length - 1, false,
                                                       id + "/batch/prefix-last-output")[0]);
                        }
                    }
                    p.dump(id + "-batched", batched, history, suffix, true);
                    p.restore(ctx.get(), state, prefix.size(), id + "/rollback/restore");
                    p.decode(ctx.get(), verification, 0, 1, length - 1, true, id + "/rollback/committed-last");
                    for (size_t i = 0; i < repair.size(); ++i) {
                        replay.push_back(p.decode(ctx.get(), repair, i, 1, length + i, true, id + "/rollback/replay")[0]);
                    }
                    maps(p.output / (id + "-batch-replay.maps"));
                }
                p.dump(id + "-replay", replay, history, repair, false);
                {
                    auto ctx = p.context(model, cp, id + "/fresh");
                    p.prefix(ctx.get(), prefix, id + "/fresh/prefix");
                    p.decode(ctx.get(), verification, 0, 1, length - 1, true, id + "/fresh/committed-last");
                    for (size_t i = 0; i < repair.size(); ++i) {
                        fresh.push_back(p.decode(ctx.get(), repair, i, 1, length + i, true, id + "/fresh/repair")[0]);
                    }
                    maps(p.output / (id + "-fresh.maps"));
                }
                p.dump(id + "-fresh", fresh, history, repair, false);
                for (size_t i = 0; i < sequential.size() + fresh.size(); ++i) {
                    const bool initial = i < sequential.size();
                    const size_t index = initial ? i : i - sequential.size();
                    auto stats = initial ? compare(sequential[index], batched[index], suppressed) : compare(fresh[index], replay[index], suppressed);
                    all_valid &= stats["valid_distributions"].get<bool>();
                    stats["cell"] = id;
                    stats["axis"] = initial ? batch_axis : "fresh-vs-rollback-replay";
                    stats["row"] = index;
                    stats["history_length"] = length + index + (initial ? 0 : 1);
                    p.report["comparisons"].push_back(stats);
                }
                write_json(p.output / "report.json", p.report);
            }
            p.report["complete"] = true;
            p.report["all_distributions_valid"] = all_valid;
            p.report["tolerance_gate"] = "not imposed; measurement is not an equivalence proof";
            write_json(p.output / "report.json", p.report);
            require(all_valid, "nonfinite logits/probabilities observed; consult raw artifacts");
        }
        llama_backend_free();
        common_log_flush(common_log_main());
        return 0;
    } catch (const std::exception & e) {
        p.report["complete"] = false;
        p.report["error"] = e.what();
        if (owns_output) {
            try { write_json(p.output / "report.json", p.report); } catch (...) { std::fprintf(stderr, "failed to preserve failure report\n"); }
        }
        if (backend_initialized) { llama_backend_free(); }
        common_log_flush(common_log_main());
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
