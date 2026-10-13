// Tests for the schema decision engine (tools/schema-decision).
//
// usage:
//   test-schema-decision --vocab DIR       schema compiler, request parser, and the token plan
//                                          against joint tokenization on DIR/ggml-vocab-*.gguf
//   test-schema-decision --model FILE...   scoring of token-level plans against a brute-force
//                                          oracle (any tokenizer, e.g. test-llama-archs fixtures)
//   test-schema-decision --e2e FILE...     text requests through the chat template against the
//                                          oracle on the joint tokenization (a real model)
//
// The oracle decodes every trie prefix from an empty memory on one sequence: no forks, no
// cached prefix, no grouping and no padding, so it checks all of those in the engine.

#include "schema-decision.h"

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace schema_decision;

static int g_failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fputc('\n', stderr); \
        g_failures++; \
    } \
} while (0)

static void quiet_log(ggml_log_level level, const char * text, void * /*user*/) {
    if (level >= GGML_LOG_LEVEL_ERROR && getenv("TEST_SCHEMA_DECISION_VERBOSE")) {
        fputs(text, stderr);
    }
}

// ---------------------------------------------------------------- schema and request

static outcome<compiled_schema> compile_text(const std::string & text) {
    const common_json j = common_json::parse_no_throw(text);
    if (j.is_discarded()) {
        outcome<compiled_schema> out;
        out.error = "unparsable test JSON";
        return out;
    }
    return compile_schema(j, "");
}

static void expect_schema_error(const char * name, const std::string & text, const char * needle) {
    const auto cs = compile_text(text);
    CHECK(!cs.ok(), "%s: accepted", name);
    CHECK(cs.error.find(needle) != std::string::npos, "%s: error '%s' lacks '%s'", name, cs.error.c_str(), needle);
}

static std::vector<std::string> encoded(const std::string & text) {
    const auto cs = compile_text(text);
    if (!cs.ok() || cs.value.specs.empty()) {
        fprintf(stderr, "compile failed: %s\n", cs.error.c_str());
        return {};
    }
    return cs.value.specs[0].encoded;
}

static std::string join(const std::vector<std::string> & v) {
    std::string s;
    for (const auto & x : v) {
        s += (s.empty() ? "" : ",") + x;
    }
    return s;
}

static void test_schema() {
    const int failures_at_start = g_failures;
    // integer bounds: width checked without overflow, iteration bounded
    CHECK(join(encoded(R"({"n": {"type": "integer", "minimum": 9223372036854775807, "maximum": 9223372036854775807, "description": "d"}})")) ==
          "9223372036854775807", "integer at LLONG_MAX");
    CHECK(encoded(R"({"n": {"type": "integer", "minimum": -9223372036854775808, "maximum": -9223372036854775554, "description": "d"}})").size() == 255,
          "255 integers from LLONG_MIN");
    expect_schema_error("full 64-bit range", R"({"n": {"type": "integer", "minimum": -9223372036854775808, "maximum": 9223372036854775807, "description": "d"}})", "1-255 values");
    expect_schema_error("256 integers", R"({"n": {"type": "integer", "minimum": 0, "maximum": 255, "description": "d"}})", "1-255 values");
    expect_schema_error("reversed bounds", R"({"n": {"type": "integer", "minimum": 5, "maximum": 4, "description": "d"}})", "1-255 values");
    expect_schema_error("fractional bound", R"({"n": {"type": "integer", "minimum": 1.5, "maximum": 4, "description": "d"}})", "integer minimum");
    expect_schema_error("unsigned beyond LLONG_MAX", R"({"n": {"type": "integer", "minimum": 0, "maximum": 18446744073709551615, "description": "d"}})", "64-bit");

    // numbers: compact step grids start at minimum; multipleOf grids are multiples of the step
    CHECK(join(encoded(R"({"x": {"type": "number", "minimum": 0.1, "maximum": 0.9, "step": 0.2, "description": "d"}})")) ==
          "0.1,0.3,0.5,0.7,0.9", "compact step grid");
    CHECK(join(encoded(R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 0.1, "maximum": 1.0, "multipleOf": 0.2}}})")) ==
          "0.2,0.4,0.6,0.8,1.0", "multipleOf grid aligned to the step");
    CHECK(join(encoded(R"({"x": {"type": "number", "minimum": -1, "maximum": 1, "step": 0.5, "description": "d"}})")) ==
          "-1.0,-0.5,0.0,0.5,1.0", "no negative zero");
    expect_schema_error("step grid just short of maximum", R"({"x": {"type": "number", "minimum": 0, "maximum": 1.00000001, "step": 0.5, "description": "d"}})", "both ends");
    expect_schema_error("step grid misses maximum", R"({"x": {"type": "number", "minimum": 0, "maximum": 1, "step": 0.3, "description": "d"}})", "both ends");
    {
        // JSON text cannot carry infinity, a caller-built object can
        common_json field = common_json::object();
        field["type"]        = "number";
        field["minimum"]     = 0.0;
        field["maximum"]     = INFINITY;
        field["step"]        = 1.0;
        field["description"] = "d";
        common_json schema   = common_json::object();
        schema["x"]          = field;
        const auto cs = compile_schema(schema, "");
        CHECK(!cs.ok() && cs.error.find("finite") != std::string::npos, "infinite bound: '%s'", cs.error.c_str());
    }
    expect_schema_error("zero step", R"({"x": {"type": "number", "minimum": 0, "maximum": 1, "step": 0, "description": "d"}})", "positive");
    expect_schema_error("too fine step", R"({"x": {"type": "number", "minimum": 0, "maximum": 1e-9, "step": 1e-11, "description": "d"}})", "9 decimals");
    expect_schema_error("huge bound", R"({"x": {"type": "number", "minimum": 0, "maximum": 1e300, "step": 1e299, "description": "d"}})", "1e15");
    CHECK(join(encoded(R"({"x": {"type": "number", "minimum": 999999999999999.5, "maximum": 999999999999999.5, "step": 1, "description": "d"}})")) ==
          "999999999999999.5", "decimals kept near 1e15");
    CHECK(join(encoded(R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 0.300000001, "maximum": 0.5, "multipleOf": 0.1}}})")) ==
          "0.4,0.5", "a multiple just below minimum is left out");
    // the double product 909090909090909 * 0.11 prints as the bound; the exact one is 99999999999999.99
    expect_schema_error("bound that is no multiple at 1e14", R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 99999999999999.98, "maximum": 99999999999999.98, "multipleOf": 0.11}}})", "multiples");
    CHECK(join(encoded(R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 9000000000000.01, "maximum": 9000000000000.03, "multipleOf": 0.11}}})")) ==
          "9000000000000.02", "an exact multiple at 9e12");
    // .99 reads back as the double of .98 at this magnitude
    expect_schema_error("grid value without a double", R"({"x": {"type": "number", "minimum": 99999999999999.97, "maximum": 100000000000000, "step": 0.01, "description": "d"}})", "no double of its own");
    expect_schema_error("bounds past 9 decimals", R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 0.30000000001, "maximum": 0.5, "multipleOf": 0.1}}})", "9 decimals");
    expect_schema_error("past 18 digits", R"({"x": {"type": "number", "minimum": 999999999999.5, "maximum": 999999999999.5, "step": 0.0000001, "description": "d"}})", "18 significant digits");
    expect_schema_error("no multiple in bounds", R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 0.1, "maximum": 0.15, "multipleOf": 0.2}}})", "multiples");

    // enums
    expect_schema_error("duplicate choice", R"({"e": {"type": "enum", "choices": ["a", "a"], "description": "d"}})", "duplicate");
    expect_schema_error("non-string choice", R"({"e": {"type": "enum", "choices": ["a", 1], "description": "d"}})", "strings");
    expect_schema_error("enum and choices", R"({"e": {"type": "enum", "enum": ["allow"], "choices": ["deny"], "description": "d"}})", "not both");
    {
        std::string many = R"({"e": {"type": "enum", "description": "d", "choices": [)";
        for (int i = 0; i < 256; i++) {
            many += (i ? "," : "") + std::string("\"v") + std::to_string(i) + "\"";
        }
        expect_schema_error("256 choices", many + "]}}", "255");
        const std::string long_value(300, 'x');
        expect_schema_error("long choice", R"({"e": {"type": "enum", "description": "d", "choices": [")" + long_value + R"("]}})", "longer");
    }

    // keywords the compiler does not implement are refused, not ignored
    expect_schema_error("boolean enum", R"({"type": "object", "properties": {"b": {"type": "boolean", "enum": [true]}}})", "does not apply");
    expect_schema_error("integer enum", R"({"n": {"type": "integer", "minimum": 0, "maximum": 3, "enum": [1], "description": "d"}})", "does not apply");
    expect_schema_error("enum bounds", R"({"e": {"type": "enum", "choices": ["a"], "minimum": 0, "description": "d"}})", "does not apply");
    CHECK(compile_text(R"({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"]})").ok(), "required property");
    expect_schema_error("required unknown", R"({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["b"]})", "not a property");
    expect_schema_error("required twice", R"({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a", "a"]})", "twice");
    expect_schema_error("required not a list", R"({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": "a"})", "array");
    expect_schema_error("exclusiveMinimum", R"({"type": "object", "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0.5, "exclusiveMinimum": 0}}})", "exclusiveMinimum");
    expect_schema_error("pattern", R"({"type": "object", "properties": {"s": {"type": "string", "enum": ["a"], "pattern": "a"}}})", "pattern");
    expect_schema_error("anyOf at the top", R"({"type": "object", "properties": {"b": {"type": "boolean"}}, "anyOf": []})", "anyOf");
    expect_schema_error("integer step", R"({"n": {"type": "integer", "minimum": 0, "maximum": 4, "step": 2, "description": "d"}})", "no step");
    expect_schema_error("aggregate and x-aggregate", R"({"n": {"type": "integer", "minimum": 0, "maximum": 3, "aggregate": "mode", "x-aggregate": "mean", "description": "d"}})", "not both");
    expect_schema_error("mean of an enum", R"({"e": {"type": "enum", "choices": ["a", "b"], "aggregate": "mean", "description": "d"}})", "aggregate");
    expect_schema_error("missing description", R"({"b": {"type": "boolean"}})", "description");
    expect_schema_error("33 fields", [] {
        std::string s = "{";
        for (int i = 0; i < 33; i++) {
            s += (i ? "," : "") + std::string("\"f") + std::to_string(i) + R"(": {"type": "boolean", "description": "d"})";
        }
        return s + "}";
    }(), "1-32 fields");

    // compact fields may be named properties or type: the two forms are told apart by structure
    {
        const auto one = compile_text(R"({"properties": {"type": "boolean", "description": "d"}})");
        CHECK(one.ok() && one.value.specs.size() == 1 && one.value.specs[0].name == "properties",
              "compact field named properties: '%s'", one.error.c_str());
        const auto two = compile_text(R"({"type": {"type": "boolean", "description": "d"},
                                          "properties": {"type": "integer", "minimum": 0, "maximum": 3, "description": "d"}})");
        CHECK(two.ok() && two.value.specs.size() == 2, "compact fields named type and properties: '%s'", two.error.c_str());
        const auto js = compile_text(R"({"properties": {"type": {"type": "boolean"}}})");
        CHECK(js.ok() && js.value.specs.size() == 1 && js.value.specs[0].name == "type",
              "JSON Schema property named type: '%s'", js.error.c_str());
    }
    expect_schema_error("JSON Schema property not an object", R"({"type": "object", "properties": {"a": "boolean"}})",
                        "\"a\" must be an object");
    expect_schema_error("JSON Schema without properties", R"({"type": "object"})", "needs \"properties\"");

    // the suffix fixes the values' common leading characters
    {
        const auto cs = compile_text(R"({"e": {"type": "enum", "choices": ["alpha", "alphabet"], "description": "d"}})");
        CHECK(cs.ok() && cs.value.inputs[0].suffix == "  \"e\": \"alpha" && cs.value.inputs[0].candidates[0] == "\"" &&
              cs.value.inputs[0].candidates[1] == "bet\"", "suffix holds the common prefix");
    }

    // the largest valid contexts at worst-case JSON escaping still fit the CLI's read limit
    {
        common_json body = common_json::object();
        body["schema"]   = common_json::parse(R"({"b": {"type": "boolean", "description": "d"}})");
        body["contexts"] = common_json::array();
        for (size_t i = 0; i < max_request_bytes / max_context_bytes; ++i) {
            body["contexts"].push_back(std::string(max_context_bytes, '\x01')); // dumped as \u0001
        }
        const std::string text = body.dump();
        CHECK(text.size() > 5 * max_request_bytes && text.size() <= max_request_text_bytes,
              "a %zu-byte request of valid contexts against the %zu-byte read limit", text.size(), max_request_text_bytes);
        CHECK(parse_request(common_json::parse(text)).ok(), "the largest valid contexts parse");
    }

    // request parsing
    const auto req = [](const std::string & extra, const std::string & contexts = R"(["c"])") {
        return parse_request(common_json::parse(R"({"schema": {"b": {"type": "boolean", "description": "d"}}, "contexts": )" +
                                                contexts + extra + "}"));
    };
    CHECK(req("").ok(), "minimal request");
    CHECK(req(R"(, "tree_max": 0)").ok() && req(R"(, "tree_max": 255)").ok(), "tree_max bounds");
    for (const char * bad : { R"(, "tree_max": -1)", R"(, "tree_max": 256)", R"(, "tree_max": 1.5)", R"(, "tree_max": "7")",
                              R"(, "tree_max": 18446744073709551615)", R"(, "mode": "fast")", R"(, "cache_prompt": 1)" }) {
        CHECK(!req(bad).ok(), "accepted %s", bad);
    }
    CHECK(!req("", "[]").ok() && !req("", R"([""])").ok() && !req("", "[1]").ok() && !req("", R"("c")").ok(), "bad contexts");
    {
        std::string many = "[";
        for (int i = 0; i < 257; i++) {
            many += (i ? "," : "") + std::string("\"c\"");
        }
        CHECK(!req("", many + "]").ok(), "257 contexts");
    }

    // aggregates over a tree distribution
    {
        auto cs = compile_text(R"({"x": {"type": "integer", "minimum": 1, "maximum": 5, "aggregate": "median", "description": "d"}})");
        CHECK(cs.ok() && cs.value.inputs.size() == 1 && cs.value.inputs[0].needs_tree, "median schema scores a tree");
        result r;
        field_result fr;
        fr.winner = 0;
        fr.tree   = true;
        fr.probs  = { 0.4f, 0.05f, 0.1f, 0.05f, 0.4f };
        r.fields  = { fr };
        auto a = assemble(cs.value, r);
        CHECK(a.ok() && a.value.at("decision").at("x").get<long long>() == 3, "median of the distribution");
        cs.value.specs[0].aggregate = "mean";
        a = assemble(cs.value, r);
        CHECK(a.ok() && a.value.at("decision").at("x").get<long long>() == 3, "grid value nearest to the mean");
        cs.value.specs[0].aggregate = "mode";
        a = assemble(cs.value, r);
        CHECK(a.ok() && a.value.at("decision").at("x").get<long long>() == 1, "mode keeps the winner");
    }
    // 64-bit integers past 2^53 stay distinct: the mean and the median select the middle value
    for (const char * agg : { "mean", "median" }) {
        auto cs = compile_text(std::string(R"({"x": {"type": "integer", "minimum": 9007199254740992, "maximum": 9007199254740994, "aggregate": ")") +
                               agg + R"(", "description": "d"}})");
        CHECK(cs.ok(), "large integer %s schema: %s", agg, cs.error.c_str());
        if (!cs.ok()) {
            continue;
        }
        result r;
        field_result fr;
        fr.winner = 0;
        fr.tree   = true;
        fr.probs  = { 0.0f, 1.0f, 0.0f };
        r.fields  = { fr };
        const auto a = assemble(cs.value, r);
        CHECK(a.ok() && a.value.at("decision").at("x").get<long long>() == 9007199254740993LL, "large integer %s: %s", agg,
              a.ok() ? a.value.at("decision").dump().c_str() : a.error.c_str());
    }
    if (g_failures == failures_at_start) {
        printf("ok: schema compiler and request parser\n");
    }
}

// ---------------------------------------------------------------- token plan

static tokens_t joint_tokens(const llama_vocab * vocab, const std::string & text) {
    tokens_t toks = common_tokenize(vocab, text, true, true);
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    return toks;
}

// every planned path must be exactly the joint text's tokenization; returns false when the plan failed
static bool check_plan(const char * label, const llama_vocab * vocab, const std::string & shared,
                       const std::vector<std::string> & texts, const compiled_schema & cs, std::string & plan_error) {
    const auto plan = plan_tokens(vocab, shared, texts, cs.inputs);
    if (!plan.ok()) {
        plan_error = plan.error;
        return false;
    }
    const token_plan & p = plan.value;
    int bad = 0;
    for (size_t i = 0; i < texts.size(); ++i) {
        const auto & fields = p.tails[p.tail_of[i]];
        for (size_t f = 0; f < cs.inputs.size(); ++f) {
            for (size_t j = 0; j < cs.inputs[f].candidates.size(); ++j) {
                tokens_t got = p.shared;
                got.insert(got.end(), p.trunks[i].begin(), p.trunks[i].end());
                got.insert(got.end(), fields[f].suffix.begin(), fields[f].suffix.end());
                got.insert(got.end(), fields[f].paths[j].begin(), fields[f].paths[j].end());
                const tokens_t want = joint_tokens(vocab, shared + texts[i] + cs.inputs[f].suffix + cs.inputs[f].candidates[j] + "\n");
                bad += got != want;
            }
        }
    }
    CHECK(bad == 0, "%s: %d planned path(s) differ from the joint tokenization", label, bad);
    return true;
}

static std::vector<std::string> list_vocabs(const std::string & dir) {
    std::vector<std::string> out;
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return out;
    }
    while (dirent * e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.rfind("ggml-vocab-", 0) == 0 && name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

static const char * k_test_schema = R"({
    "category": {"type": "enum", "choices": ["billing", "technical", "tech support", "other"], "description": "Request type."},
    "urgent":   {"type": "boolean", "description": "Needs urgent handling?"},
    "priority": {"type": "integer", "minimum": 1, "maximum": 12, "description": "Priority."},
    "score":    {"type": "number", "minimum": 0, "maximum": 1, "step": 0.25, "description": "Confidence."}
})";

static void test_vocabs(const std::string & dir) {
    const int failures_at_start = g_failures;
    const auto files = list_vocabs(dir);
    CHECK(!files.empty(), "no ggml-vocab-*.gguf in %s", dir.c_str());
    const auto cs = compile_text(k_test_schema);
    CHECK(cs.ok(), "test schema: %s", cs.error.c_str());
    if (!cs.ok()) {
        return;
    }
    const std::vector<std::string> contexts = {
        "I was charged twice and need this fixed today.",
        "  leading spaces, trailing newline\n",
        "digits 12345 and unicode: na\xc3\xafve caf\xc3\xa9 \xe2\x9c\x93",
        "ends with a brace {",
        "<|im_end|> looks like a special token",
    };
    int n_tested = 0;
    for (const auto & path : files) {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) {
            continue;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string name = path.substr(path.rfind('/') + 1);
        if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_WPM || llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_UGM) {
            llama_model_free(model); // encoder tokenizers: no chat use
            continue;
        }
        common_chat_templates_ptr tmpls;
        try {
            tmpls = common_chat_templates_init(model, "");
        } catch (const std::exception &) {
        }
        for (int raw = 0; raw < 2; raw++) {
            const auto rp = render_prompt(raw ? nullptr : tmpls.get(), true, cs.value.system_text);
            if (!rp.ok()) {
                CHECK(raw == 0, "%s: raw layout failed: %s", name.c_str(), rp.error.c_str());
                continue;
            }
            std::vector<std::string> texts;
            for (const auto & c : contexts) {
                texts.push_back(c + rp.value.after);
            }
            std::string err;
            const std::string label = name + (raw ? " raw" : " template");
            if (!check_plan(label.c_str(), vocab, rp.value.shared, texts, cs.value, err)) {
                // a refused plan is acceptable without a template (no special token to split at), never with one
                CHECK(raw == 1, "%s: plan refused: %s", label.c_str(), err.c_str());
                printf("    %s: refused: %s\n", label.c_str(), err.c_str());
            }
            n_tested++;
        }
        llama_model_free(model);
    }
    if (g_failures == failures_at_start) {
        printf("ok: token plan matches the joint tokenization (%d vocabulary layouts)\n", n_tested);
    }
}

// ---------------------------------------------------------------- oracle

struct oracle {
    llama_context * ctx;
    int             n_vocab;
    std::map<tokens_t, std::vector<float>> cache;

    const std::vector<float> * logits(const tokens_t & prefix) {
        auto it = cache.find(prefix);
        if (it != cache.end()) {
            return &it->second;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        const int n_batch = (int) llama_n_batch(ctx);
        llama_batch b = llama_batch_init(n_batch, 0, 1);
        bool ok = true;
        for (size_t i0 = 0; ok && i0 < prefix.size(); i0 += n_batch) {
            common_batch_clear(b);
            const size_t i1 = std::min(prefix.size(), i0 + n_batch);
            for (size_t i = i0; i < i1; ++i) {
                common_batch_add(b, prefix[i], (llama_pos) i, { 0 }, i + 1 == prefix.size());
            }
            ok = llama_decode(ctx, b) == 0;
        }
        llama_batch_free(b);
        if (!ok) {
            return nullptr;
        }
        const float * l = llama_get_logits_ith(ctx, -1);
        return &cache.emplace(prefix, std::vector<float>(l, l + n_vocab)).first->second;
    }
};

struct oracle_field {
    std::vector<double> probs; // tree
    int                 greedy_winner = -1;
    double              greedy_score  = 1.0;
    double              greedy_margin = 1.0; // smallest gap between the best and second option of a step
};

// the constrained distribution of one field after prompt, scored node by node from scratch
static bool oracle_score(oracle & o, const tokens_t & prompt, const std::vector<tokens_t> & paths, oracle_field & out) {
    const size_t n = paths.size();
    const auto allowed = [&](const tokens_t & prefix) {
        tokens_t opts;
        for (const auto & p : paths) {
            if (p.size() > prefix.size() && std::equal(prefix.begin(), prefix.end(), p.begin()) &&
                std::find(opts.begin(), opts.end(), p[prefix.size()]) == opts.end()) {
                opts.push_back(p[prefix.size()]);
            }
        }
        return opts;
    };
    const auto local_logp = [&](const tokens_t & prefix, const tokens_t & opts, std::vector<double> & lp) {
        tokens_t full = prompt;
        full.insert(full.end(), prefix.begin(), prefix.end());
        const std::vector<float> * l = o.logits(full);
        if (!l) {
            return false;
        }
        double mx = -INFINITY;
        for (llama_token t : opts) {
            mx = std::max(mx, (double) (*l)[t]);
        }
        double z = 0;
        for (llama_token t : opts) {
            z += std::exp((*l)[t] - mx);
        }
        lp.clear();
        for (llama_token t : opts) {
            lp.push_back((*l)[t] - mx - std::log(z));
        }
        return true;
    };

    std::vector<double> path_lp(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        for (size_t d = 0; d < paths[j].size(); ++d) {
            const tokens_t prefix(paths[j].begin(), paths[j].begin() + d);
            const tokens_t opts = allowed(prefix);
            if (opts.size() < 2) {
                continue;
            }
            std::vector<double> lp;
            if (!local_logp(prefix, opts, lp)) {
                return false;
            }
            path_lp[j] += lp[std::find(opts.begin(), opts.end(), paths[j][d]) - opts.begin()];
        }
    }
    const double mx = *std::max_element(path_lp.begin(), path_lp.end());
    double z = 0;
    for (double x : path_lp) {
        z += std::exp(x - mx);
    }
    out.probs.clear();
    for (double x : path_lp) {
        out.probs.push_back(std::exp(x - mx) / z);
    }

    tokens_t chosen;
    out.greedy_score = 1.0;
    while (true) {
        const tokens_t opts = allowed(chosen);
        if (opts.empty()) {
            break;
        }
        if (opts.size() == 1) {
            chosen.push_back(opts[0]);
            continue;
        }
        std::vector<double> lp;
        if (!local_logp(chosen, opts, lp)) {
            return false;
        }
        const size_t best = std::max_element(lp.begin(), lp.end()) - lp.begin();
        double second = 0;
        for (size_t k = 0; k < lp.size(); ++k) {
            if (k != best) {
                second = std::max(second, std::exp(lp[k]));
            }
        }
        out.greedy_margin = std::min(out.greedy_margin, std::exp(lp[best]) - second);
        out.greedy_score *= std::exp(lp[best]);
        chosen.push_back(opts[best]);
    }
    out.greedy_winner = (int) (std::find(paths.begin(), paths.end(), chosen) - paths.begin());
    return true;
}

// compare an engine result with the oracle for every context and field of a plan
static void compare_with_oracle(const char * label, oracle & o, const token_plan & plan, const batch_result & r,
                                const options & opt, double tol) {
    CHECK(r.items.size() == plan.trunks.size(), "%s: %zu results for %zu contexts", label, r.items.size(), plan.trunks.size());
    double worst = 0;
    for (size_t i = 0; i < r.items.size() && i < plan.trunks.size(); ++i) {
        const auto & fields = plan.tails[plan.tail_of[i]];
        for (size_t f = 0; f < fields.size() && f < r.items[i].fields.size(); ++f) {
            tokens_t prompt = plan.shared;
            prompt.insert(prompt.end(), plan.trunks[i].begin(), plan.trunks[i].end());
            prompt.insert(prompt.end(), fields[f].suffix.begin(), fields[f].suffix.end());
            oracle_field of;
            if (!oracle_score(o, prompt, fields[f].paths, of)) {
                CHECK(false, "%s: oracle decode failed", label);
                return;
            }
            const field_result & fr = r.items[i].fields[f];
            const bool tree = opt.mode == score_mode::tree || fields[f].needs_tree ||
                              (opt.mode == score_mode::automatic && fields[f].paths.size() <= (size_t) opt.tree_max);
            CHECK(fr.tree == tree, "%s: context %zu field %zu: tree %d, expected %d", label, i, f, fr.tree, tree);
            if (tree) {
                CHECK(fr.probs.size() == of.probs.size(), "%s: context %zu field %zu: %zu probabilities", label, i, f, fr.probs.size());
                for (size_t j = 0; j < fr.probs.size() && j < of.probs.size(); ++j) {
                    worst = std::max(worst, std::fabs(fr.probs[j] - of.probs[j]));
                }
                const int best = (int) (std::max_element(of.probs.begin(), of.probs.end()) - of.probs.begin());
                std::vector<double> sorted = of.probs;
                std::sort(sorted.rbegin(), sorted.rend());
                const bool clear_winner = sorted.size() < 2 || sorted[0] - sorted[1] > 2 * tol;
                CHECK(!clear_winner || fr.winner == best, "%s: context %zu field %zu: winner %d, oracle %d", label, i, f, fr.winner, best);
            } else if (of.greedy_margin > 2 * tol) {
                CHECK(fr.winner == of.greedy_winner, "%s: context %zu field %zu: greedy winner %d, oracle %d", label, i, f,
                      fr.winner, of.greedy_winner);
                worst = std::max(worst, std::fabs(fr.path_score - of.greedy_score));
            } else if (fr.winner != of.greedy_winner) {
                // a step within the backend's spread may go either way, and so may the path after it
                printf("    %s: context %zu field %zu: near tie (margin %.2g), winner %d, oracle %d\n", label, i, f,
                       of.greedy_margin, fr.winner, of.greedy_winner);
            }
        }
    }
    CHECK(worst <= tol, "%s: probabilities differ from the oracle by %.3g (tolerance %.3g)", label, worst, tol);
    printf("    %s: max |p - oracle| %.2g\n", label, worst);
}

// ---------------------------------------------------------------- token-level engine checks

// a request built from random tokens: multi-token tries with shared prefixes, a one-value field
// and an integer-like field whose values are prefixes of each other up to the terminator
static token_plan random_plan(std::mt19937 & rng, int n_vocab, const std::vector<int> & trunk_lens) {
    std::vector<llama_token> pool(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        pool[i] = i;
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    size_t next = 0;
    const auto tok = [&]() { return pool[next++ % pool.size()]; };
    const auto toks = [&](int n) {
        tokens_t v;
        for (int i = 0; i < n; i++) {
            v.push_back(tok());
        }
        return v;
    };

    token_plan p;
    p.shared = toks(6);
    for (int len : trunk_lens) {
        p.trunks.push_back(toks(len));
        p.tail_of.push_back(0);
    }
    const llama_token term = tok();
    const llama_token a = tok(), b = tok(), c = tok(), d = tok(), e = tok();
    std::vector<field_tokens> fields(5);
    fields[0].suffix = toks(3);
    fields[0].paths  = { { a, b, term }, { a, b, c, term }, { a, d, term }, { e, term } };
    fields[1].suffix = toks(2);
    fields[1].paths  = { { tok() }, { tok() } };
    fields[2].suffix = toks(2);
    fields[2].paths  = { { tok(), term } };
    fields[3].suffix = toks(1);
    std::vector<llama_token> digit;
    for (int i = 0; i < 10; i++) {
        digit.push_back(tok());
    }
    for (int v = 1; v <= 12; v++) { // "1".."12" with a terminator, as integers tokenize per digit
        tokens_t path;
        for (char ch : std::to_string(v)) {
            path.push_back(digit[ch - '0']);
        }
        path.push_back(term);
        fields[3].paths.push_back(path);
    }
    // two divergences in a row: a greedy walk needs a second round
    const llama_token x0 = tok(), x1 = tok(), y0 = tok(), y1 = tok();
    fields[4].suffix = toks(2);
    fields[4].paths  = { { x0, y0, term }, { x0, y1, term }, { x1, y0, term }, { x1, y1, term } };
    p.tails = { fields };
    return p;
}

// token rows a greedy walk decodes on its way to value w: the suffix and the chosen tokens at
// every divergence
static size_t greedy_rows_to(const field_tokens & ft, int w) {
    std::vector<size_t> act;
    for (size_t i = 0; i < ft.paths.size(); ++i) {
        act.push_back(i);
    }
    size_t rows = 0;
    for (size_t depth = 0; act.size() > 1 && w >= 0 && depth < ft.paths[w].size(); ++depth) {
        std::vector<llama_token> next;
        std::vector<size_t>      keep;
        for (size_t i : act) {
            if (depth < ft.paths[i].size()) {
                if (std::find(next.begin(), next.end(), ft.paths[i][depth]) == next.end()) {
                    next.push_back(ft.paths[i][depth]);
                }
                if (ft.paths[i][depth] == ft.paths[w][depth]) {
                    keep.push_back(i);
                }
            }
        }
        if (next.size() > 1) {
            rows += ft.suffix.size() + depth;
        }
        act = keep;
    }
    return rows;
}

static bool engine_seqs_empty(llama_context * ctx, llama_seq_id base, int n) {
    for (llama_seq_id s = base; s < base + n; ++s) {
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), s) >= 0) {
            return false;
        }
    }
    return true;
}

static llama_context * make_context(llama_model * model, int n_ctx, int n_seq, int n_batch, int n_ubatch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = n_seq;
    cp.kv_unified      = true;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    return llama_init_from_model(model, cp);
}

static void test_model(const std::string & path) {
    const int failures_at_start = g_failures;
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    CHECK(model != nullptr, "cannot load %s", path.c_str());
    if (!model) {
        return;
    }
    const std::string name = path.substr(path.rfind('/') + 1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const bool padded = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    printf("== %s (%s)\n", name.c_str(), padded ? "padded branches" : "plain branches");

    llama_context * octx = make_context(model, 512, 1, 512, 512);
    llama_context * ctx  = make_context(model, 1024, 16, 64, 32); // n_batch 64 splits groups
    CHECK(octx && ctx, "%s: context creation failed", name.c_str());
    if (!octx || !ctx) {
        llama_free(octx);
        llama_free(ctx);
        llama_model_free(model);
        return;
    }
    oracle o{ octx, n_vocab, {} };
    std::mt19937 rng(7);
    const token_plan plan = random_plan(rng, n_vocab, { 3, 9, 5 });
    const double tol = 2e-3;

    auto eng = engine::create(ctx, 0, 16);
    CHECK(eng.ok(), "%s: %s", name.c_str(), eng.error.c_str());
    if (eng.ok()) {
        engine & e = *eng.value;
        failure err;
        options opt;

        opt.mode = score_mode::tree;
        const batch_result tree = e.decide_plan(plan, opt, err);
        CHECK(!err, "%s: tree: %s", name.c_str(), err.message.c_str());
        compare_with_oracle("tree", o, plan, tree, opt, tol);
        CHECK(!tree.cache_hit, "first request hit the cache");

        // the one-value field is certain and scores nothing
        CHECK(tree.items.size() == 3 && tree.items[0].fields[2].winner == 0 && tree.items[0].fields[2].probs.size() == 1 &&
              tree.items[0].fields[2].probs[0] == 1.0f && tree.items[0].fields[2].scored_nodes == 0, "one-value field");

        const batch_result again = e.decide_plan(plan, opt, err);
        CHECK(!err && again.cache_hit, "%s: second request must reuse the cached prefix", name.c_str());
        compare_with_oracle("tree, cached prefix", o, plan, again, opt, tol);

        opt.allow_cache = false;
        compare_with_oracle("tree, cache off", o, plan, e.decide_plan(plan, opt, err), opt, tol);
        opt.allow_cache = true;

        opt.mode = score_mode::greedy;
        const batch_result greedy = e.decide_plan(plan, opt, err);
        CHECK(!err && greedy.rounds > 1, "%s: greedy needs several rounds (%d): %s", name.c_str(), greedy.rounds, err.message.c_str());
        compare_with_oracle("greedy", o, plan, greedy, opt, tol);
        // usage counts every replay of the suffix and the chosen tokens; on a padded model also the
        // rows that pad a branch to its group's longest, so more than the branches hold
        size_t unpadded = 0, reported = 0;
        for (size_t i = 0; i < greedy.items.size(); ++i) {
            size_t want = 0;
            for (size_t f = 0; f < greedy.items[i].fields.size(); ++f) {
                want += greedy_rows_to(plan.tails[plan.tail_of[i]][f], greedy.items[i].fields[f].winner);
            }
            CHECK(padded ? (size_t) greedy.items[i].rows >= want : (size_t) greedy.items[i].rows == want,
                  "%s: greedy context %zu reports %d rows, its branches hold %zu", name.c_str(), i, greedy.items[i].rows, want);
            unpadded += want;
            reported += (size_t) greedy.items[i].rows;
        }
        CHECK(reported == (size_t) greedy.rows, "%s: context rows sum to %zu, the request reports %d", name.c_str(), reported,
              greedy.rows);
        CHECK(!padded || reported > unpadded, "%s: padded greedy branches report %zu rows, no more than their %zu tokens",
              name.c_str(), reported, unpadded);

        opt.mode     = score_mode::automatic;
        opt.tree_max = 4; // fields 0-2 and 4 tree, field 3 greedy
        compare_with_oracle("auto, tree_max 4", o, plan, e.decide_plan(plan, opt, err), opt, tol);

        // a median or mean field: auto scores it as a tree past tree_max, greedy refuses it
        {
            token_plan agg = plan;
            agg.tails[0][3].needs_tree = true;
            const batch_result r = e.decide_plan(agg, opt, err);
            CHECK(!err && !r.items.empty() && r.items[0].fields[3].tree && r.items[0].fields[3].probs.size() == 12,
                  "%s: aggregate field scored as a tree: %s", name.c_str(), err.message.c_str());
            compare_with_oracle("auto, tree_max 4, aggregate field", o, agg, r, opt, tol);
            opt.mode = score_mode::greedy;
            e.decide_plan(agg, opt, err);
            CHECK(err.code == error_code::invalid_request && err.message.find("aggregate") != std::string::npos,
                  "%s: greedy aggregate: '%s'", name.c_str(), err.message.c_str());
            opt.mode = score_mode::automatic;
        }

        // each context alone gives the same answer as the batch
        opt.mode = score_mode::tree;
        for (size_t i = 0; i < plan.trunks.size(); ++i) {
            token_plan one = plan;
            one.trunks  = { plan.trunks[i] };
            one.tail_of = { 0 };
            const batch_result r = e.decide_plan(one, opt, err);
            CHECK(!err && r.items.size() == 1, "%s: single context %zu: %s", name.c_str(), i, err.message.c_str());
            if (r.items.size() != 1 || tree.items.size() != plan.trunks.size()) {
                continue; // reported above; nothing to compare
            }
            for (size_t f = 0; f < r.items[0].fields.size() && f < tree.items[i].fields.size(); ++f) {
                const auto & a = r.items[0].fields[f].probs;
                const auto & b = tree.items[i].fields[f].probs;
                double worst = 0;
                for (size_t j = 0; j < a.size() && j < b.size(); ++j) {
                    worst = std::max(worst, (double) std::fabs(a[j] - b[j]));
                }
                CHECK(a.size() == b.size() && worst <= tol, "%s: context %zu alone, field %zu: %.3g", name.c_str(), i, f, worst);
            }
        }

        // cancellation leaves no state behind
        int polls = 0;
        opt.should_abort = [&]() { return ++polls > 2; };
        e.decide_plan(plan, opt, err);
        CHECK(err.code == error_code::cancelled, "%s: cancel: '%s'", name.c_str(), err.message.c_str());
        CHECK(engine_seqs_empty(ctx, 0, 16), "%s: sequences left after a cancelled request", name.c_str());
        opt.should_abort = nullptr;

        // invalid plans are refused before anything is decoded
        token_plan broken = plan;
        broken.tails[0][0].paths.push_back({ broken.tails[0][0].paths[0][0] }); // a prefix of another value
        e.decide_plan(broken, opt, err);
        CHECK(err.code == error_code::invalid_request && err.message.find("colliding") != std::string::npos,
              "%s: prefix collision: '%s'", name.c_str(), err.message.c_str());
        broken = plan;
        broken.trunks[1].push_back(n_vocab);
        e.decide_plan(broken, opt, err);
        CHECK(err.code == error_code::invalid_request, "%s: out-of-vocabulary token accepted", name.c_str());
        // greedy replays the suffix and the chosen tokens at every divergence: 32 fields of a
        // 63-value comb (c^k z term) score up to 32 * (62 * 3 + 1891) = 66464 rows, over the limit
        {
            field_tokens comb;
            comb.suffix = { 4, 5, 6 };
            for (int k = 0; k < 63; ++k) {
                tokens_t path(k, 1);
                path.push_back(2);
                path.push_back(3);
                comb.paths.push_back(path);
            }
            broken = plan;
            broken.tails = { std::vector<field_tokens>(32, comb) };
            const score_mode mode = opt.mode;
            opt.mode = score_mode::greedy;
            e.decide_plan(broken, opt, err);
            CHECK(err.code == error_code::invalid_request && err.message.find("66464 token rows") != std::string::npos,
                  "%s: greedy rows over the limit: '%s'", name.c_str(), err.message.c_str());
            opt.mode = mode;
        }
        opt.tree_max = 256;
        e.decide_plan(plan, opt, err);
        CHECK(err.code == error_code::invalid_request && err.message.find("tree_max") != std::string::npos,
              "%s: tree_max 256 accepted", name.c_str());
        opt.tree_max = 128;
    }

    // the smallest sequence budget: one context and one branch per decode
    {
        auto small = engine::create(ctx, 4, 3);
        CHECK(small.ok(), "%s: %s", name.c_str(), small.error.c_str());
        options opt;
        opt.mode = score_mode::tree;
        failure err;
        if (small.ok()) {
            const batch_result r = small.value->decide_plan(plan, opt, err);
            CHECK(!err, "%s: 3 sequences: %s", name.c_str(), err.message.c_str());
            compare_with_oracle("tree, 3 sequences", o, plan, r, opt, tol);
            CHECK(r.decodes > 10, "%s: 3 sequences took only %d decodes", name.c_str(), r.decodes);
        }
        CHECK(!engine::create(ctx, 14, 3).ok(), "sequence range beyond n_seq_max accepted");
        CHECK(!engine::create(ctx, 0, 2).ok(), "2 sequences accepted");
    }

    // KV exhaustion: a context fits with its longest branch, not with all its branches at once;
    // the error leaves no state behind
    {
        llama_context * tiny = make_context(model, 256, 16, 256, 256);
        CHECK(tiny != nullptr, "%s: tiny context", name.c_str());
        if (tiny) {
            auto e = engine::create(tiny, 0, 16);
            CHECK(e.ok(), "%s: tiny engine: %s", name.c_str(), e.error.c_str());
            if (e.ok()) {
                std::mt19937 rng2(11);
                const token_plan big = random_plan(rng2, n_vocab, { 235, 235 }); // one context per group: 241 cells + 25 branch rows
                options opt;
                failure err;
                e.value->decide_plan(big, opt, err);
                CHECK(err.code == error_code::no_memory, "%s: KV exhaustion not reported: '%s'", name.c_str(), err.message.c_str());
                CHECK(engine_seqs_empty(tiny, 0, 16), "%s: sequences left after KV exhaustion", name.c_str());
                if (err) {
                    printf("    KV exhaustion: %s\n", err.message.c_str());
                }
                const batch_result r = e.value->decide_plan(plan, opt, err);
                CHECK(!err && r.items.size() == 3, "%s: request after KV exhaustion: %s", name.c_str(), err.message.c_str());
                // the longest branch decodes 3 suffix tokens and a 2-token prefix (the candidate is
                // read from the logits, the one-value field scores nothing): 6 + 244 + 5 fits 256
                std::mt19937 rng4(13);
                e.value->decide_plan(random_plan(rng4, n_vocab, { 244 }), opt, err);
                CHECK(!(err.code == error_code::invalid_request && err.message.find("positions") != std::string::npos),
                      "%s: a context that fits its longest branch was refused: '%s'", name.c_str(), err.message.c_str());
                std::mt19937 rng3(12);
                e.value->decide_plan(random_plan(rng3, n_vocab, { 300 }), opt, err);
                CHECK(err.code == error_code::invalid_request && err.message.find("positions") != std::string::npos,
                      "%s: oversized context: '%s'", name.c_str(), err.message.c_str());
            }
            llama_free(tiny);
        }
    }

    llama_free(ctx);
    llama_free(octx);
    llama_model_free(model);
    if (g_failures == failures_at_start) {
        printf("ok: %s\n", name.c_str());
    }
}

// ---------------------------------------------------------------- end to end on a real model

static void test_e2e(const std::string & path) {
    const int failures_at_start = g_failures;
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    CHECK(model != nullptr, "cannot load %s", path.c_str());
    if (!model) {
        return;
    }
    const std::string name = path.substr(path.rfind('/') + 1);
    printf("== %s end to end\n", name.c_str());
    llama_context * ctx  = make_context(model, 4096, 24, 512, 512);
    llama_context * octx  = make_context(model, 2048, 1, 512, 512);
    llama_context * octx2 = make_context(model, 2048, 1, 64, 64);
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const common_json body = common_json::parse(std::string(R"({"instructions": "Answer about this support request.", "schema": )") +
        k_test_schema + R"(, "contexts": ["I was charged twice and need this fixed today.", "The app crashes when I upload a photo, not urgent."]})");
    const auto req = parse_request(body);
    CHECK(req.ok(), "%s: %s", name.c_str(), req.error.c_str());
    auto eng = engine::create(ctx, 0, 24);
    const auto rp = render_prompt(tmpls.get(), true, req.value.schema.system_text);
    CHECK(eng.ok() && rp.ok(), "%s: setup: %s %s", name.c_str(), eng.error.c_str(), rp.error.c_str());
    if (req.ok() && eng.ok() && rp.ok()) {
        std::vector<std::string> texts;
        for (const auto & c : req.value.contexts) {
            texts.push_back(c + rp.value.after);
        }
        const auto plan = plan_tokens(vocab, rp.value.shared, texts, req.value.schema.inputs);
        CHECK(plan.ok(), "%s: plan: %s", name.c_str(), plan.error.c_str());
        std::string perr;
        check_plan(name.c_str(), vocab, rp.value.shared, texts, req.value.schema, perr);

        oracle o{ octx, llama_vocab_n_tokens(vocab), {} };
        // the backend's own spread: the same oracle with the prompt decoded in other ubatches.
        // Kernels are not batch-invariant (on the CPU the two oracles differ by up to about 1 in
        // a logit of SmolLM2-135M and Qwen3-0.6B), so the engine, whose batches differ from the
        // oracle's, is held to that spread rather than to a fixed bound.
        oracle o2{ octx2, llama_vocab_n_tokens(vocab), {} };
        double spread = 0;
        if (plan.ok()) {
            for (size_t i = 0; i < plan.value.trunks.size(); ++i) {
                const auto & fields = plan.value.tails[plan.value.tail_of[i]];
                for (size_t f = 0; f < fields.size(); ++f) {
                    tokens_t prompt = plan.value.shared;
                    prompt.insert(prompt.end(), plan.value.trunks[i].begin(), plan.value.trunks[i].end());
                    prompt.insert(prompt.end(), fields[f].suffix.begin(), fields[f].suffix.end());
                    oracle_field a, b;
                    CHECK(oracle_score(o, prompt, fields[f].paths, a) && oracle_score(o2, prompt, fields[f].paths, b), "oracle failed");
                    for (size_t j = 0; j < a.probs.size() && j < b.probs.size(); ++j) {
                        spread = std::max(spread, std::fabs(a.probs[j] - b.probs[j]));
                    }
                }
            }
        }
        const double tol = std::max(2e-3, 3 * spread);
        printf("    oracle at ubatch 512 vs 64: max |p - p'| %.2g; tolerance %.2g\n", spread, tol);
        options opt;
        opt.mode = score_mode::tree;
        failure err;
        const batch_result r = eng.value->decide_batch(rp.value.shared, texts, req.value.schema.inputs, opt, err);
        CHECK(!err, "%s: %s", name.c_str(), err.message.c_str());
        if (plan.ok() && !err) {
            compare_with_oracle("text request, tree", o, plan.value, r, opt, tol);
        }
        opt.mode = score_mode::greedy;
        const batch_result g = eng.value->decide_batch(rp.value.shared, texts, req.value.schema.inputs, opt, err);
        CHECK(!err, "%s: %s", name.c_str(), err.message.c_str());
        if (plan.ok() && !err) {
            compare_with_oracle("text request, greedy", o, plan.value, g, opt, tol);
        }

        const auto res = run(*eng.value, tmpls.get(), true, req.value);
        CHECK(res.ok(), "%s: run: %s", name.c_str(), res.error.c_str());
        if (res.ok()) {
            printf("    %s\n", res.value.at("results").at(0).at("decision").dump().c_str());
            printf("    %s\n", res.value.at("results").at(1).at("decision").dump().c_str());
            printf("    timings %s\n", res.value.at("timings").dump().c_str());
        }
    }
    llama_free(octx2);
    llama_free(octx);
    llama_free(ctx);
    llama_model_free(model);
    if (g_failures == failures_at_start) {
        printf("ok: %s end to end\n", name.c_str());
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s --vocab DIR | --model FILE... | --e2e FILE...\n", argv[0]);
        return 2;
    }
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();
    const std::string mode = argv[1];
    if (mode == "--vocab") {
        test_schema();
        test_vocabs(argv[2]);
    } else if (mode == "--model") {
        for (int i = 2; i < argc; i++) {
            test_model(argv[i]);
        }
    } else if (mode == "--e2e") {
        for (int i = 2; i < argc; i++) {
            test_e2e(argv[i]);
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    llama_backend_free();
    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all schema decision checks passed\n");
    return 0;
}
