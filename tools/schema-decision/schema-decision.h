#pragma once

// Constrained decisions for finite JSON schemas, shared by llama-schema-decision and
// llama-server's /v1/decision endpoint. Ported from thecodacus/llama.cpp decision-upstream
// (a8d96e948, tools/parallel-decision) with bounded inputs and errors returned as values.
//
// Every field of a schema has a finite set of allowed values. After a shared context, each
// field's values are scored as token paths that follow the field's own suffix. Every scored
// path runs as its own sequence forked from the context (llama_memory_seq_cp), so the fields
// are scored in batched llama_decode calls and cannot see each other. A tree field scores every
// divergence node of its token trie and returns the constrained distribution over its values;
// a greedy field walks the trie one divergence at a time.
//
// The probabilities are constrained-decoding probabilities: the softmax at each divergence
// node is taken over the allowed next tokens only. They are not calibrated probabilities that
// an answer is true, and not full-vocabulary sequence likelihoods.
//
// Thread safety: an engine is not thread safe and must be the only user of its sequence ids.
// It touches no other sequence, so it can share a context with other users that are
// serialized with it (llama-server's task loop).

#include "json.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct common_chat_templates;

namespace schema_decision {

// input limits, enforced by compile_schema, parse_request and the engine
constexpr size_t max_fields             = 32;
constexpr size_t max_values             = 255;     // per field
constexpr size_t max_contexts           = 256;     // per request
constexpr size_t max_name_bytes         = 256;     // field name
constexpr size_t max_value_bytes        = 256;     // one enum value
constexpr size_t max_description_bytes  = 4096;
constexpr size_t max_instructions_bytes = 65536;
constexpr size_t max_context_bytes      = 1 << 20; // one context
constexpr size_t max_request_bytes      = 16 << 20; // all contexts together
constexpr size_t max_path_tokens        = 64;      // one value with its terminator
constexpr size_t max_rows_per_context   = 65536;   // token rows of one context, before recurrent padding
constexpr int    max_tree_max           = 255;

// what an error is about; llama-server maps it to an HTTP status
enum class error_code {
    none,
    invalid_request, // the request cannot be scored as given
    no_memory,       // the KV cache ran out
    cancelled,       // options::should_abort returned true
    internal,        // a decode, chat template or engine failure
};

struct failure {
    error_code  code = error_code::none;
    std::string message;

    explicit operator bool() const { return code != error_code::none; }
};

// T on success, otherwise error is non-empty and value is default-constructed; code tells what
// the error is about and is only meaningful with an error
template <typename T> struct outcome {
    T           value{};
    std::string error;
    error_code  code = error_code::invalid_request;

    bool ok() const { return error.empty(); }
};

using tokens_t = std::vector<llama_token>;

// one field as the scorer sees it: the text before its value and the allowed value texts
struct field_input {
    std::string              suffix;     // e.g. '\n  "fire": '
    std::vector<std::string> candidates; // allowed values without the suffix's shared prefix
    bool                     needs_tree = false; // a median or mean aggregate reads the distribution
};

enum class score_mode { automatic, tree, greedy };

struct options {
    score_mode mode        = score_mode::automatic; // automatic: tree up to tree_max values
    int        tree_max    = 128;
    bool       allow_cache = true; // reuse the cached static prefix when it matches
    // polled before every llama_decode; returning true stops the request with an error
    std::function<bool()> should_abort;
};

struct field_result {
    int                winner       = -1;
    float              path_score   = 1.0f; // probability of the winner
    int                scored_nodes = 0;
    bool               tree         = false;
    std::vector<float> probs;               // tree fields: probability of every allowed value
};

struct result {
    std::vector<field_result> fields;
    size_t context_tokens = 0;
    int    rows           = 0;
};

// several contexts decided against one schema and one cached prefix; results keep the order
// of the contexts, timings and cache state cover the whole batch
struct batch_result {
    std::vector<result> items;
    bool   cache_hit     = false;
    size_t shared_tokens = 0; // tokens of the cached prefix
    int    rows          = 0;
    int    rounds        = 0;
    int    decodes       = 0;
    double prefill_ms    = 0;
    double scoring_ms    = 0;
};

// How a request's text becomes tokens. For every context i, field f and value j:
//   shared + trunks[i] + tails[tail_of[i]][f].suffix + paths[j]
//     == tokenize(shared_text + contexts[i] + inputs[f].suffix + inputs[f].candidates[j] + "\n")
// (with special tokens parsed and a single BOS), so every scored path is the joint text's own
// tokenization. Contexts that end in the same text share one tail entry.
struct field_tokens {
    tokens_t              suffix; // tokens every value of the field starts with
    std::vector<tokens_t> paths;  // per value, the rest including the terminator
    bool                  needs_tree = false; // see field_input
};

struct token_plan {
    tokens_t                               shared;  // cached prefix, common to every context
    std::vector<tokens_t>                  trunks;  // per context, the tokens after the prefix
    std::vector<int>                       tail_of; // per context, index into tails
    std::vector<std::vector<field_tokens>> tails;   // per distinct context tail, the fields
};

outcome<token_plan> plan_tokens(const llama_vocab * vocab, const std::string & shared_text,
                                const std::vector<std::string> & contexts, const std::vector<field_input> & inputs);

// Scores decisions on an existing context with the sequence ids [seq_base, seq_base + n_seqs):
// one keeps the cached static prefix; the rest hold one trunk (prefix + context) per context in
// flight, then branches. The context needs a unified KV cache so branches share the trunk's
// cells, and n_outputs_max_per_seq >= 1.
class engine {
  public:
    static outcome<std::unique_ptr<engine>> create(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    // shared_text is the static prefix (cached across calls); each context is the per-request
    // text that follows it, up to the point where the first field's suffix begins
    batch_result decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                              const std::vector<field_input> & fields, const options & opt, failure & error);

    // the scoring core of decide_batch, on an already tokenized request
    batch_result decide_plan(const token_plan & plan, const options & opt, failure & error);

    // forget the cached prefix and clear every sequence the engine owns
    void reset();

    // true while the static prefix of the last call holds cells in the context
    bool has_cached_prefix() const { return !cached.empty(); }

  private:
    engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    struct prompt_part {
        const tokens_t * toks;
        llama_pos        pos0;
        llama_seq_id     seq;
    };
    struct branch {
        llama_seq_id trunk;
        llama_pos    pos0;
        tokens_t     toks;
        tokens_t     cands;
    };

    llama_context     * ctx;
    const llama_vocab * vocab;
    llama_memory_t      mem;
    llama_seq_id        seq_snap;
    llama_seq_id        seq_pool;
    int                 n_pool;
    bool                pad_branches; // recurrent/hybrid model: branches in a ubatch need equal lengths
    tokens_t            cached;

    failure     decode_parts(const std::vector<prompt_part> & parts, const options & opt, int & n_decodes);
    failure     prepare_prefix(const tokens_t & shared, const options & opt, bool & hit, int & n_decodes);
    failure     score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free,
                               const options & opt, std::vector<std::vector<float>> & scores, std::vector<size_t> & rows,
                               int & n_decodes);
    void        clear_pool();
};

// ---- schema compiler

struct field_spec {
    std::string              name;
    std::string              type;      // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate; // mode | median | mean (median/mean: numeric fields)
    std::vector<common_json> values;    // typed values; index = candidate index
    std::vector<double>      numbers;   // numeric fields: value - number_base as doubles
    double                   number_base = 0; // integer fields: minimum, so 64-bit values stay exact offsets
    std::vector<std::string> encoded;   // JSON text of each value
};

struct compiled_schema {
    std::string             system_text; // fixed instructions and field catalogue (cacheable)
    std::vector<field_spec> specs;
    std::vector<field_input> inputs;
};

// Accepts compact field specs {"name": {"type": ..., "description": ..., ...}} or a JSON Schema
// object with "properties" (boolean, string enum, integer minimum/maximum, number
// minimum/maximum/multipleOf). Keywords that would restrict the values in a way the compiler
// does not implement are rejected.
outcome<compiled_schema> compile_schema(const common_json & schema, const std::string & instructions);

// Renders system and user messages with the model's chat template (thinking disabled) around a
// placeholder for the context. A request's text is shared + context + after: shared is the
// static prefix (cached across requests), after holds the end of the user turn, the generation
// prompt and the opening brace of the JSON answer. The context is inserted as is, not passed
// through the template. Without templates a plain "Context:" / "Output:" layout is used.
struct rendered_prompt {
    std::string shared;
    std::string after;
};
outcome<rendered_prompt> render_prompt(const common_chat_templates * tmpls, bool use_jinja, const std::string & system_text);

// {"decision": {...}, "fields": {...}} from the scores, applying each numeric field's aggregate
outcome<common_json> assemble(const compiled_schema & cs, const result & r);

// ---- request and response (the /v1/decision body; the CLI reads the same JSON)

struct request {
    std::string              instructions;
    compiled_schema          schema;
    std::vector<std::string> contexts;
    options                  opt;
};

outcome<request> parse_request(const common_json & body);

// renders, scores and assembles a whole request; the response object of /v1/decision
outcome<common_json> run(engine & eng, const common_chat_templates * tmpls, bool use_jinja, const request & req);

const char * mode_name(score_mode mode);

} // namespace schema_decision
