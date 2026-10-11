#pragma once

#include "llama.h"
#include "common.h"
#include "speculative-cycle.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

// Where the speculative checkpoints of one sequence of a context are kept.
struct common_speculative_checkpoint_place {
    llama_state_seq_flags flags     = 0; // state flags of the latest checkpoint, 0 before the first one
    size_t                size_copy = 0; // tensor bytes of the copy the context holds on the device
};

// Picks the state flags for the next speculative checkpoint of one sequence of ctx and records them in place.
// Call it before every checkpoint update; the load that follows has to use place.flags.
// LLAMA_STATE_SEQ_FLAGS_ON_DEVICE keeps a device copy that --fit does not measure and that is allocated
// again whenever its size changes. It is chosen only while every device of the model keeps its margin free
// after the copy has grown to the new size. Otherwise the checkpoint stays on the host.
// The whole growth is checked against each device, which overstates a device's part of a split model.
// margins are indexed like --fit-target: by the device order of model_margins, the model they were given
// for. A device that model does not use keeps the largest margin free.
llama_state_seq_flags common_speculative_checkpoint_flags(
        common_speculative_checkpoint_place & place,
        llama_context * ctx, llama_seq_id seq_id,
        const std::vector<size_t> & margins, const llama_model * model_margins);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // Optional compatibility output; P06 ownership lives in the P04 finalized snapshot.
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    int64_t request_id = -1; // server task identity, independent of slot reuse
    uint64_t proposal_id = 0;
    std::vector<std::vector<llama_token_data>> dists;

};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process a batch already evaluated by the target and update the speculative context's internal state
// returns true for a null spec; otherwise stops at the first implementation returning false
// updates made by earlier implementations are retained on failure
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// legacy llama_batch input, converted with common_batch_from_llama_batch() when spec has a target context
// returns false if conversion fails; without explicit positions, target memory must still end at this batch
// otherwise has the same result and state-update behavior as the common_batch overload
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// Immutable MTP snapshot; also available with P06 enabled and cycle logging disabled.
struct common_sampler;
struct common_speculative_sampling_config {
    float temperature = 0;
    float top_p = 1;
};
// Pure parser for initialization and environment fixtures; error names the rejected variable/value.
bool common_speculative_sampling_parse(const char * temperature, const char * top_p,
        common_speculative_sampling_config & config, std::string & error);
// Completed P06 decisions may need KV repair, but must never be sampled again.
// The caller's draft stores the selected prefix plus the unprocessed correction.
struct common_speculative_repair {
    bool pending = false;
    size_t n_draft = 0;

    bool prepare(llama_tokens & draft, const llama_tokens & selected, bool stops_at_eog);
    size_t n_decode(const llama_tokens & draft) const;
    llama_tokens finish(const llama_tokens & draft);
    void clear();
};

llama_tokens common_speculative_sample_and_accept(common_speculative * spec, llama_seq_id seq,
        common_sampler * sampler, llama_context * ctx, const std::vector<int> & idxs,
        const llama_tokens & draft, uint64_t proposal, bool replay, bool grammar_first = false,
        bool * used_distribution = nullptr);
void common_speculative_invalidate(common_speculative * spec, llama_seq_id seq);

const common_speculative_cycle_record * common_speculative_cycle_get(const common_speculative * spec, llama_seq_id seq);
void common_speculative_cycle_finish(common_speculative * spec, llama_seq_id seq, uint64_t proposal, size_t accepted);
void common_speculative_cycle_cancel(common_speculative * spec, llama_seq_id seq);
std::shared_ptr<common_speculative_cycle_phase> common_speculative_verify_begin(
        common_speculative * spec, const std::vector<common_speculative_cycle_participant> & participants);
void common_speculative_verify_end(common_speculative * spec, const std::shared_ptr<common_speculative_cycle_phase> & phase);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
