// llama-schema-decision: answers a finite JSON schema about one or more contexts by scoring the
// allowed values of every field instead of generating. Reads a /v1/decision request body (see
// README.md) and prints the response object.

#include "schema-decision.h"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

// the request text, bounded so that a wrong file cannot exhaust memory; false on failure
static bool read_request(const std::string & path, std::string & text, std::string & error) {
    constexpr size_t max_bytes = schema_decision::max_request_text_bytes;
    std::ifstream file;
    std::istream * in = &std::cin;
    if (path != "-") {
        file.open(path, std::ios::binary);
        if (!file) {
            error = "cannot open " + path;
            return false;
        }
        in = &file;
    }
    text.clear();
    char buf[65536];
    while (in->read(buf, sizeof(buf)) || in->gcount() > 0) {
        text.append(buf, (size_t) in->gcount());
        if (text.size() > max_bytes) {
            error = path + " holds more than " + std::to_string(max_bytes) + " bytes";
            return false;
        }
    }
    if (in->bad()) {
        error = "reading " + path + " failed";
        return false;
    }
    return true;
}

static int run_cli(common_params & params) {
    if (params.decision_request.empty()) {
        LOG_ERR("%s: --decision-request is required (a file, or - for stdin)\n", __func__);
        return 1;
    }
    std::string text, error;
    if (!read_request(params.decision_request, text, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }
    const common_json body = common_json::parse_no_throw(text);
    if (body.is_discarded()) {
        LOG_ERR("%s: %s is not valid JSON\n", __func__, params.decision_request.c_str());
        return 1;
    }
    auto req = schema_decision::parse_request(body);
    if (!req.ok()) {
        LOG_ERR("%s: invalid request %s: %s\n", __func__, params.decision_request.c_str(), req.error.c_str());
        return 1;
    }

    // the engine owns every sequence of the context (n_seq_max = n_parallel + n_seq_decision);
    // branches share the trunk's cells
    params.n_parallel = 0;
    params.kv_unified = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        LOG_ERR("%s: failed to load model %s\n", __func__, params.model.path.c_str());
        return 1;
    }
    llama_context * ctx = init->context();

    auto eng = schema_decision::engine::create(ctx, 0, (int) llama_n_seq_max(ctx));
    if (!eng.ok()) {
        LOG_ERR("%s: %s\n", __func__, eng.error.c_str());
        return 1;
    }
    common_chat_templates_ptr tmpls = common_chat_templates_init(init->model(), params.chat_template);

    auto res = schema_decision::run(*eng.value, tmpls.get(), params.use_jinja, req.value);
    if (!res.ok()) {
        LOG_ERR("%s: decision failed: %s\n", __func__, res.error.c_str());
        return 1;
    }
    std::printf("%s\n", res.value.dump(2).c_str());
    std::fflush(stdout);
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SCHEMA_DECISION)) {
        return 1;
    }
    common_init();

    int rc = 1;
    try {
        rc = run_cli(params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: unexpected error: %s\n", __func__, e.what());
    }
    llama_backend_free();
    common_log_flush(common_log_main());
    return rc;
}
