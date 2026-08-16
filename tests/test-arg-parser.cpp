#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

struct env_var_snapshot {
    std::string name;
    bool was_set;
    std::string value;

    explicit env_var_snapshot(const char * name) : name(name) {
        const char * current = getenv(name);
        was_set = current != nullptr;
        if (current != nullptr) {
            value = current;
        }
    }

    ~env_var_snapshot() {
        set(was_set ? value.c_str() : nullptr);
    }

    void set(const char * new_value) const {
#ifdef _WIN32
        assert(_putenv_s(name.c_str(), new_value != nullptr ? new_value : "") == 0);
#else
        if (new_value != nullptr) {
            assert(setenv(name.c_str(), new_value, true) == 0);
        } else {
            assert(unsetenv(name.c_str()) == 0);
        }
#endif
    }
};

static void test_unknown_env_classifier(void) {
    std::vector<common_arg> options = {
        common_arg(
            {"--known"},
            {"--no-known"},
            "test option",
            [](common_params &, bool) {
            }
        ).set_env("LLAMA_ARG_KNOWN"),
    };
    const std::vector<std::string> environment = {
        "LLAMA_ARG_kNoWn=on",
        "LLAMA_ARG_NO_KNOWN=on",
        "PATH=/tmp",
        "llama_arg_beta=secret=must-not-leak",
        "LLAMA_ARG_ZETA=another-secret",
        "LLAMA_ARG_zeta=third-secret",
        "LLAMA_ARG_ALPHA=value",
    };

    const std::vector<std::string> unknown =
        common_arg_utils::find_unknown_env_vars(options, environment);
#ifdef _WIN32
    assert((unknown == std::vector<std::string>{"LLAMA_ARG_ALPHA", "LLAMA_ARG_BETA", "LLAMA_ARG_ZETA"}));
#else
    assert((unknown == std::vector<std::string>{
        "LLAMA_ARG_ALPHA", "LLAMA_ARG_ZETA", "LLAMA_ARG_kNoWn", "LLAMA_ARG_zeta"}));
#endif
}

static void test(void) {
    test_unknown_env_classifier();

    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    {
        env_var_snapshot warn_unknown_env("LLAMA_ARG_WARN_UNKNOWN_ENV");
        env_var_snapshot typo_env("LLAMA_ARG_TYPO");
        warn_unknown_env.set("1");
        typo_env.set("secret");

        const std::filesystem::path log_path = std::filesystem::temp_directory_path() /
            ("llama-test-arg-parser-" + std::to_string(std::random_device{}()) + ".log");
        common_log_set_file(common_log_main(), log_path.string().c_str());

        common_params invalid_params;
        argv = {"binary_name", "-m", "model.gguf", "--prompt-cache-all", "--interactive"};
        assert(false == common_params_parse(
            argv.size(), list_str_to_char(argv).data(), invalid_params, LLAMA_EXAMPLE_COMPLETION));

        common_log_flush(common_log_main());
        common_log_set_file(common_log_main(), nullptr);

        {
            std::ifstream log_file(log_path);
            const std::string log_contents(
                (std::istreambuf_iterator<char>(log_file)),
                std::istreambuf_iterator<char>());
            assert(log_contents.find("LLAMA_ARG_TYPO") == std::string::npos);
        }

        common_log_set_file(common_log_main(), log_path.string().c_str());

        common_params valid_params;
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(
            argv.size(), list_str_to_char(argv).data(), valid_params, LLAMA_EXAMPLE_COMMON));

        common_log_flush(common_log_main());
        common_log_set_file(common_log_main(), nullptr);

        {
            std::ifstream log_file(log_path);
            const std::string log_contents(
                (std::istreambuf_iterator<char>(log_file)),
                std::istreambuf_iterator<char>());
            assert(log_contents.find("LLAMA_ARG_TYPO") != std::string::npos);
            assert(log_contents.find("secret") == std::string::npos);
        }
        std::filesystem::remove(log_path);
    }

    {
        env_var_snapshot warn_unknown_env("LLAMA_ARG_WARN_UNKNOWN_ENV");
        warn_unknown_env.set(nullptr);

        common_params default_params;
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(
            argv.size(), list_str_to_char(argv).data(), default_params, LLAMA_EXAMPLE_COMMON));
        assert(!default_params.warn_unknown_env);

        common_params cli_params;
        argv = {"binary_name", "-m", "model.gguf", "--warn-unknown-env"};
        assert(true == common_params_parse(
            argv.size(), list_str_to_char(argv).data(), cli_params, LLAMA_EXAMPLE_COMMON));
        assert(cli_params.warn_unknown_env);

        warn_unknown_env.set("1");
        common_params env_params;
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(
            argv.size(), list_str_to_char(argv).data(), env_params, LLAMA_EXAMPLE_COMMON));
        assert(env_params.warn_unknown_env);
    }

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

    argv = {"binary_name", "--api-key", "\" first-key \",\"   \",second-key,\"third-key \""};
    params.api_keys.clear();
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.api_keys.size() == 3);
    assert(params.api_keys[0] == "first-key");
    assert(params.api_keys[1] == "second-key");
    assert(params.api_keys[2] == "third-key");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    {
        env_var_snapshot api_key("LLAMA_API_KEY");
        env_var_snapshot api_key_file("LLAMA_ARG_API_KEY_FILE");
        const auto key_file = std::filesystem::temp_directory_path() /
                              ("llama-api-keys-" + std::to_string(std::random_device{}()) + ".txt");

        setenv("LLAMA_ARG_API_KEY_FILE", key_file.string().c_str(), true);
        setenv("LLAMA_API_KEY", "stale-env-key", true);
        {
            std::ofstream output(key_file);
            output << "file-key\n";
        }

        argv = {"binary_name", "--api-key", "cli-key"};
        params.api_keys.clear();
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
        assert(params.api_keys.size() == 2);
        assert(params.api_keys[0] == "file-key");
        assert(params.api_keys[1] == "cli-key");
        assert(std::filesystem::remove(key_file));
    }

    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    {
        env_var_snapshot hf_token("HF_TOKEN");
        env_var_snapshot hub_token("HUGGING_FACE_HUB_TOKEN");
        env_var_snapshot misspelled_hub_token("HUGGINGFACE_HUB_TOKEN");

        hf_token.set(nullptr);
        hub_token.set(nullptr);
        misspelled_hub_token.set(nullptr);

        printf("test-arg-parser: test Hugging Face token precedence\n\n");

        hub_token.set("fallback-token");
        {
            common_params token_params;
            const auto handler = common_models_handler_init(token_params, LLAMA_EXAMPLE_COMMON);
            assert(handler.opts.bearer_token == "fallback-token");
        }

        hf_token.set("");
        {
            common_params token_params;
            const auto handler = common_models_handler_init(token_params, LLAMA_EXAMPLE_COMMON);
            assert(handler.opts.bearer_token == "fallback-token");
        }

        hf_token.set("hf-token");
        {
            common_params token_params;
            const auto handler = common_models_handler_init(token_params, LLAMA_EXAMPLE_COMMON);
            assert(handler.opts.bearer_token == "hf-token");
        }

        {
            common_params token_params;
            token_params.hf_token = "explicit-token";
            const auto handler = common_models_handler_init(token_params, LLAMA_EXAMPLE_COMMON);
            assert(handler.opts.bearer_token == "explicit-token");
        }

        hf_token.set(nullptr);
        hub_token.set(nullptr);
        misspelled_hub_token.set("misspelled-token");
        {
            common_params token_params;
            const auto handler = common_models_handler_init(token_params, LLAMA_EXAMPLE_COMMON);
            assert(handler.opts.bearer_token.empty());
        }
    }

#ifndef LLAMA_DOWNLOAD_DISABLED
    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }
#endif // LLAMA_DOWNLOAD_DISABLED

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
