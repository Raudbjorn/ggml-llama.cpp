#include "model-cache.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <openvino/core/version.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#if defined(_WIN32)
#    include <direct.h>
#endif

namespace {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
inline uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

inline uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    return fnv1a(h, &v, sizeof(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Sampling budget for the weight fingerprint. The whole model is never hashed
// (that would cost seconds every run); instead we hash a bounded number of
// fixed-size windows spread evenly across each weight's bytes, from offset 0
// through the final window ending at nbytes. Spreading the windows (rather than
// only sampling the head and tail) means a change confined to the middle of a
// tensor still changes the fingerprint. Total sampled bytes per weight are
// capped regardless of tensor size, so cost stays bounded on multi-GB weights;
// the tradeoff is a residual (much smaller) collision risk between the sampled
// windows, which manifest re-verification does not add further protection
// against since it hashes the same windows.
constexpr size_t WEIGHT_SAMPLE_WINDOW_BYTES = 1024;
constexpr size_t WEIGHT_SAMPLE_WINDOWS = 8;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
bool is_weight_src(const ggml_tensor * src) {
    if (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        return false;
    }
    return src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Per-weight sampled fingerprint: identity (name/shape/type) + a bounded byte
// sample. Returns FNV offset basis if data is unavailable (kept deterministic).
uint64_t weight_fingerprint(const ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    const size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    if (t->data != nullptr && nbytes > 0) {
        const uint8_t * data = static_cast<const uint8_t *>(t->data);
        const size_t total_sample = WEIGHT_SAMPLE_WINDOWS * WEIGHT_SAMPLE_WINDOW_BYTES;
        if (nbytes <= total_sample) {
            // Small enough to hash in full: no unsampled gap is possible.
            h = fnv1a(h, data, nbytes);
        } else {
            // Evenly-spaced windows spanning the whole buffer, first window at
            // offset 0 and last window ending at nbytes, so the middle of the
            // tensor is covered along with the head and tail.
            for (size_t w = 0; w < WEIGHT_SAMPLE_WINDOWS; ++w) {
                const size_t start = (nbytes - WEIGHT_SAMPLE_WINDOW_BYTES) * w / (WEIGHT_SAMPLE_WINDOWS - 1);
                h = fnv1a(h, data + start, WEIGHT_SAMPLE_WINDOW_BYTES);
            }
        }
    }
    return h;
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
void for_each_weight(const ggml_cgraph * cgraph, F && fn) {
    std::vector<const ggml_tensor *> seen;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (!is_weight_src(src)) {
                continue;
            }
            bool dup = false;
            for (const auto * p : seen) {
                if (p == src) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    const ov::Version v = ov::get_openvino_version();
    return std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
bool make_dir(const std::string & path) {
#if defined(_WIN32)
    int rc = _mkdir(path.c_str());
#else
    int rc = ::mkdir(path.c_str(), 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
bool make_dirs(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        acc.push_back(c);
        const bool sep = (c == '/'
#if defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        if (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            if (!make_dir(component)) {
                return false;
            }
        }
    }
    return make_dir(path);
}

}  // namespace

std::string ggml_openvino_model_cache_dir() {
    const char * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    if (!dir || strlen(dir) == 0) {
        return std::string();
    }
    std::string path(dir);
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    if (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        return std::string();
    }
    return path;
}

uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg) {
    uint64_t h = FNV_OFFSET;

    // Everything about the graph other than weight contents: node count, and
    // for every node its op, name, element type, shape, strides, flags, view
    // offset, the raw op_params block (rope freq/scale, norm eps, permute
    // axes, clamp bounds, ... all baked into the ov::Model as attributes or
    // Constants) and the identity of each source edge. Two cgraphs that agree
    // on the op/name sequence but differ in activation geometry or an op
    // parameter must not share a key: the manifest re-checks weights only, so
    // this is the sole guard against importing a blob compiled for a different
    // graph. It also makes the index-based OV tensor names sound, since a hit
    // now implies the same node/leaf ordering.
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
        h = fnv1a_u64(h, static_cast<uint64_t>(node->type));
        h = fnv1a_u64(h, static_cast<uint64_t>(node->flags));
        h = fnv1a_u64(h, static_cast<uint64_t>(node->view_offs));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = fnv1a_u64(h, static_cast<uint64_t>(node->ne[d]));
            h = fnv1a_u64(h, static_cast<uint64_t>(node->nb[d]));
        }
        h = fnv1a(h, node->op_params, sizeof(node->op_params));
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr) {
                h = fnv1a_u64(h, 0);
                continue;
            }
            h = fnv1a_u64(h, static_cast<uint64_t>(src->op));
            h = fnv1a(h, src->name, strlen(src->name));
            h = fnv1a_u64(h, static_cast<uint64_t>(src->type));
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                h = fnv1a_u64(h, static_cast<uint64_t>(src->ne[d]));
            }
        }
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](const ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Config that changes the produced blob.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    if (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, sizeof(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    const std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    return h;
}

std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".manifest";
}

bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & input_names,
                                              const std::vector<std::string> & output_names) {
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        return false;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    // one name per line: ggml names can contain spaces (e.g. "cache_k_l0 (view)")
    f << "inputs " << input_names.size() << "\n";
    for (const auto & n : input_names) {
        f << n << "\n";
    }
    f << "outputs " << output_names.size() << "\n";
    for (const auto & n : output_names) {
        f << n << "\n";
    }
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<int>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    return f.good();
}

bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & input_names,
                                               std::vector<std::string> & output_names) {
    input_names.clear();
    output_names.clear();

    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::string tag, val, line;
    // header: fingerprint
    if (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        return false;
    }
    // header: ov_version
    if (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        return false;
    }

    // port names, one per line, each section prefixed by its count
    const auto read_names = [&](const char * section, std::vector<std::string> & out) -> bool {
        size_t n = 0;
        if (!(f >> tag >> n) || tag != section) {
            return false;
        }
        std::getline(f, line);  // consume rest of the count line
        for (size_t i = 0; i < n; ++i) {
            if (!std::getline(f, line)) {
                return false;
            }
            out.push_back(line);
        }
        return true;
    };
    if (!read_names("inputs", input_names) || !read_names("outputs", output_names)) {
        return false;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<int>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        if (idx >= expected.size() || line != expected[idx]) {
            return false;
        }
        ++idx;
    }
    return idx == expected.size();
}
