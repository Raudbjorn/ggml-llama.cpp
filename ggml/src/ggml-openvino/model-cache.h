#pragma once

// Frontend-level compiled-model cache (GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR).
//
// The OpenVINO plugin's own ov::cache_dir caches the compiled blob keyed by the
// *OV model*, but producing that model still runs the full frontend every time:
// weight requantization (incl. the large token_embd F32 transient) and the
// ggml->OV graph conversion. This cache keys off a fingerprint computed directly
// from the ggml cgraph, so a hit skips requant + convert + compile entirely and
// instead imports a previously exported CompiledModel blob.
//
// Opt-in and independent from GGML_OPENVINO_CACHE_DIR. Default off.

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

// Returns the compiled-model cache directory from GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR,
// or empty if unset/disabled. When empty, callers must not use the cache.
std::string ggml_openvino_model_cache_dir();

// Compute a stable 64-bit fingerprint identifying the model+config that a cgraph
// would compile to. Combines graph topology, a sampled hash of every weight
// tensor (name/shape/dtype + bounded byte sample), and the config that changes
// the produced blob (device, flash-attention, rope params, the compile-memory
// flags, stateful, and the OpenVINO version). `device` is the resolved device
// string; `fa` is the flash-attention flag; `rope_params`/`rope_len` cover the
// model's rope configuration; `extra_cfg` folds in any other blob-affecting bits.
uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint);

// Write/read the manifest: a header (fingerprint, OV version), the OV
// Parameter and Result names of the exported model in port order, then one
// "name ne0 ne1 ne2 ne3 type sample_hash" line per weight. Returns false on
// I/O error.
//
// The port names are persisted rather than re-derived from the imported
// CompiledModel: the frontend keys tensors by Parameter/Result friendly name,
// but OpenVINO's serializer renames colliding friendly names on export (a
// stateless KV cache is both a Parameter and a Result called cache_k_lN and
// comes back as cache_k_lN_1), so names read off the imported ports no longer
// match the decoder's keys and the output silently stays unbound. Port *order*
// is what export/import preserves, and binding is positional.
bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & input_names,
                                              const std::vector<std::string> & output_names);

// Verify that the cgraph's weights still match the stored manifest (guards the
// sampled-hash collision risk: a blob is only trusted if every weight's
// name/shape/type/sample-hash matches what was cached) and read back the port
// names. Returns true on match; the name vectors are only meaningful then.
bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & input_names,
                                               std::vector<std::string> & output_names);
