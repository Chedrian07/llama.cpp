// llama-kv-dump: extract K/V cache contents after a single decode pass
// for offline analysis (TurboQuant quantization error studies, etc.)
//
// Usage:
//   llama-kv-dump -m model.gguf [--mmproj mmproj.gguf]
//                 --cache-type-k f16 --cache-type-v f16
//                 --prompt "..." [--image file.jpg] [--image ...]
//                 --output-dir results/kv_dumps/run1
//                 -ngl 99 -fa on
//
// Outputs:
//   <output-dir>/K_layer_<L>.bin    float32 [n_tokens, n_head_kv, head_dim]
//   <output-dir>/V_layer_<L>.bin    float32 [n_tokens, n_head_kv, head_dim]
//   <output-dir>/meta.json          metadata + per-token vision/text mask

#include "arg.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#endif

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "KV cache dump tool for offline quantization analysis.\n\n"
        "Usage: %s [options] -m <model> [--mmproj <mmproj>] [--image <img>] -p <prompt> --output-dir <dir>\n\n"
        "  -m is required; --mmproj and --image are required together for VLMs\n"
        "  After a single forward pass, K and V tensors for every layer are dumped\n"
        "  as float32 binary blobs, and a meta.json describes the layout, vision\n"
        "  token mask, and runtime configuration.\n"
        "\n"
        "  Use --cache-type-k / --cache-type-v to exercise quantized KV storage.\n",
        argv[0]
    );
}

// ---------------------------------------------------------------------------
// JSON helpers (intentionally minimal; we write a very restricted subset)
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string timestamp_iso8601() {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

static std::string hostname_string() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    char buf[256] = {0};
    if (gethostname(buf, sizeof(buf) - 1) == 0) {
        return std::string(buf);
    }
#endif
    return "";
}

// ---------------------------------------------------------------------------
// Per-token vision/text mask tracking
// ---------------------------------------------------------------------------
//
// We track which token positions came from image chunks (as opposed to text
// chunks) by walking the mtmd_input_chunks list. Each chunk reports its token
// count via mtmd_input_chunk_get_n_tokens, and its type via
// mtmd_input_chunk_get_type.
//
// Note: for M-RoPE models like Qwen3-VL the chunk's n_pos may differ from
// n_tokens, but the KV cache stores one row per token regardless of the pos
// layout, so we use n_tokens here.

struct chunk_token_span {
    size_t      start;    // inclusive
    size_t      n_tokens;
    bool        is_vision;
    std::string type;     // "text", "image", "audio"
};

static const char * chunk_type_name(mtmd_input_chunk_type type) {
    switch (type) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  return "text";
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: return "image";
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: return "audio";
    }
    return "unknown";
}

static std::vector<bool> build_vision_mask_from_chunks(
        const mtmd_input_chunks * chunks,
        size_t                    n_prefix_tokens,
        size_t                    n_total_tokens,
        std::vector<chunk_token_span> * out_spans = nullptr) {
    std::vector<bool> mask(n_total_tokens, false);
    size_t cursor = n_prefix_tokens;
    const size_t n_chunks = chunks ? mtmd_input_chunks_size(chunks) : 0;

    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        // We treat image patches as "vision"; audio is tracked as a
        // separate span but does not toggle the vision mask.
        const bool is_vision = (type == MTMD_INPUT_CHUNK_TYPE_IMAGE);

        if (out_spans) {
            out_spans->push_back({cursor, n_tokens, is_vision, chunk_type_name(type)});
        }

        if (is_vision) {
            const size_t end = std::min(cursor + n_tokens, n_total_tokens);
            for (size_t j = cursor; j < end; ++j) {
                mask[j] = true;
            }
        }
        cursor += n_tokens;
        if (cursor >= n_total_tokens) {
            break;
        }
    }

    return mask;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;
    params.sampling.temp = 0.0f;
    params.n_predict     = 0; // no generation, we only want the prompt KV cache
    params.warmup        = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    if (params.model.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: --model is required\n");
        return 1;
    }

    // Output directory: we reuse --output-file as the output directory.
    // If empty, fall back to CWD.
    std::string output_dir = params.out_file;
    if (output_dir.empty()) {
        output_dir = "kv_dump_out";
    }
    {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            LOG_ERR("ERR: failed to create output dir '%s': %s\n",
                    output_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    LOG_INF("kv-dump: loading model: %s\n", params.model.path.c_str());
    LOG_INF("kv-dump: output dir:    %s\n", output_dir.c_str());

    common_init_result_ptr llama_init_res = common_init_from_params(params);
    if (!llama_init_res || !llama_init_res->model() || !llama_init_res->context()) {
        LOG_ERR("ERR: failed to load model / create context\n");
        return 1;
    }

    llama_model   * model = llama_init_res->model();
    llama_context * lctx  = llama_init_res->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int32_t n_layer   = llama_model_n_layer(model);
    const int32_t n_head    = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    const int32_t n_embd    = llama_model_n_embd(model);

    if (n_head <= 0 || n_head_kv <= 0 || n_embd <= 0) {
        LOG_ERR("ERR: unexpected model shape (n_head=%d n_head_kv=%d n_embd=%d)\n",
                n_head, n_head_kv, n_embd);
        return 1;
    }
    const int32_t head_dim = n_embd / n_head;
    if (head_dim * n_head != n_embd) {
        LOG_ERR("ERR: n_embd (%d) is not divisible by n_head (%d); "
                "head_dim inference assumes dense attention.\n", n_embd, n_head);
        return 1;
    }

    // Vision model context (only if --mmproj was supplied)
    mtmd::context_ptr ctx_vision;
    const bool use_vision = !params.mmproj.path.empty();
    if (use_vision) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = false;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = false;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;

        ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("ERR: failed to load vision model from '%s'\n", params.mmproj.path.c_str());
            return 1;
        }
    }

    // Build the prompt. If we have a vision model, use mtmd_tokenize so that
    // the image chunks get resolved correctly. Otherwise we fall back to plain
    // llama_tokenize + llama_decode.
    size_t n_past = 0;

    std::vector<bool> vision_mask;
    std::vector<chunk_token_span> spans;

    if (use_vision) {
        if (params.prompt.empty() && params.image.empty()) {
            LOG_ERR("ERR: for vision mode, at least one of --prompt / --image is required\n");
            return 1;
        }

        // Load images
        mtmd::bitmaps bitmaps;
        for (const auto & img_path : params.image) {
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
            if (!bmp.ptr) {
                LOG_ERR("ERR: failed to load image '%s'\n", img_path.c_str());
                return 1;
            }
            bitmaps.entries.push_back(std::move(bmp));
        }

        // Make sure the prompt contains the media marker for each image
        std::string prompt_text = params.prompt;
        if (prompt_text.find(mtmd_default_marker()) == std::string::npos) {
            std::string prefix;
            for (size_t i = 0; i < params.image.size(); ++i) {
                prefix += mtmd_default_marker();
            }
            prompt_text = prefix + prompt_text;
        }

        // Wrap the prompt in the model's chat template (user message) if the model has one.
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);
        std::string formatted_prompt;
        if (tmpls) {
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = prompt_text;
            std::vector<common_chat_msg> history; // empty
            formatted_prompt = common_chat_format_single(
                tmpls.get(), history, msg, /*add_ass*/ true, params.use_jinja);
        } else {
            formatted_prompt = prompt_text;
        }

        mtmd_input_text text;
        text.text          = formatted_prompt.c_str();
        text.add_special   = true;
        text.parse_special = true;

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        auto bitmaps_c = bitmaps.c_ptr();
        int32_t res = mtmd_tokenize(ctx_vision.get(),
                                    chunks.ptr.get(),
                                    &text,
                                    bitmaps_c.data(),
                                    bitmaps_c.size());
        if (res != 0) {
            LOG_ERR("ERR: mtmd_tokenize failed with code %d\n", res);
            return 1;
        }

        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(ctx_vision.get(),
                                    lctx,
                                    chunks.ptr.get(),
                                    /*n_past*/ 0,
                                    /*seq_id*/ 0,
                                    /*n_batch*/ params.n_batch,
                                    /*logits_last*/ true,
                                    &new_n_past) != 0) {
            LOG_ERR("ERR: mtmd_helper_eval_chunks failed\n");
            return 1;
        }

        const size_t total_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
        n_past = total_tokens;
        vision_mask = build_vision_mask_from_chunks(chunks.ptr.get(), /*n_prefix*/ 0, total_tokens, &spans);

        LOG_INF("kv-dump: mtmd_helper_eval_chunks done, n_past=%zu (llama pos %d)\n",
                n_past, (int) new_n_past);
    } else {
        // Plain text mode
        if (params.prompt.empty()) {
            LOG_ERR("ERR: --prompt is required in text-only mode\n");
            return 1;
        }

        std::vector<llama_token> tokens = common_tokenize(
            vocab, params.prompt, /*add_special*/ true, /*parse_special*/ true);
        if (tokens.empty()) {
            LOG_ERR("ERR: empty tokenization\n");
            return 1;
        }
        if ((int32_t) tokens.size() > params.n_batch) {
            // We decode in chunks of n_batch
            for (size_t i = 0; i < tokens.size(); i += params.n_batch) {
                size_t chunk = std::min<size_t>(params.n_batch, tokens.size() - i);
                llama_batch batch = llama_batch_get_one(tokens.data() + i, (int32_t) chunk);
                if (llama_decode(lctx, batch)) {
                    LOG_ERR("ERR: llama_decode failed at offset %zu\n", i);
                    return 1;
                }
            }
        } else {
            llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
            if (llama_decode(lctx, batch)) {
                LOG_ERR("ERR: llama_decode failed\n");
                return 1;
            }
        }
        n_past = tokens.size();
        vision_mask.assign(n_past, false); // no vision tokens
    }

    if (n_past == 0) {
        LOG_ERR("ERR: no tokens were decoded\n");
        return 1;
    }

    LOG_INF("kv-dump: decoded %zu tokens, beginning KV extraction\n", n_past);

    // Sanity-check what the context actually holds
    llama_memory_t mem = llama_get_memory(lctx);
    const llama_pos pos_max = llama_memory_seq_pos_max(mem, /*seq_id*/ 0);
    if (pos_max < 0 || (size_t) (pos_max + 1) != n_past) {
        LOG_WRN("kv-dump: llama_memory_seq_pos_max reports %d, n_past=%zu; "
                "this may be expected for M-RoPE models.\n", (int) pos_max, n_past);
    }

    // Query the KV cache shape via the debug API
    const int32_t n_kv_layers = llama_kv_self_dbg_n_layers(lctx);
    if (n_kv_layers <= 0) {
        LOG_ERR("ERR: llama_kv_self_dbg_n_layers returned %d. "
                "The context may be using an unsupported memory type (iSWA / hybrid / recurrent).\n",
                n_kv_layers);
        return 1;
    }
    const bool    v_trans     = llama_kv_self_dbg_v_trans(lctx);
    const bool    attn_rot_k  = llama_kv_self_dbg_attn_rot_k(lctx);
    const bool    attn_rot_v  = llama_kv_self_dbg_attn_rot_v(lctx);
    const int32_t k_rot_dim   = llama_kv_self_dbg_k_rot_dim(lctx);
    const int32_t v_rot_dim   = llama_kv_self_dbg_v_rot_dim(lctx);

    if (attn_rot_k || attn_rot_v) {
        LOG_WRN("kv-dump: NOTE: the KV cache applies a Hadamard rotation before storage\n");
        LOG_WRN("kv-dump:       (attn_rot_k=%d, attn_rot_v=%d, k_rot_dim=%d, v_rot_dim=%d).\n",
                (int) attn_rot_k, (int) attn_rot_v, k_rot_dim, v_rot_dim);
        LOG_WRN("kv-dump:       The dumped K/V values are in the rotated basis.\n");
        LOG_WRN("kv-dump:       Set LLAMA_ATTN_ROT_DISABLE=1 to disable this rotation and\n");
        LOG_WRN("kv-dump:       dump K/V values in the model's native basis.\n");
    }

    const size_t n_embd_gqa = (size_t) n_head_kv * (size_t) head_dim;
    const size_t n_floats_per_layer = n_past * n_embd_gqa;
    std::vector<float> kv_buf(n_floats_per_layer);

    LOG_INF("kv-dump: n_layer=%d n_kv_layers=%d n_head_kv=%d head_dim=%d n_tokens=%zu v_trans=%d\n",
            n_layer, n_kv_layers, n_head_kv, head_dim, n_past, (int) v_trans);

    // Dump each KV cache layer
    std::vector<int32_t> dumped_model_layers;
    dumped_model_layers.reserve(n_kv_layers);

    for (int32_t ikv = 0; ikv < n_kv_layers; ++ikv) {
        const int32_t il = llama_kv_self_dbg_layer_id(lctx, ikv);
        if (il < 0) {
            LOG_ERR("ERR: llama_kv_self_dbg_layer_id(%d) returned %d\n", ikv, il);
            return 1;
        }
        dumped_model_layers.push_back(il);

        // K
        {
            const size_t wrote = llama_kv_self_dbg_dump_k(
                lctx, il, (uint32_t) n_past, /*stream_id*/ 0,
                kv_buf.data(), kv_buf.size());
            if (wrote != n_floats_per_layer) {
                LOG_ERR("ERR: llama_kv_self_dbg_dump_k(layer=%d) wrote %zu / %zu floats\n",
                        il, wrote, n_floats_per_layer);
                return 1;
            }
            char name[64];
            std::snprintf(name, sizeof(name), "K_layer_%d.bin", il);
            std::filesystem::path p = std::filesystem::path(output_dir) / name;
            std::ofstream fout(p, std::ios::binary);
            if (!fout) {
                LOG_ERR("ERR: failed to open '%s' for writing\n", p.string().c_str());
                return 1;
            }
            fout.write(reinterpret_cast<const char *>(kv_buf.data()),
                       (std::streamsize) (n_floats_per_layer * sizeof(float)));
        }

        // V
        {
            const size_t wrote = llama_kv_self_dbg_dump_v(
                lctx, il, (uint32_t) n_past, /*stream_id*/ 0,
                kv_buf.data(), kv_buf.size());
            if (wrote != n_floats_per_layer) {
                LOG_ERR("ERR: llama_kv_self_dbg_dump_v(layer=%d) wrote %zu / %zu floats\n",
                        il, wrote, n_floats_per_layer);
                return 1;
            }
            char name[64];
            std::snprintf(name, sizeof(name), "V_layer_%d.bin", il);
            std::filesystem::path p = std::filesystem::path(output_dir) / name;
            std::ofstream fout(p, std::ios::binary);
            if (!fout) {
                LOG_ERR("ERR: failed to open '%s' for writing\n", p.string().c_str());
                return 1;
            }
            fout.write(reinterpret_cast<const char *>(kv_buf.data()),
                       (std::streamsize) (n_floats_per_layer * sizeof(float)));
        }

        if ((ikv + 1) % 4 == 0 || ikv + 1 == n_kv_layers) {
            LOG_INF("kv-dump:   wrote layer %d / %d\n", ikv + 1, n_kv_layers);
        }
    }

    // meta.json
    {
        std::filesystem::path meta_path = std::filesystem::path(output_dir) / "meta.json";
        std::ofstream meta(meta_path);
        if (!meta) {
            LOG_ERR("ERR: failed to open '%s' for writing\n", meta_path.string().c_str());
            return 1;
        }

        auto cache_type_k_str = ggml_type_name(params.cache_type_k);
        auto cache_type_v_str = ggml_type_name(params.cache_type_v);

        meta << "{\n";
        meta << "  \"schema_version\": 1,\n";
        meta << "  \"tool\": \"llama-kv-dump\",\n";
        meta << "  \"timestamp\": \"" << json_escape(timestamp_iso8601()) << "\",\n";
        meta << "  \"hostname\": \"" << json_escape(hostname_string()) << "\",\n";
        meta << "  \"n_tokens\": " << n_past << ",\n";
        meta << "  \"n_layers\": " << dumped_model_layers.size() << ",\n";
        meta << "  \"n_model_layers\": " << n_layer << ",\n";
        meta << "  \"n_kv_head\": " << n_head_kv << ",\n";
        meta << "  \"n_head\": " << n_head << ",\n";
        meta << "  \"n_embd\": " << n_embd << ",\n";
        meta << "  \"head_dim\": " << head_dim << ",\n";
        meta << "  \"dtype\": \"float32\",\n";
        meta << "  \"row_major_shape\": [\"n_tokens\", \"n_kv_head\", \"head_dim\"],\n";
        meta << "  \"v_trans_in_cache\": " << (v_trans ? "true" : "false") << ",\n";
        meta << "  \"attn_rot_k\": " << (attn_rot_k ? "true" : "false") << ",\n";
        meta << "  \"attn_rot_v\": " << (attn_rot_v ? "true" : "false") << ",\n";
        meta << "  \"k_rot_dim\": " << k_rot_dim << ",\n";
        meta << "  \"v_rot_dim\": " << v_rot_dim << ",\n";
        meta << "  \"cache_type_k\": \"" << json_escape(cache_type_k_str ? cache_type_k_str : "?") << "\",\n";
        meta << "  \"cache_type_v\": \"" << json_escape(cache_type_v_str ? cache_type_v_str : "?") << "\",\n";
        const char * fa_str =
            params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED  ? "on"   :
            params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "off"  :
                                                                       "auto";
        meta << "  \"flash_attn\": \"" << fa_str << "\",\n";
        meta << "  \"model_path\": \"" << json_escape(params.model.path) << "\",\n";
        meta << "  \"mmproj_path\": \"" << json_escape(params.mmproj.path) << "\",\n";
        meta << "  \"prompt\": \"" << json_escape(params.prompt) << "\",\n";

        meta << "  \"images\": [";
        for (size_t i = 0; i < params.image.size(); ++i) {
            if (i) meta << ", ";
            meta << "\"" << json_escape(params.image[i]) << "\"";
        }
        meta << "],\n";

        meta << "  \"dumped_model_layers\": [";
        for (size_t i = 0; i < dumped_model_layers.size(); ++i) {
            if (i) meta << ", ";
            meta << dumped_model_layers[i];
        }
        meta << "],\n";

        meta << "  \"chunk_spans\": [\n";
        for (size_t i = 0; i < spans.size(); ++i) {
            const auto & s = spans[i];
            meta << "    {\"start\": " << s.start
                 << ", \"n_tokens\": " << s.n_tokens
                 << ", \"type\": \"" << json_escape(s.type) << "\""
                 << ", \"is_vision\": " << (s.is_vision ? "true" : "false") << "}";
            if (i + 1 < spans.size()) meta << ",";
            meta << "\n";
        }
        meta << "  ],\n";

        meta << "  \"vision_token_mask\": [";
        for (size_t i = 0; i < vision_mask.size(); ++i) {
            if (i) meta << ", ";
            meta << (vision_mask[i] ? "true" : "false");
        }
        meta << "]\n";
        meta << "}\n";
    }

    // Summary
    const size_t total_bytes_per_layer = n_past * n_embd_gqa * sizeof(float);
    LOG_INF("kv-dump: done.\n");
    LOG_INF("kv-dump: wrote %d K/V layer pairs to %s\n",
            (int) dumped_model_layers.size(), output_dir.c_str());
    LOG_INF("kv-dump: per-layer size: %zu bytes (%.2f KiB)\n",
            total_bytes_per_layer, (double) total_bytes_per_layer / 1024.0);
    LOG_INF("kv-dump: total KV bytes: %.2f MiB\n",
            (double) total_bytes_per_layer * 2 * dumped_model_layers.size() / (1024.0 * 1024.0));

    return 0;
}
