#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <vector>

// Return zero unless the Qwen35 padded head and row map are safe for compact selection.
LLAMA_API int64_t llama_mtp_compact_rows(const llama_model * model, const std::vector<llama_token> & rows);

static inline bool llama_mtp_validate_rows(const std::vector<llama_token> & rows, const std::vector<int32_t> & inverse, int64_t n_head) {
    if (n_head <= 0 || rows.size() != (size_t) n_head || rows.size() >= inverse.size()) {
        return false;
    }
    std::vector<int32_t> expected(inverse.size(), (int32_t) n_head);
    for (int64_t i = 0; i < n_head; ++i) {
        const auto id = rows[i];
        if (id < 0 || (size_t) id >= inverse.size() || expected[id] != n_head) {
            return false;
        }
        expected[id] = i;
    }
    return inverse == expected;
}

struct llama_mtp_sampler_context {
    int64_t n_rows;
    bool supported = false;
};

static inline void llama_mtp_sampler_backend_apply(llama_sampler * smpl, ggml_context * ctx, ggml_cgraph *, llama_sampler_data * data) {
    auto * s = (llama_mtp_sampler_context *) smpl->ctx;
    // The chain's node-count probe uses a synthetic, possibly smaller vocabulary.
    auto * logits = ggml_view_1d(ctx, data->logits, std::min(s->n_rows, ggml_nelements(data->logits)), 0);
    data->sampled = ggml_argmax(ctx, logits);
    ggml_set_name(data->sampled, "mtp_compact_argmax");
    data->logits = nullptr;
    data->probs = nullptr;
    data->candidates = nullptr;
}

static inline bool llama_mtp_sampler_backend_init(llama_sampler * smpl, ggml_backend_buffer_type_t buft, uint32_t) {
    auto * s = (llama_mtp_sampler_context *) smpl->ctx;
    auto * dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        // CPU buffers can have no associated device (same convention as the core sampler).
        return s->supported = true;
    }
    ggml_init_params params = { 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true };
    auto * ctx = ggml_init(params);
    if (!ctx) {
        return s->supported = false;
    }
    auto * gf = ggml_new_graph_custom(ctx, 8, false);
    llama_sampler_data data = { ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s->n_rows), nullptr, nullptr, nullptr };
    llama_mtp_sampler_backend_apply(smpl, ctx, gf, &data);
    ggml_build_forward_expand(gf, data.sampled);
    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        supported = supported && ggml_backend_dev_supports_op(dev, ggml_graph_node(gf, i));
    }
    ggml_free(ctx);
    return s->supported = supported;
}

static inline void llama_mtp_sampler_apply(llama_sampler * smpl, llama_token_data_array * data) {
    auto * s = (llama_mtp_sampler_context *) smpl->ctx;
    data->selected = -1;
    for (size_t i = 0; i < data->size; ++i) {
        if (data->data[i].id >= 0 && data->data[i].id < s->n_rows &&
                (data->selected < 0 || data->data[i].logit > data->data[data->selected].logit)) {
            data->selected = i;
        }
    }
}

static inline llama_sampler * llama_mtp_sampler_init(int64_t n_rows);

static inline llama_sampler * llama_mtp_sampler_clone(const llama_sampler * smpl) {
    return llama_mtp_sampler_init(((const llama_mtp_sampler_context *) smpl->ctx)->n_rows);
}

static inline void llama_mtp_sampler_free(llama_sampler * smpl) {
    delete (llama_mtp_sampler_context *) smpl->ctx;
}

static inline void llama_mtp_sampler_copy(const llama_sampler * src, llama_sampler * dst) {
    GGML_ASSERT(((const llama_mtp_sampler_context *) src->ctx)->n_rows == ((const llama_mtp_sampler_context *) dst->ctx)->n_rows);
}

static inline const char * llama_mtp_sampler_name(const llama_sampler *) {
    return "mtp-compact";
}

static inline llama_sampler * llama_mtp_sampler_init(int64_t n_rows) {
    GGML_ASSERT(n_rows > 0);
    static llama_sampler_i iface = {
        /* .name              = */ llama_mtp_sampler_name,
        /* .accept            = */ nullptr,
        /* .apply             = */ llama_mtp_sampler_apply,
        /* .reset             = */ nullptr,
        /* .clone             = */ llama_mtp_sampler_clone,
        /* .free              = */ llama_mtp_sampler_free,
        /* .backend_init      = */ llama_mtp_sampler_backend_init,
        /* .backend_accept    = */ nullptr,
        /* .backend_apply     = */ llama_mtp_sampler_backend_apply,
        /* .backend_set_input = */ nullptr,
        /* .backend_reset     = */ nullptr,
        /* .copy_state        = */ llama_mtp_sampler_copy,
    };
    return llama_sampler_init(&iface, new llama_mtp_sampler_context { n_rows, false });
}
