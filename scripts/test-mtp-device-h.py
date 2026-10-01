#!/usr/bin/env python3
"""Source-contract checks only; not a substitute for compiled backend tests."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text()


class DeviceHiddenContract(unittest.TestCase):
    def test_opt_in_and_scope(self):
        s = source("common/speculative.cpp")
        self.assertIn('std::strcmp(device_h_env, "1") == 0', s)
        self.assertIn('n_seq == 1 && !chain_heads && !is_mem_shared', s)
        self.assertIn('~device_h_scope()', s)
        self.assertIn('llama_set_mtp_device_h(params.ctx_dft, false)', s)
        self.assertIn('resident_h ? nullptr : llama_get_embeddings_nextn_ith', s)
        self.assertIn('if (!resident_h)', s)

    def test_owned_snapshot(self):
        h = source("src/llama-graph.h")
        s = source("src/llama-graph.cpp")
        self.assertIn('ggml_context_ptr ctx;', h)
        self.assertIn('ggml_backend_buffer_ptr buffer;', h)
        self.assertIn('row = ggml_dup_tensor(ctx.get(), src)', s)
        self.assertNotIn('row = src;', s)
        self.assertIn('ggml_backend_buffer_is_host(src->buffer)', s)
        self.assertIn('ggml_nrows(src) != 1', s)
        self.assertLess(s.index('ggml_backend_sched_synchronize(sched)'),
                        s.index('ggml_backend_tensor_copy(src, row)'))
        self.assertIn('ggml_backend_tensor_copy(row, dst);\n    ready = false;', s)

    def test_context_isolation_and_restore(self):
        s = source("src/llama-context.cpp")
        self.assertIn('cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP', s)
        self.assertIn('cparams.n_seq_max == 1', s)
        self.assertIn('model.hparams.n_layer_nextn == 1', s)
        self.assertIn('/*.device_h    =*/ &device_h', s)
        self.assertIn('gf_res_prev_active = nullptr;', s)
        for method in ('state_read_data(llama_io_read_i & io)',
                       'state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags)'):
            self.assertIn(method + ' {\n    set_mtp_device_h(false);', s)
        for method in ('get_embeddings_nextn()', 'get_embeddings_nextn_ith(int32_t i)'):
            self.assertIn(method + ' {\n    device_h.materialize(embd_nextn.data);', s)

    def test_device_input_placement(self):
        for model in ('qwen35', 'qwen35moe'):
            s = source('src/models/' + model + '.cpp')
            self.assertIn('inp->device_h = params.device_h;', s)
            self.assertIn('if (params.device_h && params.device_h->configured)', s)
            self.assertIn('cb(inp->h, "mtp_h_input", il)', s)
        s = source('src/llama-context.cpp')
        self.assertIn('device_h.configured && il >= 0 && strcmp(name, "mtp_h_input") == 0', s)


if __name__ == '__main__':
    unittest.main()
