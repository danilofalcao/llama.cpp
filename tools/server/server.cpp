#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>
#include <chrono>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

// [multi] every instance registers its handler; a signal stops all of them
static std::vector<std::function<void(int)>> shutdown_handlers;
static std::mutex shutdown_handlers_mutex;
struct shutdown_handler_t {
    void operator=(std::function<void(int)> f) { std::lock_guard<std::mutex> lk(shutdown_handlers_mutex); shutdown_handlers.push_back(std::move(f)); }
    explicit operator bool() const { std::lock_guard<std::mutex> lk(shutdown_handlers_mutex); return !shutdown_handlers.empty(); }
    void operator()(int sig) const {
        std::vector<std::function<void(int)>> hs;
        { std::lock_guard<std::mutex> lk(shutdown_handlers_mutex); hs = shutdown_handlers; }
        for (auto & h : hs) { h(sig); }
    }
};
static shutdown_handler_t shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;
// [multi] llama_backend_free only when the last instance leaves
static std::atomic<int> live_instances{0};
static void backend_free_last() { if (--live_instances <= 0) { llama_backend_free(); } }

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    SRV_INF("%s", "initializing ...\n");

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    const int result = llama_server(params, argc, argv);
    common_log_flush(common_log_main());
    return result;
}

static int llama_server_multi(common_params & params, int argc, char ** argv, int n_inst);

int llama_server(common_params & params, int argc, char ** argv) {
    bool is_run_by_cli = (argv == nullptr);

    // [multi] LLAMA_MULTI=N: N instances (threads) sharing the model + a proxy on the base port
    {
        static const int n_multi = [] { const char * e = getenv("LLAMA_MULTI"); return e ? atoi(e) : 0; }();
        static std::atomic<bool> multi_entered{false};
        if (n_multi > 1 && !is_run_by_cli && !multi_entered.exchange(true)) {
            return llama_server_multi(params, argc, argv, n_multi);
        }
    }
    live_instances++;

    common_models_handler models_handler;

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty()
                               && params.model.docker_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized = params.kv_unified_per_slot > 0 &&
                                     params.n_ctx == 0 &&
                                     (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", params.n_parallel,
                params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = server_stream_make_get_handler();
        streams_lookup_h = server_stream_make_lookup_handler();
        stream_delete_h  = server_stream_make_delete_handler();
    }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get_h));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    std::vector<std::string> warn_names;
    if (is_router_server) {
        warn_names.push_back("router mode");
    }

    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return 1;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (!warn_names.empty()) {
        std::string features;
        for (const auto & name : warn_names) {
            if (!features.empty()) features += ", ";
            features += name;
        }
        SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", features.c_str());
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server && !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            mcp_mgr.shutdown();
            backend_free_last();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            mcp_mgr.shutdown();
            ctx_http.stop();
        };

        try {
            models_routes->models.load_startup_models();
        } catch (const std::exception & e) {
            SRV_ERR("failed to load models on startup: %s\n", e.what());
            ctx_http.stop();
            ctx_http.join();
            clean_up();
            return 1;
        }

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            ctx_http.stop();
            ctx_server.terminate();
            mcp_mgr.shutdown();
            backend_free_last();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            ctx_http.join();
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            mcp_mgr.shutdown();
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // register signal handler if not running by CLI
    if (!is_run_by_cli) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    bool uses_default_port = false;
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
        uses_default_port |= string_ends_with(address, ":8080");
    }

    // TODO: remove this in the future
    // check the string to also handle the .sock case
    if (uses_default_port) {
        SRV_WRN("%s", "notice: server default port will be changed to :9931 in a future release (ref: https://github.com/ggml-org/llama.cpp/pull/26508)\n");
    }

    if (is_router_server) {
        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        ctx_http.join(); // keep the main thread alive

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        ctx_http.join();
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}


// ============================ [multi] N instances + proxy ============================
#include "server-models.h"

static int llama_server_multi(common_params & params, int argc, char ** argv, int n_inst) {
    const int base_port = params.port;
    struct inst_t { int port; int cap = 1; std::atomic<int> busy{0}; std::thread th; int rc = 0;
                     std::atomic<size_t> held{0}; }; // held: body size of the last conversation routed here (~context it keeps)
    std::vector<std::unique_ptr<inst_t>> insts;

    auto wait_ready = [&](int port, int timeout_s) {
        httplib::Client cli("127.0.0.1", port);
        cli.set_connection_timeout(2); cli.set_read_timeout(5);
        for (int i = 0; i < timeout_s; ++i) {
            auto r = cli.Get("/health");
            if (r && r->status == 200) { return true; }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return false;
    };

    for (int i = 0; i < n_inst; ++i) {
        auto inst = std::make_unique<inst_t>();
        inst->port = base_port + 1 + i;
        common_params p = params;
        p.port = inst->port;
        inst->cap = std::max(1, p.n_parallel);
        if (i > 0) {   // extra instances: smaller context / fewer slots (LLAMA_MULTI_CTX, LLAMA_MULTI_NP)
            static const int mctx = [] { const char * e = getenv("LLAMA_MULTI_CTX"); return e ? atoi(e) : 0; }();
            static const int mnp  = [] { const char * e = getenv("LLAMA_MULTI_NP");  return e ? atoi(e) : 0; }();
            static const int mnmax = [] { const char * e = getenv("LLAMA_MULTI_DRAFT_NMAX"); return e ? atoi(e) : 0; }();
            static const int mub   = [] { const char * e = getenv("LLAMA_MULTI_UB"); return e ? atoi(e) : 0; }();
            if (mctx > 0) { p.n_ctx = mctx; if (p.kv_unified_per_slot > mctx) { p.kv_unified_per_slot = mctx; } }
            if (mnp  > 0) { p.n_parallel = mnp; inst->cap = mnp; }
            if (mnmax > 0 && p.speculative.draft.n_max > mnmax) { p.speculative.draft.n_max = mnmax; }
            if (mub > 0 && p.n_ubatch > mub) { p.n_ubatch = mub; if (p.n_batch > mub * 4) { p.n_batch = mub * 4; } }
        }
        inst_t * ip = inst.get();
        inst->th = std::thread([ip, p, argc, argv]() mutable {
            ip->rc = llama_server(p, argc, argv);
        });
        SRV_INF("[multi] instance %d starting on port %d\n", i, inst->port);
        // instances load one after the other (the first loads the weights, the rest reuse them)
        if (!wait_ready(inst->port, 1800)) {
            SRV_ERR("[multi] instance %d on port %d did not become ready\n", i, inst->port);
            return 1;
        }
        SRV_INF("[multi] instance %d ready on port %d\n", i, inst->port);
        insts.push_back(std::move(inst));
    }

    // proxy on the base port
    common_params pp = params;
    pp.port = base_port;
    pp.api_keys.clear();         // the instances check the key themselves
    server_http_context ctx_http;
    if (!ctx_http.init(pp)) {
        SRV_ERR("%s", "[multi] failed to initialize the proxy HTTP server\n");
        return 1;
    }
    // LLAMA_MULTI_ROUTE: first (first instance with a free slot), least (least loaded), hash (default: conversation
    // affinity = hash of the first user message -> instance, so a session keeps its prompt cache; spill when full)
    static const std::string route = [] { const char * e = getenv("LLAMA_MULTI_ROUTE"); return std::string(e ? e : "sticky"); }();
    auto least_loaded = [&]() -> inst_t * {
        inst_t * best = nullptr; double best_load = 1e9;
        for (auto & in : insts) {
            const double load = (double) in->busy.load() / in->cap;
            if (load < best_load) { best_load = load; best = in.get(); }
        }
        return best;
    };
    // conversation key = first user message (stable across the turns of one session)
    auto conv_key = [](const std::string & body) -> std::string {
        try {
            auto j = json::parse(body);
            if (j.contains("messages")) {
                for (const auto & m : j["messages"]) {
                    if (m.value("role", "") == "user") { return m.value("content", json()).dump(); }
                }
            }
        } catch (...) {}
        return "";
    };
    // sticky: a conversation goes back to the instance that served it (its prompt cache lives there) if that
    // instance has a free slot; otherwise first free slot; the pin follows the request
    static std::mutex pins_mutex;
    static std::unordered_map<std::string, inst_t *> pins;
    static std::deque<std::string> pins_order;
    auto pick = [&](const std::string & body) -> inst_t * {
        if (route == "first" || route == "sticky") {
            inst_t * chosen = nullptr;
            bool pinned = false;
            const std::string key = route == "sticky" ? conv_key(body) : "";
            if (!key.empty()) {
                std::lock_guard<std::mutex> lk(pins_mutex);
                auto it = pins.find(key);
                if (it != pins.end() && it->second->busy.load() < it->second->cap) { chosen = it->second; pinned = true; }
            }
            if (!chosen) {
                // PATCH(route-least-held): a conversation without a usable pin goes to the free instance that holds the
                // least context (ties -> lowest index), so a new short session does not wipe a long one kept on A
                // while B sits idle (2026-10-03: a 6K session on A dropped a 113K conversation -> 2 min re-prefill).
                for (auto & in : insts) {
                    if (in->busy.load() >= in->cap) { continue; }
                    if (route == "first" || !chosen || in->held.load() < chosen->held.load()) { chosen = in.get(); }
                    if (route == "first") { break; }
                }
            }
            if (!chosen) { chosen = least_loaded(); }
            if (!key.empty()) {
                std::string held_s;
                for (auto & in : insts) { held_s += " " + std::to_string(in->port) + "=" + std::to_string(in->held.load() / 1024) + "K"; }
                SRV_INF("[multi] route -> %d (%s), held before:%s\n", chosen->port, pinned ? "pin" : "least-held", held_s.c_str());
                chosen->held.store(body.size());
                std::lock_guard<std::mutex> lk(pins_mutex);
                if (pins.find(key) == pins.end()) { pins_order.push_back(key); }
                pins[key] = chosen;
                while (pins_order.size() > 2000) { pins.erase(pins_order.front()); pins_order.pop_front(); }
            }
            return chosen;
        }
        if (route == "hash") {
            std::string key;
            try {
                auto j = json::parse(body);
                if (j.contains("messages")) {
                    for (const auto & m : j["messages"]) {
                        if (m.value("role", "") == "user") { key = m.value("content", json()).dump(); break; }
                    }
                }
            } catch (...) {}
            if (!key.empty()) {
                inst_t * in = insts[std::hash<std::string>{}(key) % insts.size()].get();
                if (in->busy.load() < in->cap) { return in; }
                for (auto & o : insts) { if (o->busy.load() < o->cap) { return o.get(); } }
            }
        }
        return least_loaded();
    };
    auto forward = [&](const std::string & method) {
        return [&, method](const server_http_req & req) -> server_http_res_ptr {
            inst_t * in = pick(req.body);
            in->busy++;
            std::string path = req.path;
            if (!req.query_string.empty()) { path += '?' + req.query_string; }
            auto proxy = std::make_unique<server_http_proxy>(method, "http", "127.0.0.1", in->port, path,
                    req.headers, req.body, req.files, req.should_stop, params.timeout_read, params.timeout_write);
            proxy->cleanup = [in]() { in->busy--; };
            return proxy;
        };
    };
    const char * gets[]  = { "/health", "/metrics", "/props", "/models", "/v1/models", "/api/tags", "/slots", "/api/version" };
    const char * posts[] = { "/completion", "/completions", "/v1/completions", "/chat/completions", "/v1/chat/completions",
                             "/v1/messages", "/v1/responses", "/infill", "/embedding", "/embeddings", "/v1/embeddings",
                             "/rerank", "/reranking", "/v1/rerank", "/v1/reranking", "/tokenize", "/detokenize",
                             "/apply-template", "/props", "/api/show", "/slots/:id_slot" };
    for (auto p : gets)  { ctx_http.get(p, forward("GET")); }
    for (auto p : posts) { ctx_http.post(p, forward("POST")); }

    if (!ctx_http.start()) {
        SRV_ERR("%s", "[multi] proxy HTTP server failed to start\n");
        return 1;
    }
    ctx_http.is_ready.store(true);
    SRV_INF("[multi] proxy listening on port %d, %d instances on %d..%d\n", base_port, n_inst, base_port + 1, base_port + n_inst);

    shutdown_handler = [&](int) { ctx_http.stop(); };

    ctx_http.join();
    for (auto & in : insts) { if (in->th.joinable()) { in->th.join(); } }
    return 0;
}
