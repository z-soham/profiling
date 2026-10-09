// opprof: per-ggml-op wall-time profile of llama.cpp prompt processing (prefill only).
//
// perf is not installed in the Turin pods and llama.cpp has no per-op timer, so this uses the
// scheduler's eval callback: when the callback asks for every node, ggml_backend_sched computes
// the graph one node at a time (ask=true -> compute -> synchronize -> ask=false). The time
// between the two calls is that node's kernel, including the thread-pool fork/join. The per-node
// barrier already exists inside ggml-cpu, so the extra cost is small; the tool also times the
// same prompt on a second context without the callback and reports both walls.
//
// Placement: ggml_backend_sched gives an op with a weight to the highest-priority backend that
// supports the weight's buffer type and the op (ggml_backend_sched_backend_from_buffer). ZenDNN
// is an ACCEL device, ahead of CPU, so a weight op is labelled "ZenDNN" iff the ZenDNN device
// supports the op and the buffer. Ops without a weight are labelled "CPU" if ZenDNN cannot run
// them and "ZenDNN?" if it could (the scheduler then decides from neighbours; see the
// GGML_SCHED_DEBUG run for those). The stock build has no ZenDNN device: everything is "CPU".
//
// Build (headers of the same source tree as the binaries; pick the build at run time with
// LD_LIBRARY_PATH, both builds export the same libllama/libggml ABI):
//   g++ -O2 -std=c++17 -I<src>/include -I<src>/ggml/include opprof.cpp
//       -L<build>/bin -lllama -lggml -lggml-base -o opprof
// Usage:
//   opprof -m model.gguf -p 8192 -ub 512 [-b 4096] [-t 32] [-fa on|off] [-ctk f16] [-ctv f16]
//          [-r 1] [-w 1] [-lm auto|none|mmap|...] [-f prompt.txt] [-o out.tsv]
//   -f: use this text (tokenized, repeated to n_prompt tokens) instead of llama-bench's random
//       tokens. Real text routes tokens to experts differently from random token ids.
//   -w: warm-up passes over the full prompt before timing (0 = one ubatch). ZenDNNL packs weights
//       per shape on first use, so its group GEMM needs several passes to reach steady state.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct op_stat {
    int64_t calls = 0;
    int64_t us    = 0;
};

struct prof_state {
    bool               on    = false;
    int64_t            t_ask = 0;
    int64_t            sum_us = 0;
    ggml_backend_dev_t zdev  = nullptr;
    std::map<std::string, op_stat> by_key;  // op \t stem \t backend \t wtype \t wbuf \t shape
    std::map<std::string, op_stat> by_op;   // op \t backend
};

// "ffn_moe_gate_up-12 (reshaped)" -> "ffn_moe_gate_up"
static std::string name_stem(const char * name) {
    std::string s(name);
    size_t p = s.find(" (");
    if (p != std::string::npos) {
        s = s.substr(0, p);
    }
    p = s.rfind('-');
    if (p != std::string::npos && p + 1 < s.size() &&
        std::all_of(s.begin() + p + 1, s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        s = s.substr(0, p);
    }
    return s.empty() ? std::string("(unnamed)") : s;
}

static bool is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static const char * placement(const prof_state * p, const ggml_tensor * t) {
    if (p->zdev == nullptr || !ggml_backend_dev_supports_op(p->zdev, t)) {
        return "CPU";
    }
    const ggml_tensor * w = t->src[0];
    if (w != nullptr && w->view_src != nullptr) {
        w = w->view_src;
    }
    if (w != nullptr && w->buffer != nullptr &&
        ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return ggml_backend_dev_supports_buft(p->zdev, ggml_backend_buffer_get_type(w->buffer)) ? "ZenDNN" : "CPU";
    }
    return "ZenDNN?";
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * p = (prof_state *) user_data;
    if (ask) {
        p->t_ask = ggml_time_us();
        return true;  // want every node -> one node per compute call
    }
    const int64_t dt = ggml_time_us() - p->t_ask;
    if (!p->on || is_view_op(t->op)) {
        return true;
    }

    const char * be = placement(p, t);
    std::string wtype = "-", wbuf = "-", shape = "-";
    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * w = t->src[0];
        const ggml_tensor * x = t->src[1];
        wtype = ggml_type_name(w->type);
        const ggml_tensor * wb = w->view_src ? w->view_src : w;
        // "act": src0 is an activation (e.g. chunked delta-net or attention math), not a weight
        wbuf  = wb->buffer && ggml_backend_buffer_get_usage(wb->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS
                    ? ggml_backend_buffer_name(wb->buffer) : "act";
        char buf[160];
        if (t->op == GGML_OP_MUL_MAT_ID) {
            // w: [K, M, n_expert]; x: [K, n_used or 1, n_tokens]; ids: [n_used, n_tokens]
            snprintf(buf, sizeof(buf), "K=%lld M=%lld E=%lld used=%lld tok=%lld",
                     (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
                     (long long) t->src[2]->ne[0], (long long) t->src[2]->ne[1]);
        } else {
            snprintf(buf, sizeof(buf), "K=%lld M=%lld N=%lld B=%lld",
                     (long long) w->ne[0], (long long) w->ne[1], (long long) x->ne[1],
                     (long long) (x->ne[2] * x->ne[3]));
        }
        shape = buf;
    }
    const std::string op = ggml_op_desc(t);
    auto & k = p->by_key[op + "\t" + name_stem(t->name) + "\t" + be + "\t" + wtype + "\t" + wbuf + "\t" + shape];
    k.calls++;
    k.us += dt;
    auto & o = p->by_op[op + "\t" + be];
    o.calls++;
    o.us += dt;
    p->sum_us += dt;
    return true;
}

// text prompt tokens (-f); empty = random tokens
static std::vector<llama_token> g_text_tokens;

// same token stream as llama-bench's test_prompt, or the -f text from its start
static bool run_prompt(llama_context * ctx, int n_prompt, int n_batch) {
    const llama_vocab * vocab   = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens(n_batch);
    int n_done = 0;
    while (n_done < n_prompt) {
        const int n = std::min(n_prompt - n_done, n_batch);
        if (!g_text_tokens.empty()) {
            for (int i = 0; i < n; i++) {
                tokens[i] = g_text_tokens[(n_done + i) % g_text_tokens.size()];
            }
        } else {
            tokens[0] = n_done == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
            for (int i = 1; i < n; i++) {
                tokens[i] = std::rand() % n_vocab;
            }
        }
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n)) != 0) {
            fprintf(stderr, "opprof: llama_decode failed\n");
            return false;
        }
        n_done += n;
    }
    llama_synchronize(ctx);
    return true;
}

static ggml_type parse_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    fprintf(stderr, "opprof: unsupported cache type %s\n", s.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, out_path, fa = "on", ctk = "f16", ctv = "f16", load_mode = "auto", text_path;
    int n_prompt = 512, n_ubatch = 512, n_batch = 4096, n_threads = 32, reps = 1, warmup = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "opprof: missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")   model_path = next();
        else if (a == "-p")   n_prompt   = atoi(next());
        else if (a == "-ub")  n_ubatch   = atoi(next());
        else if (a == "-b")   n_batch    = atoi(next());
        else if (a == "-t")   n_threads  = atoi(next());
        else if (a == "-r")   reps       = atoi(next());
        else if (a == "-w")   warmup     = atoi(next());
        else if (a == "-fa")  fa         = next();
        else if (a == "-ctk") ctk        = next();
        else if (a == "-ctv") ctv        = next();
        else if (a == "-o")   out_path   = next();
        else if (a == "-lm")  load_mode  = next();
        else if (a == "-f")   text_path  = next();
        else { fprintf(stderr, "opprof: unknown argument %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: opprof -m model.gguf [-p N] [-ub N] [-b N] [-t N] [-fa on|off] [-ctk T] [-ctv T] [-r N] [-w N] [-lm MODE] [-f text] [-o out.tsv]\n");
        return 1;
    }
    n_batch  = std::min(n_batch, n_prompt);
    n_ubatch = std::min(n_ubatch, n_batch);

    llama_backend_init();

    prof_state prof;
    prof.zdev = ggml_backend_dev_by_name("ZenDNN");

    auto mparams = llama_model_default_params();
    mparams.load_mode = llama_load_mode_from_str(load_mode.c_str());
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "opprof: failed to load %s\n", model_path.c_str());
        return 1;
    }

    if (!text_path.empty()) {
        FILE * tf = fopen(text_path.c_str(), "rb");
        if (tf == nullptr) { fprintf(stderr, "opprof: cannot read %s\n", text_path.c_str()); return 1; }
        std::string text;
        char buf[65536];
        size_t nr;
        while ((nr = fread(buf, 1, sizeof(buf), tf)) > 0) text.append(buf, nr);
        fclose(tf);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, false);
        g_text_tokens.resize(n);
        if (llama_tokenize(vocab, text.c_str(), text.size(), g_text_tokens.data(), n, true, false) != n) {
            fprintf(stderr, "opprof: tokenization of %s failed\n", text_path.c_str());
            return 1;
        }
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = n_prompt;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k          = parse_type(ctk);
    cparams.type_v          = parse_type(ctv);
    cparams.no_perf         = false;

    // 1) reference wall time, no callback (graph computed normally)
    double clean_ms = 0;
    {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) { fprintf(stderr, "opprof: context init failed\n"); return 1; }
        for (int w = 0; w < std::max(warmup, 1); w++) {
            run_prompt(ctx, warmup > 0 ? n_prompt : std::min(n_prompt, n_ubatch), n_batch);
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        const int64_t t0 = ggml_time_us();
        if (!run_prompt(ctx, n_prompt, n_batch)) return 1;
        clean_ms = (ggml_time_us() - t0) / 1000.0;
        llama_free(ctx);
    }

    // 2) profiled runs
    cparams.cb_eval           = eval_cb;
    cparams.cb_eval_user_data = &prof;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { fprintf(stderr, "opprof: context init failed\n"); return 1; }
    run_prompt(ctx, std::min(n_prompt, n_ubatch), n_batch);  // new context: one ubatch to settle its buffers
    int64_t prof_us = 0;
    for (int r = 0; r < reps; r++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        prof.on = true;
        const int64_t t0 = ggml_time_us();
        if (!run_prompt(ctx, n_prompt, n_batch)) return 1;
        prof_us += ggml_time_us() - t0;
        prof.on = false;
    }
    const double prof_ms  = prof_us / 1000.0 / reps;
    const double nodes_ms = prof.sum_us / 1000.0 / reps;

    FILE * out = out_path.empty() ? stdout : fopen(out_path.c_str(), "w");
    if (out == nullptr) { fprintf(stderr, "opprof: cannot write %s\n", out_path.c_str()); return 1; }
    fprintf(out, "# model=%s n_prompt=%d n_batch=%d n_ubatch=%d threads=%d fa=%s ctk=%s ctv=%s reps=%d warmup=%d load_mode=%s prompt=%s zendnn_device=%s\n",
            model_path.c_str(), n_prompt, n_batch, n_ubatch, n_threads, fa.c_str(), ctk.c_str(), ctv.c_str(), reps, warmup, load_mode.c_str(),
            text_path.empty() ? "random" : text_path.c_str(), prof.zdev ? "yes" : "no");
    fprintf(out, "# clean_wall_ms=%.1f clean_tps=%.2f profiled_wall_ms=%.1f profiled_tps=%.2f sum_node_ms=%.1f\n",
            clean_ms, n_prompt * 1000.0 / clean_ms, prof_ms, n_prompt * 1000.0 / prof_ms, nodes_ms);
    fprintf(out, "op\tstem\tbackend\twtype\twbuf\tshape\tcalls\tms\tpct\n");
    std::vector<std::pair<std::string, op_stat>> rows(prof.by_key.begin(), prof.by_key.end());
    std::sort(rows.begin(), rows.end(), [](auto & a, auto & b) { return a.second.us > b.second.us; });
    for (auto & [k, s] : rows) {
        fprintf(out, "%s\t%lld\t%.2f\t%.2f\n", k.c_str(), (long long) (s.calls / reps),
                s.us / 1000.0 / reps, 100.0 * s.us / std::max<int64_t>(prof.sum_us, 1));
    }
    if (out != stdout) fclose(out);

    // short console summary
    printf("opprof: pp%d ub%d  clean %.1f ms (%.1f t/s)  profiled %.1f ms (%.1f t/s)  sum(nodes) %.1f ms\n",
           n_prompt, n_ubatch, clean_ms, n_prompt * 1000.0 / clean_ms, prof_ms, n_prompt * 1000.0 / prof_ms, nodes_ms);
    std::vector<std::pair<std::string, op_stat>> ops(prof.by_op.begin(), prof.by_op.end());
    std::sort(ops.begin(), ops.end(), [](auto & a, auto & b) { return a.second.us > b.second.us; });
    for (auto & [k, s] : ops) {
        printf("  %-40s %8lld calls %10.1f ms %6.2f%%\n", k.c_str(), (long long) (s.calls / reps),
               s.us / 1000.0 / reps, 100.0 * s.us / std::max<int64_t>(prof.sum_us, 1));
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
