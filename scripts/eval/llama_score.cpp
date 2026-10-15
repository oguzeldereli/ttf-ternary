// Score token sequences with a GGUF model through llama.cpp's own kernels (the file as shipped, no dequantisation), for
// the same windows and zero-shot items as qwen_zeroshot.py / ttf_qwen.py --ppl.
//   g++ -O2 -std=c++17 scripts/eval/llama_score.cpp -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin -lllama \
//       -Wl,-rpath,$LLAMA/build/bin -o llama_score
//   llama_score MODEL.gguf [n_gpu_layers=99] [n_ctx=4096] < requests > results
// stdin: repeated (int32 n, int32 k, n x int32 token ids), n = 0 ends. For each sequence the model state is cleared, the n
// tokens are decoded and the last k tokens scored from the positions before them. stdout: one line per sequence,
// "sum_logp all_greedy" (natural log; all_greedy 1 if every scored token is the argmax).
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL.gguf [ngl] [n_ctx]\n", argv[0]); return 1; }
    int ngl = argc > 2 ? atoi(argv[2]) : 99, n_ctx = argc > 3 ? atoi(argv[3]) : 4096;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.n_seq_max = 1; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    std::vector<int32_t> tok;
    for (;;) {
        int32_t hdr[2];
        if (fread(hdr, sizeof(int32_t), 2, stdin) != 2 || hdr[0] == 0) break;
        int n = hdr[0], k = hdr[1];
        if (n > n_ctx || k < 1 || k >= n) { fprintf(stderr, "bad sequence n %d k %d\n", n, k); return 1; }
        tok.resize(n);
        if (fread(tok.data(), sizeof(int32_t), n, stdin) != (size_t) n) { fprintf(stderr, "short read\n"); return 1; }
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n - 1;                       // the last token is only a target
        for (int i = 0; i < n - 1; i++) {
            batch.token[i] = tok[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= n - 1 - k;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        double sum = 0; int greedy = 1;
        for (int i = n - 1 - k; i < n - 1; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            int am = 0; double mx = lg[0];
            for (int v = 1; v < V; v++) if (lg[v] > mx) { mx = lg[v]; am = v; }
            double z = 0;
            for (int v = 0; v < V; v++) z += std::exp((double) lg[v] - mx);
            sum += (double) lg[tok[i + 1]] - mx - std::log(z);
            greedy &= am == tok[i + 1];
        }
        printf("%.6f %d\n", sum, greedy); fflush(stdout);
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}
