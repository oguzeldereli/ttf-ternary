// Teacher targets from a GGUF model through llama.cpp (e.g. a Q4 file that fits a 24 GB card): for each window, the top-k
// next-token log-probabilities (under the full softmax) at every position, as scripts/quant/teacher_topk.py computes them
// from the bf16 model. Driven by teacher_gguf.py, which writes the same kd_NNNNN.npz shards.
//   g++ -O3 -fopenmp -std=c++17 scripts/quant/llama_topk.cpp -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin \
//       -lllama -Wl,-rpath,$LLAMA/build/bin -o llama_topk
//   llama_topk MODEL.gguf K [n_gpu_layers=99] [n_ctx=2048] < requests > results
// stdin: repeated (int32 n, n x int32 token ids), n = 0 ends; n <= n_ctx. For each window the state is cleared and all n
// tokens decoded. stdout: per window n x K int32 ids, then n x K float16 log-probabilities (position i: the distribution
// of token i + 1), ordered by descending probability.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static uint16_t f2h(float f) {                        // IEEE half, round to nearest even (values are log-probs <= 0)
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e == 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    int ne = (int) e - 127 + 15;
    if (ne >= 31) return sign | 0x7c00;
    if (ne <= 0) {
        if (ne < -10) return sign;
        m |= 0x800000; int sh = 14 - ne;
        uint32_t h = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return sign | h;
    }
    uint32_t h = sign | (ne << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf K [ngl] [n_ctx]\n", argv[0]); return 1; }
    int K = atoi(argv[2]), ngl = argc > 3 ? atoi(argv[3]) : 99, n_ctx = argc > 4 ? atoi(argv[4]) : 2048;
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
    std::vector<int32_t> tok, ids;
    std::vector<uint16_t> lps;
    for (;;) {
        int32_t n;
        if (fread(&n, sizeof(int32_t), 1, stdin) != 1 || n == 0) break;
        if (n > n_ctx || n < 1) { fprintf(stderr, "bad window n %d\n", n); return 1; }
        tok.resize(n);
        if (fread(tok.data(), sizeof(int32_t), n, stdin) != (size_t) n) { fprintf(stderr, "short read\n"); return 1; }
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = tok[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        ids.assign((size_t) n * K, 0); lps.assign((size_t) n * K, 0);
        #pragma omp parallel
        {
            std::vector<int> idx(V);
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                const float * lg = llama_get_logits_ith(ctx, i);
                float mx = *std::max_element(lg, lg + V);
                double z = 0;
                for (int v = 0; v < V; v++) z += std::exp((double) lg[v] - mx);
                const float lz = mx + (float) std::log(z);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
                for (int j = 0; j < K; j++) {
                    ids[(size_t) i * K + j] = idx[j]; lps[(size_t) i * K + j] = f2h(lg[idx[j]] - lz);
                }
            }
        }
        fwrite(ids.data(), sizeof(int32_t), ids.size(), stdout);
        fwrite(lps.data(), sizeof(uint16_t), lps.size(), stdout);
        fflush(stdout);
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}
