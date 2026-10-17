"""Time of the TTF update per micro-batch at 27B layer shapes (17 Oct): dense (weight gradient gy^T x, then ts_layer)
against --ts_factored (ts_layer on the factors), --ts_micro M = 8 (non-final and final micro-batches), rank r.
  python scripts/tests/factored_bench.py [RANK ...]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import ttf.train as T
from ttf.flip import KernelTernaryLinear

dev = "cuda"
Tk, M = 4096, 8                                   # tokens per micro-batch (2 x 2048), micro-batches per step


def bench(N, K, r, ks):
    l = KernelTernaryLinear(K, N, int8=True, dw_mode="dense").to(dev); l.track = False; l._ts_idx = 5; l._ts_host = False
    state = {}
    c = dict(r=r, lr_ratio=1.0, theta=64.0, tau=1000, rank_s=r, b1=0.9, b2=0.95, gate=False, spend=1.0, tau_anneal=0.75,
             micro=M, factored=ks, store_a="fp12s", offload=False, nomom=True, start=0, step=10)
    gy = (torch.randn(Tk, N, device=dev) * 1e-3).to(torch.bfloat16)
    x = torch.randn(Tk, K, device=dev).to(torch.bfloat16)

    def one(i):
        l._ts_final = i == M - 1
        if ks:
            T.ts_layer(l, (gy, x), state, c)
        else:
            g = (gy.T @ x).float()                    # the dense backward's weight gradient (bf16 GEMM, then fp32)
            T.ts_layer(l, g, state, c); del g
    for i in range(2 * M):                            # warm-up (state created, kernels compiled)
        one(i % M)
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(2 * M):
        one(i % M)
    torch.cuda.synchronize()
    return (time.time() - t0) / (2 * M) * 1e3, torch.cuda.max_memory_allocated() / 2 ** 30


ranks = [int(a) for a in sys.argv[1:]] or [256, 1024]
for N, K in ((17408, 5120), (5120, 17408), (6144, 5120)):
    for r in ranks:
        torch.cuda.reset_peak_memory_stats(); d, md = bench(N, K, r, 0)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); f, mf = bench(N, K, r, 256)
        torch.cuda.empty_cache()
        print(f"{N} x {K} rank {r}: dense {d:7.2f} ms ({md:.2f} GiB peak), factored {f:7.2f} ms ({mf:.2f} GiB): {d / f:.2f}x",
              flush=True)
