"""GEMM microbenchmark at the 1.3B shapes (batch 16 x 2048 = 32768 tokens): time per pass of the ternary matmuls,
summed over the model's 168 matrices, per kind and backend. Used to project the speed of a step on hardware with native
ternary matmuls (Amdahl: only the matmuls that touch trits get faster).
    python scripts/analysis/gemm_bench.py [preset]"""
import sys, time, torch
sys.path.insert(0, ".")
from ttf import kernel as K
from ttf.config import PRESETS

c = PRESETS[sys.argv[1] if len(sys.argv) > 1 else "d2048_l24"]
M = 16 * 2048
kv = c.dim * c.n_kv_heads // c.n_heads
shapes = [(c.dim, c.dim), (kv, c.dim), (kv, c.dim), (c.dim, c.dim), (c.hidden_dim, c.dim), (c.hidden_dim, c.dim),
          (c.dim, c.hidden_dim)]


def timeit(f, n=10):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


tot = {}
for N, Kd in shapes:
    w = torch.randint(-1, 2, (N, Kd), dtype=torch.int8, device="cuda"); wp = K.pack_rows(w)
    xq, xs = K.act_quant_i8(torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16))
    gy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    wb = w.to(torch.bfloat16)
    for be in ("cublas", "int8mm"):
        K.GEMM_BACKEND[0] = be
        tot[f"forward ({be})"] = tot.get(f"forward ({be})", 0) + timeit(lambda: K.tern_gemm_i8(xq, xs, wp, Kd))
    K.GEMM_BACKEND[0] = "cublas"
    tot["input gradient (cublas)"] = tot.get("input gradient (cublas)", 0) + timeit(lambda: K.tern_gemm_dx(gy, wp, Kd))
    tot["weight gradient (dense, cublas)"] = tot.get("weight gradient (dense, cublas)", 0) + timeit(
        lambda: gy.t() @ (xq.to(torch.bfloat16) * xs[:, None].to(torch.bfloat16)))
    tot["unpack trits to bf16"] = tot.get("unpack trits to bf16", 0) + timeit(lambda: K._w_bf16(wp, Kd))
    tot["unpack trits to int8"] = tot.get("unpack trits to int8", 0) + timeit(lambda: K.unpack_rows(wp, Kd))
    tot["dense bf16 GEMM, same shape (reference)"] = tot.get("dense bf16 GEMM, same shape (reference)", 0) + timeit(
        lambda: xq.to(torch.bfloat16) @ wb.t())
print(f"GEMM benchmark, {M} tokens, per pass over all {c.n_layers * 7} ternary matrices:")
for k, v in tot.items():
    print(f"  {k:40s} {v * c.n_layers:7.3f} s")
