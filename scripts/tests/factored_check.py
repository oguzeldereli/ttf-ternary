"""--ts_factored against the dense TTF update on one layer (17 Oct): same layer, same gradients (g = gy^T x), several
steps, 16 x 1 (M = 1) and micro-batched (M = 4). Reports the relative difference of the accumulator A = U V^T and of v,
and how many moves agree, for sketch sizes K_s; as K_s grows the factored update must converge to the dense one.
  python scripts/tests/factored_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import bitnet.train as T
from bitnet.flip import KernelTernaryLinear
from bitnet.kernel import unpack_rows

dev = "cuda"
TH = [1e9]
N, K, Tk, r = 1024, 512, 2048, 64


def cfg(M, ks):
    return dict(r=r, lr_ratio=1.0, theta=TH[0], tau=1000, rank_s=r, b1=0.9, b2=0.95, gate=False, spend=1.0,
                tau_anneal=0.75, micro=M, factored=ks, store_a="", offload=False, nomom=True, start=0, step=10)


def run(M, ks, steps=6, seed=0):
    torch.manual_seed(seed)
    l = KernelTernaryLinear(K, N, int8=True, dw_mode="dense").to(dev); l.track = False; l._ts_idx = 3; l._ts_host = False
    global W0
    W0 = unpack_rows(l.wpacked, K).clone()
    state = {}
    gen = torch.Generator(device=dev).manual_seed(123)
    for s in range(steps):
        for i in range(M):
            gy = (torch.randn(Tk, N, device=dev, generator=gen) * 1e-3).to(torch.bfloat16)
            x = (torch.randn(Tk, K, device=dev, generator=gen) + 0.3).to(torch.bfloat16)
            gy[:, :8] *= 20                                            # a few loud rows
            l._ts_final = i == M - 1
            if ks:
                T.ts_layer(l, (gy, x), state, cfg(M, ks))
            else:
                T.ts_layer(l, (gy.T.float() @ x.float()).contiguous(), state, cfg(M, ks))   # what the dense backward passes
    U, V = T._load_pair(state[id(l)])
    R, C = state["_ts"][("v", id(l))]
    return U @ V.T, R, C, unpack_rows(l.wpacked, K).clone()


for th in (1e9, 6.0):
  TH[0] = th
  print(f"theta {th:g}", flush=True)
  for M in (1, 4):
    for ks in (64, 256, 1024, 4096):
        T.MICRO_SKETCH[0] = ks       # the dense path's cross-term sketch: the same maps as the factored update's
        Ad, Rd, Cd, Wd = run(M, 0)
        Af, Rf, Cf, Wf = run(M, ks)
        ea = float((Af - Ad).norm() / Ad.norm()); er = float((Rf - Rd).norm() / Rd.norm()); ec = float((Cf - Cd).norm() / Cd.norm())
        print(f"M {M} K_s {ks:5d}: |A_f - A_d| / |A_d| {ea:.4f}  v rows {er:.4f}  v cols {ec:.4f}  "
              f"trits differ {float((Wf != Wd).float().mean()):.5f} (moved {float((Wd != W0).float().mean()):.4f})", flush=True)
