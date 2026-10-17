"""Superposition in the weights of 110M networks: how near-orthogonal are the directions each layer reads and writes?
(16 Oct; the question whether ternary weights lose superposition precision)

Per linear (effective weight W_eff = layer(I), scales and adapters included): the written directions (columns of the
output and down projections, vectors in the residual stream) and the read directions (rows of q / k / v / gate / up,
vectors in the normed residual), k vectors in d dimensions. Reported, averaged over layers per type:
  mean |cos| between distinct directions (random vectors: sqrt(2 / (pi d)))
  share of pairs with |cos| > 0.2 (strong interference)
  Welch ratio: the mean squared cosine over the bound (k - d) / (d (k - 1)) that k unit vectors in d dimensions cannot beat
    (1 = an equiangular tight frame, the most superposition-friendly packing; random vectors ~ (k - 1) / (k - d) * ...)
  frame tightness: d * sum_ij (v_i . v_j)^2 / (sum_i |v_i|^2)^2 over the norm-weighted directions (1 = the directions
    spread the energy evenly over all d dimensions; large = they crowd into a few)
  python scripts/analysis/interference.py --model fp=checkpoints/fp32_baseline/ckpt.pt \
      --model ttf0=checkpoints/final/ts16full_tau1000_s0/ckpt.pt --gptq_of fp --out docs/research/analysis/geom110/interference.json
"""
import argparse, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
import models110 as M                                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help="NAME=PATH")
ap.add_argument("--gptq_of", default=""); ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
WRITE, READ = ("wo", "w_down"), ("wq", "wk", "wv", "w_gate", "w_up")


@torch.no_grad()
def eff(lin):
    K = lin.in_features if hasattr(lin, "in_features") else lin.K
    return lin(torch.eye(K, device=dev)).float().T                       # [N, K]


@torch.no_grad()
def stats(V):
    """V [k, d]: k direction vectors (rows) in d dimensions"""
    k, d = V.shape
    n = V.norm(dim=1).clamp_min(1e-12)
    U = V / n[:, None]
    G = U @ U.T
    off = ~torch.eye(k, dtype=torch.bool, device=V.device)
    c = G[off].abs()
    welch = (k - d) / (d * (k - 1)) if k > d else 0.0
    Gw = V @ V.T
    tight = float(d * (Gw ** 2).sum() / (n ** 2).sum() ** 2)
    return {"mean_abs_cos": float(c.mean()), "strong_share": float((c > 0.2).float().mean()),
            "welch_ratio": float((c ** 2).mean() / welch) if welch > 0 else float("nan"), "tightness": tight,
            "k": k, "d": d, "random_mean_abs_cos": math.sqrt(2 / (math.pi * d))}


R = {}
models = [tuple(x.split("=", 1)) for x in a.model]
if a.gptq_of:
    models.append(("gptq(" + a.gptq_of + ")", None))
for name, path in models:
    m = M.gptq_copy(dict(models)[a.gptq_of]) if path is None else M.load(path)
    per = {}
    for blk in m.layers:
        for sub, nm in M.LIN:
            W = eff(getattr(getattr(blk, sub), nm))
            V = W.T if nm in WRITE else W                                  # written: columns; read: rows
            per.setdefault(nm, []).append(stats(V))
    R[name] = {nm: {k: float(np.mean([s[k] for s in v])) for k in v[0]} for nm, v in per.items()}
    print(f"== {name}")
    for nm, s in R[name].items():
        print(f"  {nm:7s} {'write' if nm in WRITE else 'read '} k={int(s['k'])} d={int(s['d'])}  mean|cos| {s['mean_abs_cos']:.4f}"
              f" (random {s['random_mean_abs_cos']:.4f})  |cos|>0.2 {s['strong_share']:.4f}  Welch ratio {s['welch_ratio']:.2f}"
              f"  tightness {s['tightness']:.2f}", flush=True)
    del m; torch.cuda.empty_cache()
if a.out:
    json.dump(R, open(a.out, "w"), indent=1)
