"""Does ternary weight precision limit superposition? (16 Oct)

Toy model of Elhage et al. (2022), "Toy models of superposition": n sparse features x in [0, 1]^n (each present with
probability 1 - S, importance I_i = 0.9^i), compressed to m < n dimensions and read back, x' = ReLU(W^T W x + b), loss
sum_i I_i (x_i - x'_i)^2. The weights W [m, n] are
  fp        floats
  ternary   a * T with T in {-1, 0, 1} (straight-through from a float latent; per-row scale a, absmean threshold)
  ternary-c ternary with a per-column (per-feature) scale as well, W = a_r c_j T (our layers' row x column scales)
Reported per sparsity: features represented (|W_j| > 0.5 of the largest), the superposition count (sum over features of
|W_j|^2 / ... as in Elhage: dimensions per feature), mean interference among represented features, and the loss.
  python scripts/analysis/toy_superposition.py --n 80 --m 20 --out OUT.json
"""
import argparse, json
import numpy as np
import torch
torch.set_num_threads(4)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=80); ap.add_argument("--m", type=int, default=20)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="")
a = ap.parse_args()
I = 0.9 ** torch.arange(a.n).float()


def ste_ternary(w, col=False):
    """absmean ternary of w [m, n] with a straight-through gradient; row scale (and column scale)"""
    g = w.abs().mean(1, keepdim=True).clamp_min(1e-8)
    T = (w / g).round().clamp(-1, 1)
    q = T * (w * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp_min(1)
    return w + (q - w).detach()


def run(kind, S, seed):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(a.m, a.n) * (1 / a.m) ** 0.5)
    b = torch.nn.Parameter(torch.zeros(a.n))
    cs = torch.nn.Parameter(torch.ones(a.n)) if kind == "ternary-c" else None
    opt = torch.optim.AdamW([p for p in (W, b, cs) if p is not None], lr=1e-2, weight_decay=0.0)
    for t in range(a.steps):
        for g_ in opt.param_groups:
            g_["lr"] = 1e-2 * min(1.0, (t + 1) / 300) * 0.5 * (1 + np.cos(np.pi * t / a.steps))
        x = torch.rand(a.batch, a.n) * (torch.rand(a.batch, a.n) > S).float()
        We = W if kind == "fp" else ste_ternary(W)
        if cs is not None:
            We = We * cs[None, :]
        y = torch.relu(x @ We.T @ We + b)
        loss = ((y - x) ** 2 * I).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        We = W if kind == "fp" else ste_ternary(W)
        if cs is not None:
            We = We * cs[None, :]
        x = torch.rand(8192, a.n) * (torch.rand(8192, a.n) > S).float()
        L = float((((torch.relu(x @ We.T @ We + b) - x) ** 2) * I).sum(1).mean())
        nrm = We.norm(dim=0)
        rep = nrm > 0.5 * nrm.max()
        Wn = We[:, rep] / nrm[rep]
        G = (Wn.T @ Wn).abs(); G.fill_diagonal_(0)
        k = int(rep.sum())
        interf = float(G.sum() / max(k * (k - 1), 1))
        # Elhage's dimensionality per feature: D_j = |W_j|^2 / sum_k (W_j_hat . W_k)^2
        Wh = We / nrm.clamp_min(1e-8)
        D = (nrm ** 2) / ((Wh.T @ We) ** 2).sum(1).clamp_min(1e-8)
        zeros = float((We.abs() < 1e-8).float().mean())
    return {"loss": L, "represented": k, "features_per_dim": k / a.m, "mean_interference": interf,
            "sum_dimensionality": float(D[rep].sum()), "zero_fraction": zeros}


R = {}
for S in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
    for kind in ("fp", "ternary", "ternary-c"):
        rs = [run(kind, S, s) for s in range(a.seeds)]
        R[f"{kind}@S={S}"] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        r = R[f"{kind}@S={S}"]
        print(f"S={S:<5} {kind:10s} loss {r['loss']:.4f}  represented {r['represented']:5.1f} / {a.n} "
              f"({r['features_per_dim']:.2f} per dim)  interference {r['mean_interference']:.3f}  zeros {r['zero_fraction']:.2f}",
              flush=True)
    if a.out:
        json.dump({"args": vars(a), "results": R}, open(a.out, "w"), indent=1)
