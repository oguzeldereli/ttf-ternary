"""Is crowding the cause of the rounding loss? A causal toy test (16 Oct, test 2 of the superposition question).

Toy model of superposition (Elhage et al. 2022): n sparse features in m dims, x' = ReLU(W^T W x + b). Full-precision
models are trained with an overlap penalty lam * mean_{i != j} cos^2(W_i, W_j) over the represented features (columns of
W), lam = 0 (the natural, crowded solution) to large (spread). Each is then converted the way our layers are: ternary
trits with row and column scales (W ~ a_r c_j T, alternating least squares on the grid), and recovered for a short
straight-through fine-tune from that start. If crowding causes the rounding loss, the spread fp models lose less at the
same fp loss. Ternary trained from scratch (same steps as fp + recovery) is the reference.
  python scripts/analysis/toy_causal.py --out docs/research/analysis/toy_causal.json
"""
import argparse, json
import numpy as np
import torch
torch.set_num_threads(4)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=80); ap.add_argument("--m", type=int, default=20); ap.add_argument("--S", type=float, default=0.9)
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--rec", type=int, default=600)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="")
a = ap.parse_args()
I = 0.9 ** torch.arange(a.n).float()


def batch(bs=1024):
    return torch.rand(bs, a.n) * (torch.rand(bs, a.n) > a.S).float()


def lossf(W, b, x):
    return ((torch.relu(x @ W.T @ W + b) - x) ** 2 * I).sum(1).mean()


def overlap(W):
    nrm = W.norm(dim=0)
    rep = nrm > 0.5 * nrm.max()
    U = W[:, rep] / nrm[rep]
    G = U.T @ U; k = G.shape[0]
    return float((G.pow(2).sum() - G.diagonal().pow(2).sum()) / max(k * (k - 1), 1)), int(rep.sum())


def ternarize(W, iters=6):
    """W [m, n] ~ a_r c_j T: per-row threshold grid (as hf_init._row_grid) alternated with column scales (ALS)"""
    c = torch.ones(W.shape[1])
    for _ in range(iters):
        Wc = W / c[None, :]
        m_ = Wc.abs().mean(1, keepdim=True).clamp_min(1e-12)
        best = None
        for g in torch.linspace(0.3, 2.0, 35):
            T = (Wc / (g * m_)).round().clamp(-1, 1)
            ar = (Wc * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp_min(1)
            err = ((Wc - ar * T) ** 2).sum(1, keepdim=True)
            best = (T, ar, err) if best is None else (torch.where(err < best[2], T, best[0]), torch.where(err < best[2], ar, best[1]), torch.minimum(err, best[2]))
        T, ar = best[0], best[1]
        num = (W * T * ar).sum(0); den = (T * T * ar * ar).sum(0)
        c = torch.where(den > 0, num / den.clamp_min(1e-30), c).clamp_min(1e-3)
    return T, ar.squeeze(1), c


def ste_tern(V):
    """forward: the ternary of V (row absmean threshold, row scale); backward: identity"""
    g = V.abs().mean(1, keepdim=True).clamp_min(1e-8)
    T = (V / g).round().clamp(-1, 1)
    q = T * (V * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp_min(1)
    return V + (q - V).detach()


def train(lam, seed, steps):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(a.m, a.n) * (1 / a.m) ** 0.5); b = torch.nn.Parameter(torch.zeros(a.n))
    opt = torch.optim.AdamW([W, b], lr=1e-2, weight_decay=0.0)
    for t in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = 1e-2 * min(1.0, (t + 1) / 300) * 0.5 * (1 + np.cos(np.pi * t / steps))
        L = lossf(W, b, batch())
        if lam:
            U = W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)
            G = U.T @ U; w_ = (W.norm(dim=0) / W.norm(dim=0).max()).detach()     # weight pairs by how represented both are
            P = (G * G * (w_[:, None] * w_[None, :])); L = L + lam * (P.sum() - P.diagonal().sum()) / (a.n * (a.n - 1))
        opt.zero_grad(); L.backward(); opt.step()
    return W.detach(), b.detach()


def recover(T, ar, c, b, steps, seed):
    """the conversion's recovery: the trits as a latent (re-ternarised by the STE each step) plus the float scales"""
    torch.manual_seed(seed + 100)
    V = torch.nn.Parameter((ar[:, None] * c[None, :] * T).clone()); c_ = torch.nn.Parameter(c.clone()); b_ = torch.nn.Parameter(b.clone())
    opt = torch.optim.AdamW([V, c_, b_], lr=3e-3, weight_decay=0.0)
    for t in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = 3e-3 * 0.5 * (1 + np.cos(np.pi * t / steps))
        We = ste_tern(V / c_[None, :]) * c_[None, :]
        L = lossf(We, b_, batch()); opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        We = ste_tern(V / c_[None, :]) * c_[None, :]
        return float(lossf(We, b_, batch(16384)))


def scratch_ternary(seed, steps):
    torch.manual_seed(seed)
    V = torch.nn.Parameter(torch.randn(a.m, a.n) * (1 / a.m) ** 0.5); c_ = torch.nn.Parameter(torch.ones(a.n)); b = torch.nn.Parameter(torch.zeros(a.n))
    opt = torch.optim.AdamW([V, c_, b], lr=1e-2, weight_decay=0.0)
    for t in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = 1e-2 * min(1.0, (t + 1) / 300) * 0.5 * (1 + np.cos(np.pi * t / steps))
        L = lossf(ste_tern(V / c_[None, :]) * c_[None, :], b, batch()); opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        We = ste_tern(V / c_[None, :]) * c_[None, :]
        return float(lossf(We, b, batch(16384))), overlap(We)


R = {"args": vars(a), "fp": {}}
for lam in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0):
    rows = []
    for s in range(a.seeds):
        W, b = train(lam, s, a.steps)
        xe = batch(16384)
        Lfp = float(lossf(W, b, xe)); ov, k = overlap(W)
        T, ar, c = ternarize(W)
        Lr = float(lossf(ar[:, None] * c[None, :] * T, b, xe))
        Lrec = recover(T, ar, c, b, a.rec, s)
        rows.append((Lfp, ov, k, Lr, Lrec))
    m_ = np.mean(rows, 0)
    R["fp"][str(lam)] = {"fp_loss": m_[0], "overlap": m_[1], "features": m_[2], "rounded": m_[3], "recovered": m_[4]}
    print(f"lam {lam:6.1f}: fp loss {m_[0]:.4f}  overlap {m_[1]:.4f}  features {m_[2]:5.1f}  rounded {m_[3]:.4f} "
          f"(+{m_[3] - m_[0]:.4f})  recovered {m_[4]:.4f} (+{m_[4] - m_[0]:.4f})", flush=True)
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
sc = [scratch_ternary(s, a.steps + a.rec) for s in range(a.seeds)]
R["scratch_ternary"] = {"loss": float(np.mean([x[0] for x in sc])), "overlap": float(np.mean([x[1][0] for x in sc])),
                        "features": float(np.mean([x[1][1] for x in sc]))}
print(f"ternary from scratch ({a.steps + a.rec} steps): loss {R['scratch_ternary']['loss']:.4f}  overlap "
      f"{R['scratch_ternary']['overlap']:.4f}  features {R['scratch_ternary']['features']:.1f}", flush=True)
if a.out:
    json.dump(R, open(a.out, "w"), indent=1)
