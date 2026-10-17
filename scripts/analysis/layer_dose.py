"""Does a layer's crowding predict what rounding it costs? (16 Oct, test 1 of the superposition question)

110M fp model: for every one of the 84 linears, (a) its crowding: the Welch ratio and the share of pairs with
|cos| > 0.2 of its reading directions (rows of q / k / v / gate / up) or writing directions (columns of o / down), and
(b) the loss added when only that linear is rounded (GPTQ ternary, 32 calibration windows; the rest stays fp), on
--n_eval fixed validation windows. Then, within each linear type (12 blocks each), the Spearman correlation of crowding
with the added loss, and over all 84 a regression of the added loss on type + depth + crowding.
  python scripts/analysis/layer_dose.py --fp models/fp32_baseline_slim/ckpt.pt --out docs/analysis/geom110/layer_dose.json
"""
import argparse, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
import models110 as M                                                              # noqa: E402
from ttf.hf_init import ternarize, ternarize_gptq                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fp", required=True); ap.add_argument("--train", default="data/wiki32k_train.bin")
ap.add_argument("--val", default="data/wiki32k_val.bin"); ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--calib", type=int, default=32); ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"
WRITE = ("wo", "w_down")
val = np.memmap(a.val, dtype=np.uint16, mode="r")
EV = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_eval)]


@torch.no_grad()
def loss(m):
    return float(np.mean([float(m(w.to(dev)[None][:, :-1], w.to(dev)[None][:, 1:])[1]) for w in EV]))


@torch.no_grad()
def crowd(V):
    k, d = V.shape
    U = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = U @ U.T; off = ~torch.eye(k, dtype=torch.bool, device=V.device); c = G[off]
    welch = (k - d) / (d * (k - 1)) if k > d else 1.0 / d          # k <= d: random-vector level as the reference
    return float((c * c).mean() / welch), float((c.abs() > 0.2).float().mean())


m = M.load(a.fp)
base = loss(m); print(f"fp {base:.4f}", flush=True)
H, cnt, hooks = {}, {}, []
for i, blk in enumerate(m.layers):
    for sub, nm in M.LIN:
        lin = getattr(getattr(blk, sub), nm); H[(i, nm)] = torch.zeros(lin.in_features, lin.in_features, device=dev); cnt[(i, nm)] = 0
        def hk(mod, inp, key=(i, nm)):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[key].addmm_(x.T, x); cnt[key] += x.shape[0]
        hooks.append(lin.register_forward_pre_hook(hk))
tr = np.memmap(a.train, dtype=np.uint16, mode="r")
with torch.no_grad():
    for s in np.random.default_rng(4321).integers(0, len(tr) - a.seq - 1, a.calib):
        x = torch.tensor(tr[s:s + a.seq + 1].astype(np.int64), device=dev)[None]; m(x[:, :-1], x[:, 1:])
for h in hooks:
    h.remove()
rows = []
for i, blk in enumerate(m.layers):
    for sub, nm in M.LIN:
        lin = getattr(getattr(blk, sub), nm); W = lin.weight.data.float().clone()
        wr, sh = crowd(W.T if nm in WRITE else W)
        T, ra, c = ternarize(W, "als"); T, ra, c = ternarize_gptq(W, H[(i, nm)] / cnt[(i, nm)], ra, c)
        lin.weight.data = (ra[:, None] * c[None, :] * T.float()).to(lin.weight.dtype)
        dl = loss(m) - base
        lin.weight.data = W.to(lin.weight.dtype)
        rows.append({"block": i, "type": nm, "welch": wr, "strong": sh, "dloss": dl})
        print(f"block {i:2d} {nm:7s} welch {wr:7.2f} strong {sh:.4f} dloss {dl:+.4f}", flush=True)
from scipy.stats import spearmanr                                                 # noqa: E402
out = {"fp": base, "rows": rows, "within_type": {}}
for nm in sorted({r["type"] for r in rows}):
    rs = [r for r in rows if r["type"] == nm]
    rho, p = spearmanr([r["welch"] for r in rs], [r["dloss"] for r in rs])
    rho2, _ = spearmanr([r["block"] for r in rs], [r["dloss"] for r in rs])
    out["within_type"][nm] = {"rho_crowd": rho, "p": p, "rho_depth": rho2}
    print(f"{nm:7s}: Spearman(crowding, dloss) {rho:+.2f} (p {p:.3f}); Spearman(depth, dloss) {rho2:+.2f}", flush=True)
# over all 84: dloss ~ type dummies + depth + log crowding (least squares); the crowding coefficient and its t value
types = sorted({r["type"] for r in rows})
X = np.array([[1.0 if r["type"] == t else 0.0 for t in types] + [r["block"] / 11, math.log(r["welch"])] for r in rows])
y = np.array([r["dloss"] for r in rows])
beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
sig2 = float(((y - X @ beta) ** 2).sum() / (len(y) - X.shape[1]))
se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
out["regression"] = {"crowding_coef": float(beta[-1]), "crowding_t": float(beta[-1] / se[-1]), "depth_coef": float(beta[-2]),
                     "depth_t": float(beta[-2] / se[-2])}
print(f"regression over 84 (type + depth + log crowding): crowding coef {beta[-1]:+.4f} (t {beta[-1] / se[-1]:+.2f}), "
      f"depth coef {beta[-2]:+.4f} (t {beta[-2] / se[-2]:+.2f})", flush=True)
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
