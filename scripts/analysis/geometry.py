"""Basis-free geometry of 110M networks (full precision, TTF, roundings of full precision): where do they differ?

  python scripts/analysis/geometry.py --model fp=checkpoints/fp32_baseline/ckpt.pt --model ttf0=RUN/ckpt.pt --gptq_of fp

Per model and layer:
  weights   the effective weight of every linear (W_eff = layer(I): trits with their scales and adapters included),
            its singular spectrum: effective rank exp(H(s^2 / sum s^2)) / min(N, K), stable rank |W|_F^2 / s_max^2,
            the energy share of the top 1% of directions, and the slope of log s against log rank over ranks 10-200
            (heavy-tailed spectra have a steep slope); per attention head the OV circuit W_O[:, h] W_V[h, :]
            (rank <= head_dim) in the same measures
  trits     (ternary models) zero fraction, and how zeros sit: std over rows / columns of the per-row / per-column zero
            fraction (0 = spread evenly; large = whole neurons or input channels zeroed)
  stream    residual stream at every block input on --n_act windows: norm (mean |h|), outlier dimensions (max over
            dimensions of the per-dimension rms / median), mean excess kurtosis over tokens of h across dimensions,
            and anisotropy (top eigenvalue share and effective rank of the token covariance)
Light: 4 CPU threads, activations of every 4th token on the GPU. Writes --out JSON and prints a summary table.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
from ttf.kernel import unpack_rows                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help="NAME=PATH")
ap.add_argument("--gptq_of", default=""); ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--train", default="data/wiki32k_train.bin")
ap.add_argument("--n_act", type=int, default=8); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--out", default="geometry.json")
a = ap.parse_args()
import models110 as fm                                                             # noqa: E402  (load, gptq_copy)
dev = "cuda"
LIN = fm.LIN
val = np.memmap(a.val, dtype=np.uint16, mode="r")
ACT = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_act)]


def spec(W):
    s = torch.linalg.svdvals(W.float())
    p = s ** 2 / (s ** 2).sum()
    k = max(1, int(round(0.01 * len(s))))
    r = torch.arange(1, len(s) + 1, device=s.device, dtype=torch.float32)
    lo, hi = 10, min(200, len(s))
    slope = float(np.polyfit(np.log(r[lo - 1:hi].cpu().numpy()), np.log(s[lo - 1:hi].clamp_min(1e-12).cpu().numpy()), 1)[0])
    return {"eff_rank": float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum())) / len(s),
            "stable_rank": float((s ** 2).sum() / s[0] ** 2), "top1pct": float(p[:k].sum()), "slope": slope}


@torch.no_grad()
def eff_weight(lin):
    K = lin.in_features if hasattr(lin, "in_features") else lin.K
    I = torch.eye(K, device=dev)
    z = torch.zeros(1, K, device=dev)                                        # the affine part (gptqc: b - W mu) apart
    return (lin(I).float() - lin(z).float()).T                               # [N, K]


def zero_layout(T):
    z = (T == 0).float()
    return {"zeros": float(z.mean()), "row_std": float(z.mean(1).std()), "col_std": float(z.mean(0).std())}


@torch.no_grad()
def stream(m):
    res, hooks = [[] for _ in m.layers], []
    for i, blk in enumerate(m.layers):
        hooks.append(blk.register_forward_pre_hook(lambda mod, inp, i=i: res[i].append(inp[0][0, ::4].float())))
    for w in ACT:
        x = w.to(dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    out = []
    for r in res:
        H = torch.cat(r)                                                     # [T, d]
        rms = H.pow(2).mean(0).sqrt()
        z = (H - H.mean(1, keepdim=True)) / H.std(1, keepdim=True).clamp_min(1e-8)
        C = torch.cov(H.T); ev = torch.linalg.eigvalsh(C).clamp_min(0).flip(0); p = ev / ev.sum()
        out.append({"norm": float(H.norm(dim=1).mean()), "outlier": float(rms.max() / rms.median()),
                    "kurtosis": float((z ** 4).mean(1).mean() - 3), "top_eig": float(p[0]),
                    "cov_eff_rank": float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum())) / len(p)})
    return out


R = {}
names = []
paths = {}
for s in a.model:
    n, p = s.split("=", 1); names.append(n); paths[n] = p
for n in names + ([f"gptq({a.gptq_of})"] if a.gptq_of else []):
    m = fm.gptq_copy(paths[a.gptq_of], a.train) if n.startswith("gptq(") else fm.load(paths[n])
    nh = m.layers[0].attn.n_heads
    r = {"layers": []}
    for i, blk in enumerate(m.layers):
        d = {}
        Ws = {nm: eff_weight(getattr(getattr(blk, sub), nm)) for sub, nm in LIN}
        for nm, Wt in Ws.items():
            d[nm] = spec(Wt)
            lin = getattr(blk.attn if nm in ("wq", "wk", "wv", "wo") else blk.ffn, nm)
            if hasattr(lin, "wpacked"):
                d[nm].update(zero_layout(unpack_rows(lin.wpacked, lin.K)))
            elif n.startswith("gptq("):                                     # dequantised: a c T is 0 where T is
                d[nm].update(zero_layout(Wt))
        hd = Ws["wv"].shape[0] // nh
        ov = [spec(Ws["wo"][:, h * hd:(h + 1) * hd] @ Ws["wv"][h * hd:(h + 1) * hd, :]) for h in range(nh)]
        d["ov"] = {k: float(np.mean([o[k] for o in ov])) for k in ov[0]}
        r["layers"].append(d)
        del Ws
    r["stream"] = stream(m)
    R[n] = r
    del m; torch.cuda.empty_cache()
    L = r["layers"]
    mean = lambda key, nm: float(np.mean([l[nm][key] for l in L]))
    print(f"== {n}", flush=True)
    for nm in [x for _, x in LIN] + ["ov"]:
        extra = f"  zeros {mean('zeros', nm):.3f} row_std {mean('row_std', nm):.3f} col_std {mean('col_std', nm):.3f}" \
            if "zeros" in L[0][nm] else ""
        print(f"  {nm:7s} eff_rank {mean('eff_rank', nm):.3f}  stable {mean('stable_rank', nm):7.1f}  top1% "
              f"{mean('top1pct', nm):.3f}  slope {mean('slope', nm):.3f}{extra}", flush=True)
    S = r["stream"]
    print("  stream by layer: norm " + " ".join(f"{x['norm']:.0f}" for x in S), flush=True)
    print("  outlier " + " ".join(f"{x['outlier']:.1f}" for x in S) + "\n  kurtosis "
          + " ".join(f"{x['kurtosis']:.1f}" for x in S) + "\n  top_eig " + " ".join(f"{x['top_eig']:.2f}" for x in S)
          + "\n  cov_eff_rank " + " ".join(f"{x['cov_eff_rank']:.3f}" for x in S), flush=True)
json.dump(R, open(a.out, "w"), indent=1)
