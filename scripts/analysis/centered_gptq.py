"""GPTQ ternary rounding with the input mean taken out (110M test of a training-free conversion).

The fp model's residual stream carries a large constant offset (two dimensions at ~40x the median rms, ~96% of it the
same on every token), so the second moment H = E[x x^T] of every layer's input is dominated by one constant direction
and GPTQ spends its precision on reproducing W mu. Variants, each on every linear:
  plain     GPTQ with H = E[x x^T] (as now)
  biascorr  plain trits, plus the exact correction of the mean's error as a float bias b = (W - W_hat) mu
  centered  GPTQ with the covariance C = E[(x - mu)(x - mu)^T] (the varying part only), plus b = (W - W_hat) mu
so the layer computes W_hat x + b = W_hat (x - mu) + W mu: the mean passes exactly, the rounding error only touches the
varying part. Losses on --n_eval fixed validation windows (dequantised, float activations).
  python scripts/analysis/centered_gptq.py --fp checkpoints/fp32_baseline/ckpt.pt
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
import models110 as M                                                              # noqa: E402
from ttf.hf_init import ternarize, ternarize_gptq, _out_err                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fp", required=True); ap.add_argument("--train", default="data/wiki32k_train.bin")
ap.add_argument("--val", default="data/wiki32k_val.bin"); ap.add_argument("--calib", type=int, default=32)
ap.add_argument("--n_eval", type=int, default=32); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--variants", default="plain,biascorr,centered"); ap.add_argument("--out", default="")
ap.add_argument("--only", default="", help="round only these: comma list of linear names (wq, ..., w_down) and / or "
                "block ranges a-b; everything else stays fp (where does the rounding loss come from)")
a = ap.parse_args()
dev = "cuda"
val = np.memmap(a.val, dtype=np.uint16, mode="r")
EV = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_eval)]


@torch.no_grad()
def loss(m):
    return float(np.mean([float(m(w.to(dev)[None][:, :-1], w.to(dev)[None][:, 1:])[1]) for w in EV]))


@torch.no_grad()
def stats(m):
    """per linear: E[x x^T], E[x] over the calibration windows (inputs of the fp model)"""
    S, mu, cnt, hooks = {}, {}, {}, []
    for i, blk in enumerate(m.layers):
        for sub, nm in M.LIN:
            lin = getattr(getattr(blk, sub), nm); K = lin.in_features
            S[(i, nm)] = torch.zeros(K, K, device=dev, dtype=torch.float64); mu[(i, nm)] = torch.zeros(K, device=dev, dtype=torch.float64)
            cnt[(i, nm)] = 0

            def hk(mod, inp, key=(i, nm)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                S[key] += x.T @ x; mu[key] += x.sum(0); cnt[key] += x.shape[0]
            hooks.append(lin.register_forward_pre_hook(hk))
    tr = np.memmap(a.train, dtype=np.uint16, mode="r")
    for s in np.random.default_rng(4321).integers(0, len(tr) - a.seq - 1, a.calib):
        x = torch.tensor(tr[s:s + a.seq + 1].astype(np.int64), device=dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    return {k: (S[k] / cnt[k], mu[k] / cnt[k]) for k in S}


R = {}
m0 = M.load(a.fp); R["fp"] = loss(m0)
st = stats(m0)
mean_share = [float((mu @ mu) / torch.trace(H)) for H, mu in st.values()]
R["mean_share_of_input_energy"] = float(np.mean(mean_share))
print(f"fp {R['fp']:.4f}; the input mean's share of E|x|^2, mean over linears {R['mean_share_of_input_energy']:.3f} "
      f"(max {max(mean_share):.3f})", flush=True)
del m0
for var in a.variants.split(","):
    m = M.load(a.fp)
    errs = []
    with torch.no_grad():
        names_ = [x for x in a.only.split(",") if x and "-" not in x]
        ranges_ = [tuple(map(int, x.split("-"))) for x in a.only.split(",") if "-" in x]
        for i, blk in enumerate(m.layers):
            for sub, nm in M.LIN:
                if (names_ and nm not in names_) or (ranges_ and not any(lo <= i <= hi for lo, hi in ranges_)):
                    continue
                lin = getattr(getattr(blk, sub), nm)
                W = lin.weight.data.float(); H2, mu = st[(i, nm)]
                C = (H2 - torch.outer(mu, mu)).float(); H2, mu = H2.float(), mu.float()
                T, ra, c = ternarize(W, "als")
                T, ra, c = ternarize_gptq(W, C if var == "centered" else H2, ra, c)
                Wh = ra[:, None] * c[None, :] * T.float()
                errs.append(_out_err(W, T, ra, c, H2))
                lin.weight.data = Wh
                if var in ("biascorr", "centered"):          # the mean's output error, corrected exactly by a bias
                    b = (W - Wh) @ mu
                    lin.register_forward_hook(lambda mod, inp, out, b=b: out + b.to(out.dtype))
    R[var] = {"loss": loss(m), "out_err_full_input": float(np.mean(errs))}
    print(f"{var}: loss {R[var]['loss']:.4f} (fp + {R[var]['loss'] - R['fp']:.4f}), output error on the full input "
          f"{R[var]['out_err_full_input']:.4f}", flush=True)
    del m; torch.cuda.empty_cache()
if a.out:
    json.dump(R, open(a.out, "w"), indent=1)
