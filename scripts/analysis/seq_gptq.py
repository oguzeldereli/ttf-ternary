"""Sequential, error-aware ternary GPTQ (110M test, training-free).

Plain GPTQ calibrates every layer on the inputs x of the clean fp model. In the converted model a layer receives the
already-rounded network's inputs x_hat, so errors pile up towards the top (on 110M the last third of the blocks carries
half the rounding loss). Here the linears are rounded in network order (per block: q k v, then o, then gate up, then
down), each calibrated on x_hat from the partly converted model, centred (input mean out, W mu as a float bias), with
the target either
  sym    the layer's own weight W on x_hat:              min |W_hat (x_hat - mu_hat) - W (x_hat - mu_hat)|
  asym   the fp layer's output on the clean input:       min |W_hat (x_hat - mu_hat) + b - W x|
         = GPTQ of W* = W C(x, x_hat) C(x_hat, x_hat)^-1 on C(x_hat, x_hat), b = W mu (the layer repairs error from below)
Losses on --n_eval fixed windows (dequantised, float activations), against the one-shot centred GPTQ.
  python scripts/analysis/seq_gptq.py --fp checkpoints/fp32_baseline/ckpt.pt --variants sym,asym
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
import models110 as M                                                              # noqa: E402
from bitnet.hf_init import ternarize, ternarize_gptq                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fp", required=True); ap.add_argument("--train", default="data/wiki32k_train.bin")
ap.add_argument("--val", default="data/wiki32k_val.bin"); ap.add_argument("--calib", type=int, default=32)
ap.add_argument("--n_eval", type=int, default=32); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--variants", default="sym,asym"); ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
val = np.memmap(a.val, dtype=np.uint16, mode="r"); tr = np.memmap(a.train, dtype=np.uint16, mode="r")
EV = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_eval)]
CAL = [torch.tensor(tr[s:s + a.seq + 1].astype(np.int64)) for s in
       np.random.default_rng(4321).integers(0, len(tr) - a.seq - 1, a.calib)]
STAGES = [[("attn", "wq"), ("attn", "wk"), ("attn", "wv")], [("attn", "wo")], [("ffn", "w_gate"), ("ffn", "w_up")],
          [("ffn", "w_down")]]


@torch.no_grad()
def loss(m):
    return float(np.mean([float(m(w.to(dev)[None][:, :-1], w.to(dev)[None][:, 1:])[1]) for w in EV]))


@torch.no_grad()
def moments(clean, q, i, stage):
    """for the linears of one stage of block i: E[xh xh^T], E[x xh^T], E[xh], E[x] (x clean, xh from q), same tokens"""
    names = [nm for _, nm in stage]
    sub = lambda nm: "attn" if nm in ("wq", "wk", "wv", "wo") else "ffn"
    keys = {}
    for nm in names:
        K = getattr(getattr(clean.layers[i], sub(nm)), nm).in_features
        keys[nm] = dict(Hh=torch.zeros(K, K, device=dev, dtype=torch.float64), Hx=torch.zeros(K, K, device=dev, dtype=torch.float64),
                        mh=torch.zeros(K, device=dev, dtype=torch.float64), mx=torch.zeros(K, device=dev, dtype=torch.float64), n=0)
    buf = {}
    hc = [getattr(getattr(clean.layers[i], sub(nm)), nm).register_forward_pre_hook(
        lambda mod, inp, nm=nm: buf.__setitem__(nm, inp[0].reshape(-1, inp[0].shape[-1]).double())) for nm in names]

    def qhook(mod, inp, nm):
        xh = inp[0].reshape(-1, inp[0].shape[-1]).double(); x = buf[nm]; d = keys[nm]
        d["Hh"] += xh.T @ xh; d["Hx"] += x.T @ xh; d["mh"] += xh.sum(0); d["mx"] += x.sum(0); d["n"] += xh.shape[0]
    hq = [getattr(getattr(q.layers[i], sub(nm)), nm).register_forward_pre_hook(lambda mod, inp, nm=nm: qhook(mod, inp, nm))
          for nm in names]
    for w in CAL:
        x = w.to(dev)[None]
        clean(x[:, :-1], x[:, 1:]); q(x[:, :-1], x[:, 1:])
    for h in hc + hq:
        h.remove()
    out = {}
    for nm, d in keys.items():
        n = d["n"]; mh, mx = d["mh"] / n, d["mx"] / n
        out[nm] = (d["Hh"] / n - torch.outer(mh, mh), d["Hx"] / n - torch.outer(mx, mh), mh, mx)
    return out


R = {}
clean = M.load(a.fp); R["fp"] = loss(clean)
print(f"fp {R['fp']:.4f}", flush=True)
for var in a.variants.split(","):
    q = M.load(a.fp)
    for i in range(len(q.layers)):
        for stage in STAGES:
            mo = moments(clean, q, i, stage)
            for sub, nm in stage:
                lin = getattr(getattr(q.layers[i], sub), nm)
                W = getattr(getattr(clean.layers[i], sub), nm).weight.data.double()
                Ch, Cxh, mh, mx = mo[nm]
                Cr = Ch + a.damp * torch.diagonal(Ch).mean() * torch.eye(Ch.shape[0], device=dev, dtype=Ch.dtype)
                Wt = (W @ Cxh @ torch.linalg.inv(Cr)) if var == "asym" else W     # the target weight on x_hat
                Wt = Wt.float(); Chf = Ch.float()
                T, ra, c = ternarize(Wt, "als")
                T, ra, c = ternarize_gptq(Wt, Chf, ra, c)
                Wh = ra[:, None] * c[None, :] * T.float()
                b = (W @ mx).float() if var == "asym" else (W @ mh).float()  # asym: the clean output's mean
                lin.weight.data = Wh
                lin._mu = mh.float()
                lin.register_forward_pre_hook(lambda mod, inp: (inp[0] - mod._mu.to(inp[0].dtype),))
                lin.register_forward_hook(lambda mod, inp, out, b=b: out + b.to(out.dtype))
        print(f"  {var}: block {i} done", flush=True)
    R[var] = loss(q)
    print(f"{var}: loss {R[var]:.4f} (fp + {R[var] - R['fp']:.4f})", flush=True)
    del q; torch.cuda.empty_cache()
if a.out:
    json.dump(R, open(a.out, "w"), indent=1)
