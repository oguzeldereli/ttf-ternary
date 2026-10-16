"""Training-free GPTQ ternary variants on the 110M fp model (16 Oct): zero fraction, input rotation, calibration.

Each variant rounds every linear and reports the loss on --n_eval fixed validation windows (dequantised, float
activations), the realised zero fraction (by linear type) and the output-space error. A variant is NAME:key=val,...
  zf=0.23        GPTQ rounding threshold at that per-row quantile of |W / c| (ternarize_gptq zero_frac): fewer zeros,
                 the cut-off part of a large weight goes on to the other columns as rounding error does
  rot=down|out|all  a random orthogonal Q on the layer's input before rounding (exact in fp: W x = (W Q^T)(Q x)); the
                 rounded layer is W'_hat Q. down = w_down only, out = wo + w_down (inputs not on the residual stream),
                 all = every linear. Spreads each input's energy over all dimensions (float activations here, so the
                 int8 activation side of a rotation is not measured).
  cen=1          centred GPTQ (covariance, W mu as exact bias), as gptqc
  calib=N        calibration windows (default 32); damp=X GPTQ damping (default 0.01); several options joined by ||
  python scripts/analysis/gptq_variants.py --fp models/fp32_baseline_slim/ckpt.pt \
      --variants "plain:,zf23:zf=0.23,zf30:zf=0.30,rotdown:rot=down"
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
import models110 as M                                                              # noqa: E402
from bitnet.hf_init import ternarize, ternarize_gptq, _out_err                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fp", required=True); ap.add_argument("--train", default="data/wiki32k_train.bin")
ap.add_argument("--val", default="data/wiki32k_val.bin"); ap.add_argument("--n_eval", type=int, default=32)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--variants", required=True)
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
val = np.memmap(a.val, dtype=np.uint16, mode="r")
EV = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_eval)]


@torch.no_grad()
def loss(m):
    return float(np.mean([float(m(w.to(dev)[None][:, :-1], w.to(dev)[None][:, 1:])[1]) for w in EV]))


@torch.no_grad()
def stats(m, n):
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
    for s in np.random.default_rng(4321).integers(0, len(tr) - a.seq - 1, n):
        x = torch.tensor(tr[s:s + a.seq + 1].astype(np.int64), device=dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    return {k: ((S[k] / cnt[k]).float(), (mu[k] / cnt[k]).float()) for k in S}


def parse(v):
    name, _, kv = v.partition(":")
    o = {"zf": None, "rot": "none", "cen": 0, "calib": 32, "damp": 0.01}
    for p in filter(None, kv.split("|")):
        k, x = p.split("=")
        o[k] = x if k == "rot" else (int(x) if k in ("cen", "calib") else float(x))
    return name, o


ROT = {"none": set(), "down": {"w_down"}, "out": {"wo", "w_down"}, "all": {n for _, n in M.LIN}}
R = {}
m0 = M.load(a.fp); R["fp"] = loss(m0); print(f"fp {R['fp']:.4f}", flush=True)
ST = {}
for v in a.variants.split(","):
    name, o = parse(v)
    if o["calib"] not in ST:
        ST[o["calib"]] = stats(m0, o["calib"])
    st = ST[o["calib"]]
    m = M.load(a.fp)
    errs, zeros = [], {}
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for i, blk in enumerate(m.layers):
            for sub, nm in M.LIN:
                lin = getattr(getattr(blk, sub), nm)
                W = lin.weight.data.float(); H2, mu = st[(i, nm)]
                Q = None
                if nm in ROT[o["rot"]]:
                    Q = torch.linalg.qr(torch.randn(W.shape[1], W.shape[1], generator=g))[0].to(dev)
                    W, H2, mu = W @ Q.T, Q @ H2 @ Q.T, Q @ mu
                Hg = (H2 - torch.outer(mu, mu)) if o["cen"] else H2
                T, ra, c = ternarize(W, "als")
                T, ra, c = ternarize_gptq(W, Hg, ra, c, damp=o["damp"], zero_frac=o["zf"])
                Wh = ra[:, None] * c[None, :] * T.float()
                errs.append(_out_err(W, T, ra, c, H2))
                zeros.setdefault(nm, []).append(float((T == 0).float().mean()))
                b = (W - Wh) @ mu if o["cen"] else None
                lin.weight.data = Wh @ Q if Q is not None else Wh
                if b is not None:
                    lin.register_forward_hook(lambda mod, inp, out, b=b: out + b.to(out.dtype))
    R[name] = {"opts": o, "loss": loss(m), "out_err": float(np.mean(errs)),
               "zeros": {k: round(float(np.mean(x)), 3) for k, x in zeros.items()}}
    print(f"{name} {o}: loss {R[name]['loss']:.4f} (fp + {R[name]['loss'] - R['fp']:.4f}), output error "
          f"{R[name]['out_err']:.4f}, zeros {R[name]['zeros']}", flush=True)
    del m; torch.cuda.empty_cache()
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
