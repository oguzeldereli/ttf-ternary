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
  src=ones|lam=L  no data: inputs assumed to share one common part, H = I + (L / K) 1 1^T (error diffusion that keeps
                 each row's sum; the user's 16 Oct proposal)
  src=train|random|self|wprior|eye  calibration source: training windows, uniform random tokens, text sampled from
                 the fp model itself, input covariances derived from the weights alone (no data), identity (no data)
  slen=L         length of the self-generated windows (default --seq; sampling recomputes the prefix, so short is fast)
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
def windows(m, n, src, slen=0):
    """calibration token windows: train (random windows of --train), random (uniform token ids), self (sampled from
    the fp model itself, temperature 1, from BOS: no external data)"""
    if src == "train":
        tr = np.memmap(a.train, dtype=np.uint16, mode="r")
        return [torch.tensor(tr[s:s + a.seq + 1].astype(np.int64), device=dev)[None]
                for s in np.random.default_rng(4321).integers(0, len(tr) - a.seq - 1, n)]
    V = m.tok_emb.weight.shape[0]
    g = torch.Generator(device=dev).manual_seed(4321)
    if src == "random":
        return [torch.randint(0, V, (1, a.seq + 1), generator=g, device=dev) for _ in range(n)]
    out = []
    for b0 in range(0, n, 16):          # self-generation, no KV cache (110M: minutes)
        nb = min(16, n - b0)
        x = torch.full((nb, 1), 1, dtype=torch.long, device=dev)
        while x.shape[1] < (slen or a.seq) + 1:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lg = m(x[:, -a.seq:])[0][:, -1].float()
            x = torch.cat([x, torch.multinomial(torch.softmax(lg, -1), 1, generator=g)], 1)
        out += [x[i:i + 1] for i in range(nb)]
    return out


@torch.no_grad()
def wprior(m):
    """data-free input covariances from the weights alone: the residual stream starts as the embedding table's
    covariance (uniform over the vocabulary) and gains W_o C W_o^T and W_down C W_down^T per block (sources taken as
    independent); each norm rescales it to unit mean variance and applies its gain; attention outputs are taken as
    mixtures of values (C_v = W_v C_x W_v^T, repeated per kv group); the FFN hidden as silu(g) u with silu(g) ~ g / 2
    (Isserlis: C_d = (C_gg o C_uu + C_gu o C_ug) / 4)"""
    E = m.tok_emb.weight.float()
    Ch = (E - E.mean(0)).T @ (E - E.mean(0)) / E.shape[0]
    out = {}
    def normed(C, gw):
        g_ = gw.float(); return (C / (torch.trace(C) / C.shape[0])) * g_[:, None] * g_[None, :]
    z = lambda K: torch.zeros(K, device=dev)
    for i, blk in enumerate(m.layers):
        at, ff = blk.attn, blk.ffn
        Cx = normed(Ch, blk.attn_norm.weight)
        for nm in ("wq", "wk", "wv"):
            out[(i, nm)] = (Cx, z(Cx.shape[0]))
        Wv = at.wv.weight.float(); Cv = Wv @ Cx @ Wv.T
        hd, rep = at.head_dim, at.rep
        idx = torch.arange(at.n_heads * hd, device=dev); kv = (idx // hd) // rep * hd + idx % hd
        Co = Cv[kv][:, kv]
        out[(i, "wo")] = (Co, z(Co.shape[0]))
        Wo = at.wo.weight.float(); Ch = Ch + Wo @ Co @ Wo.T
        Cx2 = normed(Ch, blk.ffn_norm.weight)
        for nm in ("w_gate", "w_up"):
            out[(i, nm)] = (Cx2, z(Cx2.shape[0]))
        Wg, Wu = ff.w_gate.weight.float(), ff.w_up.weight.float()
        Cgg, Cuu, Cgu = Wg @ Cx2 @ Wg.T, Wu @ Cx2 @ Wu.T, Wg @ Cx2 @ Wu.T
        Cd = (Cgg * Cuu + Cgu * Cgu.T) / 4
        out[(i, "w_down")] = (Cd, z(Cd.shape[0]))
        Wd = ff.w_down.weight.float(); Ch = Ch + Wd @ Cd @ Wd.T
    return out


@torch.no_grad()
def stats(m, n, src="train", slen=0):
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
    for x in windows(m, n, src, slen):
        m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    return {k: ((S[k] / cnt[k]).float(), (mu[k] / cnt[k]).float()) for k in S}


def parse(v):
    name, _, kv = v.partition(":")
    o = {"zf": None, "rot": "none", "cen": 0, "calib": 32, "damp": 0.01, "src": "train", "lam": 1.0, "slen": 0}
    for p in filter(None, kv.split("|")):
        k, x = p.split("=")
        o[k] = x if k in ("rot", "src") else (int(x) if k in ("cen", "calib") else float(x))
    return name, o


ROT = {"none": set(), "down": {"w_down"}, "out": {"wo", "w_down"}, "all": {n for _, n in M.LIN}}
R = {}
m0 = M.load(a.fp); R["fp"] = loss(m0); print(f"fp {R['fp']:.4f}", flush=True)
ST = {}
for v in a.variants.split(","):
    name, o = parse(v)
    key = (o["calib"], o["src"], o["lam"], o["slen"])
    if key not in ST:
        if o["src"] == "wprior":
            ST[key] = wprior(m0)
        elif o["src"] == "ones":   # data-free error diffusion: inputs assumed to share one common part (H = I + lam 1 1^T),
            # so GPTQ rounds each weight and pushes the sign-flipped difference evenly onto the rest of the row (keeps
            # the row sum, the more so the larger lam: lam = K times the common part's share)
            ST[key] = {k: (torch.eye(v[0].shape[0], device=dev) + o["lam"] * torch.ones(v[0].shape[0], v[0].shape[0], device=dev)
                           / v[0].shape[0], torch.zeros(v[0].shape[0], device=dev)) for k, v in wprior(m0).items()}
        elif o["src"] == "eye":
            ST[key] = {k: (torch.eye(v[0].shape[0], device=dev), torch.zeros(v[0].shape[0], device=dev))
                       for k, v in wprior(m0).items()}
        else:
            ST[key] = stats(m0, o["calib"], o["src"], int(o["slen"]))
    st = ST[key]
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
