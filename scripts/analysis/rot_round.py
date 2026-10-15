"""Does rounding the full-precision model in a different residual basis give better ternary weights?

With the RMSNorm gains folded into the weights that read the normalised stream (and the tied head untied as E diag(g)),
an orthogonal Q on the residual stream is an exact symmetry: embedding E Q, head H Q, readers (wq wk wv w_gate w_up)
W Q, writers (wo w_down) Q^T W. The rotated fp model computes the same function (checked), but its weights round
differently. Rotations tried, each followed by GPTQ (input second moments from the rotated model) on every linear:
  identity  the gains folded, no rotation (the fold alone)
  random    a random orthogonal Q (QuaRot-like incoherence)
  ttf       orthogonal Procrustes from the fp residual stream onto a TTF model's (block inputs, same tokens): the basis
            the ternary network chose
Reports loss (dequantised, float activations, --n_eval fixed windows), output-space error, zero fraction, and for
the ttf rotation the trit agreement with the TTF model on the residual-side axes (FFN hidden units matched by
activation correlation; attention heads unaligned).

  python scripts/analysis/rot_round.py --fp checkpoints/fp32_baseline/ckpt.pt --ttf RUN/ckpt.pt
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from bitnet.model import BitTransformer                                          # noqa: E402
from bitnet.hf_init import ternarize, ternarize_gptq, _out_err                    # noqa: E402
from bitnet.kernel import unpack_rows                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fp", required=True); ap.add_argument("--ttf", required=True)
ap.add_argument("--train", default="data/wiki32k_train.bin"); ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--calib", type=int, default=32); ap.add_argument("--n_eval", type=int, default=32)
ap.add_argument("--n_align", type=int, default=4); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="rot_round.json")
a = ap.parse_args()
dev = "cuda"
torch.set_num_threads(4)                          # leave the machine usable (laptop)
READ = [("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("ffn", "w_gate"), ("ffn", "w_up")]
WRITE = [("attn", "wo"), ("ffn", "w_down")]
LIN = READ[:3] + WRITE[:1] + READ[3:] + WRITE[1:]
val = np.memmap(a.val, dtype=np.uint16, mode="r"); train = np.memmap(a.train, dtype=np.uint16, mode="r")
EVAL = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_eval)]
rng = np.random.default_rng(4321)
CAL = [torch.tensor(train[s:s + a.seq + 1].astype(np.int64)) for s in rng.integers(0, len(train) - a.seq - 1, a.calib)]

ck = torch.load(a.fp, map_location="cpu", weights_only=False, mmap=True)
cfg, sd0 = ck["cfg"], ck["model"]
del ck["opt"]                                    # the optimizer state is not needed


def build(sd, untie):
    import dataclasses
    c = dataclasses.replace(cfg, tie_embeddings=not untie)
    m = BitTransformer(c, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
    m.load_state_dict(sd, strict=True)
    return m.eval()


@torch.no_grad()
def loss(m):
    ls = []
    for w in EVAL:
        x = w.to(dev)[None]
        ls.append(float(m(x[:, :-1], x[:, 1:])[1]))
    return float(np.mean(ls))


def folded(sd):
    """gains folded into the readers, head untied as E diag(g_final); all gains 1"""
    s = {k: v.clone().float() for k, v in sd.items()}
    for i in range(cfg.n_layers):
        g1, g2 = s[f"layers.{i}.attn_norm.weight"], s[f"layers.{i}.ffn_norm.weight"]
        for n in ("wq", "wk", "wv"):
            s[f"layers.{i}.attn.{n}.weight"] *= g1[None, :]
        for n in ("w_gate", "w_up"):
            s[f"layers.{i}.ffn.{n}.weight"] *= g2[None, :]
        s[f"layers.{i}.attn_norm.weight"] = torch.ones_like(g1); s[f"layers.{i}.ffn_norm.weight"] = torch.ones_like(g2)
    s["lm_head.weight"] = s["tok_emb.weight"] * s["norm.weight"][None, :]
    s["norm.weight"] = torch.ones_like(s["norm.weight"])
    return s


def rotated(s, Q):
    r = dict(s)
    r["tok_emb.weight"] = s["tok_emb.weight"] @ Q; r["lm_head.weight"] = s["lm_head.weight"] @ Q
    for i in range(cfg.n_layers):
        for sub, n in READ:
            r[f"layers.{i}.{sub}.{n}.weight"] = s[f"layers.{i}.{sub}.{n}.weight"] @ Q
        for sub, n in WRITE:
            r[f"layers.{i}.{sub}.{n}.weight"] = Q.T @ s[f"layers.{i}.{sub}.{n}.weight"]
    return r


@torch.no_grad()
def residuals(m, windows):
    res, ffn, hooks = [[] for _ in m.layers], [[] for _ in m.layers], []
    for i, blk in enumerate(m.layers):
        hooks.append(blk.register_forward_pre_hook(lambda mod, inp, i=i: res[i].append(inp[0][0, ::4].float())))
        hooks.append(blk.ffn.w_down.register_forward_pre_hook(       # every 4th token, kept on the GPU
            lambda mod, inp, i=i: ffn[i].append(inp[0].reshape(-1, inp[0].shape[-1])[::4].float())))
    for w in windows:
        x = w.to(dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    return [torch.cat(r) for r in res], [torch.cat(f) for f in ffn]


@torch.no_grad()
def gptq_round(m):
    H, cnt, hooks = {}, {}, []
    for i, blk in enumerate(m.layers):
        for sub, n in LIN:
            lin = getattr(getattr(blk, sub), n)
            H[(i, n)] = torch.zeros(lin.in_features, lin.in_features, device=dev); cnt[(i, n)] = 0

            def hk(mod, inp, key=(i, n)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[key].addmm_(x.T, x); cnt[key] += x.shape[0]
            hooks.append(lin.register_forward_pre_hook(hk))
    for w in CAL:
        x = w.to(dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    trits, errs, zeros = {}, [], []
    for i, blk in enumerate(m.layers):
        for sub, n in LIN:
            lin = getattr(getattr(blk, sub), n)
            W = lin.weight.data.float(); Hk = H[(i, n)] / cnt[(i, n)]
            T, ra, c = ternarize(W, "als")
            T, ra, c = ternarize_gptq(W, Hk, ra, c)
            errs.append(_out_err(W, T, ra, c, Hk)); zeros.append(float((T == 0).float().mean()))
            trits[(i, n)] = T.cpu()
            lin.weight.data = ra[:, None] * c[None, :] * T.float()
    return trits, float(np.mean(errs)), float(np.mean(zeros))


def agree(T1, T2):
    eq = float((T1 == T2).float().mean())
    p = [float((T1 == v).float().mean()) for v in (-1, 0, 1)]; q = [float((T2 == v).float().mean()) for v in (-1, 0, 1)]
    nz = (T1 != 0) & (T2 != 0)
    return eq, sum(x * y for x, y in zip(p, q)), float((T1[nz] == T2[nz]).float().mean())


R = {}
base = build(sd0, untie=False); R["fp"] = loss(base); del base
fs = folded(sd0)
fm = build(fs, untie=True); R["fp_folded"] = loss(fm)
print(f"fp {R['fp']:.4f}, folded {R['fp_folded']:.4f}", flush=True)
from final_eval import load_model                                                  # noqa: E402
tm = load_model(a.ttf, dev)[0].eval()
R["ttf_model"] = loss(tm)
fres, fffn = residuals(fm, EVAL[:a.n_align]); tres, tffn = residuals(tm, EVAL[:a.n_align])
X = torch.cat(fres); Y = torch.cat(tres)
X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8); Y = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-8)
U, S, Vh = torch.linalg.svd(X.T @ Y)
Q_ttf = (U @ Vh).float().cpu()                              # argmin |X Q - Y|, Q orthogonal
R["procrustes_fit"] = float(1 - (X @ Q_ttf.to(dev) - Y).pow(2).sum() / (X.pow(2).sum() + Y.pow(2).sum()))
del X, Y, fres, tres, U, S, Vh
g = torch.Generator().manual_seed(a.seed)
Q_rand, _ = torch.linalg.qr(torch.randn(cfg.dim, cfg.dim, generator=g))
T_ttf = {(i, n): unpack_rows(getattr(getattr(blk, sub), n).wpacked, getattr(getattr(blk, sub), n).K).cpu().to(torch.int8)
         for i, blk in enumerate(tm.layers) for sub, n in LIN}
del tm; torch.cuda.empty_cache()
for name, Q in (("identity", torch.eye(cfg.dim)), ("random", Q_rand), ("ttf", Q_ttf)):
    rs = rotated(fs, Q)
    m = build(rs, untie=True)
    l_exact = loss(m)
    trits, oe, z = gptq_round(m)
    R[name] = {"loss_before_rounding": l_exact, "loss": loss(m), "out_err": oe, "zeros": z}
    if name == "ttf":       # residual axes now shared; FFN hidden matched by activation correlation
        from scipy.optimize import linear_sum_assignment
        mr = build(rs, untie=True); rres, rffn = residuals(mr, EVAL[:a.n_align]); del mr, rres
        ags = []
        for i in range(cfg.n_layers):
            A_ = (rffn[i] - rffn[i].mean(0)) / rffn[i].std(0).clamp_min(1e-8)
            B_ = (tffn[i] - tffn[i].mean(0)) / tffn[i].std(0).clamp_min(1e-8)
            C = (A_.T @ B_) / A_.shape[0]
            _, ph = linear_sum_assignment(-C.abs().cpu().numpy()); ph = torch.as_tensor(ph)
            for sub, n in LIN:
                Tt = T_ttf[(i, n)]
                if n in ("w_gate", "w_up"):
                    Tt = Tt[ph, :]
                if n == "w_down":
                    Tt = Tt[:, ph]
                ags.append(agree(trits[(i, n)], Tt))
        R[name]["agree_with_ttf"] = [float(np.mean([x[k] for x in ags])) for k in range(3)]
    print(f"{name}: exact {l_exact:.4f} -> GPTQ {R[name]['loss']:.4f}, out err {oe:.4f}, zeros {z:.3f}"
          + (f", agreement with TTF (agree, chance, sign) {R[name]['agree_with_ttf']}" if name == "ttf" else ""), flush=True)
    del m, rs; torch.cuda.empty_cache()
print(f"TTF {R["ttf_model"]:.4f}; Procrustes fit {R['procrustes_fit']:.3f}")
json.dump(R, open(a.out, "w"), indent=1)
