"""How do the trits a ternary network learns from scratch (TTF) relate to the trits obtained by rounding a full-precision
network of the same architecture, data and schedule?

  python scripts/analysis/trit_compare.py --fp checkpoints/fp32_baseline/ckpt.pt --ttf RUN/ckpt.pt [--ttf RUN2/ckpt.pt]

From the fp32 model: per-layer ternary roundings absmean (BitNet's), ALS (row and column scales) and GPTQ onto the ALS
grid (input second moments from --calib windows of the training data). For every trit set: loss on --n_eval fixed
validation windows (rounded fp models dequantised, float activations; TTF models through their own forward), the zero
fraction per layer, and pairwise agreement with chance levels. Since two separately trained networks are related only up
to the symmetries of the architecture, the TTF trits are also compared after aligning them to the fp model:
  - residual stream: one signed permutation for the whole network (RMSNorm gains, the tied embedding and every
    read / write of the residual commute with it), from the correlation of residual activations at all block inputs;
  - FFN hidden units: one permutation per block from the correlation of the SwiGLU products (gate and up rows, down
    columns move together); attention is left unaligned within heads (RoPE pairs), heads are not permuted.
Also linear CKA of the block outputs (no alignment needed) and sign(TTF trit) vs the fp weight it lands on.
Writes --out JSON.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from ttf.model import BitTransformer                                          # noqa: E402
from ttf.hf_init import ternarize, ternarize_gptq, _out_err                    # noqa: E402
from ttf.kernel import unpack_rows                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fp", required=True); ap.add_argument("--ttf", action="append", default=[])
ap.add_argument("--train", default="data/wiki32k_train.bin"); ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--calib", type=int, default=64); ap.add_argument("--n_eval", type=int, default=64)
ap.add_argument("--n_align", type=int, default=4, help="windows for the activation statistics")
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--out", default="trit_compare.json")
a = ap.parse_args()
dev = "cuda"
LIN = [("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "wo"), ("ffn", "w_gate"), ("ffn", "w_up"), ("ffn", "w_down")]
val = np.memmap(a.val, dtype=np.uint16, mode="r"); train = np.memmap(a.train, dtype=np.uint16, mode="r")
EVAL = [torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64)) for i in range(a.n_eval)]
rng = np.random.default_rng(4321)
CAL = [torch.tensor(train[s:s + a.seq + 1].astype(np.int64)) for s in rng.integers(0, len(train) - a.seq - 1, a.calib)]


def fp_model(sd, cfg):
    m = BitTransformer(cfg, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
    m.load_state_dict({k: v for k, v in sd.items()}, strict=True)
    return m.eval()


@torch.no_grad()
def loss(m, windows=EVAL):
    ls = []
    for w in windows:
        x = w.to(dev)[None]
        out = m(x[:, :-1], x[:, 1:])
        ls.append(float(out[1] if isinstance(out, tuple) else out))
    return float(np.mean(ls))


@torch.no_grad()
def trace(m, windows, want_ffn=True):
    """residual activations at every block input [L, T, d] and the SwiGLU products [L, T, h] (concatenated windows)"""
    res, ffn, hooks = [[] for _ in m.layers], [[] for _ in m.layers], []
    for i, blk in enumerate(m.layers):
        hooks.append(blk.register_forward_pre_hook(lambda mod, inp, i=i: res[i].append(inp[0][0].float().cpu())))
        if want_ffn:
            hooks.append(blk.ffn.w_down.register_forward_pre_hook(
                lambda mod, inp, i=i: ffn[i].append(inp[0].reshape(-1, inp[0].shape[-1]).float().cpu())))
    outs = [[] for _ in m.layers]
    for i, blk in enumerate(m.layers):
        hooks.append(blk.register_forward_hook(
            lambda mod, inp, out, i=i: outs[i].append((out[0] if isinstance(out, tuple) else out)[0].float().cpu())))
    for w in windows:
        x = w.to(dev)[None]
        m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    return ([torch.cat(r) for r in res], [torch.cat(f) for f in ffn] if want_ffn else None, [torch.cat(o) for o in outs])


def corr(A, B):
    A = (A - A.mean(0)) / A.std(0).clamp_min(1e-8); B = (B - B.mean(0)) / B.std(0).clamp_min(1e-8)
    return (A.T @ B) / A.shape[0]


def match(C):
    """permutation maximising |C| (rows: fp units, cols: other units) -> perm, sign"""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-C.abs().numpy())
    return torch.as_tensor(c), torch.sign(C[torch.as_tensor(r), torch.as_tensor(c)])


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def agree(T1, T2):
    eq = float((T1 == T2).float().mean())
    p = [float((T1 == v).float().mean()) for v in (-1, 0, 1)]; q = [float((T2 == v).float().mean()) for v in (-1, 0, 1)]
    nz = (T1 != 0) & (T2 != 0)
    sgn = float((T1[nz] == T2[nz]).float().mean()) if nz.any() else float("nan")
    return {"agree": eq, "chance": sum(x * y for x, y in zip(p, q)), "sign_agree_nz": sgn}


# ---- the fp32 model and its roundings
ck = torch.load(a.fp, map_location="cpu", weights_only=False, mmap=True)
assert ck["mode"] == "fp32", ck["mode"]
cfg, sd = ck["cfg"], ck["model"]
fpm = fp_model(sd, cfg)
R = {"fp": {"loss": loss(fpm)}}
print(f"fp32 loss {R['fp']['loss']:.4f} on {a.n_eval} windows", flush=True)
H, hooks, cnt = {}, [], {}
for i, blk in enumerate(fpm.layers):
    for sub, name in LIN:
        lin = getattr(getattr(blk, sub), name)
        H[(i, name)] = torch.zeros(lin.in_features, lin.in_features, device=dev); cnt[(i, name)] = 0

        def hk(mod, inp, key=(i, name)):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[key].addmm_(x.T, x); cnt[key] += x.shape[0]
        hooks.append(lin.register_forward_pre_hook(hk))
with torch.no_grad():
    for w in CAL:
        x = w.to(dev)[None]; fpm(x[:, :-1], x[:, 1:])
for h in hooks:
    h.remove()
TR = {k: {} for k in ("absmean", "als", "gptq")}
oerr = {k: [] for k in TR}
for i, blk in enumerate(fpm.layers):
    for sub, name in LIN:
        W = getattr(getattr(blk, sub), name).weight.data.float()
        Hk = H[(i, name)] / cnt[(i, name)]
        for meth in ("absmean", "als"):
            T, ra, c = ternarize(W, meth); TR[meth][(i, name)] = (T.cpu(), ra.cpu(), c.cpu())
        T0, ra, c = TR["als"][(i, name)]
        Tg, ag, cg = ternarize_gptq(W, Hk, ra.to(dev), c.to(dev)); TR["gptq"][(i, name)] = (Tg.cpu(), ag.cpu(), cg.cpu())
        for meth in TR:
            T_, a_, c_ = TR[meth][(i, name)]
            oerr[meth].append(_out_err(W, T_.to(dev), a_.to(dev), c_.to(dev), Hk))
for meth in TR:
    m2 = fp_model(sd, cfg)
    for i, blk in enumerate(m2.layers):
        for sub, name in LIN:
            T_, a_, c_ = TR[meth][(i, name)]
            getattr(getattr(blk, sub), name).weight.data = (a_[:, None] * c_[None, :] * T_.float()).to(dev)
    R[meth] = {"loss": loss(m2), "out_err_mean": float(np.mean(oerr[meth]))}
    print(f"{meth}: loss {R[meth]['loss']:.4f}, mean output-space error {R[meth]['out_err_mean']:.4f}", flush=True)
    del m2
torch.cuda.empty_cache()
R["zeros"] = {meth: float(np.mean([float((T == 0).float().mean()) for T, _, _ in TR[meth].values()])) for meth in TR}
R["pairs"] = {}
for m1, m2 in (("absmean", "als"), ("als", "gptq"), ("absmean", "gptq")):
    R["pairs"][f"{m1}~{m2}"] = {k: float(np.mean([agree(TR[m1][key][0], TR[m2][key][0])[k] for key in TR[m1]]))
                               for k in ("agree", "chance", "sign_agree_nz")}
print("zero fraction:", R["zeros"], "\npairs:", R["pairs"], flush=True)

# ---- TTF models
from final_eval import load_model                                                  # noqa: E402
fres, fffn, fout = trace(fpm, EVAL[:a.n_align])
for path in a.ttf:
    tm = load_model(path, dev)[0].eval()
    name = os.path.basename(os.path.dirname(path))
    r = {"loss": loss(tm)}
    tres, tffn, tout = trace(tm, EVAL[:a.n_align])
    r["cka_block_out"] = [cka(x, y) for x, y in zip(fout, tout)]
    # residual: one signed permutation for the whole stack
    C = sum(corr(x, y) for x, y in zip(fres, tres)) / len(fres)
    pr, sr = match(C)
    r["res_match_corr"] = float(C[torch.arange(C.shape[0]), pr].abs().mean())
    raw, aligned, wsign, zeros = [], [], [], []
    for i, blk in enumerate(tm.layers):
        ph, sh = match(corr(fffn[i], tffn[i]))
        for sub, nm in LIN:
            lin = getattr(getattr(blk, sub), nm)
            T = unpack_rows(lin.wpacked, lin.K).cpu().to(torch.int8)
            Tg = TR["gptq"][(i, nm)][0]
            Wf = getattr(getattr(fpm.layers[i], sub), nm).weight.data.float().cpu()
            zeros.append(float((T == 0).float().mean()))
            raw.append(agree(Tg, T))
            # align: rows / columns that live in the residual stream take (pr, sr); FFN hidden rows / columns take ph
            Ta = T.clone()
            if nm in ("wq", "wk", "wv", "w_gate", "w_up"):           # reads the residual: columns
                Ta = Ta[:, pr] * sr.to(torch.int8)[None, :]
            if nm in ("wo", "w_down"):                                # writes the residual: rows
                Ta = Ta[pr, :] * sr.to(torch.int8)[:, None]
            if nm in ("w_gate", "w_up"):
                Ta = Ta[ph, :] * (sh.to(torch.int8)[:, None] if nm == "w_up" else 1)
            if nm == "w_down":
                Ta = Ta[:, ph]
            aligned.append(agree(Tg, Ta))
            nz = Ta != 0
            wsign.append(float((torch.sign(Wf[nz]) == Ta[nz].float()).float().mean()))
    for k, lst in (("raw_vs_gptq", raw), ("aligned_vs_gptq", aligned)):
        r[k] = {q: float(np.mean([d[q] for d in lst])) for q in ("agree", "chance", "sign_agree_nz")}
    r["zeros"] = float(np.mean(zeros)); r["aligned_sign_vs_fp_weight"] = float(np.mean(wsign))
    r["_trits"] = [unpack_rows(getattr(getattr(blk, sub), nm).wpacked, getattr(getattr(blk, sub), nm).K).cpu().to(torch.int8)
                   for blk in tm.layers for sub, nm in LIN]
    R[name] = r
    print(f"{name}: loss {r['loss']:.4f}, zeros {r['zeros']:.3f}, residual match |corr| {r['res_match_corr']:.3f}, "
          f"CKA {np.mean(r['cka_block_out']):.3f}\n  raw vs gptq {r['raw_vs_gptq']}\n  aligned vs gptq {r['aligned_vs_gptq']}"
          f"\n  aligned sign vs fp weight {r['aligned_sign_vs_fp_weight']:.3f}", flush=True)
    del tm
    torch.cuda.empty_cache()
names = [n for n in R if isinstance(R[n], dict) and "_trits" in R[n]]
for x in range(len(names)):                      # TTF runs against each other (raw: different seeds share nothing)
    for y in range(x + 1, len(names)):
        d = [agree(t1, t2) for t1, t2 in zip(R[names[x]]["_trits"], R[names[y]]["_trits"])]
        R["pairs"][f"{names[x]}~{names[y]}"] = {q: float(np.mean([e[q] for e in d])) for q in ("agree", "chance", "sign_agree_nz")}
        print(f"{names[x]} ~ {names[y]}: {R['pairs'][f'{names[x]}~{names[y]}']}", flush=True)
for n in names:
    del R[n]["_trits"]
json.dump(R, open(a.out, "w"), indent=1)
