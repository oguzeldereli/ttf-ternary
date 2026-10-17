"""Do two 110M networks (full precision, TTF, a rounding of full precision) learn the same features?

  python scripts/analysis/feature_match.py --model fp=checkpoints/fp32_baseline/ckpt.pt \
      --model ttf0=checkpoints/final/ts16full_tau1000_s0/ckpt.pt --model ttf2=.../eval.pt --gptq_of fp

FFN neurons are a privileged basis (the SwiGLU product acts unit by unit), so each neuron of one network is matched to
the other's by activation correlation on the same tokens (inputs of w_down): per layer, the mean best |corr| (one to
many), the mean |corr| of a one-to-one assignment (Hungarian) and the fraction of neurons with a match above 0.5 / 0.8
(convergent learning, Li et al. 2015). The residual stream has no privileged basis (rotations are a symmetry), so it is
compared by linear CKA and by stitching: run network B up to block k, map its residual stream into A's by least
squares (fit on --n_fit windows, ridge 1e-3) and continue in A; the stitched loss against A's own (on other windows).
--gptq_of NAME adds NAME's GPTQ rounding (dequantised, float activations) as another model. Light: activations of
every 4th token, kept on the GPU in fp16; 4 CPU threads.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
torch.set_num_threads(4)
from ttf.model import BitTransformer                                          # noqa: E402
from ttf.hf_init import ternarize, ternarize_gptq                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help="NAME=PATH")
ap.add_argument("--gptq_of", default=""); ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--train", default="data/wiki32k_train.bin")
ap.add_argument("--n_act", type=int, default=16); ap.add_argument("--n_fit", type=int, default=8)
ap.add_argument("--n_eval", type=int, default=8); ap.add_argument("--stitch", default="3,6,9")
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--out", default="feature_match.json")
a = ap.parse_args()
dev = "cuda"
val = np.memmap(a.val, dtype=np.uint16, mode="r")
W = lambda i: torch.tensor(val[i * a.seq:(i + 1) * a.seq + 1].astype(np.int64))
ACT = [W(i) for i in range(a.n_act)]                                   # features
FIT = [W(a.n_act + i) for i in range(a.n_fit)]                         # stitch maps
EV = [W(a.n_act + a.n_fit + i) for i in range(a.n_eval)]               # stitched losses
LIN = [("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "wo"), ("ffn", "w_gate"), ("ffn", "w_up"), ("ffn", "w_down")]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if ck["mode"] == "fp32":
        m = BitTransformer(ck["cfg"], grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
        m.load_state_dict(ck["model"], strict=True)
        return m.eval()
    from final_eval import load_model
    return load_model(path, dev)[0].eval()


@torch.no_grad()
def gptq_copy(path):
    m = load(path)
    H, cnt, hooks = {}, {}, []
    for i, blk in enumerate(m.layers):
        for sub, n in LIN:
            lin = getattr(getattr(blk, sub), n)
            H[(i, n)] = torch.zeros(lin.in_features, lin.in_features, device=dev); cnt[(i, n)] = 0

            def hk(mod, inp, key=(i, n)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[key].addmm_(x.T, x); cnt[key] += x.shape[0]
            hooks.append(lin.register_forward_pre_hook(hk))
    tr = np.memmap(a.train, dtype=np.uint16, mode="r")
    for s in np.random.default_rng(4321).integers(0, len(tr) - a.seq - 1, 32):
        x = torch.tensor(tr[s:s + a.seq + 1].astype(np.int64), device=dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    for i, blk in enumerate(m.layers):
        for sub, n in LIN:
            lin = getattr(getattr(blk, sub), n); Wt = lin.weight.data.float()
            T, ra, c = ternarize(Wt, "als"); T, ra, c = ternarize_gptq(Wt, H[(i, n)] / cnt[(i, n)], ra, c)
            lin.weight.data = ra[:, None] * c[None, :] * T.float()
    del H
    return m


@torch.no_grad()
def acts(m, windows):
    """FFN products [L][T/4, h] and residual block inputs [L][T/4, d], fp16 on the GPU"""
    ffn, res, hooks = [[] for _ in m.layers], [[] for _ in m.layers], []
    for i, blk in enumerate(m.layers):
        hooks.append(blk.ffn.w_down.register_forward_pre_hook(
            lambda mod, inp, i=i: ffn[i].append(inp[0].reshape(-1, inp[0].shape[-1])[::4].half())))
        hooks.append(blk.register_forward_pre_hook(lambda mod, inp, i=i: res[i].append(inp[0][0, ::4].half())))
    for w in windows:
        x = w.to(dev)[None]; m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    return [torch.cat(f) for f in ffn], [torch.cat(r) for r in res]


def zs(X):
    X = X.float(); return (X - X.mean(0)) / X.std(0).clamp_min(1e-6)


def neuron_match(F1, F2):
    from scipy.optimize import linear_sum_assignment
    out = []
    for A, B in zip(F1, F2):
        C = (zs(A).T @ zs(B)) / A.shape[0]                      # [h1, h2]
        best = C.abs().max(1).values
        r, c = linear_sum_assignment(-C.abs().cpu().numpy())
        one = C.abs()[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)]
        out.append({"best": float(best.mean()), "hungarian": float(one.mean()), "gt05": float((best > 0.5).float().mean()),
                    "gt08": float((best > 0.8).float().mean())})
        del C
    return out


def cka(X, Y):
    X = X.float() - X.float().mean(0); Y = Y.float() - Y.float().mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


@torch.no_grad()
def stitched_loss(mB, mA, k, M, windows):
    """B up to block k, residual mapped by M into A, A from block k on"""
    ls = []
    for w in windows:
        x = w.to(dev)[None]; inp, tgt = x[:, :-1], x[:, 1:]
        h = mB.tok_emb(inp); fc = mB.freqs_cis.to(dev)
        for blk in mB.layers[:k]:
            h = blk(h, fc)
        h = (h.float() @ M).to(h.dtype)
        fa = mA.freqs_cis.to(dev)
        for blk in mA.layers[k:]:
            h = blk(h, fa)
        h = mA.norm(h)
        wh = mA.tok_emb.weight if mA.lm_head is None else mA.lm_head.weight
        ls.append(float(nn.functional.cross_entropy((h @ wh.T).float().view(-1, wh.shape[0]), tgt.reshape(-1))))
    return float(np.mean(ls))


@torch.no_grad()
def own_loss(m, windows):
    return float(np.mean([float(m(w.to(dev)[None][:, :-1], w.to(dev)[None][:, 1:])[1]) for w in windows]))


names, paths = [], {}
for s in a.model:
    n, p = s.split("=", 1); names.append(n); paths[n] = p
A_ffn, A_res, A_fit, losses = {}, {}, {}, {}
models = {}
for n in names + ([f"gptq({a.gptq_of})"] if a.gptq_of else []):
    m = gptq_copy(paths[a.gptq_of]) if n.startswith("gptq(") else load(paths[n])
    A_ffn[n], A_res[n] = acts(m, ACT)
    A_fit[n] = acts(m, FIT)[1]
    losses[n] = own_loss(m, EV)
    models[n] = m.cpu()                                          # back to the CPU between uses (small: 110M)
    torch.cuda.empty_cache()
    print(f"{n}: loss {losses[n]:.4f}", flush=True)
R = {"loss": losses, "pairs": {}}
ks = [int(k) for k in a.stitch.split(",")]
allm = list(A_ffn)
for x in range(len(allm)):
    for y in range(x + 1, len(allm)):
        p, q = allm[x], allm[y]
        nm = neuron_match(A_ffn[p], A_ffn[q])
        ck_ = [cka(u, v) for u, v in zip(A_res[p], A_res[q])]
        st = {}
        mp, mq = models[p].to(dev), models[q].to(dev)
        for k in ks:                                             # q's stream into p (q up to k, then p)
            X, Y = A_fit[q][k].float(), A_fit[p][k].float()
            M = torch.linalg.solve(X.T @ X + 1e-3 * torch.eye(X.shape[1], device=dev) * X.pow(2).mean() * X.shape[0], X.T @ Y)
            st[k] = stitched_loss(mq, mp, k, M, EV)
        models[p], models[q] = mp.cpu(), mq.cpu(); torch.cuda.empty_cache()
        R["pairs"][f"{p}~{q}"] = {"neurons": nm, "cka": ck_, "stitch_q_into_p": st}
        print(f"{p} ~ {q}: neuron best |corr| {np.mean([d['best'] for d in nm]):.3f}, one-to-one "
              f"{np.mean([d['hungarian'] for d in nm]):.3f}, >0.8 {np.mean([d['gt08'] for d in nm]):.3f}, "
              f"CKA {np.mean(ck_):.3f}; stitched ({q} -> {p}) " + " ".join(f"k{k} {v:.4f}" for k, v in st.items())
              + f" (own {losses[p]:.4f})", flush=True)
json.dump(R, open(a.out, "w"), indent=1)
