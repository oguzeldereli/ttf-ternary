"""How does master move, compared with the summed gradient and with stateless flips?

Ternary MLP LM. Train master (BitNet STE + AdamW) and stateless flips (r = 0.02) to S0 steps,
then over windows of N steps record, per layer, cosine similarities (averaged over layers):
  master: latent displacement vs -sum(g) | trit change vs -sum(g) | trit change vs latent
          displacement | one Adam step vs -g of that step
  flips:  trit change vs -sum(g) | one step's flips vs -g of that step
sum(g) is the sum of the gradients each run saw along its own trajectory (the "added
gradient vector"). Trit changes are in trit units; cosines are scale-free within a layer.

  python -m scripts.mlp.geometry
"""
import math, os, numpy as np, torch, torch.nn.functional as F
from ttf.master import MasterTernaryLinear
from ttf.layers import KernelTernaryLinear, set_flip_rate
from ttf.kernel import unpack_rows
from scripts.mlp.mlp_lab import MLPLM, windows

dev, BATCH, LR, WARM, TOTAL, RATE, S0 = "cuda", 4096, 1.5e-3, 100, 6000, 0.02, 300
EVERY = 200
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0).item()


def lr_at(s):
    if s < WARM: return LR * (s + 1) / WARM
    return 1.5e-4 + 0.5 * (LR - 1.5e-4) * (1 + math.cos(math.pi * (s - WARM) / (TOTAL - WARM)))


def rate_at(s):
    if s < 10: return RATE * s / 10
    return RATE * 0.5 * (1 + math.cos(math.pi * (s - 10) / (TOTAL - 10)))


def run(mode):
    torch.manual_seed(0)
    mk = (lambda i, o: MasterTernaryLinear(i, o)) if mode == "master" else \
         (lambda i, o: KernelTernaryLinear(i, o, rate=RATE, g_ref=3.0, beta=True, int8=True, dw_mode="dense"))
    m = MLPLM(mk).to(dev).train()
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in m.named_parameters() if "norm" not in n], "weight_decay": 0.1},
         {"params": [p for n, p in m.named_parameters() if "norm" in n], "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95))
    Ls = [l for l in m.modules() if isinstance(l, (MasterTernaryLinear, KernelTernaryLinear))]
    gen = torch.Generator().manual_seed(1234)
    gla = torch.Generator().manual_seed(4242)

    def trits():
        if mode == "master":
            out = []
            for l in Ls:
                w = l.weight.detach().float(); gm = w.abs().mean().clamp_min(1e-5)
                out.append((w / gm).round().clamp(-1, 1))
            return out
        return [unpack_rows(l.wpacked, l.K).float() for l in Ls]

    def one_step(s, record):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        if mode != "master": set_flip_rate(m, rate_at(s))
        for l in Ls:
            if mode != "master": l.capture = record         # gradient only; flips applied below
        x, y = windows(train, BATCH, 8, gen, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x, y)[1].backward()
        if mode == "master":
            gr = [l.weight.grad.float().clone() for l in Ls] if record else None
            w0 = [l.weight.detach().float().clone() for l in Ls] if record else None
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            upd = [l.weight.detach().float() - a for l, a in zip(Ls, w0)] if record else None
            return gr, upd
        gr = None
        if mode == "xb2":
            from ttf.train import lookahead_step
            gr = [l.gw.float().clone() for l in Ls]
            lookahead_step(m, x, y, [p for p in m.parameters() if p.requires_grad], dev, s, 2, 0,
                           lambda: windows(train, BATCH, 8, gla, dev))
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0); opt.step()
            return gr, None
        if record:
            from ttf.kernel import fused_flip
            gr = [l.gw.float().clone() for l in Ls]
            for i, l in enumerate(Ls):
                fused_flip(l.wpacked, l.gw, l.rate, l.g_ref, 10_000 + s * 97 + i,
                           gmean=l.gw.abs().mean().clamp_min(1e-8).item())
                l.gw = None; l.capture = False
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0); opt.step()
        return gr, None

    # full run: every step records its gradient; every EVERY steps report the alignment of
    # (a) the last EVERY-step window and (b) everything since step 0, with the summed gradient
    out = []
    T00 = trits(); Tw = T00
    W00 = [l.weight.detach().float().clone() for l in Ls] if mode == "master" else None
    Ww = W00
    gcum = [torch.zeros_like(t) for t in T00]; gwin = [torch.zeros_like(t) for t in T00]
    for s in range(TOTAL):
        gr, _ = one_step(s, True)
        for a, b, g in zip(gcum, gwin, gr): a += g; b += g
        if (s + 1) % EVERY == 0:
            T = trits()
            r = {"step": s + 1,
                 "win_trit": np.mean([cos(t - t0, -g) for t, t0, g in zip(T, Tw, gwin)]),
                 "cum_trit": np.mean([cos(t - t0, -g) for t, t0, g in zip(T, T00, gcum)]),
                 "win_changed_%": np.mean([((t - t0) != 0).float().mean().item() * 100 for t, t0 in zip(T, Tw)])}
            # projection onto the top-r singular directions of this window's summed gradient:
            # share of the trit move inside them and the alignment there
            for rr in (16, 64):
                fr, cr = [], []
                for t, t0, g in zip(T, Tw, gwin):
                    d = t - t0
                    U, S, Vh = torch.linalg.svd(g.float(), full_matrices=False)
                    Ur, Vr = U[:, :rr], Vh[:rr].T
                    Pd = Ur @ (Ur.T @ d @ Vr) @ Vr.T
                    gr_ = (Ur * S[:rr]) @ Vh[:rr]
                    fr.append((Pd.norm() / d.norm().clamp_min(1e-12)).item())
                    cr.append(cos(Pd, -gr_))
                r[f"frac_top{rr}"] = float(np.mean(fr)); r[f"cos_top{rr}"] = float(np.mean(cr))
            if mode == "master":
                W = [l.weight.detach().float().clone() for l in Ls]
                r["win_latent"] = np.mean([cos(w - w0, -g) for w, w0, g in zip(W, Ww, gwin)])
                r["cum_latent"] = np.mean([cos(w - w0, -g) for w, w0, g in zip(W, W00, gcum)])
                Ww = W
            out.append(r); Tw = T
            gwin = [torch.zeros_like(t) for t in T00]
    return out


import json, sys
modes = sys.argv[1:] or ["master", "flip"]
path = os.environ.get("GEOM_OUT", "checkpoints/mlp/geometry_full.json")
res = json.load(open(path)) if os.path.exists(path) else {}
for mode in modes:
    res[mode] = run(mode)
    print(f"\n=== {mode}: alignment with the summed gradient along the run ===")
    for r in res[mode]:
        print("  " + " | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    json.dump(res, open(path, "w"))
