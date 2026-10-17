"""Does master choose its trit changes by the *accumulated* push? Master's latent W reaches a rounding boundary because the
gradient pushed it there over many steps; on the step it crosses, the instantaneous gradient can be small. Here: master is
replayed from step A with its saved AdamW state, latent snapshots are kept at steps S - w for each window w, and on step S
-> S+1 its trit changes are compared with the drift of each weight's latent over the window, d_w = (W_S - W_{S-w}) / gamma
(in units of the rounding step). For each window: are the changed weights the ones with the largest |d_w| (quartile
shares within their layer, 25% each = no preference), and does the change follow the drift's sign? Also how |d_w| relates
to the instantaneous true gradient |T| at step S (rank correlation, sampled).
  python -m scripts.analysis.master_drift MASTER_CKPT_AT_A STEPS_TO_REPLAY
"""
import sys, numpy as np, torch
from ttf.master import build_master_transformer
from ttf.train import split_params, get_batch

path, NREP = sys.argv[1], int(sys.argv[2])
WIN = [w for w in (1000, 300, 100, 30, 10, 3, 1) if w <= NREP]
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
b = torch.load(path, map_location="cpu", weights_only=False)
mm = build_master_transformer(b["cfg"], grad_checkpoint=True); mm.load_state_dict(b["model"]); mm = mm.cuda().train()
master, emb, norms = split_params(mm)
opt = torch.optim.AdamW([{"params": master, "weight_decay": 0.1}, {"params": emb, "weight_decay": 0.1},
                         {"params": norms, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95))
opt.load_state_dict(b["opt"])
lin = [m for m in mm.modules() if hasattr(m, "ternary_weight")]
lr0 = opt.param_groups[0]["lr"]
snaps = {}
gen = torch.Generator().manual_seed(123)
def step():
    x, y = get_batch(tr, 16, 2048, "cuda", gen)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = mm(x, y)[1]
    loss.backward(); opt.step()
for k in range(NREP):                      # replay (lr held at the saved value: the schedule moves little over the window)
    if NREP - k in WIN:
        snaps[NREP - k] = [m.weight.detach().float().clone() for m in lin]
    step()
WS = [m.weight.detach().float().clone() for m in lin]
gam = [w.abs().mean().clamp_min(1e-5) for w in WS]
T0 = [(w / g).round().clamp(-1, 1) for w, g in zip(WS, gam)]
# instantaneous true gradient at S (latent gradient, 8 batches)
G = [torch.zeros_like(w) for w in WS]
for _ in range(8):
    x, y = get_batch(tr, 16, 2048, "cuda", gen)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = mm(x, y)[1]
    loss.backward()
    for g, m in zip(G, lin): g += m.weight.grad.float() / 8
step()                                     # the step S -> S+1 whose trit changes we explain
T1 = [(m.weight.detach().float() / m.weight.detach().float().abs().mean().clamp_min(1e-5)).round().clamp(-1, 1) for m in lin]
D = [t1 - t0 for t0, t1 in zip(T0, T1)]
n = sum(int((d != 0).sum()) for d in D)
print(f"replayed {NREP} steps from {path} (lr {lr0:.2e}); step S -> S+1: {n / 1e3:.0f}k trit changes")
print(f"{'window':>7s} {'|drift| quartile shares of the changes (Q1 small .. Q4 large)':>58s} {'follows drift':>14s} {'rank corr |drift|,|T|':>22s}")
for w in WIN:
    q = np.zeros(4); fol = tot = 0; rc = []
    for i, (d, ws, s0, g) in enumerate(zip(D, WS, snaps[w], gam)):
        dr = (ws - s0) / g
        nz = d != 0
        a = dr.abs(); qs = torch.quantile(a.flatten()[::97], torch.tensor([0.25, 0.5, 0.75], device=a.device))
        if nz.any():
            q += np.bincount(torch.bucketize(a[nz], qs).cpu().numpy(), minlength=4)[:4]
            fol += int((torch.sign(dr[nz]) == d[nz].sign()).sum()); tot += int(nz.sum())
        idx = torch.randperm(a.numel(), device=a.device)[:20000]
        ra = a.flatten()[idx].argsort().argsort().float(); rg = G[i].abs().flatten()[idx].argsort().argsort().float()
        rc.append(float(torch.corrcoef(torch.stack([ra, rg]))[0, 1]))
    print(f"{w:7d} {' / '.join(f'{100 * x / max(tot, 1):5.1f}' for x in q):>58s} {100 * fol / max(tot, 1):13.1f}% {np.mean(rc):+22.3f}")
qT = np.zeros(4)
for d, g in zip(D, G):
    nz = d != 0; a = g.abs(); qs = torch.quantile(a.flatten()[::97], torch.tensor([0.25, 0.5, 0.75], device=a.device))
    if nz.any(): qT += np.bincount(torch.bucketize(a[nz], qs).cpu().numpy(), minlength=4)[:4]
print(f"instantaneous |T| quartile shares of the changes: {' / '.join(f'{100 * x / max(n, 1):.1f}' for x in qT)}")
