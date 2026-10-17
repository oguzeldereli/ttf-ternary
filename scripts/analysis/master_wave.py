"""The swing test for master weights (fp32 latent + STE + AdamW), for comparison with our flip rules. From a master
checkpoint, 41 real AdamW steps (the trainer's groups, betas, LR schedule and saved optimizer state); before each step
the true gradient T (32 batches) of the ternary layers. "Momentum" = Adam's first moment (exp_avg); a "flip" = a trit
of the ternarized latent that changed in the step. Reports, as scripts.analysis.wave: autocorrelation of T, momentum
vs later gradients, flips per step, uphill share (and by |T| quartile), one-step overshoot, per-weight reversals.
  MW_SUB=0.25 python -m scripts.analysis.master_wave RUN STEP
"""
import os, sys, math, numpy as np, torch
from ttf.master import build_master_transformer, split_params, MasterTernaryLinear
from ttf.train import get_batch
from ttf.config import TrainConfig

RUN, ST = sys.argv[1], int(sys.argv[2])
NT, K, SUB = 32, int(os.environ.get("MW_K", "41")), float(os.environ.get("MW_SUB", "0.25"))
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
b = torch.load(f"checkpoints/{RUN}/ckpt_{ST}.pt", map_location="cpu", weights_only=False)
model = build_master_transformer(b["cfg"], grad_checkpoint=True).cuda()
model.load_state_dict(b["model"]); model.train()
master, emb, norms = split_params(model)
tc = TrainConfig()
opt = torch.optim.AdamW([{"params": master, "weight_decay": tc.weight_decay},
                         {"params": emb, "weight_decay": tc.weight_decay},
                         {"params": norms, "weight_decay": 0.0}], lr=tc.lr, betas=(tc.beta1, tc.beta2))
opt.load_state_dict(b["opt"])
Ls = [m for m in model.modules() if isinstance(m, MasterTernaryLinear)]
VB = [get_batch(val, 8, 2048, "cuda", torch.Generator().manual_seed(555 + i)) for i in range(16)]


def lr_at(step):   # the trainer's schedule for the 300M run (9155 steps, warmup 305)
    if step < 305: return 1.5e-3 * (step + 1) / 305
    r = (step - 305) / (9155 - 305)
    return 1.5e-4 + 0.5 * (1.5e-3 - 1.5e-4) * (1 + math.cos(math.pi * r))


def fwd_bwd(x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)[1]
    loss.backward()


def truth(seed):
    g = torch.Generator().manual_seed(seed); acc = None
    for _ in range(NT):
        opt.zero_grad(set_to_none=True); fwd_bwd(*get_batch(train, 16, 2048, "cuda", g))
        gr = torch.cat([l.weight.grad.float().flatten() for l in Ls]); acc = gr if acc is None else acc + gr
    opt.zero_grad(set_to_none=True)
    return acc / NT


def held():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([model(x, y)[1].item() for x, y in VB]))


def trits(): return torch.cat([l.ternary_weight()[0].flatten() for l in Ls])


NTOT = sum(l.weight.numel() for l in Ls)
IDX = torch.randperm(NTOT, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))[:int(SUB * NTOT)]
keep = lambda v: v[IDX].half().cpu()
L0 = held(); Ts, Ms, MV, nflip, upshare, CMT, CLT = [], [], [], [], [], [], []
gs = torch.Generator().manual_seed(5151)
for k in range(K):
    Tf = truth(900 + k)
    Ts.append(keep(Tf)); Ms.append(keep(torch.cat([opt.state[l.weight]["exp_avg"].float().flatten() for l in Ls])))
    w0 = trits(); lat0 = torch.cat([l.weight.detach().float().flatten() for l in Ls])
    opt.zero_grad(set_to_none=True); fwd_bwd(*get_batch(train, 16, 2048, "cuda", gs))
    for gr in opt.param_groups: gr["lr"] = lr_at(ST + k)
    opt.step(); opt.zero_grad(set_to_none=True)
    d = (trits() - w0).float(); nz = d != 0
    nflip.append(int(nz.sum())); upshare.append(float(((d * Tf) > 0)[nz].float().mean()) if nz.any() else 0.0)
    CMT.append(float(-(d @ Tf) / (d.norm() * Tf.norm()).clamp_min(1e-30)))
    lat = torch.cat([l.weight.detach().float().flatten() for l in Ls]) - lat0
    CLT.append(float(-(lat @ Tf) / (lat.norm() * Tf.norm()).clamp_min(1e-30)))
    MV.append(keep(d)); del Tf, w0, d, lat, lat0
    if k % 10 == 0: print(f"  step {k} measured", flush=True)
L1 = held()
print(f"\nheld-out loss {L0:.4f} -> {L1:.4f} ({L1 - L0:+.4f}) over {K} steps; trit changes per step: mean "
      f"{np.mean(nflip) / 1e3:.0f}k (first 5 {[round(n / 1e3) for n in nflip[:5]]}k)")
print(f"{RUN} @{ST} (master: AdamW on the latent, lr {lr_at(ST):.2e}), true gradient from {NT} batches")


def dots(A, B_):
    return torch.tensor([[float(a.float() @ b_.float()) for b_ in B_] for a in A], dtype=torch.float64)


G = dots(Ts, Ts); nT = G.diagonal().sqrt(); G = G / nT[:, None] / nT[None, :]
MT = dots(Ms, Ts); nM = torch.tensor([float(m.float().norm()) for m in Ms], dtype=torch.float64)
MT = MT / nM[:, None].clamp_min(1e-30) / nT[None, :]
print("lag  mean cos(T_t, T_t+lag)   mean cos(M_t, T_t+lag)   (M = Adam's first moment)")
for lag in (0, 1, 2, 3, 5, 8, 10, 13, 16, 20):
    if lag < K: print(f"{lag:3d}  {float(torch.diagonal(G, lag).mean()):+22.3f}   {float(torch.diagonal(MT, lag).mean()):+22.3f}")
print(f"\nalignment with the true descent direction: trit changes cos(move, -T) {np.mean(CMT):+.4f}, latent (AdamW) step "
      f"cos(step, -T) {np.mean(CLT):+.4f}")
print(f"trit changes that move uphill on the true gradient: mean {100 * np.mean(upshare):.1f}%")
qf = np.zeros(4); qu = np.zeros(4); wup = wall = 0.0
for t in range(K):
    D = MV[t].float(); Tt = Ts[t].float(); m = D != 0
    if not m.any(): continue
    a = D[m] * Tt[m]; mag = Tt.abs(); edges = mag[::8].quantile(torch.tensor([.25, .5, .75]))
    q = torch.bucketize(mag[m], edges)
    for j in range(4): sel = q == j; qf[j] += float(sel.sum()); qu[j] += float((a[sel] > 0).sum())
    wup += float(a.clamp_min(0).sum()); wall += float(a.abs().sum())
print("uphill by |T| quartile (small -> large): " + "  ".join(
    f"Q{j + 1}: {100 * qf[j] / max(qf.sum(), 1):4.1f}% of changes, {100 * qu[j] / max(qf[j], 1):4.1f}% uphill" for j in range(4)))
print(f"weighted by |T|: {100 * wup / max(wall, 1e-30):.1f}% uphill")
print("changes judged before and after (share of all trit changes):")
for k2 in (1, 2, 3):
    A = B_ = C = 0.0; n = 0
    for t in range(K - k2):
        D = MV[t].float(); m = D != 0
        if not m.any(): continue
        a = (D * Ts[t].float())[m]; c = (D * Ts[t + k2].float())[m]
        A += float((a > 0).float().mean()); B_ += float(((a < 0) & (c > 0)).float().mean()); C += float(((a < 0) & (c < 0)).float().mean()); n += 1
    print(f"  after {k2} step(s): uphill when made {100 * A / n:.1f}%, overshoot {100 * B_ / n:.1f}%, downhill both {100 * C / n:.1f}%")
st = [(float(MV[t].float() @ Ts[t].float()), float(MV[t].float() @ Ts[t + 1].float())) for t in range(K - 1)]
print(f"  the step as a whole (move . T): before {np.mean([x for x, _ in st]):+.3e}, one step later {np.mean([y for _, y in st]):+.3e}; "
      f"steps turning uphill one step later: {100 * np.mean([(x < 0) and (y > 0) for x, y in st]):.0f}%")
ci = torch.randperm(Ts[0].numel(), generator=torch.Generator().manual_seed(7))[:200_000]
Tc_ = torch.stack([t.float()[ci] for t in Ts]); Mc_ = torch.stack([m.float()[ci] for m in Ms])
sT, sM = Tc_.sign(), Mc_.sign()
print(f"per weight, momentum sign = true-gradient sign: {float((sT == sM).float().mean()):.3f} of weights")
big = Tc_.abs() > Tc_.abs().median()
delays, cens, ratio = [], 0, []
for t in range(1, K):
    idx = ((sT[t] != sT[t - 1]) & big[t] & big[t - 1]).nonzero().flatten()
    if idx.numel() == 0: continue
    ratio.append((Mc_[t, idx].abs() / (1 - tc.beta1) / Tc_[t, idx].abs()).median().item())   # exp_avg as a sum of gradients
    d = torch.full((idx.numel(),), -1)
    for dd in range(0, K - t): hit = (sM[t + dd, idx] == sT[t, idx]) & (d < 0); d[hit] = dd
    cens += int((d < 0).sum()); delays.append(d[d >= 0])
D_ = torch.cat(delays).float(); tot = D_.numel() + cens
print(f"true-gradient reversals: {tot}; momentum already on the new side {100 * float((D_ == 0).sum()) / tot:.1f}%, "
      f"within 1-3 steps {100 * float(((D_ >= 1) & (D_ <= 3)).sum()) / tot:.1f}%, 4+ {100 * float((D_ >= 4).sum()) / tot:.1f}%, "
      f"never {100 * cens / tot:.1f}%; |M|/|T| at a reversal {np.median(ratio):.1f}")
