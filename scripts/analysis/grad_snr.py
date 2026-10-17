"""Gradient signal-to-noise per layer at one TTF checkpoint: is the deep-layer deficit at 2.9B (never moved 0.45-0.48
in the deepest sixth vs ~0.2 in the middle; 1.3B has none) a weaker, less consistent gradient there?

TTF accumulates u = g / sqrt(v) (factored second moment from the checkpoint) and, with its leak and rank-r projection,
fires only on a drift that is consistent over ~tau steps and lies in the top-r subspace. So per layer, over n
micro-batch gradients u_b at the frozen checkpoint (the kernel path's own dL/dW, layer.capture):
  - signal energy sum mu^2 (debiased: mean^2 - var / n) and noise energy sum var, as a per-step SNR at the training
    batch (micro-batch noise / (batch / micro));
  - rms drift per entry sqrt(sum mu^2 / NK), in units of g / sqrt(v), against the fire bar theta / (tau_eff a);
  - rank capture: the share of the signal inside a rank-r subspace, (i) the top-r SVD subspace of one third of the
    gradients, measured on the other two (<P G2, P G3> / <G2, G3>, so neither the noise nor the fitted subspace's own
    noise counts), (ii) the checkpoint's accumulator subspace span(U) x span(V).
A frozen checkpoint gives the instantaneous mean gradient, not whether it persists over the ~tau steps the accumulator
integrates: --save_mu FILE keeps each layer's mean (bf16) and debiased signal energy, and --persist FILE_A FILE_B
compares two such files (checkpoints ~1000 steps apart): cos = <mu_a, mu_b> / sqrt(sig_a sig_b) (independent batches,
so the cross product carries no noise; the energies are debiased), the share of the drift that keeps its direction.
    python scripts/analysis/grad_snr.py CKPT --data data/fwedu32k_train.bin [--n 96] [--micro 4] [--batch 16] [--save_mu F]
    python scripts/analysis/grad_snr.py --persist A.pt B.pt
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.eval.final_eval import load_model                      # noqa: E402
from ttf.flip import KernelTernaryLinear                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?"); ap.add_argument("--data")
ap.add_argument("--save_mu", default=""); ap.add_argument("--persist", nargs=2, default=None)
ap.add_argument("--save_thirds", action="store_true", help="--save_mu also keeps the means of the three independent "
                "thirds (bf16), for within-checkpoint cross spectra (rank_spectrum.py)")
ap.add_argument("--seed", type=int, default=1234, help="data windows (another seed gives independent batches)")
ap.add_argument("--every", type=int, default=1, help="keep every k-th ternary layer only (memory at 1.3B / 2.9B)")
ap.add_argument("--n", type=int, default=96, help="micro-batch gradients (a multiple of 3)")
ap.add_argument("--micro", type=int, default=4); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--no_svd", action="store_true")
a = ap.parse_args()
dev = "cuda"
if a.persist:
    A_, B_ = (torch.load(f, map_location="cpu", weights_only=False) for f in a.persist)
    print(f"persistence of the mean gradient: {a.persist[0]} (step {A_['step']}) vs {a.persist[1]} (step {B_['step']})")
    cs = []
    for i, (ma, mb) in enumerate(zip(A_["mu"], B_["mu"])):
        c = (ma.float() * mb.float()).sum().item() / max(A_["sig"][i] * B_["sig"][i], 1e-30) ** 0.5
        cs.append(c)
    nl = len(cs) // 7; six = np.array_split(np.arange(nl), 6)
    print("cos by depth sixth: " + " ".join(f"{np.mean([cs[7 * l + j] for l in t for j in range(7)]):.3f}" for t in six))
    for j, t in enumerate(["q", "k", "v", "o", "gate", "up", "down"]):
        print(f"  {t:5s} " + " ".join(f"{np.mean([cs[7 * l + j] for l in tt]):.3f}" for tt in six))
    sys.exit()
t0 = time.time()
model, ck = load_model(a.ckpt, dev)
cfg = ck.get("args") or {}
step = ck.get("step")
layers = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
names = [n for n, m in model.named_modules() if isinstance(m, KernelTernaryLinear)]
for m in layers:
    m.capture = True
for p in model.parameters():
    p.requires_grad_(p.dim() == 2 and p.shape[0] > 30000)          # the embedding only: keeps the backward running
ts = ck.get("ts_state"); lr = ck.get("lowrank")
print(f"{a.ckpt}: step {step}, {len(layers)} ternary layers, ts_state {'yes' if ts else 'no'}, "
      f"loaded in {time.time() - t0:.0f}s", flush=True)

# fire bar theta / (tau_eff a), tau_eff = tau / rho (annealed leak), a = rho^p, from the run's own metrics
mp = os.path.join(os.path.dirname(a.ckpt), "metrics.jsonl")
rho = None
if os.path.exists(mp):
    lrs = {}
    for l in open(mp):
        d = json.loads(l)
        if "lr" in d:
            lrs[d["step"]] = d["lr"]
    if lrs:
        rho = lrs.get(step, lrs[min(lrs, key=lambda s: abs(s - step))]) / max(lrs.values())

vfac = []                           # factored v from the checkpoint, kept as (R, C, scale): a full N x K copy per layer
for i, m in enumerate(layers):      # would not fit next to the accumulators at 2.9B
    f = None
    if ts and ts[i] and "v" in ts[i]:
        R, C = (x.to(dev).float() for x in ts[i]["v"])
        t = int(ts[i]["t"]) if "t" in ts[i] else 10 ** 6
        f = (R, C, 1.0 / (R.mean().clamp_min(1e-30) * (1 - 0.95 ** max(t, 1))))
    vfac.append(f)
model.grad_checkpoint = True; model.ckpt_skip = 0; model.train()   # activations of a 2.9B micro-batch otherwise ~20 GB

data = np.memmap(a.data, dtype=np.uint16, mode="r")
rng = np.random.default_rng(a.seed)
n3 = a.n // 3
KEEP = [i % a.every == 0 for i in range(len(layers))]
S = [[torch.zeros(m.N, m.K, device=dev) for _ in range(3)] if KEEP[i] else None for i, m in enumerate(layers)]
SQ = [torch.zeros(m.N, m.K, device=dev) if KEEP[i] else None for i, m in enumerate(layers)]
EMP = [None] * len(layers)          # empirical second moment, if the checkpoint has no v
for b in range(3 * n3):
    starts = rng.integers(0, len(data) - a.seq - 1, a.micro)
    x = torch.tensor(np.stack([data[s:s + a.seq + 1].astype(np.int64) for s in starts]), device=dev)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    for i, m in enumerate(layers):
        g = m.gw; m.gw = None
        if not KEEP[i]:
            del g
            continue
        if vfac[i] is None:
            EMP[i] = g * g if EMP[i] is None else EMP[i].add_(g * g)
            u = g
        else:
            R, C, sc = vfac[i]
            u = g.div_((R[:, None] * C[None, :] * sc).sqrt_().add_(1e-8))
        S[i][b % 3].add_(u); SQ[i].addcmul_(u, u)
        del g, u
    if b % 12 == 11:
        print(f"  {b + 1}/{3 * n3} micro-batches, loss {loss.item():.3f}, {time.time() - t0:.0f}s", flush=True)

n = 3 * n3
k_noise = a.batch / a.micro            # micro-batch noise variance / training-batch noise variance
cg = cfg if isinstance(cfg, dict) else vars(cfg)
p_, theta, tau = float(cg.get("ts_anneal", 0.5)), float(cg.get("ts_theta", 16.0)), float(cg.get("ts_tau", 1000.0))
tan = bool(cg.get("ts_tau_anneal", False))
# the leak per step is a / tau with --ts_tau_anneal (a = rho^p), 1 / tau otherwise: bar theta / (tau_eff a)
bar = (theta / tau if tan else theta / (tau * rho ** p_)) if rho else None
print(f"\nrho {rho}, fire bar theta / (tau_eff a) = {bar} (theta {theta}, tau {tau}{' annealed' if tan else ''}, p {p_})")
rows = []; MU, SIG, TH = [], [], []
for i, m in enumerate(layers):
    if not KEEP[i]:
        continue
    if vfac[i] is None:                # normalise by the empirical factored second moment at the training batch
        e = EMP[i] / n; R = e.mean(1); C = e.mean(0)
        w = 1.0 / ((R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt() + 1e-8)
        for j in range(3):
            S[i][j].mul_(w)
        SQ[i].mul_(w * w); EMP[i] = None
    tot = S[i][0] + S[i][1] + S[i][2]
    mu = tot / n
    var = (SQ[i] - n * mu * mu).clamp_min_(0) / (n - 1)
    sig = (mu * mu - var / n).sum().item(); noise = var.sum().item() / k_noise
    NK = m.N * m.K
    G = [S[i][j] / n3 for j in range(3)]
    cross = (G[1] * G[2]).sum().item()                              # unbiased signal energy (independent thirds)
    cap_svd = cap_acc = float("nan")
    r = lr[i][0].shape[1] if lr and lr[i] is not None else 0
    if r and not a.no_svd:
        Uq, _, Vh = torch.linalg.svd(G[0], full_matrices=False)
        Pu, Pv = Uq[:, :r], Vh[:r].T
        cap_svd = ((Pu.T @ G[1] @ Pv) * (Pu.T @ G[2] @ Pv)).sum().item() / cross
        Qu = torch.linalg.qr(lr[i][0].to(dev).float())[0]; Qv = torch.linalg.qr(lr[i][1].to(dev).float())[0]
        cap_acc = ((Qu.T @ G[1] @ Qv) * (Qu.T @ G[2] @ Qv)).sum().item() / cross
        del Uq, Vh, Pu, Pv, Qu, Qv
    rows.append(dict(i=i, name=names[i], N=m.N, K=m.K, snr=sig / max(noise, 1e-30), drift=(max(sig, 0) / NK) ** 0.5,
                     noise_rms=(noise / NK) ** 0.5, cross=cross, cap_svd=cap_svd, cap_acc=cap_acc, r=r))
    if a.save_mu:
        MU.append(mu.to(torch.bfloat16).cpu()); SIG.append(sig)
        if a.save_thirds:
            TH.append([x.to(torch.bfloat16).cpu() for x in G])
    S[i] = None; SQ[i] = None; del tot, mu, var, G
    torch.cuda.empty_cache()

if a.save_mu:
    torch.save({"step": step, "mu": MU, "sig": SIG, "names": [nm for nm, k in zip(names, KEEP) if k], **({"thirds": TH} if a.save_thirds else {})},
               a.save_mu); MU = TH = None
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
nl = len(rows) // 7
six = np.array_split(np.arange(nl), 6)
print("per-step SNR (training batch) | rms drift per entry (g / sqrt v) | drift / bar | signal share in top-r SVD "
      "subspace | in the accumulator's subspace")
for k, t in enumerate(six):
    sel = [r for r in rows if r["i"] // 7 in t]
    f = lambda key: np.nanmean([r[key] for r in sel])
    print(f"  sixth {k} (layers {t[0]}-{t[-1]}): SNR {f('snr'):.4f}  drift {f('drift'):.4f}"
          + (f"  drift/bar {f('drift') / bar:.2f}" if bar else "")
          + f"  top-r {f('cap_svd'):.3f}  acc {f('cap_acc'):.3f}")
print("\nby type: deepest sixth vs middle (sixths 2-3): SNR, drift, top-r share")
for j, t in enumerate(TYPES):
    d = [r for r in rows if r["i"] % 7 == j and r["i"] // 7 in six[-1]]
    mm = [r for r in rows if r["i"] % 7 == j and (r["i"] // 7 in six[2] or r["i"] // 7 in six[3])]
    g = lambda s, key: np.nanmean([r[key] for r in s])
    print(f"  {t:5s} deep SNR {g(d, 'snr'):.4f} drift {g(d, 'drift'):.4f} top-r {g(d, 'cap_svd'):.3f} | "
          f"middle SNR {g(mm, 'snr'):.4f} drift {g(mm, 'drift'):.4f} top-r {g(mm, 'cap_svd'):.3f}")
print("\nper layer: i name N K snr drift noise_rms top-r acc")
for r in rows:
    print(f"  {r['i']:3d} {r['name']:28s} {r['N']:5d} {r['K']:5d} {r['snr']:.4f} {r['drift']:.4f} {r['noise_rms']:.3f} "
          f"{r['cap_svd']:.3f} {r['cap_acc']:.3f}")
print(f"done in {time.time() - t0:.0f}s")
