"""What sets the share of a step's energy a halving of the rank drops (~0.15 at 110M / 340M / 1.3B / 2.9B, and not the
batch)? A share that is the same for every halving means the step's singular energy is spread evenly over octaves
(sigma_i^2 ~ 1 / i). Hypothesis: one step's gradient G = sum_t delta_t x_t^T (per token: the error at the layer's output
times its input), so its spectrum follows from the input covariance Cx = sum x x^T [K, K] and the error covariance
Cd = sum delta delta^T [N, N], both of which decay as power laws in trained LLMs.

At one checkpoint, for one training batch (--batch tokens-sequences, in micro-batches of --micro), per ternary layer:
  measured   kept share of ||u||^2 in the top-r singular directions of u itself, u = G / sqrt(v) (the checkpoint's
             factored v, as TTF steps), and of G
  model      the same for G' = Cd^(1/2) Z Cx^(1/2) (Z iid Gaussian), normalised by the same v: a random gradient
             with the measured input and error covariances and nothing else
  spectra    the power-law slope of the eigenvalues of Cx and Cd (fit over ranks 4..K/4), and the energy per octave
             of u's singular values
Energy-weighted over layers, per matrix type and overall.
    python scripts/analysis/step_spectrum.py CKPT [--data data/wiki32k_train.bin] [--batch 16] [--micro 4] [--seed 7]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.eval.final_eval import load_model                      # noqa: E402
from bitnet.flip import KernelTernaryLinear                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--data", default="data/wiki32k_train.bin")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--every", type=int, default=1, help="keep every k-th ternary layer only")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
t0 = time.time()
model, ck = load_model(a.ckpt, dev)
layers = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
names = [n for n, m in model.named_modules() if isinstance(m, KernelTernaryLinear)]
KEEP = [i % a.every == 0 for i in range(len(layers))]
ts = ck.get("ts_state")
for p in model.parameters():
    p.requires_grad_(p.dim() == 2 and p.shape[0] > 30000)          # the embedding only: keeps the backward running
model.grad_checkpoint = False; model.train()

CX = [torch.zeros(m.K, m.K, device=dev, dtype=torch.float64) if KEEP[i] else None for i, m in enumerate(layers)]
CD = [torch.zeros(m.N, m.N, device=dev, dtype=torch.float64) if KEEP[i] else None for i, m in enumerate(layers)]
G = [torch.zeros(m.N, m.K, device=dev) if KEEP[i] else None for i, m in enumerate(layers)]


def fwd_hook(i):
    def h(mod, inp):
        x = inp[0].detach().float().reshape(-1, mod.K)
        if mod.col_scale is not None:
            x = x * mod.col_scale.detach().float()
        CX[i].add_((x.T @ x).double())
    return h


def bwd_hook(i):
    def h(mod, gin, gout):
        d = gout[0].detach().float().reshape(-1, mod.N)
        if mod.row_scale is not None:
            d = d * mod.row_scale.detach().float()
        CD[i].add_((d.T @ d).double())
    return h


for i, m in enumerate(layers):
    m.capture = True
    if KEEP[i]:
        m.register_forward_pre_hook(fwd_hook(i)); m.register_full_backward_hook(bwd_hook(i))

data = np.memmap(a.data, dtype=np.uint16, mode="r")
rng = np.random.default_rng(a.seed)
for b in range(a.batch // a.micro):
    starts = rng.integers(0, len(data) - a.seq - 1, a.micro)
    x = torch.tensor(np.stack([data[s:s + a.seq + 1].astype(np.int64) for s in starts]), device=dev)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    for i, m in enumerate(layers):
        g = m.gw; m.gw = None
        if KEEP[i]:
            G[i].add_(g)
        del g
print(f"{a.ckpt}: step {ck.get('step')}, one batch of {a.batch} x {a.seq} tokens, loss {loss.item():.3f}, "
      f"{time.time() - t0:.0f}s", flush=True)

RANKS = [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048]


def kept(M):
    s2 = torch.linalg.svdvals(M.float()).double().pow(2)
    c = s2.cumsum(0) / s2.sum()
    return [float(c[min(r, len(c)) - 1]) for r in RANKS], s2


def slope(ev):
    ev = ev.clamp_min(1e-30).cpu().numpy()[::-1].copy()          # eigvalsh ascends: largest first
    n = len(ev); lo, hi = 4, max(8, n // 4)
    i = np.arange(1, n + 1)
    return float(np.polyfit(np.log(i[lo:hi]), np.log(ev[lo:hi]), 1)[0])


rows = []
for i, m in enumerate(layers):
    if not KEEP[i]:
        continue
    if ts and ts[i] and "v" in ts[i]:
        R, C = (t_.to(dev).float() for t_ in ts[i]["v"])
        sv = (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().add_(1e-12)
    else:
        sv = torch.ones(m.N, m.K, device=dev)
    u = G[i] / sv
    ku, s2u = kept(u)
    kg, _ = kept(G[i])
    ex, Ux = torch.linalg.eigh(CX[i]); ed, Ud = torch.linalg.eigh(CD[i])
    Sx = (Ux * ex.clamp_min(0).sqrt()) @ Ux.T; Sd = (Ud * ed.clamp_min(0).sqrt()) @ Ud.T
    Z = torch.randn(m.N, m.K, device=dev, dtype=torch.float64)
    Gp = (Sd @ Z @ Sx).float()
    kp, _ = kept(Gp / sv)
    oct_ = [float(s2u[2 ** j - 1:2 ** (j + 1) - 1].sum() / s2u.sum()) for j in range(int(np.log2(len(s2u))) + 1)]
    rows.append(dict(name=names[i], typ=names[i].split(".")[-1], e=float(u.pow(2).sum()), ku=ku, kg=kg, kp=kp,
                     ax=slope(ex), ad=slope(ed), oct=oct_, n=min(m.N, m.K)))
    del u, Gp, Z, Sx, Sd, Ux, Ud


def agg(rs, key):
    w = np.array([r["e"] for r in rs]); M = np.array([r[key] for r in rs])
    return (w[:, None] * M).sum(0) / w.sum()


print("\nkept share of one step's energy within rank r (energy-weighted over layers)")
print("  r              " + " ".join(f"{r:6d}" for r in RANKS))
for key, lab in (("ku", "u measured"), ("kp", "u model Cd,Cx"), ("kg", "G measured")):
    print(f"  {lab:14s} " + " ".join(f"{x:6.3f}" for x in agg(rows, key)))
print("\nper halving (kept(2r) - kept(r)), u measured / model")
ku, kp = agg(rows, "ku"), agg(rows, "kp")
print("  r              " + " ".join(f"{r:6d}" for r in RANKS[:-1]))
print("  measured       " + " ".join(f"{ku[j + 1] - ku[j]:6.3f}" for j in range(len(RANKS) - 1)))
print("  model          " + " ".join(f"{kp[j + 1] - kp[j]:6.3f}" for j in range(len(RANKS) - 1)))
print("\nby type: kept at r 64 / 256 measured vs model; eigenvalue slopes of Cx, Cd (lambda_i ~ i^slope)")
for t in sorted({r["typ"] for r in rows}):
    rs = [r for r in rows if r["typ"] == t]
    k1, k2 = agg(rs, "ku"), agg(rs, "kp")
    print(f"  {t:7s} {k1[3]:.3f} / {k1[5]:.3f}  vs  {k2[3]:.3f} / {k2[5]:.3f}   Cx {np.mean([r['ax'] for r in rs]):+.2f}"
          f"  Cd {np.mean([r['ad'] for r in rs]):+.2f}")
nmin = min(len(r["oct"]) for r in rows)
oc = np.mean([r["oct"][:nmin] for r in rows], axis=0)
print("\nenergy of u per octave of singular values (ranks 1, 2-3, 4-7, ...): " + " ".join(f"{x:.3f}" for x in oc))
print(f"done in {time.time() - t0:.0f}s")
