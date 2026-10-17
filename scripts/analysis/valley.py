"""Is the loss a narrow valley where the true gradient zigzags across and progress runs along? Two tests at a
no-look-ahead snapshot (saved momentum, float tail frozen, plain momentum with the trainer's flip rule and rate).

zigzag   10 steps of flips; before each step's flips, the true gradient T_t (64 batches). With the window mean
         Tbar = mean_t T_t: the share of each T_t's energy along Tbar, and the residuals r_t = T_t - (T_t . u) u
         (u = Tbar / |Tbar|): cos(r_t, r_t+1) < 0 means the across-direction flips sign step to step. Also
         cos(T_t, T_t+1), and momentum against the instantaneous vs the window-mean true gradient.
labels   the online selector's label vs the true one, for momentum's proposals at 2x the rate at one step: truth
         label = downhill on the true gradient at the decision point; online label = the next 8 batches' gradients
         (after the step, with a random half of the proposals applied, as in the selector's warm-up, and 8 more
         normal steps) still push in the proposed direction. Agreement, per applied / not applied, and how each
         label relates to |g|, |M| and the trit value.
  python -m scripts.analysis.valley RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
BETA, NT = 0.97, 64
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
T0 = [unpack_rows(l.wpacked, l.K).to(torch.int8).clone() for l in Ls]
warm = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]


def grad(b):
    g = B.grad(b); return [g[n] for n in names]


def truth(seed):
    s = batches(seed); acc = None
    for _ in range(NT):
        g = grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    return [a / NT for a in acc]


cat = lambda xs: torch.cat([x.flatten().float() for x in xs])
cosv = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def flip_all(F, k, mult=1.0):
    for i, (l, f) in enumerate(zip(Ls, F)):
        fused_flip(l.wpacked, f.contiguous(), RATE * mult, G_REF, 3000 + 131 * k + i, gmean=f.abs().mean().clamp_min(1e-12))


print(f"{RUN} @{ST}, flip rate {RATE:.4f}", flush=True)
print("\n== zigzag: 10 steps of plain momentum flips, true gradient before each step", flush=True)
F = [U @ V.T for U, V in warm]; s = batches(5151); Ts, Ms = [], []
for k in range(1, 11):
    Ts.append(cat(truth(700 + k))); Ms.append(cat(F))
    g = grad(next(s)); F = [BETA * f + x for f, x in zip(F, g)]
    flip_all(F, k)
Tbar = torch.stack(Ts).mean(0); u = Tbar / Tbar.norm()
res = [t - (t @ u) * u for t in Ts]
print("  step  share of |T_t|^2 along the window mean  cos(T_t, T_t+1)  cos(r_t, r_t+1)  cos(M_t, T_t)  cos(M_t, window mean)", flush=True)
for t in range(10):
    along = float((Ts[t] @ u) ** 2 / (Ts[t] @ Ts[t]))
    ct = cosv(Ts[t], Ts[t + 1]) if t < 9 else float("nan")
    cr = cosv(res[t], res[t + 1]) if t < 9 else float("nan")
    print(f"  {t + 1:4d}  {along:38.3f}  {ct:+15.3f}  {cr:+15.3f}  {cosv(Ms[t], Ts[t]):+13.3f}  {cosv(Ms[t], Tbar):+21.3f}", flush=True)

print("\n== labels: online selector label vs true-gradient label", flush=True)
for l, t in zip(Ls, T0): l.wpacked.copy_(pack_rows(t))
F = [U @ V.T for U, V in warm]; s = batches(6161)
g = grad(next(s)); F = [BETA * f + x for f, x in zip(F, g)]
Tt = truth(8080)
props, applied = [], []
gen = torch.Generator(device="cuda").manual_seed(1)
for i, (l, f) in enumerate(zip(Ls, F)):
    w0 = unpack_rows(l.wpacked, l.K).to(torch.int8); tmp = l.wpacked.clone()
    fused_flip(tmp, f.contiguous(), RATE * 2, G_REF, 4000 + i, gmean=f.abs().mean().clamp_min(1e-12))
    mv = unpack_rows(tmp, l.K).to(torch.int8) - w0
    keep = (mv != 0) & (torch.rand(mv.shape, device="cuda", generator=gen) < 0.5)
    l.wpacked.copy_(pack_rows((w0 + torch.where(keep, mv, torch.zeros_like(mv))).clamp_(-1, 1)))
    props.append((mv, w0)); applied.append(keep)
acc = [torch.zeros_like(f) for f in F]
for k in range(8):
    g = grad(next(s)); acc = [a + x for a, x in zip(acc, g)]
    F = [BETA * f + x for f, x in zip(F, g)]; flip_all(F, 100 + k)
mvs = cat([p[0] for p in props]); m = mvs != 0
truth_lab = (mvs * cat(Tt) < 0)[m]; online_lab = (mvs * cat(acc) < 0)[m]
app = cat(applied)[m] > 0
gabs = cat([x.abs() / x.abs().mean() for x in g])[m]
Mabs = cat([f.abs() / f.abs().mean() for f in F])[m]
trit0 = cat([p[1] for p in props])[m]
print(f"  proposals {int(m.sum())}: truth says downhill {float(truth_lab.float().mean()):.3f}, online label good "
      f"{float(online_lab.float().mean()):.3f}, agreement {float((truth_lab == online_lab).float().mean()):.3f} (0.5 = unrelated)", flush=True)
for name, sel in (("applied", app), ("not applied", ~app)):
    print(f"  {name:12s}: truth downhill {float(truth_lab[sel].float().mean()):.3f}  online good {float(online_lab[sel].float().mean()):.3f}"
          f"  agreement {float((truth_lab[sel] == online_lab[sel]).float().mean()):.3f}", flush=True)
for name, lab in (("truth label", truth_lab), ("online label", online_lab)):
    good, bad = lab, ~lab
    print(f"  {name:12s}: |M| good {float(Mabs[good].mean()):.2f} vs bad {float(Mabs[bad].mean()):.2f}; "
          f"trit at 0 among good {float((trit0[good] == 0).float().mean()):.2f} vs bad {float((trit0[bad] == 0).float().mean()):.2f}", flush=True)
