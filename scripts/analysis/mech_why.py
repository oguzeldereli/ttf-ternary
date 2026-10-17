"""Where does momentum lose alignment with the true gradient once flips start? From a no-look-ahead snapshot with its
saved momentum, run plain momentum (F = beta F + g, flips from F at the snapshot's rate, float tail frozen) for 33
steps, recording each weight's last flip (step and direction). Then, against a fresh 64-batch true gradient T, per
group of weights:
  flipped 1 / 2-5 / 6-10 / 11-33 steps ago, not flipped in the window, stuck (trit at +-1 and the signal pushes past it)
  share of weights, share of the signal's energy, cosine and top-1% sign precision of the signal vs T in the group,
  and for flipped groups the share where T now pushes the weight back (undo = the flip overshot or went stale)
Also: the whole signal's cosine with and without the recently flipped weights.
  python -m scripts.analysis.mech_why RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from ttf.kernel import fused_flip, unpack_rows
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
BETA, NT, K = 0.97, 64, 33
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]


prog = min(1.0, (ST - 30) / (9155 - 30)); rate = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names


def grad(b):
    g = B.grad(b); return [g[n] for n in names]


F = [(U.cuda().float() @ V.cuda().float().T) for U, V in B.b["lowrank"]]
last = [torch.full(l.wpacked.shape[:1] + (l.K,), -1, dtype=torch.int16, device="cuda") for l in Ls]
dirn = [torch.zeros(l.wpacked.shape[0], l.K, dtype=torch.int8, device="cuda") for l in Ls]
s = batches(5151)
for k in range(1, K + 1):
    g = grad(next(s))
    F = [BETA * f + x for f, x in zip(F, g)]
    for i, (l, f) in enumerate(zip(Ls, F)):
        before = unpack_rows(l.wpacked, l.K).to(torch.int8)
        fused_flip(l.wpacked, f.contiguous(), rate, G_REF, 3000 + 131 * k + i, gmean=f.abs().mean().clamp_min(1e-12))
        mv = unpack_rows(l.wpacked, l.K).to(torch.int8) - before
        nz = mv != 0
        last[i][nz] = k; dirn[i][nz] = mv[nz]
acc = None; st2 = batches(777)
for _ in range(NT):
    gg = grad(next(st2)); acc = gg if acc is None else [a + b for a, b in zip(acc, gg)]
T = torch.cat([(a / NT).flatten() for a in acc]); Fv = torch.cat([f.flatten() for f in F])
Lv = torch.cat([x.flatten() for x in last]).int(); Dv = torch.cat([d.flatten() for d in dirn]).float()
Tr = torch.cat([unpack_rows(l.wpacked, l.K).flatten().float() for l in Ls])
age = torch.where(Lv >= 0, K + 1 - Lv, torch.full_like(Lv, 10_000))      # 1 = flipped at the last step
stuck = (Lv < 0) & (Tr != 0) & ((-Fv.sign()) == Tr)                     # at the bound, signal pushes past it
groups = {"flipped 1 step ago": age == 1, "flipped 2-5 ago": (age >= 2) & (age <= 5),
          "flipped 6-10 ago": (age >= 6) & (age <= 10), "flipped 11-33 ago": (age >= 11) & (age <= K),
          "not flipped, stuck at +-1": stuck, "not flipped, movable": (Lv < 0) & ~stuck}
E = float((Fv * Fv).sum())
a = Fv.abs(); thr = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)].quantile(0.99); top = a >= thr
print(f"{RUN} @{ST}, rate {rate:.4f}, {K} steps of plain momentum flips; whole signal vs truth: "
      f"cos {float(Fv @ T / (Fv.norm() * T.norm())):+.3f}, top-1% precision {float(((Fv > 0) == (T > 0))[top].float().mean()):.3f}")
print(f"{'group':28s} {'weights':>8s} {'signal energy':>13s} {'cos':>7s} {'prec top1%':>10s} {'undo (T pushes back)':>21s}")
for name, m in groups.items():
    if m.sum() == 0: continue
    f, t = Fv[m], T[m]
    c = float(f @ t / (f.norm() * t.norm()).clamp_min(1e-30))
    tm = m & top; p = float(((Fv > 0) == (T > 0))[tm].float().mean()) if tm.any() else float("nan")
    und = ""
    if name.startswith("flipped"):
        und = f"{float(((T * Dv)[m] > 0).float().mean()):.3f}"          # gradient opposes the move: step back
    print(f"{name:28s} {float(m.float().mean()):8.4f} {float((f * f).sum()) / E:13.3f} {c:+7.3f} {p:10.3f} {und:>21s}")
keep = Lv < 0
print(f"signal vs truth on weights NOT flipped in the window: cos {float(Fv[keep] @ T[keep] / (Fv[keep].norm() * T[keep].norm())):+.3f}")
keep2 = keep & ~stuck
print(f"... and not stuck either (movable, unflipped): cos {float(Fv[keep2] @ T[keep2] / (Fv[keep2].norm() * T[keep2].norm())):+.3f}")
