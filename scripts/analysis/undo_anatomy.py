"""How many flips does --undo take back, and how many more could a stronger undo take back? From a mid-training
checkpoint of the recipe (rank-r momentum, beta 1, vnorm, gate, spend 3), the trainer's step is replayed exactly (one
subspace step of the momentum with the new batch, factored v, sign gate, p = rate * min(|S| / (3 mean|S|), 1), moves
blocked at +-1, spend U += c * mean|M| * (D V); the dry friction only rescales M and is left out), with the flip count
set to the run's own count at that step (from its train.log).

Part 1, one step: step A flips (moves d); step B's batch and momentum judge them. Undo rules on A's moves:
  current   d * g_B > 0 and sign(S_B) = d        (what --undo does)
  M only    sign(S_B) = d                         (M_B without spend: how much of that is spend's doing)
  g only    d * g_B > 0                           (one batch)
  g4        d * mean(g_B + 3 more batches) > 0
  oracle    d * T > 0, T = mean gradient of 16 batches at the new state (also the ground truth: "uphill")
For each: share of A's moves, precision (share truly uphill), recall (share of uphill moves caught), first-order gain
sum(d * T), and the real held-out change of taking them back.
Part 2, ten sequential steps with each undo rule applied every step (window k: moves made in the last k steps), same
batches and random numbers for all; held-out change after the ten steps.
  python -m scripts.analysis.undo_anatomy RUN STEP [SPEND]
"""
import sys, re, numpy as np, torch
from scripts.analysis.testbench import Bench
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
SPEND = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
NT = 16
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
log = open(f"checkpoints/{RUN}/train.log").read()
FR = float(re.findall(rf"step +{ST} .*?flip ([\d.]+)%", log)[0]) / 100
NTOT = sum(l.N * l.K for l in Ls); PER = FR * NTOT
LR0 = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]
gen = torch.Generator().manual_seed(4242)


def gmean(D, n, g_=None, sq=False):
    """mean gradient of n batches at state D (and the mean of squares' row / column means)"""
    B.set_trits(D if D is not None else [torch.zeros_like(t) for t in B.T0])
    m = r = c = None
    for _ in range(n):
        o = B.grad([get_batch(tr, 16, 2048, "cuda", g_ or gen)]); o = [o[k].float() / n for k in names]
        m = o if m is None else [a + b for a, b in zip(m, o)]
        if sq:
            r = [(x * x).mean(1) * n for x in o] if r is None else [a + (x * x).mean(1) * n for a, x in zip(r, o)]
            c = [(x * x).mean(0) * n for x in o] if c is None else [a + (x * x).mean(0) * n for a, x in zip(c, o)]
        del o
    B.set_trits([torch.zeros_like(t) for t in B.T0])
    return (m, list(zip(r, c))) if sq else m


# factored v from NT batches at the start (the trainer keeps an EMA 0.99; not in the checkpoint)
_, RC0 = gmean(None, NT, sq=True)
L0 = B.held_out()
print(f"{RUN} @{ST}: {NTOT / 1e6:.0f}M ternary weights, {PER / 1e3:.0f}k flips per step ({FR * 100:.3f}%, from its "
      f"log); spend {SPEND}; held-out {L0:.4f}", flush=True)


def mom_step(st, g):
    """the trainer's momentum update (beta 1) and Adam-normalized signal; returns S (pre-gate), gm (raw mean|M|)"""
    S, GM = [], []
    for i in range(len(Ls)):
        U, V = st["UV"][i]
        Vn = torch.linalg.qr(V @ (U.T @ U) + g[i].T @ U)[0]
        U = U @ (V.T @ Vn) + g[i] @ Vn; V = Vn; st["UV"][i] = (U, V)
        R, C = st["RC"][i]; g2 = g[i] ** 2
        R = 0.99 * R + 0.01 * g2.mean(1); C = 0.99 * C + 0.01 * g2.mean(0); st["RC"][i] = (R, C)
        M = U @ V.T; GM.append(M.abs().mean().clamp_min(1e-12))
        S.append(M / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30))
    return S, GM


def flip(st, S, GM, g, D, rng):
    """gate, p ~ min(|S| / (3 mean|S|), 1), scaled to PER flips; moves applied to D; spend; returns the moves"""
    P, MV = [], []
    for i in range(len(Ls)):
        s = S[i]; gmS = s.abs().mean().clamp_min(1e-12)
        s = s * (s.sign() == g[i].sign()); mv = -s.sign()
        cur = B.T0[i] + D[i]
        ok = ((cur.float() + mv).abs() <= 1) & (s != 0)
        P.append((s.abs() / (3 * gmS)).clamp(max=1) * ok); MV.append(mv)
    sc = PER / max(sum(float(p.sum()) for p in P), 1e-12)
    moves = []
    for i in range(len(Ls)):
        fire = torch.rand(P[i].shape, generator=rng, device="cuda") < (P[i] * sc).clamp(max=1)
        d = torch.where(fire, MV[i], torch.zeros_like(MV[i])).to(torch.int8)
        D[i] = (D[i] + d).to(torch.int8); moves.append(d)
        if SPEND:
            U, V = st["UV"][i]; st["UV"][i] = (U + SPEND * GM[i] * (d.float() @ V), V)
    return moves


def new_state():
    return {"UV": [(U.clone(), V.clone()) for U, V in LR0], "RC": [(R.clone(), C.clone()) for R, C in RC0]}


# ---------------- part 1: one step
st = new_state(); st_ns = new_state(); D = [torch.zeros_like(t) for t in B.T0]
rng = torch.Generator(device="cuda").manual_seed(1)
gA = gmean(D, 1)
S, GM = mom_step(st, gA); mom_step(st_ns, gA)
dA = flip(st, S, GM, gA, D, rng)
LA = B.held_out(D)
gB = gmean(D, 1); g4 = [(a + 3 * b) / 4 for a, b in zip(gB, gmean(D, 3))]
T = gmean(D, NT)
SB, _ = mom_step(st, gB); SBns, _ = mom_step(st_ns, gB)
nA = sum(int((d != 0).sum()) for d in dA)
up = [(d != 0) & (d.float() * T[i] > 0) for i, d in enumerate(dA)]
nup = sum(int(u.sum()) for u in up)
print(f"\npart 1: step A made {nA / 1e3:.0f}k moves, held-out {LA - L0:+.5f}; uphill on T (16 batches) after the move: "
      f"{100 * nup / nA:.1f}%")
rules = {
    "current (g_B and S_B)": lambda i, d: (d.float() * gB[i] > 0) & (SB[i].sign() == d),
    "S_B only": lambda i, d: SB[i].sign() == d,
    "S_B only, no spend": lambda i, d: SBns[i].sign() == d,
    "g_B only": lambda i, d: d.float() * gB[i] > 0,
    "g4 (4 batches)": lambda i, d: d.float() * g4[i] > 0,
    "g4 and S_B": lambda i, d: (d.float() * g4[i] > 0) & (SB[i].sign() == d),
    "oracle (T)": lambda i, d: d.float() * T[i] > 0,
}
print(f"{'undo rule':24s} {'undone':>12s} {'precision':>10s} {'recall':>8s} {'1st-order gain':>15s} {'held-out':>10s}")
for nm, f in rules.items():
    sel = [(d != 0) & f(i, d) for i, d in enumerate(dA)]
    n = sum(int(s.sum()) for s in sel); tp = sum(int((s & u).sum()) for s, u in zip(sel, up))
    gain = sum(float((dA[i].float() * T[i] * sel[i]).sum()) for i in range(len(Ls)))
    Du = [(D[i] - dA[i] * sel[i]).to(torch.int8) for i in range(len(Ls))]
    print(f"{nm:24s} {n / 1e3:6.0f}k {100 * n / nA:4.1f}% {100 * tp / max(n, 1):9.1f}% {100 * tp / max(nup, 1):7.1f}% "
          f"{gain:+15.3e} {B.held_out(Du) - LA:+10.5f}", flush=True)
del T, g4, SB, SBns, st, st_ns

# ---------------- part 2: ten sequential steps with each undo rule
K = 10
def run(rule, win=1, nb=1):
    st = new_state(); D = [torch.zeros_like(t) for t in B.T0]
    rng = torch.Generator(device="cuda").manual_seed(2); bg = torch.Generator().manual_seed(777)
    age = [torch.full_like(t, 99) for t in B.T0]; last = [torch.zeros_like(t) for t in B.T0]
    nund = nmov = 0
    for k in range(K):
        g = gmean(D, 1, bg)          # the training batch: the same sequence for every rule
        gu = g if nb <= 1 else [(a + (nb - 1) * b) / nb for a, b in zip(g, gmean(D, nb - 1, torch.Generator().manual_seed(5000 + k)))]
        if rule == "oracle":
            gu = gmean(D, NT, torch.Generator().manual_seed(6000 + k))
        S, GM = mom_step(st, g)
        if rule != "none" and k > 0:
            for i in range(len(Ls)):
                d = last[i]; cand = (d != 0) & (age[i] < win) & (d.float() * gu[i] > 0)
                if rule == "current": cand &= S[i].sign() == d
                D[i] = (D[i] - d * cand).to(torch.int8); last[i] = torch.where(cand, 0, last[i]).to(torch.int8)
                nund += int(cand.sum())
        mv = flip(st, S, GM, g, D, rng)
        for i in range(len(Ls)):
            m = mv[i] != 0
            last[i] = torch.where(m, mv[i], last[i]).to(torch.int8); age[i] = torch.where(m, 0, age[i] + 1)
            nmov += int(m.sum())
    nnet = sum(int((d != 0).sum()) for d in D)
    return B.held_out(D) - L0, nund, nmov, nnet


print(f"\npart 2: {K} sequential steps of ~{PER / 1e3:.0f}k flips, each undo rule applied every step")
print(f"{'undo rule':34s} {'held-out':>10s} {'undone':>9s} {'flips':>8s} {'net changes':>12s}")
for nm, rule, win, nb in (("none", "none", 1, 1), ("current (g and S, last step)", "current", 1, 1),
                          ("current, moves of the last 4 steps", "current", 4, 1),
                          ("g only, last step", "g", 1, 1), ("g4 (4 batches), last step", "g", 1, 4),
                          ("g4, last 4 steps", "g", 4, 4), ("oracle T (16 batches), last step", "oracle", 1, 1)):
    d, nu, nm_, nn = run(rule, win, nb)
    print(f"{nm:34s} {d:+10.5f} {nu / 1e3:8.0f}k {nm_ / 1e3:7.0f}k {nn / 1e3:11.0f}k", flush=True)
print("done")
