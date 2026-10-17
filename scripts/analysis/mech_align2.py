"""Why don't the mechanisms improve alignment? For one arm at one snapshot and one flip-rate multiple: run 33 steps of
the arm (float tail frozen, warm momentum), and at step 33 score the flip signal of that step against the true
gradient (64 batches) measured both BEFORE its flips (at the point where the flips are chosen: decision quality)
and AFTER them (what earlier measurements used). Also: signal vs this step's batch gradient, trits changed.
Arms: V0 plain, V2 move correction (gain 1), V1 and U1 (your design, gain 1) through the trainer's mech_step.
  python -m scripts.analysis.mech_align2 ARM RUN STEP RATE_MULT
"""
import sys, math, types, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from ttf.kernel import fused_flip, unpack_rows
from ttf.train import get_batch, mech_step

ARM, RUN, ST, MULT = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
BETA, NT, K = 0.97, 64, 33
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog)) * MULT
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
for l in Ls: l.rate = RATE; l.g_ref = G_REF; l.track = False
tail = [p for _, p in B.tail]


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
    return torch.cat([(a / NT).flatten() for a in acc])


def score(sv, T):
    c = float(sv @ T / (sv.norm() * T.norm()))
    a = sv.abs(); thr = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)].quantile(0.99)
    return c, float(((sv > 0) == (T > 0))[a >= thr].float().mean())


warm = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]
s = batches(4040 + ST); changed = 0
if ARM in ("V1", "U1"):
    args = types.SimpleNamespace(mech="v1" if ARM == "V1" else "user", lr_beta=BETA, mech_beta_d=0.0, mech_beta_s=0.8,
                                 mech_gain=1.0)
    mstate = {"warm": {i: uv for i, uv in enumerate(warm)}}
else:
    F = [U @ V.T for U, V in warm]; prev = None
for k in range(1, K + 1):
    batch = next(s); x, y = batch[0]
    Tpre = truth(900 + ST) if k == K else None                    # true gradient where step K's flips are chosen
    before = [unpack_rows(l.wpacked, l.K).clone() for l in Ls]
    if ARM in ("V1", "U1"):
        for l in Ls: l.capture = True
        for p in tail: p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            B.m(x, y)[1].backward()
        g = [l.gw.float().clone() for l in Ls]
        mech_step(B.m, mstate, ST + k, args, x, y, tail)
        for p in tail: p.grad = None
        sig = [(-mstate[i]["D"]) if (ARM == "V1" and "D" in mstate[i]) else mstate[i]["F"] for i in range(len(Ls))]
    else:
        g = grad(batch)
        if ARM == "V2" and prev is not None:
            pb, pg = prev
            F = [f + (a - b) for f, a, b in zip(F, grad(pb), pg)]
        F = [BETA * f + x_ for f, x_ in zip(F, g)]
        sig = F
        for i, (l, sg) in enumerate(zip(Ls, sig)):
            fused_flip(l.wpacked, sg.contiguous(), RATE, G_REF, 7000 + 131 * k + i, gmean=sg.abs().mean().clamp_min(1e-12))
        prev = (batch, g)
    changed += sum(int((unpack_rows(l.wpacked, l.K) != b).sum()) for l, b in zip(Ls, before))
    if k == K:
        sv = torch.cat([x_.flatten() for x_ in sig]); gv = torch.cat([x_.flatten() for x_ in g])
        Tpost = truth(900 + ST)
        cpre, ppre = score(sv, Tpre); cpost, ppost = score(sv, Tpost)
        rot = float(Tpre @ Tpost / (Tpre.norm() * Tpost.norm()))
        cb = float(sv @ gv / (sv.norm() * gv.norm()))
        print(f"{ARM:3s} {RUN} @{ST} rate x{MULT:<5}: BEFORE its flips cos {cpre:+.3f} prec {ppre:.3f} | AFTER cos {cpost:+.3f} "
              f"prec {ppost:.3f} | truth before vs after one step {rot:+.3f} | signal vs batch {cb:+.3f} | "
              f"trits changed in 33 steps {changed / 1e6:.2f}M", flush=True)
