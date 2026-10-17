"""True-gradient alignment and noise of each momentum mechanism along its own training run. At a snapshot, the
arm's mechanism is started from the saved low-rank momentum (warm) and runs 66 steps of flips (float tail frozen,
the snapshot's flip rate); at steps 33 and 66 its flip signal is scored against a fresh 64-batch true gradient:
  cos         cosine(signal, true gradient now)
  prec        sign right on the signal's top 1% (0.5 = chance)
  snr         cos^2 / (1 - cos^2): energy of the signal along the true gradient vs the rest
  cos_batch   cosine(signal, this step's single-batch gradient): how much of the signal is the current batch
  rot         cosine(true gradient at step 66, true gradient at step 33): how fast the target moves
  dL          held-out loss change since the snapshot
Arms: V0 plain momentum (F = beta F + g); V2 move correction alone (F += gain delta before the update, delta =
same-batch gradient change caused by the last move); V3 the bench's both-arm (target from the transported
average, curvature re-measured every step, unfloored); V1 / U1 / U0 = the trainer's own mech_step (v1; user with
gain 1; user with gain 0).
  python -m scripts.analysis.mech_along ARM RUN STEP[,STEP...]      e.g. U1 mech_user_b131 5000,7000,9000
"""
import sys, math, types, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from ttf.kernel import fused_flip, unpack_rows
from ttf.train import get_batch, mech_step

ARM, RUN, STEPS = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
import os
BETA, NT, K = 0.97, int(os.environ.get('NT', 64)), int(os.environ.get('K', 66))
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]          # one micro-batch of 16 (as the trainer)


def rate_at(step):                                              # the trainer's cosine flip-rate schedule
    prog = min(1.0, (step - 30) / (9155 - 30))
    return 0.02 * 0.5 * (1 + math.cos(math.pi * prog))


def cat(xs): return torch.cat([x.flatten().float() for x in xs])


for st in STEPS:
    B = Bench(f"checkpoints/{RUN}/ckpt_{st}.pt")
    Ls, names = B.Ls, B.names
    rate = rate_at(st)
    for l in Ls: l.rate = rate; l.g_ref = G_REF; l.track = False
    tail = [p for _, p in B.tail]

    def grad(batch):
        g = B.grad(batch); return [g[n] for n in names]

    def truth(seed):
        s = batches(seed); acc = None
        for _ in range(NT):
            g = grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
        return cat([a / NT for a in acc])

    def held():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return float(np.mean([B.m(x, y)[1].item() for x, y in B.VB]))

    warm = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]
    L0 = held(); s = batches(4040 + st); out = []; T33 = None
    if ARM in ("V1", "U1", "U0"):
        args = types.SimpleNamespace(mech="v1" if ARM == "V1" else "user", lr_beta=BETA, mech_beta_d=0.0,
                                     mech_beta_s=0.8, mech_gain=1.0 if ARM == "U1" else 0.0)
        mstate = {"warm": {i: uv for i, uv in enumerate(warm)}}
    else:
        F = [U @ V.T for U, V in warm]; S = 1 / (1 - BETA); prev = None; D = None; h = None
    for k in range(1, K + 1):
        batch = next(s); x, y = batch[0]
        if ARM in ("V1", "U1", "U0"):
            for l in Ls: l.capture = True
            B.m.train()
            for p in tail: p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                B.m(x, y)[1].backward()
            g = [l.gw.float().clone() for l in Ls]
            mech_step(B.m, mstate, st + k, args, x, y, tail)
            for p in tail: p.grad = None
            sig = [(-mstate[i]["D"]) if (ARM == "V1" and "D" in mstate[i]) else mstate[i]["F"] for i in range(len(Ls))]
        else:
            g = grad(batch)
            if ARM in ("V2", "V3") and prev is not None:
                pb, pg, pm = prev
                d = [a - b for a, b in zip(grad(pb), pg)]
                F = [f + d_ for f, d_ in zip(F, d)] if ARM == "V2" else [f + S * d_ for f, d_ in zip(F, d)]
                if ARM == "V3":
                    h = [max(float((dd * mm).sum()) / max(float((mm * mm).sum()), 1e-30), 1e-12) for dd, mm in zip(d, pm)]
            F = [BETA * f + x_ for f, x_ in zip(F, g)]; S = BETA * S + 1
            if ARM == "V3" and h is not None:
                D = [-(f / S) / hh for f, hh in zip(F, h)] if D is None else \
                    [BETA * (dd - mm) + (1 - BETA) * (-(f / S) / hh) for dd, mm, f, hh in zip(D, prev[2], F, h)]
            sig = [-d for d in D] if (ARM == "V3" and D is not None) else F
            before = [unpack_rows(l.wpacked, l.K).float() for l in Ls]
            for i, (l, sg) in enumerate(zip(Ls, sig)):
                fused_flip(l.wpacked, sg.contiguous(), rate, G_REF, 7000 + 131 * k + i, gmean=sg.abs().mean().clamp_min(1e-12))
            move = [unpack_rows(l.wpacked, l.K).float() - b for l, b in zip(Ls, before)]
            prev = (batch, g, move)
        if k in (K // 2, K):
            T = truth(900 + k + st); sv = cat(sig); gv = cat(g)
            c = float(sv @ T / (sv.norm() * T.norm()))
            a = sv.abs(); thr = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)].quantile(0.99)
            prec = float(((sv > 0) == (T > 0))[a >= thr].float().mean())
            cb = float(sv @ gv / (sv.norm() * gv.norm()))
            rot = float(T @ T33 / (T.norm() * T33.norm())) if T33 is not None else float("nan")
            T33 = T if k == K // 2 else T33
            print(f"{ARM} {RUN} @{st} (+{k:2d}): cos {c:+.3f}  prec {prec:.3f}  snr {c * c / max(1 - c * c, 1e-9):.3f}  "
                  f"cos_batch {cb:+.3f}  rot33-66 {rot:+.3f}  dL {held() - L0:+.4f}", flush=True)
    del B; torch.cuda.empty_cache()
