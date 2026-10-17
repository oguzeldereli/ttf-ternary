"""Does the look-ahead filter get better like sqrt(n) with more batches?

At a --lowrank checkpoint: propose flips from M and from one batch's g (the flip rule at the run's own
rate there), then keep flip i iff  D_i * (g_i + sum_{j<=k} g'_ij) < 0  with g'_j the gradient at the
proposed point on k fresh batches (k = 1 is the trainer's single-pass cross-batch test). For each k,
score the kept set against the true gradient gbar (mean over TRUTH held-out batches, at the original
point): precision (share with D * gbar < 0), true first-order gain per flip, flips kept, and the
held-out loss change of applying it.

  python -m scripts.analysis.filter_sqrt_n CKPT RATE [TRUTH]
"""
import sys, json, math, numpy as np, torch
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch

dev = "cuda"
CKPT, RATE = sys.argv[1], float(sys.argv[2])
TRUTH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
BS, MICRO, SEQ, G_REF, NVAL = 16, 8, 2048, 3.0, 8
KS = (1, 2, 4, 8, 16, 32)

b = torch.load(CKPT, map_location="cpu", weights_only=False)
m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                             int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"), g_ref=G_REF)
for p in m.float_tail_parameters():
    p.data = p.data.float()
m.load_state_dict(b["model"], strict=False)
m = m.to(dev).train()
Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
T0 = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]
P0 = [l.wpacked.clone() for l in Ls]
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
print(f"checkpoint {CKPT} step {b['step']}, rate {RATE}, truth {TRUTH} batches", flush=True)


def stream(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, MICRO, SEQ, dev, g) for _ in range(BS // MICRO)]


def grad(batch):
    acc = [torch.zeros(l.N, l.K, device=dev) for l in Ls]
    for x, y in batch:
        for l in Ls: l.capture = True
        for p in m.parameters(): p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x, y)[1].backward()
        for a, l in zip(acc, Ls):
            a += l.gw.float(); l.gw = None
    for l in Ls: l.capture = False
    for p in m.parameters(): p.grad = None
    return [a / len(batch) for a in acc]


def set_trits(D):
    for l, t0, d in zip(Ls, T0, D):
        l.wpacked.copy_(pack_rows((t0 + d).to(torch.int8)))


VB = [get_batch(val, BS // 2, SEQ, dev, torch.Generator().manual_seed(555 + i)) for i in range(NVAL)]


def held_out():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([m(x, y)[1].item() for x, y in VB]))


truth_s, g_s = stream(404), stream(101)
g = grad(next(g_s))                               # the step's own batch
gbar = [torch.zeros_like(t) for t in g]
for _ in range(TRUTH):
    for a, t in zip(gbar, grad(next(truth_s))): a += t / TRUTH
L0 = held_out()
Ms = [(U.to(dev).float() @ V.to(dev).float().T) for U, V in b["lowrank"]]
print(f"held-out loss at the checkpoint {L0:.4f}", flush=True)


def propose(signals, seed=1234):
    for l, p in zip(Ls, P0): l.wpacked.copy_(p)
    for i, (l, s) in enumerate(zip(Ls, signals)):
        fused_flip(l.wpacked, s, RATE, G_REF, seed + i, gmean=s.abs().mean().clamp_min(1e-12))
    return [unpack_rows(l.wpacked, l.K).to(torch.int8) - t0 for l, t0 in zip(Ls, T0)]


def score(D):
    n = sum(int((d != 0).sum()) for d in D)
    down = sum(int(((d != 0) & (d.float() * gb < 0)).sum()) for d, gb in zip(D, gbar))
    gain = -sum(float((d.float() * gb).sum()) for d, gb in zip(D, gbar))
    return n, down / max(n, 1), gain / max(n, 1)


out = {"ckpt": CKPT, "step": b["step"], "rate": RATE, "L0": L0, "rows": []}
for name, sig in (("M", Ms), ("g", g)):
    D = propose(sig)
    n, prec, gpf = score(D)
    print(f"\n{name} proposals: {n / 1e3:.1f}k flips, precision {prec:.3f}, gain/flip {gpf:.3e}", flush=True)
    set_trits(D)                                   # look-ahead gradients at the proposed point
    la_s = stream(202 if name == "M" else 303)
    acc = [a.clone() for a in g]                  # g + sum of look-ahead gradients
    kept = {}
    for j in range(1, max(KS) + 1):
        for a, t in zip(acc, grad(next(la_s))): a += t
        if j in KS:
            kept[j] = [d * ((d.float() * a) < 0).to(torch.int8) for d, a in zip(D, acc)]
    print(f"  {'k':>3s} {'kept':>9s} {'precision':>10s} {'gain/flip':>11s} {'held-out dL':>12s}")
    for k in KS:
        n, prec, gpf = score(kept[k])
        set_trits(kept[k]); dl = held_out() - L0
        print(f"  {k:3d} {n / 1e3:8.1f}k {prec:10.3f} {gpf:11.3e} {dl:+12.4f}", flush=True)
        out["rows"].append({"prop": name, "k": k, "kept": n, "precision": prec, "gain_per_flip": gpf, "dL": dl})
    set_trits([torch.zeros_like(t) for t in T0])
json.dump(out, open(f"checkpoints/sqrtn_step{b['step']}.json", "w"))
