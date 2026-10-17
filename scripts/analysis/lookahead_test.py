"""One-step test of a look-ahead flip filter at a mid-training checkpoint (kernel path).

A lone flip is never right-sized (curv_single.py: median |g|/H ~ 60), so the overshoot
seen when flips are taken together comes from their interaction. Judge each flip
against the step everyone else is taking:
  1. propose flips with the current rule (Delta),
  2. gradient g' at W + Delta on the same batch,
  3. keep flip i iff the midpoint slope still descends:  Delta_i * (g_i + g'_i) < 0
     (exact per-flip contribution for a quadratic, given all the other flips).
Stateless: g is only held within the step. Held-out loss change is measured on
EVAL_BATCHES fresh batches; a random subset of the proposal with the same count is
the control (fewer flips alone is known to help).
"""
import sys, numpy as np, torch
from ttf.layers import build_kernel_transformer, KernelTernaryLinear
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch
dev = "cuda"
CKPT = "checkpoints/lr_ctl/ckpt.pt"
BS, SEQ, EVAL_BATCHES, G_REF = 16, 2048, 16, 3.0
RATES = [float(r) for r in sys.argv[1:]] or [0.0194, 0.005, 0.05]

b = torch.load(CKPT, map_location="cpu", weights_only=False)
m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                             int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"),
                             g_ref=G_REF)
m.load_state_dict(b["model"], strict=False); m = m.to(dev).train()
Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
Q0 = [unpack_rows(l.wpacked, l.K).clone() for l in Ls]
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
xa, ya = get_batch(train, BS, SEQ, dev, torch.Generator().manual_seed(1001))
gv = torch.Generator().manual_seed(2002)
EV = [get_batch(val, BS, SEQ, dev, gv) for _ in range(EVAL_BATCHES)]


def set_q(Q):
    for l, q in zip(Ls, Q): l.wpacked.copy_(pack_rows(q))


def grad():
    for l in Ls: l.capture = True
    for p in m.parameters(): p.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(xa, ya)[1]
    loss.backward()
    for l in Ls: l.capture = False
    for p in m.parameters(): p.grad = None
    return loss.item(), [l.gw for l in Ls]


def held_out():
    tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in EV: tot += m(x, y)[1].item()
    return tot / len(EV)


set_q(Q0); L0 = held_out(); l0, g = grad()
g = [t.clone() for t in g]
print(f"held-out L0 {L0:.4f}  train-batch {l0:.4f}", flush=True)
print(f"{'rate':>7} {'variant':>22} {'flips':>10} {'dL held-out':>12}")
for ri, rate in enumerate(RATES):
    torch.manual_seed(ri)
    D = []
    for li, (l, q0, gw) in enumerate(zip(Ls, Q0, g)):
        wp = pack_rows(q0)
        fused_flip(wp, gw, rate, G_REF, 777 + 1000 * ri + li, gmean=gw.abs().mean().item())
        D.append((unpack_rows(wp, l.K) - q0).to(torch.int8))
    Q1 = [q0 + d for q0, d in zip(Q0, D)]
    set_q(Q1); L1 = held_out(); _, g1 = grad()
    keep_mid = [(d != 0) & (d.float() * (a + c) < 0) for d, a, c in zip(D, g, g1)]
    keep_end = [(d != 0) & (d.float() * c < 0) for d, c in zip(D, g1)]
    n1 = sum(int((d != 0).sum()) for d in D)
    print(f"{rate:7.4f} {'proposal (current rule)':>22} {n1:10d} {L1 - L0:+12.4f}", flush=True)
    for lab, K in (("midpoint filter", keep_mid), ("endpoint filter", keep_end)):
        nk = sum(int(k.sum()) for k in K)
        set_q([q0 + d * k.to(torch.int8) for q0, d, k in zip(Q0, D, K)]); Lk = held_out()
        # control: random subset of the proposal, same count per layer
        R = []
        for d, k in zip(D, K):
            idx = (d != 0).flatten().nonzero().squeeze(1)
            sel = idx[torch.randperm(idx.numel(), device=dev)[:int(k.sum())]]
            r = torch.zeros(d.numel(), dtype=torch.bool, device=dev); r[sel] = True
            R.append(r.view_as(d))
        set_q([q0 + d * r.to(torch.int8) for q0, d, r in zip(Q0, D, R)]); Lr = held_out()
        print(f"{'':7} {lab:>22} {nk:10d} {Lk - L0:+12.4f}   random same-count {Lr - L0:+.4f}", flush=True)
    del g1
    set_q(Q0)
