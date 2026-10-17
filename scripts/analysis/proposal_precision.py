"""Why do momentum proposals beat gradient proposals under the same look-ahead filter?

At a --lowrank checkpoint: propose flips with the rule from M (A) and from one batch's g (B),
filter both with cross-batch look-ahead x2 (the trainer's test, same batches for both), and score
every set against the true gradient gbar (mean over K held-out training batches):
  precision   share of flips with Delta * gbar < 0 (downhill on the true gradient)
  gain/flip   -sum(Delta * gbar) per flip (first-order true loss decrease per flip)
  M-agree     share of B's flips whose direction -sign(M) M would also propose
and the held-out loss change of one step for A-kept, B-kept, and a random subset of B-kept
matched to A-kept's count (is fewer flips alone the effect?).

  python -m scripts.analysis.proposal_precision [ckpt] [K]
"""
import sys, numpy as np, torch
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch

dev = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/lm_lowrank256_xb2_diag/ckpt_340.pt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BS, MICRO, SEQ, RATE, G_REF, NVAL = 16, 4, 2048, 0.0199, 3.0, 8

b = torch.load(CKPT, map_location="cpu", weights_only=False)
m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                             int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"), g_ref=G_REF)
for p in m.float_tail_parameters():
    p.data = p.data.float()
m.load_state_dict(b["model"], strict=False)
m = m.to(dev).train()
Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
P0 = [l.wpacked.clone() for l in Ls]
T0 = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]
UV = b["lowrank"]
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
print(f"checkpoint step {b['step']}, K={K}")


def batches(seed, n):
    g = torch.Generator().manual_seed(seed)
    return [[get_batch(train, MICRO, SEQ, dev, g) for _ in range(BS // MICRO)] for _ in range(n)]


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


def set_packed(P):
    for l, p in zip(Ls, P): l.wpacked.copy_(p)


VB = [get_batch(val, BS // 2, SEQ, dev, torch.Generator().manual_seed(555 + i)) for i in range(NVAL)]


def held_out():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([m(x, y)[1].item() for x, y in VB]))


# the step's own batch g, two look-ahead batches, and K truth batches (all different)
B0, LA1, LA2 = batches(101, 1)[0], batches(202, 1)[0], batches(303, 1)[0]
set_packed(P0)
g = grad(B0)
gbar = [torch.zeros_like(t) for t in g]
for bt in batches(404, K):
    for a, t in zip(gbar, grad(bt)): a += t / K
L0 = held_out()


def propose(signals, seed):
    set_packed(P0)
    for i, (l, s) in enumerate(zip(Ls, signals)):
        fused_flip(l.wpacked, s, RATE, G_REF, seed + i, gmean=s.abs().mean().clamp_min(1e-12).item())
    return [unpack_rows(l.wpacked, l.K).to(torch.int8) - t0 for l, t0 in zip(Ls, T0)]


def lookahead(D):
    """the trainer's filter: keep flip iff D * (g + g') < 0, twice, on the two look-ahead batches"""
    for bt in (LA1, LA2):
        set_packed([pack_rows((t0 + d).to(torch.int8)) for t0, d in zip(T0, D)])
        g2 = grad(bt)
        D = [d * ((d.float() * (a + c)) < 0).to(torch.int8) for d, a, c in zip(D, g, g2)]
    return D


def score(D, name):
    n = sum(int((d != 0).sum()) for d in D)
    down = sum(int(((d != 0) & (d.float() * gb < 0)).sum()) for d, gb in zip(D, gbar))
    gain = -sum(float((d.float() * gb).sum()) for d, gb in zip(D, gbar))
    Mag = sum(int(((d != 0) & (d.float() * M.sign() < 0)).sum()) for d, M in zip(D, Ms))
    set_packed([pack_rows((t0 + d).to(torch.int8)) for t0, d in zip(T0, D)])
    dl = held_out() - L0
    print(f"  {name:34s} flips {n / 1e3:8.1f}k  precision {down / max(n, 1):.3f}  gain/flip {gain / max(n, 1):.3e}"
          f"  M-agree {Mag / max(n, 1):.3f}  held-out dL {dl:+.4f}")
    return n


Ms = [(U.to(dev).float() @ V.to(dev).float().T) for U, V in UV]
DA = propose(Ms, 1234)
DB = propose(g, 1234)
print(f"held-out loss at the checkpoint {L0:.4f}")
nA = score(DA, "A proposals (from M)")
score(DB, "B proposals (from g)")
KA, KB = lookahead(DA), lookahead(DB)
nKA = score(KA, "A kept (M proposes, look-ahead)")
nKB = score(KB, "B kept (g proposes, look-ahead)")
gen = torch.Generator(device=dev).manual_seed(9)
frac = nKA / max(nKB, 1)
KBs = [d * (torch.rand(d.shape, device=dev, generator=gen) < frac).to(torch.int8) for d in KB]
score(KBs, f"B kept, random {frac:.2f} subset")
both = [a * (a == c).to(torch.int8) for a, c in zip(KA, KB)]
score(both, "kept by both A and B")
