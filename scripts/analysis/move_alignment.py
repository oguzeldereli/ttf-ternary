"""Where did the trits actually move? Net displacement D = T_end - T_start between two checkpoints of a
run, compared with the momentum M and the true gradient gbar (mean over K batches) at the start and at
the end. For each reference R: the share of moved trits that went the way -sign(R) says (0.5 = chance),
and cos(D, -R) (0 = unrelated; in ~1M dimensions a random direction gives ~0.001).

  python -m scripts.analysis.move_alignment START.pt END.pt [K]
"""
import sys, numpy as np, torch
import torch.nn.functional as F
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.kernel import unpack_rows
from ttf.train import get_batch

dev = "cuda"
A, B = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
BS, MICRO, SEQ = 16, 4, 2048
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")


def load(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                                 int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"), g_ref=3.0)
    for p in m.float_tail_parameters():
        p.data = p.data.float()
    m.load_state_dict(b["model"], strict=False)
    m = m.to(dev).train()
    return b, m, [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]


def gbar(m, Ls):
    gen = torch.Generator().manual_seed(4242)
    acc = [torch.zeros(l.N, l.K, device=dev) for l in Ls]
    for _ in range(K * BS // MICRO):
        x, y = get_batch(train, MICRO, SEQ, dev, gen)
        for l in Ls: l.capture = True
        for p in m.parameters(): p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x, y)[1].backward()
        for a, l in zip(acc, Ls):
            a += l.gw.float(); l.gw = None
    for l in Ls: l.capture = False
    return [a / (K * BS // MICRO) for a in acc]


bA, mA, LA = load(A)
TA = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in LA]
GA = gbar(mA, LA)
MA = bA.get("lowrank")
del mA
bB, mB, LB = load(B)
TB = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in LB]
GB = gbar(mB, LB)
MB = bB.get("lowrank")
print(f"{A} (step {bA['step']}) -> {B} (step {bB['step']}), {bB['step'] - bA['step']} steps, K={K}")
D = [(tb - ta).float() for ta, tb in zip(TA, TB)]
moved = sum(int((d != 0).sum()) for d in D); tot = sum(d.numel() for d in D)
print(f"moved trits: {moved / 1e6:.2f}M of {tot / 1e6:.1f}M ({moved / tot * 100:.1f}%)")

refs = {"gbar at start": GA, "gbar at end": GB}
if MA is not None:
    refs["M at start"] = [(U.to(dev).float() @ V.to(dev).float().T) for U, V in MA]
if MB is not None:
    refs["M at end"] = [(U.to(dev).float() @ V.to(dev).float().T) for U, V in MB]
print(f"  {'reference R':16s} {'moves along -sign(R)':>22s} {'cos(D, -R)':>12s}   (per-layer mean; chance 0.5 / 0)")
for name, R in refs.items():
    agree, cs = [], []
    for d, r in zip(D, R):
        nz = d != 0
        agree.append(((d[nz] * r[nz]) < 0).float().mean().item())
        cs.append(F.cosine_similarity(d.flatten(), -r.flatten(), 0).item())
    print(f"  {name:16s} {np.mean(agree):22.3f} {np.mean(cs):12.4f}")
