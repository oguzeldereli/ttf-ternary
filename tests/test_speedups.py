"""Correctness of the speed changes (tiny sizes, light GPU load).

  python -m tests.test_speedups
"""
import torch
import torch.nn.functional as F
from ttf.kernel import pack_rows, unpack_rows, lookahead_filter, _P3
from ttf.config import PRESETS
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf import model as M

dev = "cuda"
torch.manual_seed(0)


def pack_ref(w):
    N, K = w.shape
    K5 = (K + 4) // 5
    x = (w.to(torch.int16) + 1)
    if K5 * 5 - K:
        x = torch.cat([x, x.new_zeros(N, K5 * 5 - K)], 1)
    return (x.view(N, K5, 5) * torch.tensor(_P3, dtype=torch.int16, device=w.device)).sum(2).to(torch.uint8)


# 1. pack_rows cache: identical bytes
for K in (768, 770, 3072):
    w = torch.randint(-1, 2, (64, K), device=dev, dtype=torch.int8)
    assert torch.equal(pack_rows(w), pack_ref(w)), K
print("pack_rows: identical")

# 2. fused look-ahead filter == reference (2 passes)
for K in (768, 771):
    N = 96
    q0 = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8)
    step = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8) * (torch.rand(N, K, device=dev) < 0.2)
    q1 = (q0 + step).clamp(-1, 1).to(torch.int8)
    D = (q1 - q0).to(torch.int8)
    g, g2, g3 = (torch.randn(N, K, device=dev) for _ in range(3))
    # reference (the previous implementation)
    keep = D != 0
    keep &= D.float() * (g + g2) < 0
    ref1 = pack_rows(q0 + D * keep.to(torch.int8))
    keep &= D.float() * (g + g3) < 0
    ref2 = pack_rows(q0 + D * keep.to(torch.int8))
    # fused
    p0, pc = pack_rows(q0), pack_rows(q1)
    c1 = lookahead_filter(p0, pc, g, g2)
    assert torch.equal(pc, ref1), "pass 1"
    assert int(c1[0]) == int((D != 0).sum())
    c2 = lookahead_filter(p0, pc, g, g3)
    assert torch.equal(pc, ref2), "pass 2"
    assert int(c2[1]) == int(keep.sum())
print("lookahead_filter: identical to reference over 2 passes")


# 3/4. compile and partial checkpointing: same loss and same weight gradients
def run(compile_=False, ckpt_skip=0):
    torch.manual_seed(0)
    m = build_kernel_transformer(PRESETS["small"], grad_checkpoint=True, beta=True,
                                 int8=True).to(dev).train()
    m.ckpt_skip = ckpt_skip
    Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
    for l in Ls: l.capture = True                      # gradients only, no flips
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 32000, (2, 256), generator=g).to(dev)
    y = torch.randint(0, 32000, (2, 256), generator=g).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, y)[1]
    loss.backward()
    return loss.item(), [l.gw.float().clone() for l in Ls]


base_l, base_g = run()
# rounding reference: SwiGLU with a single rounding (what a fused kernel does), eager
M._FN["swiglu"] = lambda a, b: (F.silu(a.float()) * b.float()).to(a.dtype)
ref_l, ref_g = run()
ref_cos = min(F.cosine_similarity(a.flatten(), b.flatten(), 0).item() for a, b in zip(base_g, ref_g))
print(f"rounding reference (eager, one-rounding SwiGLU): loss {ref_l:.6f}, min grad cosine {ref_cos:.6f}")
M._FN["swiglu"] = M._swiglu
for name, kw in (("ckpt_skip=6", dict(ckpt_skip=6)), ("compile", dict(compile_=True))):
    if kw.get("compile_"):
        M.enable_compile()
    l, gs = run(ckpt_skip=kw.get("ckpt_skip", 0))
    cos = min(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()
              for a, b in zip(base_g, gs))
    print(f"{name}: loss {l:.6f} vs {base_l:.6f}, min per-layer grad cosine {cos:.6f}")
    assert abs(l - base_l) < 2e-3 and cos > min(0.999, ref_cos - 0.002), name
print("OK")
