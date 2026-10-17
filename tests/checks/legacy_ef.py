"""Spatial error feedback: plumbing and invariants.

1. alpha = 0 is bit-identical to EF off (same flips, same grad_x)
2. EF never changes WHICH weights flip at this layer (flip happens before the
   feedback is folded in), only the grad_x sent to earlier layers
3. alpha > 0 changes grad_x, stays finite, and the added term is gy-scaled
4. guards: evidence mode / grad accumulation are rejected
"""
import torch
from ttf.legacy import flip as F
from ttf.legacy.flip import KernelTernaryLinear, set_err_feedback, build_kernel_transformer
from ttf.legacy.config import ModelConfig

dev = "cuda"
torch.manual_seed(0)
K, N, B, T = 768, 2048, 4, 256
x0 = torch.randn(B, T, K, device=dev, dtype=torch.bfloat16)
gy = torch.randn(B, T, N, device=dev, dtype=torch.bfloat16)


def run(int8, ef, alpha, seed=1000):
    torch.manual_seed(1)
    lay = KernelTernaryLinear(K, N, beta=True, rate=2e-2, int8=int8).to(dev)
    lay.err_feedback, lay.ef_alpha = ef, alpha
    x = x0.clone().requires_grad_(True)
    y = lay(x)
    F._FLIP_SEED[0] = seed
    y.backward(gy)
    return x.grad.float(), lay.wpacked.clone()


for int8 in (True, False):
    g_off, w_off = run(int8, False, 0.0)
    g_a0, w_a0 = run(int8, True, 0.0)
    g_a1, w_a1 = run(int8, True, 1.0)
    tag = "int8" if int8 else "bf16"
    assert torch.equal(w_off, w_a0) and torch.equal(w_off, w_a1), f"{tag}: flips changed"
    assert torch.equal(g_off, g_a0), f"{tag}: alpha=0 is not identical to EF off"
    assert torch.isfinite(g_a1).all(), f"{tag}: non-finite grad_x"
    rel = ((g_a1 - g_off).norm() / g_off.norm()).item()
    cos = ((g_a1 * g_off).sum() / (g_a1.norm() * g_off.norm())).item()
    assert 0.05 < rel < 5.0, f"{tag}: feedback term has implausible size {rel}"
    flips = (w_off != run(int8, False, 0.0, seed=1000)[1]).sum().item()
    print(f"{tag}: alpha=0 identical to off | flips unchanged by EF | "
          f"alpha=1: |dgx|/|gx| = {rel:.3f}, cos(gx_ef, gx) = {cos:.3f}")

# guards
c = ModelConfig(vocab_size=1000, dim=64, n_layers=1, n_heads=4, n_kv_heads=4,
                hidden_dim=128, max_seq_len=32)
m = build_kernel_transformer(c, evidence=True, ev_bits=3)
try:
    set_err_feedback(m, True); raise AssertionError("evidence mode not rejected")
except ValueError:
    pass
m = build_kernel_transformer(c)
for l in m.modules():
    if isinstance(l, KernelTernaryLinear):
        l.accum_steps = 4
try:
    set_err_feedback(m, True); raise AssertionError("grad accumulation not rejected")
except ValueError:
    pass
print("guards: evidence mode and grad_accum>1 rejected")
# 5. sign: the feedback must help the earlier layer compensate, not amplify.
#    Old (wrong) sign = negative alpha. It must be clearly more harmful.
import torch.nn.functional as Fn
def post_step_delta(alpha, seeds=20):
    out = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        x = torch.randn(1, 1024, 256, device=dev)
        A = torch.nn.Linear(256, 256, bias=False).to(dev)
        L = KernelTernaryLinear(256, 256, beta=True, rate=0.3, int8=True).to(dev)
        L.err_feedback, L.ef_alpha = True, alpha
        tgt = torch.randn(1, 1024, 256, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l0 = Fn.mse_loss(L(A(x)).float(), tgt)
        F._FLIP_SEED[0] = 10_000 + seed
        l0.backward()
        with torch.no_grad():
            A.weight -= 0.05 * A.weight.grad / A.weight.grad.norm() * A.weight.norm()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append((Fn.mse_loss(L(A(x)).float(), tgt) - l0).item())
    return sum(out) / len(out)
d0, dfix, dold = post_step_delta(0.0), post_step_delta(1.0), post_step_delta(-1.0)
assert (dold - d0) > 2 * (dfix - d0), (d0, dfix, dold)
print(f"sign: harm vs no-feedback  fixed {dfix - d0:+.4f}  old {dold - d0:+.4f}")
print("OK: spatial error feedback")
