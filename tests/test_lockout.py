"""Per-weight flip lockout: invariants.

1. mode "once": a weight that flipped this epoch cannot flip again until reset
2. mode "noreversal": later flips in an epoch must keep the first flip's direction
3. the lock mask popcount equals the number of weights that flipped
4. reset_lock() reopens everything
"""
import torch
from ttf import flip as F
from ttf.kernel import unpack_rows, popcount, trit_diff, expand_trit_mask
from ttf.flip import KernelTernaryLinear

dev = "cuda"; K, N, M = 512, 768, 2048
torch.manual_seed(0)
x = torch.randn(1, M, K, device=dev, dtype=torch.bfloat16)
gy = torch.randn(1, M, N, device=dev, dtype=torch.bfloat16)


def step(lay, seed):
    xx = x.clone().requires_grad_(True)
    before = lay.wpacked.clone()
    F._FLIP_SEED[0] = seed
    lay(xx).backward(gy)
    return expand_trit_mask(trit_diff(before, lay.wpacked), lay.K), unpack_rows(before, lay.K)


for mode in ("once", "noreversal"):
    torch.manual_seed(1)
    lay = KernelTernaryLinear(K, N, beta=True, rate=0.25, int8=True).to(dev)
    lay.enable_lockout(10**9, mode)
    m1, w0 = step(lay, 7)
    w1 = unpack_rows(lay.wpacked, lay.K)
    dir1 = (w1 - w0)                       # +1 / -1 / 0
    n1 = int(m1.sum())
    lockbits = int(popcount(lay.lock).item())
    assert lockbits == n1, (lockbits, n1)
    ever, dirs = m1.clone(), dir1.clone()
    for s in range(2, 6):
        mk, wb = step(lay, 7 + s)
        wa = unpack_rows(lay.wpacked, lay.K)
        if mode == "once":
            assert not (mk & ever).any(), f"{mode}: a locked weight flipped again"
        else:
            d = (wa - wb)
            clash = mk & ever & (torch.sign(d) != torch.sign(dirs)) & (d != 0)
            assert not clash.any(), f"{mode}: {int(clash.sum())} reversals slipped through"
        dirs = torch.where(mk & ~ever, torch.sign(d if mode != "once" else (wa - wb)), dirs)
        ever = ever | mk
    print(f"{mode}: first step flipped {n1:,} | lock popcount matches | "
          f"{int(ever.sum()):,} distinct weights moved over 5 steps")
    lay.reset_lock()
    assert int(popcount(lay.lock).item()) == 0
    mk, _ = step(lay, 99)
    assert (mk & ever).any(), "after reset, previously locked weights still cannot flip"
    print(f"{mode}: reset reopens the mask ({int((mk & ever).sum()):,} re-flipped)")
print("OK: flip lockout")
