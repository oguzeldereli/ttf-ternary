"""TTF's accumulator at one checkpoint, by depth and matrix type: how close is the stored evidence A = U V^T to the
threshold? With a leak (tau), an entry whose mean step is m settles near tau * m; if that stays below theta the weight
never moves. So: |A| / theta quantiles, the share of entries above theta / 2, and the same restricted to the weights
that have never moved since init (the flip-tracking mask saved in the checkpoint), per depth sixth and type.
Runs on the CPU from a memory-mapped checkpoint (no GPU needed).
    python scripts/analysis/acc_state.py CKPT [--theta 16]
"""
import argparse, os, sys
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--theta", type=float, default=16.0)
a = ap.parse_args()
torch.set_num_threads(min(16, os.cpu_count()))
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
lr = ck["lowrank"]; touched = ck.get("touched") or {}
names = [n for n in touched] if touched else []
print(f"{a.ckpt}: step {ck['step']}, {len(lr)} ternary layers, flip masks {len(touched)}")
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
rows = []
g = torch.Generator().manual_seed(0)
for i, uv in enumerate(lr):
    if uv is None:
        continue
    U, V = (x.float() for x in uv[:2])
    A = (U @ V.T).abs().div_(a.theta)                     # N x K, in units of theta
    flat = A.view(-1)
    idx = torch.randint(0, flat.numel(), (2_000_000,), generator=g)   # sample for the quantiles
    s = flat[idx]
    q = torch.quantile(s, torch.tensor([0.5, 0.9, 0.99, 0.999])).tolist()
    half = (s > 0.5).float().mean().item()
    nev = None
    if names:
        m = touched[names[i]]                                 # packed uint8 [N, K5]: bit j of byte c = trit 5 c + j
        bits = ((m.long().unsqueeze(-1) >> torch.arange(5)) & 1).view(m.shape[0], -1)[:, :A.shape[1]]
        nm = (bits == 0).reshape(-1)                          # never moved since init
        if nm.numel() == flat.numel() and nm.any():
            sn = flat[nm][:2_000_000]
            nev = (nm.float().mean().item(), torch.quantile(sn[:1_000_000], torch.tensor([0.5, 0.9, 0.99])).tolist(),
                   (sn > 0.5).float().mean().item())
    rows.append((i, q, half, nev))
    del A, flat

nl = len(rows) // 7
six = np.array_split(np.arange(nl), 6)
def agg(sel, f):
    v = [f(r) for r in rows if r[0] // 7 in sel[0] and r[0] % 7 in sel[1] and f(r) is not None]
    return np.mean(v, 0) if v else None
print("\n|A| / theta by depth sixth (all types): p50 p90 p99 p99.9 | share > 0.5 | never-moved: share, p50 p90 p99, share > 0.5")
for k, t in enumerate(six):
    q = agg((set(t), set(range(7))), lambda r: r[1]); h = agg((set(t), set(range(7))), lambda r: r[2])
    nv = agg((set(t), set(range(7))), lambda r: [r[3][0], *r[3][1], r[3][2]] if r[3] else None)
    print(f"  sixth {k}: " + " ".join(f"{x:.3f}" for x in q) + f" | {h:.4f}" +
          (" | " + " ".join(f"{x:.3f}" for x in nv) if nv is not None else ""))
print("\nby type, deepest sixth vs middle (sixths 2-3): p99 of |A| / theta, share > 0.5")
for j, t in enumerate(TYPES):
    d = agg((set(six[-1]), {j}), lambda r: [r[1][2], r[2]]); m = agg((set(six[2]) | set(six[3]), {j}), lambda r: [r[1][2], r[2]])
    print(f"  {t:5s} deep p99 {d[0]:.3f} >0.5 {d[1]:.4f} | middle p99 {m[0]:.3f} >0.5 {m[1]:.4f}")
