"""Which layers take the trit moves in a run, and when? From a run's metrics.jsonl (flip_frac_layers: per ternary layer,
the share of its weights that moved that step), per layer: the cumulative share of its weights moved (sum over steps),
its share of the run's moves (weighted by layer size, from the checkpoint), and the step range where it moved.
Layers in the kernel order: 7 per block (q, k, v, o, gate, up, down).
    python scripts/analysis/layer_moves.py RUN_DIR [--top 15] [--from STEP]
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--top", type=int, default=15); ap.add_argument("--from_step", type=int, default=0)
a = ap.parse_args()
T = ["q", "k", "v", "o", "gate", "up", "down"]
steps, F = [], []
for l in open(os.path.join(a.run, "metrics.jsonl")):
    d = json.loads(l)
    if "flip_frac_layers" in d and d.get("step", 0) >= a.from_step:
        steps.append(d["step"]); F.append(d["flip_frac_layers"])
F = np.array(F); steps = np.array(steps)
n = F.shape[1]
sizes = None
ck = os.path.join(a.run, "ckpt.pt")
if os.path.exists(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)["model"]
    ws = [k for k in sd if k.endswith("wpacked") and "tok_emb" not in k and "mag_" not in k]
    if len(ws) == n:
        sizes = np.array([sd[k].shape[0] * sd[k].shape[1] * 5 for k in ws], dtype=float)   # 5 trits per byte
        names = ws
cum = F.sum(0)
print(f"{a.run}: steps {steps.min()}-{steps.max()} ({len(steps)} logged), {n} layers")
w = cum * sizes if sizes is not None else cum
order = np.argsort(-cum)
print(f"\ntop {a.top} layers by share of their weights moved (cumulative):")
print("  layer          moved   share of all moves   active steps (first / median / last move-weighted)")
for i in order[:a.top]:
    f = F[:, i]; c = np.cumsum(f) / max(f.sum(), 1e-30)
    s_first, s_med, s_last = (steps[np.searchsorted(c, q)] for q in (0.05, 0.5, 0.95))
    print(f"  {i // 7:3d}.{T[i % 7]:5s}   {cum[i]:8.4f}   {w[i] / w.sum():8.3f}            {s_first} / {s_med} / {s_last}")
print("\nby type (mean cumulative moved share over depth thirds):")
thirds = np.array_split(np.arange(n // 7), 3)
for j, t in enumerate(T):
    print(f"  {t:5s} " + " ".join(f"{np.mean([cum[7 * b + j] for b in tt]):.4f}" for tt in thirds))
print("\nblock 0-3 by type:")
for b in range(min(4, n // 7)):
    print(f"  block {b}: " + " ".join(f"{T[j]} {cum[7 * b + j]:.4f}" for j in range(7)))
