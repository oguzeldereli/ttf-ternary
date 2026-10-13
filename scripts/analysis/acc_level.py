"""How full is each layer's accumulator? From a TTF checkpoint (lowrank state: A = U V^T per ternary layer), per layer:
its shape, the state's rank against min(N, K) (a rank >= min(N, K) holds every step exactly), rms(A) / theta and the
share of entries within reach of the bar (|A| >= theta / 2). Grouped by matrix type and depth third, with the run's
cumulative moved share per layer (metrics.jsonl) alongside, to see whether a layer's moves follow its accumulator level.
    python scripts/analysis/acc_level.py RUN_DIR --theta 64
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--theta", type=float, required=True)
a = ap.parse_args()
T = ["q", "k", "v", "o", "gate", "up", "down"]
ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location="cpu", weights_only=False)
st = ck["lowrank"]
F = [json.loads(l)["flip_frac_layers"] for l in open(os.path.join(a.run, "metrics.jsonl")) if "flip_frac_layers" in l]
cum = np.array(F).sum(0)
rows = []
for i, s in enumerate(st):
    U, V = s[0].float(), s[1].float()
    A = U @ V.T
    N, K = A.shape; r = U.shape[1]
    x = A.abs() / a.theta
    rows.append(dict(i=i, typ=T[i % 7], N=N, K=K, r=r, full=r >= min(N, K), rms=float(x.pow(2).mean().sqrt()),
                     near=float((x >= 0.5).float().mean()), moved=float(cum[i])))
print(f"{a.run}: step {ck.get('step')}, theta {a.theta}")
print("  type   shape         rank  full   rms(A)/theta   |A|>=theta/2   moved (cumulative), by depth third")
for t in T:
    rs = [r for r in rows if r["typ"] == t]
    th = np.array_split(rs, 3)
    print(f"  {t:5s}  {rs[0]['N']:5d}x{rs[0]['K']:<5d}  {rs[0]['r']:4d}  {str(rs[0]['full']):5s}  "
          + " ".join(f"{np.mean([r['rms'] for r in g]):.3f}" for g in th) + "   "
          + " ".join(f"{np.mean([r['near'] for r in g]):.4f}" for g in th) + "   "
          + " ".join(f"{np.mean([r['moved'] for r in g]):.4f}" for g in th))
ms = np.array([r["moved"] for r in rows]); rm = np.array([r["rms"] for r in rows]); nr = np.array([r["near"] for r in rows])
print(f"\ncorrelation over layers (Spearman): moved vs rms(A)/theta {np.corrcoef(np.argsort(np.argsort(ms)), np.argsort(np.argsort(rm)))[0, 1]:+.2f}, "
      f"moved vs near-bar share {np.corrcoef(np.argsort(np.argsort(ms)), np.argsort(np.argsort(nr)))[0, 1]:+.2f}")
