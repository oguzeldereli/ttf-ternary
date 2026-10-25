"""Paired comparison of final evaluations (scripts/eval/final_eval.py) on the same windows.

    python scripts/eval/compare.py A_DIR B_DIR [--set wiki|fw] [--block 4] [--n 10000]
    python scripts/eval/compare.py --groups "TTF=dirA1,dirA2" "master=dirB1,dirB2" [--set wiki]

Difference of mean loss (A - B) with a 95% interval from a paired block bootstrap over windows (blocks of --block
consecutive windows, since neighbouring windows can come from the same article). With --groups, each group is the mean
over its runs (seeds), and the spread across seeds is reported next to the window interval.
"""
from __future__ import annotations
import argparse, os
import numpy as np


def load(d, which):
    z = np.load(os.path.join(d, "final_eval.npz"))
    return z[which]


def boot(diff, block, n, rng):
    """95% interval of mean(diff) by resampling blocks of consecutive windows"""
    nb = len(diff) // block
    blocks = diff[:nb * block].reshape(nb, block).mean(1)
    idx = rng.integers(0, nb, size=(n, nb))
    means = blocks[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--groups", nargs="*", help='"NAME=dir1,dir2" (mean over seeds)')
    ap.add_argument("--set", default="wiki")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    if a.groups:
        G = {}
        for g in a.groups:
            name, ds = g.split("=", 1)
            runs = [load(d, a.set) for d in ds.split(",")]
            G[name] = (np.mean(runs, 0), [r.mean() for r in runs])
        names = list(G)
        for nm in names:
            m, per = G[nm]
            print(f"{nm:24s} mean {m.mean():.4f}  seeds {' '.join(f'{x:.4f}' for x in per)}  "
                  f"spread {np.std(per, ddof=1) if len(per) > 1 else float('nan'):.4f}")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = G[names[i]][0] - G[names[j]][0]
                lo, hi = boot(d, a.block, a.n, rng)
                print(f"{names[i]} - {names[j]}: {d.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
                      f"(windows: {len(d)}, {np.mean(d < 0) * 100:.0f}% favour {names[i]})")
        return
    A, B = a.dirs[0], a.dirs[1]
    x, y = load(A, a.set), load(B, a.set)
    d = x - y
    lo, hi = boot(d, a.block, a.n, rng)
    print(f"{os.path.basename(A.rstrip('/'))} {x.mean():.4f} - {os.path.basename(B.rstrip('/'))} {y.mean():.4f} = "
          f"{d.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  ({len(d)} windows, {np.mean(d < 0) * 100:.0f}% favour A)")


if __name__ == "__main__":
    main()
