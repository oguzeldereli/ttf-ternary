"""Analysis of the shadow-run matrix dumps (--ts_shadows ... --ts_dump_steps): per dumped layer and step,
- rank holding 90 / 95 / 99% of the energy of the full-rank accumulator A_full and momentum M_full;
- rank-r accumulators (A_r64 / A_r128 / A_r512) against A_full: cosine, sign agreement where |A_full| >= theta / 2,
  would-move agreement (|A| >= theta and the move stays in [-1, 1]): precision and recall;
- |A_full| / theta distribution and how many would-move decisions change when A is rounded to K mantissa bits.
    python scripts/analysis/dump_analysis.py checkpoints/final/RUN/acc_dumps [theta]
"""
import glob, os, sys
import numpy as np

D = sys.argv[1]; TH = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0


def energy_rank(X, fracs=(0.9, 0.95, 0.99)):
    s2 = np.linalg.svd(X.astype(np.float64), compute_uv=False) ** 2
    c = np.cumsum(s2) / s2.sum()
    return [int(np.searchsorted(c, f) + 1) for f in fracs], X.shape


def moves(A, w):
    return (np.abs(A) >= TH) & (np.abs(w + np.sign(A)) <= 1)


def fpm(X, k):
    m, e = np.frexp(X); return np.ldexp(np.round(m * 2 ** (k + 1)) / 2 ** (k + 1), e)


rows = []
for f in sorted(glob.glob(os.path.join(D, "*.npz")), key=lambda p: (int(os.path.basename(p).split("_")[0][1:]), p)):
    z = np.load(f); name = os.path.basename(f)[:-4]
    Af = z["A_full"].astype(np.float32); Mf = z["M_full"].astype(np.float32); w = z["w"].astype(np.float32)
    ra, shp = energy_rank(Af); rm, _ = energy_rank(Mf)
    mf = moves(Af, w); big = np.abs(Af) >= TH / 2
    line = [f"{name:22s} {shp[0]}x{shp[1]}  A rank 90/95/99%: {ra[0]}/{ra[1]}/{ra[2]}  M rank: {rm[0]}/{rm[1]}/{rm[2]}"]
    for key in ("A_r64", "A_r128", "A_r512", "A_run"):
        if key not in z.files: continue
        A = z[key].astype(np.float32); mv = moves(A, w); both = (mv & mf).sum()
        cos = (A * Af).sum() / (np.linalg.norm(A) * np.linalg.norm(Af) + 1e-30)
        line.append(f"   {key:7s} cos {cos:.2f} sign {np.mean(np.sign(A[big]) == np.sign(Af[big])):.2f} "
                    f"moves {mv.sum():6d} (full {mf.sum():6d}) precision {both / max(mv.sum(), 1):.2f} recall {both / max(mf.sum(), 1):.2f}")
    a = np.abs(Af) / TH
    line.append(f"   |A_full|/theta: >=1 {np.mean(a >= 1):.4f}, 0.5-1 {np.mean((a >= 0.5) & (a < 1)):.4f}, 0.9-1 {np.mean((a >= 0.9) & (a < 1)):.4f}; "
                + " ".join(f"m{k}: {np.mean(moves(fpm(Af, k), w) != mf):.2e}" for k in (3, 4, 5, 7)) + " of decisions change")
    print("\n".join(line))
