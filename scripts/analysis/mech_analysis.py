"""Why the low-rank mechanisms (4 Oct: rescale, refresh, exact table, exact spend, tiles) do or do not help: tests of
each one's premise on the matrices dumped by the shadow runs (A_full = full-rank accumulator fed the same steps and spent
by the run's moves; A_rK = rank-K shadow accumulators; w = the trits), q / down of two blocks at steps 500 / 2000 / 5000 /
9000.

    python scripts/analysis/mech_analysis.py checkpoints/final/ts64an05_shadows_s0/acc_dumps [rank] [theta]

Per dump (rank K, default 64):
  rescale  precision / recall of the moves of A_rK at threshold theta * s against the moves of A_full (same sign), for
           s from 1 to 0.25, and the s the mechanism would use (sqrt of the energy share A_rK keeps)
  table    share of A_full's moves that fall in the top 2% of |A_rK| (what a table selected by the low-rank value could
           hold) and recall of an oracle table (the exact A_full on the top 2% of |A_full|)
  tiles    energy of A_full captured by its best rank-K approximation vs by 2 x 2 tiles of best rank K/2 each
  spend    share of a move pattern D (A_full's moves) that the rank-K basis holds: what a low-rank spend removes at the
           moved entries, and how much it removes elsewhere instead
  refresh  energy of A_full captured in A_rK's own basis vs in A_full's best rank-K basis (is the tracked basis stale?)
"""
import glob, os, sys
import numpy as np

D = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
TH = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0


def moves(A, w, th):
    s = np.sign(A)
    return (np.abs(A) >= th) & (np.abs(w + s) <= 1), s


def pr(F, sF, G, sG):
    both = (F & G & (sF == sG)).sum()
    return both / max(F.sum(), 1), both / max(G.sum(), 1)


def topk_energy(A, k):
    sv = np.linalg.svd(A, compute_uv=False) ** 2
    return sv[:k].sum() / sv.sum()


for f in sorted(glob.glob(os.path.join(D, "*.npz")), key=lambda p: (int(os.path.basename(p).split("_")[0][1:]), p)):
    z = np.load(f)
    if "A_full" not in z or f"A_r{K}" not in z:
        continue
    Af, Ar, w = z["A_full"].astype(np.float64), z[f"A_r{K}"].astype(np.float64), z["w"].astype(np.float64)
    name = os.path.basename(f)[:-4]
    Ff, sf = moves(Af, w, TH)
    nf = Ff.sum()
    print(f"\n{name}  {Af.shape}  full-rank moves {nf} ({nf / Af.size:.2%})")
    # rescale
    keep = np.linalg.norm(Ar) / np.linalg.norm(Af)
    row = []
    for s in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25):
        F, sF = moves(Ar, w, TH * s)
        p, r = pr(F, sF, Ff, sf)
        row.append(f"s{s:.2f}: n{F.sum()} P{p:.2f} R{r:.2f}")
    print(f"  rescale  energy kept {keep**2:.2f} (s = {keep:.2f}) | " + " | ".join(row))
    # sign of A_rK on the entries full rank would move
    if nf:
        print(f"           on full-rank moves: sign agrees {np.mean(np.sign(Ar[Ff]) == sf[Ff]):.2f}, |A_r{K}| / |A_full| "
              f"median {np.median(np.abs(Ar[Ff]) / np.abs(Af[Ff])):.2f}")
    # table
    k2 = max(1, int(0.02 * Af.size))
    top_r = np.zeros(Af.size, bool); top_r[np.argpartition(-np.abs(Ar).ravel(), k2)[:k2]] = True; top_r = top_r.reshape(Af.shape)
    top_f = np.zeros(Af.size, bool); top_f[np.argpartition(-np.abs(Af).ravel(), k2)[:k2]] = True; top_f = top_f.reshape(Af.shape)
    Ao = np.where(top_f, Af, Ar)
    Fo, so = moves(Ao, w, TH)
    po, ro = pr(Fo, so, Ff, sf)
    print(f"  table    full-rank moves inside the top 2% of |A_r{K}|: {(Ff & top_r).sum() / max(nf, 1):.2f}; "
          f"inside the top 2% of |A_full|: {(Ff & top_f).sum() / max(nf, 1):.2f}; oracle table P{po:.2f} R{ro:.2f}")
    # tiles
    n, m = Af.shape
    e_glob = topk_energy(Af, K)
    tiles = [Af[i * n // 2:(i + 1) * n // 2, j * m // 2:(j + 1) * m // 2] for i in range(2) for j in range(2)]
    e_tile = sum(topk_energy(T, K // 2) * (T ** 2).sum() for T in tiles) / (Af ** 2).sum()
    print(f"  tiles    energy of A_full: best rank {K} {e_glob:.2f}; 2 x 2 tiles of rank {K // 2} {e_tile:.2f}")
    # spend and refresh: A_rK's own basis (its top-K right singular vectors)
    _, sv_r, Vt_r = np.linalg.svd(Ar, full_matrices=False)
    V = Vt_r[:K].T
    if nf:
        Dm = np.where(Ff, sf, 0.0)
        P = Dm @ V @ V.T
        held = (P[Ff] * Dm[Ff]).mean()
        off = np.linalg.norm(np.where(Ff, 0.0, P)) / np.linalg.norm(Dm)
        print(f"  spend    a low-rank spend removes {held:.2f} theta at a moved entry (exact: 1.00); elsewhere it removes "
              f"{off:.2f} of ||theta D|| from entries that did not move")
    e_own = (np.linalg.norm(Af @ V) / np.linalg.norm(Af)) ** 2
    print(f"  refresh  energy of A_full in A_r{K}'s basis {e_own:.2f} vs in the best rank-{K} basis {e_glob:.2f}")
