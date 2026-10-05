"""Which linear domain holds the full-rank evidence best at the memory of ranks K? (oracle, on the shadow dumps)

The low-rank accumulator tracks A in its own (row x column) domain. Any fixed invertible rearrangement T is as cheap to
track (T is linear, so the leaky sum and the subspace iteration commute with it): keep T(A) low rank, fire on
T^-1(U V^T). This compares, at the same number of floats as ranks K of A, the best (SVD) approximation in:

  plain        A itself (what TTF tracks; the tracked A_rK is printed too)
  kron n1xk1   Van Loan's rearrangement: A ~ sum_k B_k (n1 x k1) kron C_k (N/n1 x K/k1), i.e. low rank of the
               (n1 k1) x (N K / n1 k1) matrix whose rows are the blocks' positions and whose columns are the entries
               inside a block (n1 = 12 on the 768 side: one block per attention head)

For each: share of A_full's energy kept, and precision / recall (same sign) of its moves at theta against A_full's.

    python scripts/analysis/domain_oracle.py checkpoints/final/ts64an05_shadows_s0/acc_dumps [K] [theta]
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


def rearr(A, n1, k1):
    N, Kc = A.shape
    n2, k2 = N // n1, Kc // k1
    return A.reshape(n1, n2, k1, k2).transpose(0, 2, 1, 3).reshape(n1 * k1, n2 * k2)


def unrearr(R, N, Kc, n1, k1):
    n2, k2 = N // n1, Kc // k1
    return R.reshape(n1, k1, n2, k2).transpose(0, 2, 1, 3).reshape(N, Kc)


def best_lowrank(X, floats):
    r = max(1, int(floats // sum(X.shape)))
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    r = min(r, len(s))
    return (u[:, :r] * s[:r]) @ vt[:r], r


tot = {}
for f in sorted(glob.glob(os.path.join(D, "*.npz")), key=lambda p: (int(os.path.basename(p).split("_")[0][1:]), p)):
    z = np.load(f)
    if "A_full" not in z:
        continue
    Af, w = z["A_full"].astype(np.float64), z["w"].astype(np.float64)
    N, Kc = Af.shape
    budget = K * (N + Kc)
    Ff, sf = moves(Af, w, TH)
    nf, ef = Ff.sum(), (Af ** 2).sum()
    print(f"\n{os.path.basename(f)[:-4]}  {Af.shape}  full-rank moves {nf} ({nf / Af.size:.2%})")
    cands = {}
    if f"A_r{K}" in z:
        cands[f"tracked r{K}"] = (z[f"A_r{K}"].astype(np.float64), K)
    cands["plain svd"] = best_lowrank(Af, budget)
    for n1 in (4, 12, 24, 48):
        for k1 in (4, 12, 24, 48):
            if N % n1 or Kc % k1:
                continue
            R = rearr(Af, n1, k1)
            Rh, r = best_lowrank(R, budget)
            cands[f"kron {n1}x{k1}"] = (unrearr(Rh, N, Kc, n1, k1), r)
    for name, (Ah, r) in cands.items():
        e = 1 - ((Af - Ah) ** 2).sum() / ef
        F, sF = moves(Ah, w, TH)
        p, rc = pr(F, sF, Ff, sf)
        # moves at the threshold that gives as many moves as full rank (is the ranking right, whatever the scale?)
        if nf:
            thq = np.quantile(np.abs(Ah[np.abs(w + np.sign(Ah)) <= 1]), 1 - nf / Af.size)
            Fq, sFq = moves(Ah, w, thq)
            pq, _ = pr(Fq, sFq, Ff, sf)
        else:
            pq = float("nan")
        tot.setdefault(name, []).append((e, p, rc, pq))
        print(f"  {name:14s} rank {r:4d}  energy {e:.3f}  moves {F.sum():6d}  P {p:.2f}  R {rc:.2f}  | matched-count P=R {pq:.2f}")

print(f"\nmean over dumps (ranks-{K} memory):")
for name, v in tot.items():
    v = np.array(v)
    print(f"  {name:14s} energy {v[:, 0].mean():.3f}  P {v[:, 1].mean():.2f}  R {v[:, 2].mean():.2f}  matched-count {np.nanmean(v[:, 3]):.2f}")
