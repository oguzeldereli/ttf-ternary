"""Where in the spectrum is the gradient signal a rank-r accumulator keeps? One curve per model size, to compare at the
same absolute rank and at the same share of the width.

Input: pairs of mean-gradient files from grad_snr.py --save_mu (u = g / sqrt(v) per ternary layer, averaged over n
micro-batches at a frozen checkpoint), the two of a pair ~1000 steps apart. Each mean carries noise (micro-batch noise
/ n); the two are independent, so their cross products do not:
  - persistent spectrum: eigenvalues of the symmetrised cross Gram (Ma Mb^T + Mb Ma^T) / 2, which estimate the squared
    singular values of the drift that both checkpoints share; cumulative share E(r) = sum_{k<=r} lambda_k / sum lambda.
    Noise adds +- eigenvalue pairs, so the most negative eigenvalues measure the noise floor: "net" subtracts the k-th
    most negative from the k-th largest, and n_sig counts the eigenvalues above the largest negative one.
  - captured share: the top-r singular subspace of Ma (chosen on its own noisy estimate, as an accumulator chooses its
    subspace from its own noisy evidence), applied to the persistent signal: <P Ma Q, P Mb Q> / <Ma, Mb>.
  - move gain: TTF moves single trits by -sign(A) where |A| is largest, so what a rank-r state costs is in which entries
    it moves, not in Frobenius energy. With A_r = the rank-r truncation of Ma, the top fraction f of entries by |A_r|
    moved by -sign(A_r), scored on the independent Mb: gain(r) = sum_sel sign(A_r) Mb, relative to the full-rank
    choice (r = d). Also for a random choice of the same count (sign from Ma) as the floor.
Energies are summed over the layers of each matrix type, then over all layers (energy-weighted).

Moves must be legal (a trit at +-1 cannot move further out): with the checkpoint's trits given, the selection and the
random floor use legal moves only. "within" pairs use the three thirds of one checkpoint (grad_snr.py --save_thirds):
selection on the mean of two, scoring on the third, so the signal is whatever is consistent across batches at that
point, persistent or not.

    python scripts/analysis/rank_spectrum.py OUT.json NAME=WIDTH:A_mu.pt:B_mu.pt[:CKPT] [...] [--plot OUT.png]
    python scripts/analysis/rank_spectrum.py OUT.json NAME=WIDTH:A_mu.pt:within[:CKPT]

"keep": how much of one new training step a rank-r basis holds, the basis being the top-r singular subspace of the
accumulated evidence (ACC: a mean over many batches, as the accumulator is) and the step one batch's u (the three
thirds of a grad_snr.py --n 12 --micro 4 --save_thirds run with another --seed: three single training batches).
Left (||U_r^T u||^2), right (||u V_r||^2) and two-sided shares of ||u||^2, and the step's own top-r share (the best
any rank-r basis could do for that step alone).
    python scripts/analysis/rank_spectrum.py OUT.json NAME=WIDTH:ACC_mu.pt:keep:STEPS_mu.pt
"""
import json
import sys

import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

RANKS = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]
FRACS = [0.001, 0.01]
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def ltype(name):
    for t, keys in (("q", ("wq",)), ("k", ("wk",)), ("v", ("wv",)), ("o", ("wo",)), ("gate", ("w_gate", "w1")),
                    ("up", ("w_up", "w3")), ("down", ("w_down", "w2"))):
        if any(name.endswith(k) for k in keys):
            return t
    return "other"


def layer_stats(Ma, Mb, T=None):
    """per-layer cumulative cross energy at RANKS (persistent spectrum, net, captured), total cross, n_sig"""
    if Ma.shape[0] > Ma.shape[1]:
        Ma, Mb = Ma.T.contiguous(), Mb.T.contiguous()
        T = T.T.contiguous() if T is not None else None
    d = Ma.shape[0]
    C = Ma @ Mb.T
    C = 0.5 * (C + C.T)
    lam = torch.linalg.eigvalsh(C.double()).flip(0)                # descending
    tot = float(lam.sum())
    neg = (-lam.flip(0)).clamp_min(0)                               # magnitudes of the most negative, largest first
    cum = torch.cumsum(lam, 0); cumnet = torch.cumsum(lam - neg, 0)
    n_sig = int((lam > neg[0]).sum())
    U, S, Vh = torch.linalg.svd(Ma, full_matrices=False)
    A_ = U.T @ Ma @ Vh.T; B_ = U.T @ Mb @ Vh.T                     # in Ma's singular basis (r x r leading blocks)
    cap = []
    for r in RANKS:
        k = min(r, d)
        cap.append(float((A_[:k, :k] * B_[:k, :k]).sum()))
    del A_, B_
    gain = {f: [] for f in FRACS}
    ne = Ma.numel()
    for r in RANKS:
        k = min(r, d)
        Ar = (U[:, :k] * S[:k]) @ Vh[:k] if k < d else Ma
        sc = Ar.abs()
        if T is not None:                      # the move -sign(A) is legal unless the trit is already there
            sc[(T * Ar.sign()) == -1] = 0      # T = +1 and A < 0 (move up) or T = -1 and A > 0 (move down)
        for f in FRACS:
            idx = sc.view(-1).topk(max(1, int(f * ne))).indices
            gain[f].append(float((Ar.view(-1)[idx].sign() * Mb.view(-1)[idx]).sum()))
        del Ar, sc
    g_ = torch.Generator(device=Ma.device).manual_seed(0)
    rnd = {}
    ok = (T * Ma.sign() != -1).view(-1).nonzero().squeeze(1) if T is not None else None
    for f in FRACS:
        idx = torch.randint(0, ne, (max(1, int(f * ne)),), device=Ma.device, generator=g_)
        if ok is not None:
            idx = ok[torch.randint(0, ok.numel(), (max(1, int(f * ne)),), device=Ma.device, generator=g_)]
        rnd[f] = float((Ma.view(-1)[idx].sign() * Mb.view(-1)[idx]).sum())
    pick = lambda c: [float(c[min(r, d) - 1]) for r in RANKS]
    lam_at = [float(lam[min(r, d) - 1]) for r in RANKS]
    return dict(d=d, tot=tot, cum=pick(cum), net=pick(cumnet), cap=cap, n_sig=n_sig, lam=lam[:64].tolist(),
                lam_at=lam_at, gain={str(f): v for f, v in gain.items()}, rnd={str(f): v for f, v in rnd.items()})


def run_keep(name, width, fa, fs):
    A = torch.load(fa, map_location="cpu", weights_only=False)
    St = torch.load(fs, map_location="cpu", weights_only=False)
    ia = {n: i for i, n in enumerate(A["names"])}
    rows = []
    for j, nm in enumerate(St["names"]):
        M = A["mu"][ia[nm]].to(dev).float()
        tr = M.shape[0] > M.shape[1]
        if tr:
            M = M.T.contiguous()
        U, _, Vh = torch.linalg.svd(M, full_matrices=False); V = Vh.T; d = M.shape[0]
        del M
        acc = {k: [0.0] * len(RANKS) for k in ("left", "right", "two", "self")}; tot = 0.0
        for g in St["thirds"][j]:
            G = g.to(dev).float()
            G = G.T.contiguous() if tr else G
            tot += float((G * G).sum())
            L = U.T @ G; R_ = G @ V; s2 = torch.linalg.svdvals(G).pow(2)
            cl = torch.cumsum(L.pow(2).sum(1), 0); cr = torch.cumsum(R_.pow(2).sum(0), 0); cs = torch.cumsum(s2, 0)
            for q, r in enumerate(RANKS):
                k = min(r, d)
                acc["left"][q] += float(cl[k - 1]); acc["right"][q] += float(cr[k - 1]); acc["self"][q] += float(cs[k - 1])
                acc["two"][q] += float((L[:k] @ V[:, :k]).pow(2).sum())
            del G, L, R_
        rows.append(dict(name=nm, type=ltype(nm), d=d, tot=tot, **acc))
        del U, V
    agg = {}
    for grp in TYPES + ["all"]:
        sel = [r for r in rows if grp == "all" or r["type"] == grp]
        if sel:
            T = sum(r["tot"] for r in sel)
            agg[grp] = {k: [sum(r[k][q] for r in sel) / T for q in range(len(RANKS))] for k in ("left", "right", "two", "self")}
    return dict(name=name, width=int(width), mode="keep", rows=rows, agg=agg)


def run(spec):
    from bitnet.kernel import unpack_rows
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    width, fa, fb = parts[:3]
    if fb == "keep":
        return run_keep(name, width, fa, parts[3])
    ck = parts[3] if len(parts) > 3 else None
    A = torch.load(fa, map_location="cpu", weights_only=False)
    within = fb == "within"
    B = A if within else torch.load(fb, map_location="cpu", weights_only=False)
    sd = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)["model"] if ck else None
    names = A.get("names") or [str(i) for i in range(len(A["mu"]))]
    rows = []
    for i in range(len(A["mu"])):
        if within:
            t = A["thirds"][i]
            Ma = 0.5 * (t[0].to(dev).float() + t[1].to(dev).float()); Mb = t[2].to(dev).float()
        else:
            Ma = A["mu"][i].to(dev).float(); Mb = B["mu"][i].to(dev).float()
        T = None
        if sd is not None:
            wp = sd[names[i] + ".wpacked"]
            T = unpack_rows(wp.to(dev), Ma.shape[1]).float()
        s = layer_stats(Ma, Mb, T); s.update(i=i, name=names[i], type=ltype(names[i]), N=Ma.shape[0], K=Ma.shape[1])
        rows.append(s)
        del Ma, Mb, T
        if i % 28 == 0:
            print(f"  {name} layer {i}/{len(A['mu'])}", flush=True)
    out = dict(name=name, width=int(width), rows=rows)
    agg = {}
    for grp in TYPES + ["all"]:
        sel = [r for r in rows if grp == "all" or r["type"] == grp]
        if not sel:
            continue
        T = sum(r["tot"] for r in sel)
        agg[grp] = {k: [sum(r[k][j] for r in sel) / T for j in range(len(RANKS))] for k in ("cum", "net", "cap")}
        agg[grp]["n_sig_mean"] = float(np.mean([r["n_sig"] for r in sel]))
        for f in FRACS:                 # move gain relative to the full-rank choice, summed over the group's layers
            full = sum(r["gain"][str(f)][-1] for r in sel)
            agg[grp][f"gain{f}"] = [sum(r["gain"][str(f)][j] for r in sel) / full for j in range(len(RANKS))]
            agg[grp][f"rnd{f}"] = sum(r["rnd"][str(f)] for r in sel) / full
        agg[grp]["d"] = int(np.median([r["d"] for r in sel]))
    out["agg"] = agg
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    plot = sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None
    if plot:
        args = [a for a in args if a != plot]
    outp, specs = args[0], args[1:]
    res = [run(s) for s in specs]
    for r in res:
        if r.get("mode") == "keep":
            print(f"\n{r['name']} (width {r['width']}): share of one new step's energy a rank-r basis of the accumulated evidence holds")
            print("  r          " + " ".join(f"{x:>6d}" for x in RANKS))
            for grp in ["all", "q", "k", "v", "o", "gate", "up", "down"]:
                if grp in r["agg"]:
                    for k in ("left", "right", "two", "self"):
                        print(f"  {grp:5s}{k:6s}" + " ".join(f"{x:6.3f}" for x in r["agg"][grp][k]))
            continue
        print(f"\n{r['name']} (width {r['width']}): energy-weighted share of the persistent signal within rank r")
        print("  r      " + " ".join(f"{x:>6d}" for x in RANKS))
        for grp in ["all", "q", "k", "v", "o", "gate", "up", "down"]:
            if grp in r["agg"]:
                g = r["agg"][grp]
                for k in ("cap", "net") + tuple(f"gain{f}" for f in FRACS):
                    extra = f"   n_sig {g['n_sig_mean']:.0f}" if k == "net" else (f"   random {g['rnd' + k[4:]]:.3f}" if k.startswith("gain") else "")
                    print(f"  {grp:5s}{k[:4] + k[6:] if k.startswith('gain') else k:9s}" + " ".join(f"{x:6.3f}" for x in g[k]) + extra)
    json.dump([{k: v for k, v in r.items()} for r in res], open(outp, "w"))
    if plot and not any(r.get("mode") == "keep" for r in res):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for j, r in enumerate(res):
            g = r["agg"]["all"]; w = r["width"]
            x = np.array(RANKS, float); m = x <= w
            for k, ls in (("cap", "-"), ("net", ":")):
                y = 1 - np.array(g[k])
                ax[0].plot(x[m], y[m], ls, color=f"C{j}", label=f"{r['name']} ({k})")
                ax[1].plot(x[m] / w, y[m], ls, color=f"C{j}")
        for a_, xl in zip(ax, ["absolute rank r", "rank / width"]):
            a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel(xl); a_.grid(alpha=0.3)
        ax[0].set_ylabel("persistent signal outside rank r"); ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(plot, dpi=150)


if __name__ == "__main__":
    main()
