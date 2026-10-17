"""After conversion training: how much of the fp model's precision is still missing, and does moving it into extra
float adapter rank help? (17 Oct; training-free, on a trained 110M conversion checkpoint)

For every ternary linear of the trained model, y = r o ((beta T + A B^T)(c o (x - mu))) + bias, the weight the fp
model has in the same coordinates is G_fp = diag(1 / r) W diag(1 / c); the missing part G' = G_fp - (beta T + A B^T).
Its best rank-R fit in the norm of the trained model's own adapter inputs x' = c o (x - mu) (covariance from
calibration windows through the trained model) is appended to the adapter (rank 16 -> 16 + R), and the loss on the
first 16 val windows is compared with the trained model's.
  python scripts/analysis/resid_adapter.py CKPT FP_HF_DIR [--R 16,64]
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import final_eval as FE                                                     # noqa: E402
from ttf.flip import KernelTernaryLinear                                # noqa: E402
from ttf.hf_init import load_hf_tensors, _unpermute, _LIN              # noqa: E402
from ttf.kernel import unpack_rows, trit_beta                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("fp")
ap.add_argument("--R", default="16,64,256"); ap.add_argument("--calib", type=int, default=32)
ap.add_argument("--val", default="data/wiki32k_val.bin"); ap.add_argument("--train", default="data/wiki32k_train.bin")
a = ap.parse_args()
dev = "cuda"
val = np.memmap(a.val, dtype=np.uint16, mode="r"); tr = np.memmap(a.train, dtype=np.uint16, mode="r")


def sym_sqrt(H, rel=1e-6):
    """H^(1/2) and H^(-1/2) (symmetric, eigenvalues clamped at rel x the largest): the covariance of a trained model's
    inputs can be numerically singular, where a Cholesky factor fails"""
    lam, Q = torch.linalg.eigh(H)
    lam = lam.clamp_min(rel * float(lam.max()))
    return (Q * lam.sqrt()) @ Q.T, (Q / lam.sqrt()) @ Q.T


@torch.no_grad()
def loss16(m):
    ls = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(16):
            w = torch.from_numpy(val[i * 2048:(i + 1) * 2048 + 1].astype(np.int64)).to(dev)[None]
            ls.append(float(m(w[:, :-1], w[:, 1:])[1]))
    return float(np.mean(ls))


m, ck = FE.load_model(a.ckpt, dev)
base = loss16(m)
print(f"trained model, first 16 val windows: {base:.4f}", flush=True)
sd = load_hf_tensors(a.fp); c_ = ck["cfg"]
lins = {}
for i, blk in enumerate(m.layers):
    for hf, (sub, name) in _LIN.items():
        W = sd[f"model.layers.{i}.{hf}.weight"].float()
        W = _unpermute(W, c_.n_heads) if name == "wq" else (_unpermute(W, c_.n_kv_heads) if name == "wk" else W)
        lins[getattr(getattr(blk, sub), name)] = W
# covariance of each layer's adapter input x' = c o (x - mu) through the trained model
H, n = {l: torch.zeros(l.K, l.K, device=dev, dtype=torch.float64) for l in lins}, {l: 0 for l in lins}
hooks = []
for l in lins:
    def pre(mod, inp, l=l):
        x = inp[0].reshape(-1, l.K).float()
        if getattr(l, "in_mean", None) is not None:
            x = x - l.in_mean
        x = x * l.col_scale.float()
        H[l] += (x.T @ x).double(); n[l] += x.shape[0]
    hooks.append(l.register_forward_pre_hook(pre))
rng = np.random.default_rng(4321)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for s in rng.integers(0, len(tr) - 2049, a.calib):
        w = torch.from_numpy(tr[s:s + 2048].astype(np.int64)).to(dev)[None]
        m(w)
for h in hooks:
    h.remove()
fits = {}
for l, W in lins.items():
    beta = float(trit_beta(l.wpacked, l.K)) if l.use_beta else 1.0
    Gcur = beta * unpack_rows(l.wpacked, l.K).double() + (l.mag_A.double() @ l.mag_B.double().T)
    Gfp = W.to(dev).double() / l.row_scale.double().clamp_min(1e-12)[:, None] / l.col_scale.double().clamp_min(1e-12)[None, :]
    Gm = Gfp - Gcur
    Hp = H[l] / max(n[l], 1)
    L, Li = sym_sqrt(Hp)
    U, S, Vh = torch.linalg.svd(Gm @ L, full_matrices=False)
    fits[l] = (U, S, Vh, Li, float((torch.linalg.norm(Gm @ L) / torch.linalg.norm(Gfp @ L)).item()))
print(f"missing part, output-space relative norm |G' L| / |G_fp L|: mean {np.mean([f[4] for f in fits.values()]):.4f}",
      flush=True)
A0 = {l: l.mag_A.data.clone() for l in lins}; B0 = {l: l.mag_B.data.clone() for l in lins}
for R in [int(r) for r in a.R.split(",")]:
    kept = []
    for l, (U, S, Vh, L, _) in fits.items():
        A2 = (U[:, :R] * S[:R]).float(); B2 = (L @ Vh[:R].T).float()          # L here: the inverse square root
        l.mag_A.data = torch.cat([A0[l], A2], 1); l.mag_B.data = torch.cat([B0[l], B2], 1)
        kept.append(float((S[:R] ** 2).sum() / (S ** 2).sum()))
    print(f"+ rank {R} from the missing part (takes {np.mean(kept):.3f} of it): {loss16(m):.4f} (trained {base:.4f})",
          flush=True)
    for l in lins:
        l.mag_A.data = A0[l]; l.mag_B.data = B0[l]
