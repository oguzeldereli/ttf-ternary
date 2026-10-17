"""Where does each model's validation loss come from? Per-token loss on 24 validation sequences of 2048 tokens,
split by whether the pair (current token, next token) already occurred earlier in the context (copyable) and by
how predictable the next token is from corpus bigram statistics alone, p(next | current) from 50M training tokens:
  copy-rare     copyable and p < 0.01: only the context can explain it (what an induction head is for)
  copy-common   copyable and p >= 0.01: bigram statistics predict it anyway
  other-rare / other-common   the same split for non-copyable tokens
  python -m scripts.analysis.copy_split [only-run-substring]"""
import os, sys, numpy as np, torch, torch.nn.functional as F
from scripts.analysis.induction_heads import load
from ttf.train import get_batch

RUNS = [("master, 164M", "checkpoints/curve_master/ckpt_5000.pt", "master", ""),
        ("master, 98M", "checkpoints/curve_master/ckpt_3000.pt", "master", ""),
        ("no LA then LA at 131M, 164M", "checkpoints/nola_then_la/ckpt_5000.pt", "kernel", ""),
        ("no look-ahead, 164M", "checkpoints/nola_lab/ckpt_5000.pt", "kernel", ""),
        ("look-ahead + additive r4, 164M", "checkpoints/magadd_full/ckpt_5000.pt", "kernel", "add"),
        ("no look-ahead, 295M", "checkpoints/nola_lab/ckpt_9000.pt", "kernel", ""),
        ("look-ahead + additive r4, 295M", "checkpoints/magadd_full/ckpt_9000.pt", "kernel", "add")]
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
XY = [get_batch(val, 4, 2048, "cuda", torch.Generator().manual_seed(21 + i)) for i in range(6)]
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")[:50_000_000].astype(np.int64)
uni = np.bincount(tr, minlength=32000).astype(np.float64)
pu, pc = np.unique(tr[:-1] * 32000 + tr[1:], return_counts=True)
def pbi(x, y):
    k = x * 32000 + y; i = np.searchsorted(pu, k); i = np.minimum(i, len(pu) - 1)
    c = np.where(pu[i] == k, pc[i], 0)
    return c / np.maximum(uni[x], 1)
masks = []
for x, y in XY:
    M = torch.zeros_like(y, dtype=torch.bool)
    for b in range(x.shape[0]):
        seen = set(); xs, ys = x[b].tolist(), y[b].tolist()
        for t in range(len(xs)):
            if (xs[t], ys[t]) in seen: M[b, t] = True
            seen.add((xs[t], ys[t]))
    P = torch.from_numpy(pbi(x.cpu().numpy(), y.cpu().numpy())).to(M.device)
    masks.append({"copy-rare": M & (P < 0.01), "copy-common": M & (P >= 0.01),
                  "other-rare": ~M & (P < 0.01), "other-common": ~M & (P >= 0.01)})
K = list(masks[0])
tot = sum(int(m[K[0]].numel()) for m in masks)
print("share of tokens: " + "  ".join(f"{k} {100 * sum(int(m[k].sum()) for m in masks) / tot:.1f}%" for k in K))
print(f"{'model':32s} {'all':>7s} " + " ".join(f"{k:>12s}" for k in K))
only = sys.argv[1] if len(sys.argv) > 1 else ""
for name, path, kind, mag in RUNS:
    if only and only not in name: continue
    if not os.path.exists(path): print(f"{name:32s} (missing {path})"); continue
    os.environ["MAG_KIND"] = mag or "add"
    s, m, V = load(path, kind)
    parts = {k: [] for k in K}; allv = []
    for (x, y), M in zip(XY, masks):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lp = torch.cat([F.cross_entropy(m(x[i:i + 1])[0].float().transpose(1, 2), y[i:i + 1], reduction="none")
                            for i in range(x.shape[0])])
        allv.append(lp.flatten())
        for k in K: parts[k].append(lp[M[k]])
    print(f"{name:32s} {float(torch.cat(allv).mean()):7.3f} " +
          " ".join(f"{float(torch.cat(parts[k]).mean()):12.3f}" for k in K), flush=True)
    del m; torch.cuda.empty_cache()
