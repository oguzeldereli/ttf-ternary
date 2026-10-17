"""How concentrated are trit changes across rows and columns? Between two snapshots of a run, count the changed
trits per output row and per input column of every ternary matrix; report the share of all changes that falls in
the busiest 1% / 10% of rows and of columns (uniform spreading: 1% / 10%), pooled over layers.
  python -m scripts.analysis.flip_concentration RUN A B [kernel|master]"""
import sys, torch
from ttf.kernel import unpack_rows

run, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else "kernel"


def trits(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg, m = blob["cfg"], blob["model"]
    out = []
    if kind == "kernel":
        for k in sorted(x for x in m if x.endswith("wpacked")):
            K = cfg.hidden_dim if "w_down" in k else cfg.dim
            out.append((k, unpack_rows(m[k], K).to(torch.int8)))
    else:
        for k in sorted(x for x in m if x.endswith(".weight") and ("attn" in x or "ffn" in x) and m[x].dim() == 2):
            w = m[k].float(); g = w.abs().mean().clamp_min(1e-5)
            out.append((k, (w / g).round().clamp_(-1, 1).to(torch.int8)))
    return out


A, B = trits(f"checkpoints/{run}/ckpt_{a}.pt"), trits(f"checkpoints/{run}/ckpt_{b}.pt")
rows, cols, tot = [], [], 0
for (ka, ta), (kb, tb) in zip(A, B):
    ch = (ta != tb)
    rows.append(ch.sum(1).float()); cols.append(ch.sum(0).float()); tot += int(ch.sum())


def share(counts, q):
    c = torch.cat(counts); s = c.sort(descending=True).values; n = max(1, int(len(s) * q))
    return float(s[:n].sum() / s.sum().clamp_min(1))


print(f"{run} {a}->{b}: {tot / 1e6:.2f}M trits changed; share of changes in the busiest 1% / 10% of rows: "
      f"{share(rows, 0.01):.3f} / {share(rows, 0.10):.3f}; of columns: {share(cols, 0.01):.3f} / {share(cols, 0.10):.3f}")
