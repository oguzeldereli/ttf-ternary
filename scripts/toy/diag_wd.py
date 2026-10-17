"""Why does adapter weight decay 0.1 turn the toy from 2.3 to 7.2 nats of induction while barely shrinking A?
Compares add_la_wd01 with mom_add_la (no decay) and mom_la (no adapter) over snapshots:
  A, B        rms, and the effective rank of each layer's adapter A B^T (singular value entropy)
  trits       share changed since the previous snapshot (churn) and the share at zero
  alignment   |cos| between the adapter's top singular direction and the trit matrix's top one (does the adapter
              fit the same directions the trits need?)
  ratio       rms of adapter output / trit output per layer on the eval sequences
  python -m scripts.toy.diag_wd
"""
import torch, math
from ttf.kernel import unpack_rows

RUNS = ("mom_la", "mom_add_la", "add_la_wd01")
STEPS = (1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 7000, 10000, 14000)
NAMES = ["L0.wq", "L0.wk", "L0.wv", "L0.wo", "L0.gate", "L0.up", "L0.down",
         "L1.wq", "L1.wk", "L1.wv", "L1.wo", "L1.gate", "L1.up", "L1.down"]


def layers(st, run):
    b = torch.load(f"checkpoints/toy/{run}/ckpt_{st}.pt", map_location="cuda", weights_only=False)
    m, cfg = b["model"], b["cfg"]
    keys = sorted({k.rsplit(".", 1)[0] for k in m if k.endswith("wpacked")},
                  key=lambda k: (int(k.split(".")[1]), ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn.w_gate",
                                                        "ffn.w_up", "ffn.w_down"].index(k.split(".", 2)[2])))
    out = []
    for k in keys:
        K = cfg.hidden_dim if k.endswith("w_down") else cfg.dim
        T = unpack_rows(m[k + ".wpacked"], K).float()
        A, B = m.get(k + ".mag_A"), m.get(k + ".mag_B")
        out.append((T, A.float() if A is not None else None, B.float() if B is not None else None))
    return out


def erank(M):
    s = torch.linalg.svdvals(M); p = s / s.sum()
    return math.exp(-(p * p.clamp_min(1e-12).log()).sum().item())


for run in RUNS:
    print(f"\n=== {run}")
    prev = None
    for st in STEPS:
        L = layers(st, run)
        churn = 0 if prev is None else sum(int((a[0] != b[0]).sum()) for a, b in zip(L, prev)) / sum(a[0].numel() for a in L)
        zeros = sum(int((a[0] == 0).sum()) for a in L) / sum(a[0].numel() for a in L)
        line = f"step {st:5d}  churn {100 * churn:5.1f}%  zeros {100 * zeros:4.1f}%"
        if L[0][1] is not None:
            Arms = torch.cat([a[1].flatten() for a in L]).pow(2).mean().sqrt().item()
            er = [erank(a[1] @ a[2].T) for a in L]
            al = []
            for T, A, B in L:
                u = torch.linalg.svd(A @ B.T, full_matrices=False)[0][:, 0]
                ut = torch.linalg.svd(T, full_matrices=False)[0][:, :4]
                al.append((ut.T @ u).norm().item())               # share of the adapter's top direction in the trits' top-4
            line += f"  A rms {Arms:.3f}  adapter eff-rank mean {sum(er) / len(er):.2f}  top-dir overlap mean {sum(al) / len(al):.2f}"
            line += "  | L1 wq/wk eff-rank " + " ".join(f"{er[i]:.2f}" for i in (7, 8))
        print(line, flush=True)
        prev = L
