"""Previous-token heads, induction heads and random-copy gain for every checkpoint of a run (ckpt_*.pt), plus the
per-head temperatures if the run has them.

  python -m scripts.analysis.heads_over_time RUN_DIR [kernel|master]
"""
import sys, glob, json, re, torch, numpy as np
import ttf.probe as P
from scripts.analysis.induction_heads import load, CAP, ON
P._last_loss.__defaults__ = (16,)
run, kind = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "kernel")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
paths = sorted(glob.glob(f"{run}/ckpt_*.pt"), key=lambda p: int(re.findall(r"ckpt_(\d+)", p)[0]))
g = np.random.default_rng(4321); out = []
for path in paths:
    s, m, V = load(path, kind)
    rnd = torch.from_numpy(g.integers(1000, V - 1000, size=(16, 64))).cuda(); x = torch.cat([rnd, rnd], 1)
    CAP.clear(); ON[0] = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    ON[0] = False
    t = torch.arange(1, 128, device="cuda"); t2 = torch.arange(65, 128, device="cuda")
    prev = [A.mean(0)[:, t, t - 1].mean(-1).max().item() for A in CAP]
    ind = [A.mean(0)[:, t2, t2 - 63].mean(-1).max().item() for A in CAP]
    r = P.probe(m, val, "cuda")
    temps = [float(a.qk_logscale.exp().max()) for a in m.modules() if getattr(a, "qk_logscale", None) is not None]
    row = {"step": s, "tokens": (s + 1) * 32768, "prev_best": max(prev), "prev_n_over_0.8": sum(p > 0.8 for p in prev),
           "ind_best": max(ind), "copy_gain": r["copy_gain"], "ctx64_512": r["loss_ctx64"] - r["loss_ctx512"],
           "max_temp": max(temps) if temps else None, "prev_layers": prev, "ind_layers": ind}
    out.append(row)
    print(f"{row['tokens'] / 1e6:5.0f}M  prev best {row['prev_best']:.2f} (layers > 0.8: {row['prev_n_over_0.8']})  "
          f"ind best {row['ind_best']:.2f}  copy gain {row['copy_gain']:+.3f}  ctx64-512 {row['ctx64_512']:.3f}"
          + (f"  max temp {row['max_temp']:.2f}" if temps else ""), flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open(f"{run}/heads.json", "w"))
