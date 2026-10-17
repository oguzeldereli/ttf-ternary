"""Additive adapter on text: per snapshot, rms of the trit path (beta T x) vs the adapter path ((x B) A^T) per
layer type (mean | max over blocks), A/B rms, on one validation batch.
  MAG_KIND=add python -m scripts.analysis.adapter_diag RUN_DIR STEP [STEP ...]"""
import sys, numpy as np, torch
from scripts.analysis.induction_heads import load
from ttf.layers import KernelTernaryLinear, _KernelTernFn
from ttf.train import get_batch

run, steps = sys.argv[1], sys.argv[2:]
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
names = ["wq", "wk", "wv", "wo", "gate", "up", "down"]
for st in steps:
    s, m, V = load(f"{run}/ckpt_{st}.pt", "kernel")
    x, y = get_batch(val, 4, 2048, "cuda", torch.Generator().manual_seed(3))
    Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
    stats = []
    def hook(l, inp, out):
        xi = inp[0]
        with torch.no_grad():
            t = _KernelTernFn.apply(xi, l.wpacked, l).float()
            a = ((xi @ l.mag_B.to(xi.dtype)) @ l.mag_A.to(xi.dtype).T).float()
        stats.append((t.pow(2).mean().sqrt().item(), a.pow(2).mean().sqrt().item()))
    hs = [l.register_forward_hook(hook) for l in Ls]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, y)[1].item()
    for h in hs: h.remove()
    A = torch.cat([l.mag_A.float().flatten() for l in Ls]); B = torch.cat([l.mag_B.float().flatten() for l in Ls])
    r = np.array([a / max(t, 1e-12) for t, a in stats]).reshape(-1, 7)
    print(f"step {s:5d}  val-batch loss {loss:.3f}  A rms {A.pow(2).mean().sqrt():.4f}  B rms {B.pow(2).mean().sqrt():.4f}")
    print("   adapter/trit rms by type (mean | max over blocks): " +
          "  ".join(f"{n} {r[:, j].mean():.2f}|{r[:, j].max():.2f}" for j, n in enumerate(names)), flush=True)
    del m; torch.cuda.empty_cache()
