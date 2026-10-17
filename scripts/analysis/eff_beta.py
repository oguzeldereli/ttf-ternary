"""What decay does dry friction on the vector amount to? At a checkpoint: the momentum's total size ||M|| (all layers
as one vector; V orthonormal so ||M|| = ||U||) and the batch gradient's total size ||g|| (4 fresh batches). With
--dry_vec D the step is M <- f (M + g), f = 1 - D ||g||_ema / ||M + g||, so the effective decay is beta_eff = f and the
memory 1 / (1 - f). A fixed beta at pure noise would sit at ||M|| / ||g|| = 1 / sqrt(1 - beta^2).
  python -m scripts.analysis.eff_beta RUN STEP [D]
"""
import sys, numpy as np, torch
from scripts.analysis.testbench import Bench
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2]); D = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0303
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
mn = float(sum(U.float().pow(2).sum() for U, _ in B.b["lowrank"])) ** 0.5
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
gs, cs = [], []
for k in range(4):
    gd = B.grad([get_batch(train, 16, 2048, "cuda", torch.Generator().manual_seed(900 + k))])
    g = [gd[n].float() for n in B.names]
    gs.append(float(sum(x.pow(2).sum() for x in g)) ** 0.5)
    dot = sum(float(((x @ V.cuda().float()) * U.cuda().float()).sum()) for x, (U, V) in zip(g, B.b["lowrank"]))
    cs.append(dot / (gs[-1] * mn))
gn = float(np.mean(gs)); r = mn / gn
f = 1 - D * gn / mn
print(f"{RUN} @{ST}: ||M|| {mn:.4g}, ||g|| {gn:.4g}, ratio {r:.1f}, cos(g, M) {np.mean(cs):+.3f}; with dry {D}: "
      f"beta_eff {f:.5f}, memory {1 / max(1 - f, 1e-9):.0f} steps; a fixed beta with this ratio at pure noise: "
      f"{(1 - 1 / r ** 2) ** 0.5:.5f}")
