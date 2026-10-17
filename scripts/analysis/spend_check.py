"""Why did --lr_spend not change anything? On the step-4000 bench: how much of the kept flip pattern D survives
the rank-r projection used by spend, how much it actually shrinks M at the flipped entries versus elsewhere,
and whether M's largest entries sit on weights that cannot move further (already at +-1 in M's direction).

  python -m scripts.analysis.spend_check
"""
import json, torch
from scripts.analysis.testbench import OUT
dev = "cuda" if torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
b = torch.load(json.load(open(f"{OUT}/meta.json"))["ckpt"], map_location="cpu", weights_only=False)
flips = torch.load(f"{OUT}/flips.pt")
from ttf.kernel import unpack_rows
import numpy as np
G = torch.load(f"{OUT}/gbar.pt"); names = json.load(open(f"{OUT}/meta.json"))["ternary_layers"]
sd = b["model"]
C = 2.0
rows = []
for i, (U, V) in enumerate(b["lowrank"]):
    U, V = U.to(dev).float(), V.to(dev).float()
    M = U @ V.T
    gm = M.abs().mean()
    D = flips["M_kept"][i].to(dev).float()
    nz = D != 0
    if nz.sum() == 0: continue
    DV = D @ V
    kept_energy = (DV.pow(2).sum() / D.pow(2).sum()).item()          # share of D inside the row space of V
    dM = C * gm * (DV @ V.T)                                          # the low-rank spend update
    on = (dM[nz] * D[nz]).mean().item() / (C * gm).item()             # shrink of |M| at flipped entries / intended
    off = dM[~nz].abs().mean().item() / (C * gm).item()               # average change elsewhere / intended
    # weights already at +-1 in the direction M pushes: -sign(M) * t == 1 -> cannot move further
    key = [k for k in sd if k.endswith("wpacked")][i]
    t = unpack_rows(sd[key].to(dev), V.shape[0]).float()
    push = -M.sign()
    stuck = (push * t == 1)
    a = M.abs().flatten(); thr = a.kthvalue(int(a.numel() * 0.99)).values
    top = M.abs() >= thr
    g = G[names[i]].to(dev)
    rows.append((kept_energy, on, off, stuck.float().mean().item(), stuck[top].float().mean().item(),
                 ((push * -g.sign()) > 0)[top & ~stuck].float().mean().item()))
R = np.array(rows)
print(f"layers {len(R)}; spend constant c = {C}")
print(f"share of the flip pattern D inside M's rank-256 row space:        {R[:, 0].mean():.4f}")
print(f"shrink of |M| at the flipped entries / intended (c * mean|M|):    {R[:, 1].mean():.4f}")
print(f"change of M elsewhere / intended:                                 {R[:, 2].mean():.4f}")
print(f"weights that M pushes but that are already at +-1 that way (all): {R[:, 3].mean():.3f}")
print(f"  ... among M's top 1% entries:                                   {R[:, 4].mean():.3f}")
print(f"  top-1% entries that CAN move: share going with the true gradient {R[:, 5].mean():.3f}")
