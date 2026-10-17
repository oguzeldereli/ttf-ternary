"""Where does M's energy sit at step 4000, and how much of the true gradient pushes weights past +-1?
(a) entries flipped this step (kept set), (b) weights already at +-1 in M's push direction (stuck),
(c) the rest (movable). Also the share of the true gradient's energy on weights it pushes past +-1.

  python -m scripts.analysis.stuck_check
"""
import json, torch, numpy as np
from scripts.analysis.testbench import OUT
from ttf.kernel import unpack_rows
dev = "cuda"
meta = json.load(open(f"{OUT}/meta.json")); names = meta["ternary_layers"]
b = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
flips = torch.load(f"{OUT}/flips.pt"); G = torch.load(f"{OUT}/gbar.pt")
keys = [k for k in b["model"] if k.endswith("wpacked")]
E = np.zeros(3); Etot = 0; gst = gtot = 0.0; prec = []
for i, (U, V) in enumerate(b["lowrank"]):
    U, V = U.to(dev).float(), V.to(dev).float(); M = U @ V.T
    t = unpack_rows(b["model"][keys[i]].to(dev), V.shape[0]).float()
    D = flips["M_kept"][i].to(dev) != 0
    stuck = (-M.sign() * t == 1) & ~D
    mov = ~stuck & ~D
    e = M.pow(2)
    E += [e[D].sum().item(), e[stuck].sum().item(), e[mov].sum().item()]; Etot += e.sum().item()
    g = G[names[i]].to(dev)
    gstuck = (-g.sign() * t == 1)                      # the true gradient pushes these past +-1
    gst += g[gstuck].pow(2).sum().item(); gtot += g.pow(2).sum().item()
    prec.append(((M.sign() == g.sign()) & mov).sum().item() / max(mov.sum().item(), 1))
print(f"M energy: just flipped {E[0] / Etot:.4f} | stuck at +-1 in M's direction {E[1] / Etot:.3f} | movable {E[2] / Etot:.3f}")
print(f"true gradient energy on weights it pushes past +-1: {gst / gtot:.3f}")
print(f"movable entries where M agrees in sign with the true gradient: {np.mean(prec):.3f}")
