"""Why does the additive magnitude term stall the toy? Per snapshot of a run: per layer, rms of the trit path
(beta T x) vs the additive path ((x B) A^T), and on one training batch the gradient norms of the float tail
groups (embedding, norm gains, mag_A, mag_B), which share one clip at 1.0.

  MAG_KIND=add python -m scripts.toy.diag_add RUN_DIR STEP [STEP ...]
"""
import sys, numpy as np, torch
from scripts.analysis.induction_heads import load
from ttf.layers import KernelTernaryLinear, _KernelTernFn
from ttf.train import get_batch

run, steps = sys.argv[1], sys.argv[2:]
data = np.memmap("data/toy_ind_v256_train.bin", dtype=np.uint16, mode="r")
for st in steps:
    s, m, V = load(f"{run}/ckpt_{st}.pt", "kernel"); m.train()
    x, y = get_batch(data, 32, 256, "cuda", torch.Generator().manual_seed(3))
    stats = []
    def hook(l, inp, out):
        xi = inp[0]
        if l.col_scale is not None: xi = xi * l.col_scale.to(xi.dtype)
        with torch.no_grad():
            t = _KernelTernFn.apply(xi, l.wpacked, l).float()
            a = ((xi @ l.mag_B.to(xi.dtype)) @ l.mag_A.to(xi.dtype).T).float() if getattr(l, "mag_A", None) is not None else torch.zeros_like(t)
        stats.append((t.pow(2).mean().sqrt().item(), a.pow(2).mean().sqrt().item()))
    Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
    hs = [l.register_forward_hook(hook) for l in Ls]
    for l in Ls: l.capture = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, y)[1]
    loss.backward()
    for h in hs: h.remove()
    names = ["wq", "wk", "wv", "wo", "gate", "up", "down"]
    print(f"step {s}  loss {loss.item():.3f}")
    print("  layer    trit rms   add rms   add/trit")
    for i, (t, a) in enumerate(stats):
        print(f"  L{i // 7}.{names[i % 7]:5s} {t:9.4f} {a:9.4f} {a / max(t, 1e-12):9.3f}")
    grp = {"embedding": [], "norm gains": [], "mag_A": [], "mag_B": []}
    for n, p in m.named_parameters():
        if p.grad is None: continue
        k = "mag_A" if n.endswith("mag_A") else "mag_B" if n.endswith("mag_B") else "embedding" if "emb" in n else "norm gains"
        grp[k].append(p.grad.float().pow(2).sum().item())
    tot = sum(sum(v) for v in grp.values()) ** 0.5
    print("  grad norm: " + "  ".join(f"{k} {sum(v) ** 0.5:.4f}" for k, v in grp.items()) + f"  | total {tot:.4f} (clip 1.0)")
    gw = [l.gw.float().abs().mean().item() for l in Ls if l.gw is not None]
    print(f"  trit gradient mean |g| per layer: {' '.join(f'{v:.2e}' for v in gw)}")
    del m
