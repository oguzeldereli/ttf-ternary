"""The no-look-ahead unigram plateau: per snapshot of a run, on 8 validation sequences,
  full      model loss
  direct    loss with every block skipped (embedding -> final norm -> tied head)
  -blk i    loss with block i skipped (how much that block contributes)
  contrib   rms of the blocks' total residual contribution relative to the embedding's rms
  trits     share of trits changed since the previous snapshot
  python -m scripts.analysis.plateau RUN_DIR [kernel|master]"""
import sys, glob, re, numpy as np, torch
from scripts.analysis.induction_heads import load
from ttf.flip import KernelTernaryLinear
from ttf.kernel import unpack_rows
from ttf.train import get_batch

run, kind = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "kernel")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
XY = [get_batch(val, 4, 2048, "cuda", torch.Generator().manual_seed(11 + i)) for i in range(2)]
paths = sorted(glob.glob(f"{run}/ckpt_*.pt"), key=lambda p: int(re.findall(r"ckpt_(\d+)", p)[0]))
prev = None


def loss(m, skip=()):
    orig = {}
    for i in skip:
        orig[i] = m.layers[i].forward
        m.layers[i].forward = lambda h, fc, ckpt=False: h
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        L = float(np.mean([m(x, y)[1].item() for x, y in XY]))
    for i, f in orig.items(): m.layers[i].forward = f
    return L


for p in paths:
    s, m, V = load(p, kind)
    n = len(m.layers)
    full, direct = loss(m), loss(m, range(n))
    per = [loss(m, [i]) - full for i in range(n)]
    x = XY[0][0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        e = m.tok_emb(x).float(); h = e.to(torch.bfloat16); fc = m.freqs_cis
        for layer in m.layers: h = layer(h, fc)
        contrib = ((h.float() - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    T = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in m.modules() if isinstance(l, KernelTernaryLinear)] \
        if kind == "kernel" else None
    ch = "" if prev is None or T is None else \
        f"{100 * sum(int((a != b).sum()) for a, b in zip(T, prev)) / sum(a.numel() for a in T):5.2f}%"
    prev = T
    print(f"step {s:4d} ({(s + 1) * 32768 / 1e6:4.1f}M)  full {full:.3f}  direct {direct:.3f}  "
          f"contrib/emb {contrib:6.2f}  trits changed {ch:>6s}  |  skip-block dL: " + " ".join(f"{d:+.2f}" for d in per), flush=True)
    del m; torch.cuda.empty_cache()
