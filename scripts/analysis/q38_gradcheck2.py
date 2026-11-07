"""q38_gradcheck2.py (for a Flash-Next run in which the attention / DeltaNet projections' trit moves hurt and the shared
expert's and routed experts' did not): in real layers (s600 state, the real hidden state of a val window), the TTF
factor gradient (sum_t gyb^T x) of every dense module, once with the run's fast paths (compile_hc: compiled GatedResidual,
SwiGLU and mixer norms, the fused causal conv; fused Pre / Post; grouped GEMMs) and once eager (none of them: the
reference's norms and conv, padded bmm path). Same input, same loss <out, R>. A correct fast path agrees to rounding
(cos ~1, norm ratio ~1) in every module.

  python scripts/analysis/q38_gradcheck2.py --model BASE --slim S600 --val val8.pt [--layers 2,3] [--tokens 1024] [--mem-gb 12]"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim
from ttf.q38.train import tern_modules, _get, DENSE
from ttf.q38 import moe as MT



def grads(L, h, pe, ids, R, paths):
    for _, u in tern_modules(L):
        u.capture = False; u._fac = []; u._ad_leaf = None
    mods = {p: _get(L, p) for p in paths}
    for t in mods.values():
        t.capture = True
    with torch.enable_grad():
        out = L(h.detach(), pe, ids)
        (out.float() * R).sum().backward()
    G = {}
    for p, t in mods.items():
        o, gy, x = t.take_grouped()
        G[p] = (gy.float().T @ x.float()).cpu()
        t.capture = False; t._ad_leaf = None
    fl = {n: p.grad.float().cpu().clone() for n, p in L.named_parameters() if p.grad is not None}
    for p in L.parameters():
        p.grad = None
    return G, out.detach().float(), fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--layers", default="2,3"); ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--mem-gb", type=float, default=0.0)
    a = ap.parse_args()
    dev = "cuda"
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    want = [int(x) for x in a.layers.split(",")]
    m = load_slim(a.model, a.slim, dev, layers=max(want) + 1, resident=0)
    b = torch.load(a.val, weights_only=False)[0]
    ids = b["ids"][:, :a.tokens].to(dev)
    pe = m.rope(ids.shape[1], dev)
    hs = {}
    with torch.no_grad():
        h = m.embed(ids)
        for i, L in enumerate(m.layers):
            if i in want:
                hs[i] = h.clone()
            L.to(dev)
            h = L(h, pe, ids)
            for _, t in tern_modules(L):
                t._wc = None
            L.to("cpu")
    torch.manual_seed(0)
    res = {}
    for cfg in ("eager", "fast"):
        if cfg == "eager":
            MT.FUSED_DENSE[0] = False; MT.GROUPED[0] = False
        else:
            MT.FUSED_DENSE[0] = True; MT.GROUPED[0] = True
            QM.compile_hc()
        for i in want:
            L = m.layers[i]; L.to(dev)
            for _, t in tern_modules(L):
                t.invalidate()
            paths = [p for p in DENSE if _get(L, p) is not None and hasattr(_get(L, p), "wpacked")]
            g = torch.Generator(device=dev).manual_seed(i)
            R = torch.randn(hs[i].shape, device=dev, dtype=torch.float32, generator=g)
            res[(cfg, i)] = grads(L, hs[i], pe, ids, R, paths)
            for _, t in tern_modules(L):
                t._wc = None
            L.to("cpu")
        print(f"{cfg} done", flush=True)
    for i in want:
        Ge, oe, fe = res[("eager", i)]; Gf, of, ff = res[("fast", i)]
        print(f"L{i} ({m.layers[i].kind}): out rel diff fast vs eager {float((of - oe).norm() / oe.norm()):.2e}", flush=True)
        for p in Ge:
            a_, b_ = Ge[p], Gf[p]
            cos = float((a_ * b_).sum() / (a_.norm() * b_.norm()))
            print(f"  {p}: cos {cos:.5f}  |fast|/|eager| {float(b_.norm() / a_.norm()):.4f}", flush=True)
        for n in sorted(fe):
            if n in ff:
                a_, b_ = fe[n], ff[n]
                cos = float((a_ * b_).sum() / (a_.norm() * b_.norm() + 1e-30))
                if cos < 0.999 or abs(float(b_.norm() / (a_.norm() + 1e-30)) - 1) > 0.01:
                    print(f"  float {n}: cos {cos:.5f}  |fast|/|eager| {float(b_.norm() / (a_.norm() + 1e-30)):.4f}", flush=True)


if __name__ == "__main__":
    main()
