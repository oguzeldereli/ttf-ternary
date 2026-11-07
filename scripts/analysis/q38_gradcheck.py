"""q38_gradcheck.py: in a real layer (s600 state, real hidden state from a val window),
the TTF factor gradient each dense module hands its accumulator (sum_t gyb^T x over the collected factors) against
autograd through a plain float re-implementation of that module (same centring, column / row scales, act quant with a
straight-through estimator, adapter, out bias), loss = <layer output, R> with a fixed random R.
Correct factors: cosine ~1 and norm ratio ~1 for every module.

  python scripts/analysis/q38_gradcheck.py --model BASE --slim S600 --val val8.pt [--layers 2,3] [--tokens 1024] [--mem-gb 12]"""
import argparse, os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim
from ttf.q38.train import tern_modules, _set, _get, DENSE
from ttf.q38.moe import unpack3, act_quant_i8


class RefLin(nn.Module):
    """float reference of a TernLinear (E = 1): y = rs * (beta * deq(q(xt)) W^T + adapter(xt)) + ob, xt = (x - mu) * cs"""
    def __init__(self, t):
        super().__init__()
        self.t = t
        self.W = nn.Parameter(unpack3(t.wpacked, t.K)[0].float())

    def forward(self, x):
        t = self.t; sh = x.shape
        x2 = x.reshape(-1, t.K).float()
        xt = x2 - t.in_mean[0].float() if t.in_mean is not None else x2
        xt = xt * t.col_scale[0].float()
        q, s = act_quant_i8(xt.to(torch.bfloat16), t.act_bits)
        deq = q.float() * s.float()[:, None]
        xs = xt + (deq - xt).detach()                      # straight through the act quant (the core's / TTF's rule)
        y = t.beta[0].float() * (xs @ self.W.T)
        if t.ra:
            TA = unpack3(t.ta_packed, t.ra)[0].float(); TB = unpack3(t.tb_packed, t.K)[0].float()
            y = y + ((xt @ TB.T) * t.ad_cb[0].float()) @ TA.T * t.ad_sa[0].float()
        y = y * t.row_scale[0].float()
        if t.out_bias is not None:
            y = y + t.out_bias[0].float()
        return y.to(x.dtype).view(*sh[:-1], t.N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--layers", default="2,3"); ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--mem-gb", type=float, default=0.0)
    a = ap.parse_args()
    dev = "cuda"
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    QM.compile_hc()
    want = [int(x) for x in a.layers.split(",")]
    m = load_slim(a.model, a.slim, dev, layers=max(want) + 1, resident=0)
    b = torch.load(a.val, weights_only=False)[0]
    ids = b["ids"][:, :a.tokens].to(dev)
    pe = m.rope(ids.shape[1], dev)
    with torch.no_grad():
        h = m.embed(ids)
    torch.manual_seed(0)
    for i, L in enumerate(m.layers):
        L.to(dev)
        for _, t in tern_modules(L):
            t.invalidate()
        if i in want:
            R = torch.randn(h.shape, device=dev, dtype=torch.float32)
            for path in DENSE:
                t = _get(L, path)
                if t is None or not hasattr(t, "wpacked"):
                    continue
                # (1) the TTF path: capture on this module only, the factor gradient
                for _, u in tern_modules(L):
                    u.capture = u is t; u._fac = []; u._ad_leaf = None
                with torch.enable_grad():
                    out = L(h.detach(), pe, ids)
                    (out.float() * R).sum().backward()
                g = t.take_grouped()
                t.capture = False; t._ad_leaf = None
                o, gy, x = g
                G = (gy.float().T @ x.float())                       # [N, K]
                # (2) the float reference module in its place
                ref = RefLin(t).to(dev)
                _set(L, path, ref)
                with torch.enable_grad():
                    out2 = L(h.detach(), pe, ids)
                    (out2.float() * R).sum().backward()
                Gr = ref.W.grad
                _set(L, path, t)
                cos = float((G * Gr).sum() / (G.norm() * Gr.norm()))
                rel_out = float((out2.float() - out.float()).norm() / out.float().norm())
                print(f"L{i} {path}: cos {cos:.4f}  |G|/|Gref| {float(G.norm() / Gr.norm()):.4f}  out rel diff {rel_out:.2e}  "
                      f"sign agree (top 1%) {float(((G.sign() == Gr.sign()).float()[Gr.abs() >= Gr.abs().quantile(0.99) if Gr.numel() < 16_000_000 else Gr.abs() >= Gr.abs().flatten()[::7].quantile(0.99)]).mean()):.4f}",
                      flush=True)
                for p in L.parameters():
                    p.grad = None
        with torch.no_grad():
            h = L(h, pe, ids)
        for _, t in tern_modules(L):
            t._wc = None
        L.to("cpu")


if __name__ == "__main__":
    main()
