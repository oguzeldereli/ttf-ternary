"""q38_align.py: the first-order effect of the run's actual trit moves 600 -> 700 under the
true losses. On the s600 model, a full-model backward of the run's val windows (layer inputs stashed on the host, then
layer by layer from the top), once for the KD loss (kd_weight 1, renorm, as training) and once for the next-token
CE; per dense module the factor gradient G = sum_t gyb^T x (= dL/dT, as the accumulator sees it) against the moves
dT = T700 - T600 (from the run's s700 delta):
  lin  = sum G * dT / n_tokens       (first-order change of the per-token loss from that module's moves; < 0 = descent)
  cos  = -<G, dT> / (|G| |dT|)
  frac = the share of moved entries whose sign agrees with -G
Random moves give lin ~ 0, cos ~ 0, frac ~ 0.5.

  python scripts/analysis/q38_align.py --model BASE --slim S600 --delta DELTA --val val8.pt [--windows 2] [--mem-gb 16] [--out align.jsonl]"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim
from ttf.q38.train import tern_modules, _get, DENSE
from ttf.q38.moe import unpack3
from ttf.qwen import _kd_topk_chunk


def dT_of(t, d):
    """T700 - T600 (int8) of module t (state s600) from its delta entry ("delta", idx, new bytes)"""
    p6 = t.wpacked.detach().cpu()
    x = p6.clone().reshape(-1); x[d[1]] = d[2]
    return (unpack3(x.view(p6.shape), t.K)[0].to(torch.int16) - unpack3(p6, t.K)[0].to(torch.int16)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", required=True)
    ap.add_argument("--delta", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--windows", type=int, default=2); ap.add_argument("--mem-gb", type=float, default=0.0)
    ap.add_argument("--out", default="align.jsonl")
    a = ap.parse_args()
    dev = "cuda"
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    QM.compile_hc()
    m = load_slim(a.model, a.slim, dev, layers=0, resident=0)
    for p in m.parameters():
        p.requires_grad_(False)
    mbs = torch.load(a.val, weights_only=False)[:a.windows]
    nL = len(m.layers)
    acc = {}       # (loss, layer, path) -> [sum G*dT, sum G^2 (for |G|), sum dT^2, agree, moved]
    ntok = 0
    for w, b in enumerate(mbs):
        t0 = time.time()
        ids = b["ids"].to(dev); T = ids.shape[1]; pe = m.rope(T, dev)
        stash = []
        with torch.no_grad():
            h = m.embed(ids)
            for L in m.layers:
                stash.append(h.cpu())
                L.to(dev)
                for _, t in tern_modules(L):
                    t.invalidate()
                h = L(h, pe, ids)
                for _, t in tern_modules(L):
                    t._wc = None
                L.to("cpu")
        tgt, kid, klw = b["tgt"].to(dev), b["kd_ids"].to(dev), b["kd_lw"].to(dev)
        n = int((tgt != -1).sum()); ntok += n
        for loss in ("kd", "ce"):
            hf = h.detach().requires_grad_(True)
            with torch.enable_grad():
                x = m.head_in(hf)[0]
                xd = x.detach().requires_grad_(True)
                tot = 0.0
                for c0 in range(0, T, 512):
                    sl = slice(c0, c0 + 512)
                    l = _kd_topk_chunk(xd[sl], m.lm_head.weight, tgt[sl], kid[sl], klw[sl], 1.0 if loss == "kd" else 0.0,
                                       mode="renorm")
                    l.backward(); tot += float(l)
                x.backward(xd.grad)
            g = hf.grad
            for i in range(nL - 1, -1, -1):
                L = m.layers[i]; L.to(dev)
                for _, t in tern_modules(L):
                    t.invalidate(); t._fac = []; t._ad_leaf = None
                paths = [p for p in DENSE if _get(L, p) is not None and hasattr(_get(L, p), "wpacked")]
                for p in paths:
                    _get(L, p).capture = True
                hin = stash[i].to(dev).requires_grad_(True)
                with torch.enable_grad():
                    out = L(hin, pe, ids)
                    out.backward(g)
                g = hin.grad.detach()
                d700 = None
                for p in paths:
                    t = _get(L, p)
                    o, gy, xx = t.take_grouped()
                    t.capture = False; t._ad_leaf = None
                    G = (gy.float().T @ xx.float()).cpu()
                    if d700 is None:
                        d700 = torch.load(os.path.join(a.delta, f"L{i}.pt"), map_location="cpu", weights_only=False)
                    dT = dT_of(t, d700[p + ".wpacked"])
                    mv = dT != 0
                    s = acc.setdefault((loss, i, p), [0.0, None, 0.0, 0, 0])
                    s[0] += float((G * dT).sum())
                    s[1] = G.clone() if s[1] is None else s[1] + G
                    s[2] = float((dT ** 2).sum())
                    s[3] += int(((-G).sign()[mv] == dT.sign()[mv]).sum()); s[4] += int(mv.sum())
                for _, t in tern_modules(L):
                    t._wc = None
                for q in L.parameters():
                    q.grad = None
                L.to("cpu")
            print(f"window {w} {loss} loss {tot / n:.4f} done ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for (loss, i, p), s in sorted(acc.items()):
        G = s[1]; dTn = s[2] ** 0.5
        rows.append({"loss": loss, "layer": i, "module": p, "lin": s[0] / ntok, "cos": -s[0] / (float(G.norm()) * dTn + 1e-30),
                     "frac": s[3] / max(s[4], 1), "moved": s[4] // len(mbs)})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for loss in ("kd", "ce"):
        for fam in ("linear_attn", "self_attn", "mlp.shared_expert"):
            rs = [r for r in rows if r["loss"] == loss and r["module"].startswith(fam)]
            if rs:
                print(f"{loss} {fam}: sum lin {sum(r['lin'] for r in rs):+.5f}  mean cos {sum(r['cos'] for r in rs) / len(rs):+.4f}  "
                      f"agree {sum(r['frac'] * r['moved'] for r in rs) / max(sum(r['moved'] for r in rs), 1):.4f}", flush=True)
        for mod in DENSE:
            rs = [r for r in rows if r["loss"] == loss and r["module"] == mod]
            if rs:
                print(f"  {loss} {mod}: lin {sum(r['lin'] for r in rs):+.5f} cos {sum(r['cos'] for r in rs) / len(rs):+.4f} "
                      f"agree {sum(r['frac'] * r['moved'] for r in rs) / max(sum(r['moved'] for r in rs), 1):.4f}", flush=True)


if __name__ == "__main__":
    main()
