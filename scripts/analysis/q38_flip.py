"""q38_flip.py: in a Flash-Next run (600 ->
700), what flips the dense modules' gradient: the trit moves or the floats' drift? (15M: the moves alone reset every
row's gradient within 25 steps, s_j ~ -1.2; the floats add to the flip.)

The true KD factor gradient G (sum_t gyb^T x, as the accumulator sees it; q38_align's streaming backward) of every dense
module on the run's val windows, in four states built from s600 + delta700 (q38_val_lab.mix):
  600 | T7 = trits 700, floats 600 (the moves alone) | F7 = floats 700, trits 600 (the floats alone) | 700
Windows split in halves A / B (independent noise). Per dense module family, for each state X:
  rho_X = <G_X, dT> / <G_600, dT>      along the dense moves dT = T700 - T600 (1 = still downhill, < 0 = overshot)
  s_X   = sum_j <G_X,B - G_600,B, G_600,A>_j / sum_j <G_600,A, G_600,B>_j   on unmoved entries (cross-half: noise-free;
          -1 = the old gradient gone, < -1 = flipped)
  c_X   = <G_600,A, G_X,B> / sqrt(<G_600,A, G_600,B> <G_X,A, G_X,B>)   signal correlation of the gradients (unmoved)
Per-state gradients go to --tmp in bf16 (layer files), then a stats pass.

  python scripts/analysis/q38_flip.py --model BASE --slim S600 --delta DELTA --val val8.pt --tmp DIR [--windows 4] [--mem-gb 20]"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim
from ttf.q38.train import tern_modules, _get, DENSE
from q38_val_lab import mix
from q38_align import dT_of
from ttf.qwen import _kd_topk_chunk

STATES = {"600": set(), "T7": {"adapter", "dense", "expert"}, "F7": {"float"}, "700": {"float", "adapter", "dense", "expert"}}


def load_state(m, a, use):
    for i, L in enumerate(m.layers):
        b600 = torch.load(os.path.join(a.slim, f"L{i}.pt"), map_location="cpu", mmap=True, weights_only=False)["layer"]
        d700 = torch.load(os.path.join(a.delta, f"L{i}.pt"), map_location="cpu", weights_only=False)
        L.load_state_dict(mix(b600, d700, use), strict=False)


def grads(m, a, dev, mbs, tag):
    """G per dense module, summed over each half of the windows, written to tmp/tag_{A,B}_L{i}.pt"""
    nL = len(m.layers)
    acc = [{} for _ in range(nL)]
    half = len(mbs) // 2
    for w, b in enumerate(mbs):
        hk = "A" if w < half else "B"
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
        hf = h.detach().requires_grad_(True)
        with torch.enable_grad():
            x = m.head_in(hf)[0]
            xd = x.detach().requires_grad_(True)
            for c0 in range(0, T, 512):
                sl = slice(c0, c0 + 512)
                l = _kd_topk_chunk(xd[sl], m.lm_head.weight, tgt[sl], kid[sl], klw[sl], 1.0, mode="renorm")
                l.backward()
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
            for p in paths:
                t = _get(L, p)
                o, gy, xx = t.take_grouped()
                t.capture = False; t._ad_leaf = None
                G = (gy.float().T @ xx.float()).cpu()
                k = (hk, p)
                acc[i][k] = G if k not in acc[i] else acc[i][k] + G
            for _, t in tern_modules(L):
                t._wc = None
            for q in L.parameters():
                q.grad = None
            L.to("cpu")
        print(f"{tag} window {w} done ({time.time() - t0:.0f} s)", flush=True)
    for i in range(nL):
        for hk in ("A", "B"):
            torch.save({p: G.to(torch.bfloat16) for (h_, p), G in acc[i].items() if h_ == hk},
                       os.path.join(a.tmp, f"{tag}_{hk}_L{i}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", required=True)
    ap.add_argument("--delta", required=True); ap.add_argument("--val", required=True); ap.add_argument("--tmp", required=True)
    ap.add_argument("--windows", type=int, default=4); ap.add_argument("--mem-gb", type=float, default=0.0)
    ap.add_argument("--states", default="600,T7,F7,700"); ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out", default="flip.json")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.tmp, exist_ok=True)
    states = a.states.split(",")
    m = None
    if not a.stats_only:
        if a.mem_gb:
            torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        m = load_slim(a.model, a.slim, dev, layers=0, resident=0)
        for p in m.parameters():
            p.requires_grad_(False)
        mbs = torch.load(a.val, weights_only=False)[:a.windows]
        for s in states:
            if os.path.exists(os.path.join(a.tmp, f"{s}_B_L{len(m.layers) - 1}.pt")):
                print(f"{s}: gradients on disk, skipped", flush=True)
                continue
            load_state(m, a, STATES[s])
            grads(m, a, dev, mbs, s)
        load_state(m, a, set())
    # stats: per family f and state X
    fam = lambda p: "linear_attn" if p.startswith("linear_attn") else "self_attn" if p.startswith("self_attn") else "shared_expert"
    nL = len([f for f in os.listdir(a.slim) if f.startswith("L") and f.endswith(".pt")])
    if m is None:
        m = load_slim(a.model, a.slim, "cpu", layers=0, resident=0)
    S = {}
    for i in range(nL):
        L = m.layers[i]
        d700 = torch.load(os.path.join(a.delta, f"L{i}.pt"), map_location="cpu", weights_only=False)
        G = {(s, h): torch.load(os.path.join(a.tmp, f"{s}_{h}_L{i}.pt"), weights_only=False) for s in states for h in ("A", "B")}
        for p in G[(states[0], "A")]:
            t = _get(L, p)
            dT = dT_of(t, d700[p + ".wpacked"])
            um = dT == 0
            g0A = G[("600", "A")][p].float(); g0B = G[("600", "B")][p].float()
            a0 = float(((g0A + g0B) * dT).sum())
            den = float((g0A * g0B * um).sum())
            for s in states:
                gA = G[(s, "A")][p].float(); gB = G[(s, "B")][p].float()
                e = S.setdefault((fam(p), s), dict(a=0.0, a0=0.0, num=0.0, den=0.0, c01=0.0, cxx=0.0))
                e["a"] += float(((gA + gB) * dT).sum()); e["a0"] += a0
                e["num"] += float(((gB - g0B) * g0A * um).sum()); e["den"] += den
                e["c01"] += float((g0A * gB * um).sum()); e["cxx"] += float((gA * gB * um).sum())
        print(f"stats L{i}", flush=True)
    rows = []
    for (f, s), e in sorted(S.items()):
        r = dict(family=f, state=s, rho=e["a"] / e["a0"] if e["a0"] else None, s=e["num"] / e["den"] if e["den"] else None,
                 c=e["c01"] / (max(e["den"], 1e-30) * max(e["cxx"], 1e-30)) ** 0.5 if e["den"] > 0 and e["cxx"] > 0 else None,
                 a0=e["a0"])
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
