"""q38_val_lab.py: the run's val (KD renorm, kd_weight 1) and val_ce on its 8 val windows, for mixes of two
checkpoints: s600 (slim) and the s600 -> s700 delta (a delta file: packed buffers as changed bytes, the rest
in full). Variants take each part from 600 or 700:
  A s600 | B s700 | C trits 600 + floats 700 | D trits 700 + floats 600 |
  E s700, adapter trits at 600 | F s700, dense trits at 600 | G s700, expert trits at 600
If the trit moves of 600 -> 700 (~12M dense, ~43M adapter, ~2M expert) helped, C is worse than B.

  python scripts/analysis/q38_val_lab.py --model BASE --slim S600 --delta DELTA --val val8.pt [--variants ABCDEFG] [--sub 2]"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38.load import load_slim
from ttf.q38.train import tern_modules
from ttf.qwen import _kd_topk_chunk


def part(k):
    if not k.endswith("packed"):
        return "float"
    if k.endswith("ta_packed") or k.endswith("tb_packed"):
        return "adapter"
    if k.startswith("mlp.gu.") or k.startswith("mlp.dn."):
        return "expert"
    return "dense_" + ("la" if k.startswith("linear_attn.") else "sa" if k.startswith("self_attn.") else "se")


# which parts come from 700, per variant
V = {"A": set(), "B": {"float", "adapter", "dense", "expert"}, "C": {"float"}, "D": {"adapter", "dense", "expert"},
     "E": {"float", "dense", "expert"}, "F": {"float", "adapter", "expert"}, "G": {"float", "adapter", "dense"},
     "H": {"float", "adapter", "expert", "dense:rand"}, "I": {"float", "adapter", "expert", "dense:rev"},
     "J": {"float", "adapter", "expert", "dense_sa", "dense_se"},      # s700, linear_attn trits at 600
     "K": {"float", "adapter", "expert", "dense_la", "dense_se"},      # s700, self_attn trits at 600
     "L": {"float", "adapter", "expert", "dense_la", "dense_sa"}}      # s700, shared_expert trits at 600


def _dense_alt(k, v, d, mode, K):
    """dense trits of 600 with the 600 -> 700 moves replaced: "rand" the same count at random entries (a 0 goes to
    +-1, a +-1 to 0), "rev" each move reversed (T600 - (T700 - T600), clipped to [-1, 1])"""
    from ttf.q38.moe import unpack3, pack3
    t6 = unpack3(v, K)
    x = v.clone().reshape(-1); x[d[1]] = d[2]
    t7 = unpack3(x.view(v.shape), K)
    if mode == "rev":
        t = (2 * t6.to(torch.int16) - t7.to(torch.int16)).clamp(-1, 1).to(torch.int8)
    else:
        n = int((t6 != t7).sum())
        g = torch.Generator().manual_seed(hash(k) % (2**31))
        idx = torch.randperm(t6.numel(), generator=g)[:n]
        f = t6.reshape(-1).clone(); cur = f[idx]
        pm = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.int8)
        f[idx] = torch.where(cur == 0, pm, torch.zeros_like(cur))
        t = f.view(t6.shape)
    return pack3(t).view(v.shape)


def mix(b600, d700, use):
    out = {}
    alt = next((u for u in use if u.startswith("dense:")), None)
    use = set(use) | ({"dense_la", "dense_sa", "dense_se"} if "dense" in use else set())
    for k, v in b600.items():
        p = part(k)
        if alt and p.startswith("dense"):
            K = b600[k.rsplit(".", 1)[0] + ".col_scale"].shape[-1]
            out[k] = _dense_alt(k, v, d700[k], alt.split(":")[1], K)
            continue
        if p not in use:
            out[k] = v
            continue
        d = d700[k]
        if d[0] == "full":
            out[k] = d[1]
        else:
            x = v.clone().reshape(-1); x[d[1]] = d[2]; out[k] = x.view(v.shape)
    return out


@torch.no_grad()
def val(m, a, dev, mbs, use):
    T = mbs[0]["ids"].shape[1]
    pe = m.rope(T, dev)
    groups = [mbs[j:j + a.sub] for j in range(0, len(mbs), a.sub)]
    gi = [torch.cat([b["ids"] for b in g]).to(dev) for g in groups]
    hs = [m.embed(x) for x in gi]
    for i, L in enumerate(m.layers):
        b600 = torch.load(os.path.join(a.slim, f"L{i}.pt"), map_location="cpu", mmap=True, weights_only=False)["layer"]
        d700 = torch.load(os.path.join(a.delta, f"L{i}.pt"), map_location="cpu", weights_only=False)
        L.load_state_dict(mix(b600, d700, use), strict=False)
        L.to(dev)
        for _, t in tern_modules(L):
            t.invalidate()
        hs = [L(h, pe, x) for h, x in zip(hs, gi)]
        for _, t in tern_modules(L):
            t._wc = None
        L.to("cpu")
    W = m.lm_head.weight
    kd = ce = n = 0.0
    for g, h in zip(groups, hs):
        x = m.head_in(h)
        for r, b in enumerate(g):
            tgt, ids, lw = b["tgt"].to(dev), b["kd_ids"].to(dev), b["kd_lw"].to(dev)
            for c0 in range(0, T, 512):
                sl = slice(c0, c0 + 512)
                kd += float(_kd_topk_chunk(x[r, sl], W, tgt[sl], ids[sl], lw[sl], 1.0, mode="renorm"))
                ce += float(_kd_topk_chunk(x[r, sl], W, tgt[sl], ids[sl], lw[sl], 0.0, mode="renorm"))
            n += int((tgt != -1).sum())
    return kd / n, ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--slim", required=True)
    ap.add_argument("--delta", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--variants", default="ABCDEFG"); ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--out", default="val_mix.jsonl")
    ap.add_argument("--mem-gb", type=float, default=0.0, help="cap this process's GPU memory (shared GPU)")
    a = ap.parse_args()
    dev = "cuda"
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    QM.compile_hc()        # (the run used --compile-hc)
    m = load_slim(a.model, a.slim, dev)
    mbs = torch.load(a.val, weights_only=False)
    for v in a.variants:
        t = time.time()
        kd, ce = val(m, a, dev, mbs, V[v])
        r = {"variant": v, "from700": sorted(V[v]), "val": kd, "val_ce": ce, "s": round(time.time() - t)}
        print(json.dumps(r), flush=True)
        open(a.out, "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
