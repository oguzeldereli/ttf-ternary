"""move_stats.py (6 Nov 2026): stats from move_capture.py's gradient files (first used on 0.8B conversion snapshots).
Per family (DeltaNet / attention / MLP; main trits vs adapter trits) and pair (b, b + l):
  agreement  sign(D) = sign(-mu) on resolved entries (|mu| > 2 sd), at b and at b + l
  rho        sum mu(b + l) D / sum mu(b) D     (1 = still downhill, 0 = at the minimum along D, < 0 = overshot)
  rho_cf     the same with mu of snapshot b holding the trits of b + l (the moves alone, floats of b)
  bins       moved entries by quintile of z = |muA(b)| / sd(b) (half A); on half B: a0 = sum muB(b) D,
             a1 = sum muB(b + l) D, net = (a0 + a1) / 2 (trapezoid: the loss change from the moves along the path)
  python scripts/analysis/move_stats.py DIR OUT.json [--pairs 300:325,300:350,300:400,325:350,350:400] [--cf 300:325:cf300]"""
import argparse, json, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("out")
ap.add_argument("--pairs", default="300:325,300:350,300:400,325:350,350:400")
ap.add_argument("--cf", default="300:325:cf300")
a = ap.parse_args()


def fam(n):
    f = "deltanet" if "linear_attn" in n else "attn" if "self_attn" in n else "mlp"
    return f + ("_ad" if ".ad." in n else "")


_cache = {}


def g(tag):
    if tag not in _cache:
        if len(_cache) > 3:
            _cache.pop(next(iter(_cache)))
        p = os.path.join(a.dir, f"g_{tag}.pt")
        _cache[tag] = torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None
    return _cache[tag]


rows = []
cfmap = {}
for s in a.cf.split(","):
    if s:
        b_, l_, t_ = s.split(":"); cfmap[(b_, l_)] = t_
for pr in a.pairs.split(","):
    b, l = pr.split(":")
    G0, G1 = g(b), g(l)
    if G0 is None or G1 is None:
        print(f"{pr}: missing", flush=True); continue
    Gc = g(cfmap[(b, l)]) if (b, l) in cfmap else None
    acc = {}
    for n in G0["mu"]:
        D = (G1["W"][n].float() - G0["W"][n].float())
        mv = D != 0
        if not bool(mv.any()):
            continue
        m0, m1 = G0["mu"][n], G1["mu"][n]
        sd0, sd1 = G0["var"][n].sqrt(), G1["var"][n].sqrt()
        e = acc.setdefault(fam(n), dict(n=0, N=0, ag0=0, r0=0, ag1=0, r1=0, a0=0.0, a1=0.0, acf=0.0,
                                         bins=[[0.0, 0.0, 0] for _ in range(5)]))
        e["n"] += int(mv.sum()); e["N"] += D.numel()
        r0 = mv & (m0.abs() > 2 * sd0); r1 = mv & (m1.abs() > 2 * sd1)
        e["ag0"] += int(((-m0).sign() == D.sign())[r0].sum()); e["r0"] += int(r0.sum())
        e["ag1"] += int(((-m1).sign() == D.sign())[r1].sum()); e["r1"] += int(r1.sum())
        e["a0"] += float((m0 * D).sum()); e["a1"] += float((m1 * D).sum())
        if Gc is not None:
            e["acf"] += float((Gc["mu"][n] * D).sum())
        z = (G0["muA"][n].abs() / (sd0 * 2 ** 0.5 + 1e-30))[mv]   # (half A: sd of a half mean ~ sqrt 2 sd)
        q = torch.quantile(z[torch.randperm(z.numel())[:1_000_000]], torch.tensor([0.2, 0.4, 0.6, 0.8]))
        bi = torch.bucketize(z, q)
        a0B = (G0["muB"][n] * D)[mv]; a1B = (G1["muB"][n] * D)[mv]
        for k in range(5):
            s_ = bi == k
            e["bins"][k][0] += float(a0B[s_].sum()); e["bins"][k][1] += float(a1B[s_].sum()); e["bins"][k][2] += int(s_.sum())
    for f, e in sorted(acc.items()):
        r = dict(pair=pr, family=f, moved=e["n"], share=e["n"] / max(e["N"], 1),
                 agree_b=e["ag0"] / max(e["r0"], 1), agree_bl=e["ag1"] / max(e["r1"], 1),
                 a0=e["a0"], rho=e["a1"] / e["a0"] if e["a0"] else None,
                 rho_cf=(e["acf"] / e["a0"] if (Gc is not None and e["a0"]) else None),
                 bins_net=[(x[0] + x[1]) / 2 for x in e["bins"]], bins_a0=[x[0] for x in e["bins"]],
                 bins_n=[x[2] for x in e["bins"]])
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if k not in ("bins_net", "bins_a0", "bins_n")}), flush=True)
        print("   bins (z quintile, low->high) net:", " ".join(f"{x:+.2e}" for x in r["bins_net"]),
              "| a0:", " ".join(f"{x:+.2e}" for x in r["bins_a0"]), flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
