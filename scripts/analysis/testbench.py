"""Fixed test bench for flip-proposal methods at one checkpoint of the momentum + look-ahead run.

build: saves to OUT/
  gbar.pt, gbar_a.pt, gbar_b.pt   true gradient: mean over NB training batches (and its two halves), every
                                  parameter (ternary layers as dL/dT, the float tail as p.grad), fp32
  g_step.pt                       one batch's gradient (the step's own), every parameter
  g_la1.pt, g_la2.pt              the two cross-batch look-ahead gradients, taken at the proposed point
  flips.pt                        proposed / kept flip sets of the current method (int8 per ternary layer)
  meta.json                       checkpoint, rate, seeds, batch counts
eval: scores flip sets against gbar: count, precision (share with D * gbar < 0), true first-order gain
  -sum(D * gbar) in total and per flip, cos(D, -gbar), held-out loss change of applying the set (float
  tail fixed); references: oracle (best flips by -gbar at the same count, respecting the trit bounds)
  and random (same count).

  python -m scripts.analysis.testbench build [CKPT] [RATE] [NB]
  python -m scripts.analysis.testbench eval
"""
import sys, os, json, numpy as np, torch
import torch.nn.functional as F
from ttf.layers import build_kernel_transformer, KernelTernaryLinear
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch

dev = "cuda"
OUT = "checkpoints/testbench_4000"
BS, MICRO, SEQ, G_REF, NVAL = 16, 8, 2048, 3.0, 16
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")


def load_model(ckpt):
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                                 int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"), g_ref=G_REF)
    if any(k.endswith("row_scale") for k in b["model"]):     # --rc_scale runs: learned row/column scales
        from ttf.layers import enable_rc_scales
        enable_rc_scales(m)
    for p in m.float_tail_parameters():
        p.data = p.data.float()
    magA = [k for k in b["model"] if k.endswith("mag_A")]          # --lowrank_mag runs (additive unless MAG_KIND=mul)
    if magA:
        import os
        from ttf.layers import KernelTernaryLinear
        r = b["model"][magA[0]].shape[1]
        for l in m.modules():
            if isinstance(l, KernelTernaryLinear):
                l.enable_lowrank_mag(os.environ.get("MAG_KIND", "add"), r)
                l.mag_cap = float(os.environ.get("MAG_CAP", 0))
    if any(k.endswith("qk_logscale") for k in b["model"]):          # --qk_temp runs
        import ttf.model as BM
        for a in m.modules():
            if isinstance(a, BM.Attention):
                a.qk_logscale = torch.nn.Parameter(torch.zeros(a.n_heads))
    missing, unexpected = m.load_state_dict(b["model"], strict=False)
    assert not unexpected, f"checkpoint keys the model does not have: {unexpected[:5]}"
    m = m.to(dev).train()
    return b, m


class Bench:
    def __init__(self, ckpt):
        self.b, self.m = load_model(ckpt)
        self.Ls = [l for l in self.m.modules() if isinstance(l, KernelTernaryLinear)]
        self.names = [n for n, mod in self.m.named_modules() if isinstance(mod, KernelTernaryLinear)]
        self.tail = [(n, p) for n, p in self.m.named_parameters() if p.requires_grad]
        self.T0 = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in self.Ls]
        self.VB = [get_batch(val, BS // 2, SEQ, dev, torch.Generator().manual_seed(555 + i)) for i in range(NVAL)]

    def grad(self, batch):
        """every parameter's gradient for one full batch (micro-batched): ternary layers dL/dT, tail p.grad"""
        acc = {n: torch.zeros(l.N, l.K, device=dev) for n, l in zip(self.names, self.Ls)}
        acc.update({n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.tail})
        for x, y in batch:
            for l in self.Ls: l.capture = True
            for _, p in self.tail: p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.m(x, y)[1].backward()
            for n, l in zip(self.names, self.Ls):
                acc[n] += l.gw.float(); l.gw = None
            for n, p in self.tail:
                if p.grad is not None: acc[n] += p.grad.float()
        for l in self.Ls: l.capture = False
        for _, p in self.tail: p.grad = None
        return {n: a / len(batch) for n, a in acc.items()}

    def set_trits(self, D):
        for l, t0, d in zip(self.Ls, self.T0, D):
            l.wpacked.copy_(pack_rows((t0 + d).to(torch.int8)))

    def held_out(self, D=None):
        self.set_trits(D if D is not None else [torch.zeros_like(t) for t in self.T0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            L = float(np.mean([self.m(x, y)[1].item() for x, y in self.VB]))
        self.set_trits([torch.zeros_like(t) for t in self.T0])
        return L

    def propose(self, signals, rate, seed=1234):
        self.set_trits([torch.zeros_like(t) for t in self.T0])
        for i, (l, s) in enumerate(zip(self.Ls, signals)):
            fused_flip(l.wpacked, s, rate, G_REF, seed + i, gmean=s.abs().mean().clamp_min(1e-12))
        D = [unpack_rows(l.wpacked, l.K).to(torch.int8) - t0 for l, t0 in zip(self.Ls, self.T0)]
        self.set_trits([torch.zeros_like(t) for t in self.T0])
        return D


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, MICRO, SEQ, dev, g) for _ in range(BS // MICRO)]


def build(ckpt, rate, nb):
    os.makedirs(OUT, exist_ok=True)
    B = Bench(ckpt)
    tern = lambda G: [G[n] for n in B.names]
    s = batches(4242)
    ga = gb = None
    for i in range(nb):
        g = B.grad(next(s))
        if i < nb // 2:
            ga = g if ga is None else {k: ga[k] + g[k] for k in g}
        else:
            gb = g if gb is None else {k: gb[k] + g[k] for k in g}
        if (i + 1) % 32 == 0: print(f"  true gradient: {i + 1}/{nb} batches", flush=True)
    ga = {k: v / (nb // 2) for k, v in ga.items()}; gb = {k: v / (nb - nb // 2) for k, v in gb.items()}
    gbar = {k: (ga[k] + gb[k]) / 2 for k in ga}
    for name, G in (("gbar", gbar), ("gbar_a", ga), ("gbar_b", gb)):
        torch.save({k: v.cpu() for k, v in G.items()}, f"{OUT}/{name}.pt")
    g_step = B.grad(next(batches(101)))
    torch.save({k: v.cpu() for k, v in g_step.items()}, f"{OUT}/g_step.pt")
    # the current method: M proposes, cross-batch look-ahead x2 keeps (the trainer's test)
    Ms = [(U.to(dev).float() @ V.to(dev).float().T) for U, V in B.b["lowrank"]]
    flips = {}
    for name, sig in (("M", Ms), ("g", tern(g_step))):
        D = B.propose(sig, rate)
        flips[f"{name}_proposed"] = [d.cpu() for d in D]
        cur = D
        for j, seed in enumerate((202, 303)):
            B.set_trits(cur)
            g2 = B.grad(next(batches(seed + (0 if name == "M" else 1000))))
            if name == "M":
                torch.save({k: v.cpu() for k, v in g2.items()}, f"{OUT}/g_la{j + 1}.pt")
            cur = [d * ((d.float() * (a + c)) < 0).to(torch.int8) for d, a, c in zip(cur, tern(g_step), tern(g2))]
        B.set_trits([torch.zeros_like(t) for t in B.T0])
        flips[f"{name}_kept"] = [d.cpu() for d in cur]
    torch.save(flips, f"{OUT}/flips.pt")
    json.dump({"ckpt": ckpt, "step": B.b["step"], "rate": rate, "true_batches": nb, "truth_seed": 4242,
               "step_seed": 101, "la_seeds": [202, 303], "g_ref": G_REF,
               "ternary_layers": B.names, "tail_params": [n for n, _ in B.tail]}, open(f"{OUT}/meta.json", "w"), indent=1)
    print("saved", OUT, flush=True)


def score(B, D, gbar):
    n = sum(int((d != 0).sum()) for d in D)
    dot = sum(float((d.float() * g).sum()) for d, g in zip(D, gbar))
    down = sum(int(((d != 0) & (d.float() * g < 0)).sum()) for d, g in zip(D, gbar))
    num = -dot; den = (sum(float((d.float() ** 2).sum()) for d in D) ** 0.5) * \
                      (sum(float((g ** 2).sum()) for g in gbar) ** 0.5)
    return {"flips": n, "precision": down / max(n, 1), "gain": -dot, "gain_per_flip": -dot / max(n, 1),
            "cos": num / max(den, 1e-30), "held_out_dL": B.held_out(D) - B.L0}


def oracle(B, gbar, n):
    """the n flips with the largest true first-order gain |gbar|, in the direction -sign(gbar), that stay in {-1,0,1}"""
    cand = []
    for t0, g in zip(B.T0, gbar):
        d = (-g.sign()).to(torch.int8)
        ok = (t0 + d).abs() <= 1
        cand.append(torch.where(ok, g.abs(), torch.zeros_like(g)).flatten())
    allv = torch.cat(cand)
    thr = allv.kthvalue(max(1, allv.numel() - n)).values
    return [torch.where(((t0 + (-g.sign()).to(torch.int8)).abs() <= 1) & (g.abs() > thr),
                        (-g.sign()).to(torch.int8), torch.zeros_like(t0)) for t0, g in zip(B.T0, gbar)]


def evaluate():
    meta = json.load(open(f"{OUT}/meta.json"))
    B = Bench(meta["ckpt"])
    G = torch.load(f"{OUT}/gbar.pt"); Ga = torch.load(f"{OUT}/gbar_a.pt"); Gb = torch.load(f"{OUT}/gbar_b.pt")
    gbar = [G[n].to(dev) for n in B.names]
    cos = lambda a, b: F.cosine_similarity(torch.cat([a[n].flatten() for n in B.names]),
                                           torch.cat([b[n].flatten() for n in B.names]), 0).item()
    print(f"step {meta['step']}, rate {meta['rate']}, truth = {meta['true_batches']} batches; "
          f"cos(half A, half B) = {cos(Ga, Gb):.3f} -> cos(truth, exact) ~ {((2 * cos(Ga, Gb)) / (1 + cos(Ga, Gb))) ** 0.5:.3f}")
    gs = torch.load(f"{OUT}/g_step.pt")
    print(f"cos(one batch, truth) = {cos(gs, G):.3f}")
    B.L0 = B.held_out()
    flips = torch.load(f"{OUT}/flips.pt")
    rows = {}
    for k, D in flips.items():
        rows[k] = score(B, [d.to(dev) for d in D], gbar)
    nk = rows["M_kept"]["flips"]
    rows["oracle (M_kept count)"] = score(B, oracle(B, gbar, nk), gbar)
    gen = torch.Generator(device=dev).manual_seed(7)
    D = [d.to(dev) for d in flips["M_proposed"]]
    frac = nk / max(rows["M_proposed"]["flips"], 1)
    rows["random subset of M proposals (M_kept count)"] = score(
        B, [d * (torch.rand(d.shape, device=dev, generator=gen) < frac).to(torch.int8) for d in D], gbar)
    print(f"\nheld-out loss at the checkpoint {B.L0:.4f}")
    print(f"{'set':46s} {'flips':>8s} {'precision':>9s} {'gain':>10s} {'gain/flip':>10s} {'cos':>8s} {'held-out dL':>11s}")
    for k, r in rows.items():
        print(f"{k:46s} {r['flips']:8d} {r['precision']:9.3f} {r['gain']:10.3e} {r['gain_per_flip']:10.3e} "
              f"{r['cos']:8.4f} {r['held_out_dL']:+11.4f}")
    json.dump(rows, open(f"{OUT}/scores.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        ck = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/r4090_replay_11M_205M/ckpt_4000.pt"
        rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01202
        nb = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        build(ck, rate, nb)
    else:
        evaluate()
