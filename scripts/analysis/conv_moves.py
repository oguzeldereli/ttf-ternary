"""Can a single trit move help a converted model? At a checkpoint (trits from the conversion, floats trained), per trit
in trit units (the kernel's own dL/dT, row / column scales included): the mean gradient g over n micro-batches and the
Gauss-Newton diagonal h (capture_h: sum over tokens of (beta gy)^2 x^2, times the tokens per micro-batch to put it on
the mean loss's scale). A move d = -sign(g) of one trit changes the loss by about g d + h / 2 = -|g| + h / 2: it helps
only if the Newton displacement |g| / h exceeds 1/2 (the weight wants to go more than half a step). Rounding to the
nearest trit puts every weight within half a step of its full-precision value, so at the conversion point this should be
rare; whatever TTF moves on evidence alone overshoots.

Reported: the distribution of |g| / h over legal moves, per layer type; then held-out loss changes for move sets of the
same size chosen by
  evidence   largest |g| / sqrt(h)   (what an accumulator of normalised gradients fires first)
  newton     largest predicted gain |g| - h / 2, only where it is positive
  random     random legal moves in the direction -sign(g)
    python scripts/analysis/conv_moves.py CKPT [--n 48] [--every 2] [--fracs 1e-5,1e-4,1e-3]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.eval.final_eval import load_model, window_losses       # noqa: E402
from ttf.layers import KernelTernaryLinear                          # noqa: E402
from ttf.kernel import unpack_rows, pack_rows                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--data", default="data/wiki32k_train.bin")
ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--n", type=int, default=48); ap.add_argument("--micro", type=int, default=2)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--every", type=int, default=2)
ap.add_argument("--fracs", default="1e-5,1e-4,1e-3"); ap.add_argument("--nval", type=int, default=32)
ap.add_argument("--locate", type=float, default=0.0, help="random legal moves at this fraction: (i) one layer at a time "
                "(held-out change per layer), (ii) all layers with the input channels of the largest second moment "
                "excluded (top 0.1 / 1 / 5%%), (iii) only those channels")
ap.add_argument("--blocks", default="", help="keep only these decoder blocks (all 7 linears each), e.g. 0,1,2,3,10,21")
ap.add_argument("--bydepth", default="", help="FRAC: per measured layer, its TTF-chosen moves applied alone (loss change "
                "per move), with the layer's gradient consistency (per-entry SNR mean^2 / var) and evidence size")
ap.add_argument("--bytype", default="", help="FRAC: also apply each rule's moves one matrix type at a time at FRAC")
a = ap.parse_args()
dev = "cuda"
t0 = time.time()
model, ck = load_model(a.ckpt, dev)
layers = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
names = [n for n, m in model.named_modules() if isinstance(m, KernelTernaryLinear)]
KEEP = [i % a.every == 0 for i in range(len(layers))]
if a.blocks:
    _bl = {int(x) for x in a.blocks.split(",")}
    KEEP = [int(n.split(".")[1]) in _bl for n in names]
for i, m in enumerate(layers):
    m.capture = True; m.capture_h = KEEP[i]
for p in model.parameters():
    p.requires_grad_(p.dim() == 2 and p.shape[0] > 30000)
model.grad_checkpoint = True; model.ckpt_skip = 0; model.train()
G = [torch.zeros(m.N, m.K, device=dev) if KEEP[i] else None for i, m in enumerate(layers)]
Hd = [torch.zeros(m.N, m.K, device=dev) if KEEP[i] else None for i, m in enumerate(layers)]
SQ = [torch.zeros(m.N, m.K, device=dev) if KEEP[i] else None for i, m in enumerate(layers)]
data = np.memmap(a.data, dtype=np.uint16, mode="r")
rng = np.random.default_rng(777)
tok = a.micro * a.seq
for b in range(a.n):
    st = rng.integers(0, len(data) - a.seq - 1, a.micro)
    x = torch.tensor(np.stack([data[s:s + a.seq + 1].astype(np.int64) for s in st]), device=dev)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    for i, m in enumerate(layers):
        g = m.gw; m.gw = None
        if KEEP[i]:
            G[i].add_(g); Hd[i].add_(m.hw); SQ[i].addcmul_(g, g); m.hw = None
        del g
    if b % 12 == 11:
        print(f"  {b + 1}/{a.n} micro-batches, loss {loss.item():.3f}, {time.time() - t0:.0f}s", flush=True)
for i in range(len(layers)):
    if KEEP[i]:
        G[i] /= a.n; Hd[i] *= tok / a.n          # mean-loss gradient; GN diagonal on the mean loss's scale
        SQ[i] = (SQ[i] / a.n - G[i] * G[i]).clamp_min_(1e-30)   # per-entry variance across micro-batches
for m in layers:
    m.capture = False; m.capture_h = False
model.eval()
ts = ck.get("ts_state") or []
VF = {}                      # TTF's own normaliser: the run's factored second moment R_i C_j / mean R
for i in range(len(layers)):
    if KEEP[i] and i < len(ts) and ts[i] and "v" in ts[i]:
        R, C = (x.to(dev).float() for x in ts[i]["v"])
        VF[i] = (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt_().view(-1)
print(f"factored v from the checkpoint for {len(VF)} of {sum(KEEP)} measured layers")

TYPES = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]
typ = lambda n: next(t for t in TYPES if n.endswith(t))
T0 = {i: unpack_rows(layers[i].wpacked, layers[i].K).clone() for i in range(len(layers)) if KEEP[i]}
print(f"\nNewton displacement |g| / h over legal moves (d = -sign g), {a.n} x {a.micro} x {a.seq} tokens")
print("  type     share >1/2   >1/4   median   p99    p99.99")
stats = {}
for t in TYPES:
    vals = []
    for i in T0:
        if typ(names[i]) != t:
            continue
        d = -torch.sign(G[i]); legal = (T0[i].float() + d).abs() <= 1
        r = (G[i].abs() / Hd[i].clamp_min(1e-30))[legal]
        vals.append(r[torch.randint(0, r.numel(), (min(r.numel(), 2_000_000),), device=dev)])
    v = torch.cat(vals)
    q = torch.quantile(v[:5_000_000].float(), torch.tensor([0.5, 0.99, 0.9999], device=dev)).tolist()
    stats[t] = ((v > 0.5).float().mean().item(), (v > 0.25).float().mean().item(), q)
    print(f"  {t:7s} {stats[t][0]:10.5f} {stats[t][1]:7.4f} {q[0]:8.4f} {q[1]:7.4f} {q[2]:8.3f}")

val = np.memmap(a.val, dtype=np.uint16, mode="r")
base = window_losses(model, val, a.seq, a.nval, dev)[0].mean()
print(f"\nheld-out loss {base:.4f} ({a.nval} windows); moves on the {sum(KEEP)} measured layers")
tot = sum(T0[i].numel() for i in T0)


def apply(sel):
    """sel: {i: (flat idx, signs)} -> held-out loss change, then restore"""
    for i, (idx, d) in sel.items():
        T = T0[i].clone().view(-1); T[idx] = (T[idx] + d).clamp(-1, 1); layers[i].wpacked.copy_(pack_rows(T.view_as(T0[i])))
    L = window_losses(model, val, a.seq, a.nval, dev)[0].mean()
    for i in sel:
        layers[i].wpacked.copy_(pack_rows(T0[i]))
    return L - base


RULES = ("ttf", "evidence", "newton", "random") if VF else ("evidence", "newton", "random")


def choose(rule, f, only=None):
    sel, pred = {}, 0.0
    for i in T0:
        if only is not None and typ(names[i]) != only:
            continue
        if True:
            g, h = G[i].view(-1), Hd[i].view(-1)
            d = -torch.sign(g); legal = (T0[i].view(-1).float() + d).abs() <= 1
            k = max(1, int(f * g.numel()))
            if rule == "ttf":            # what TTF's accumulator fires first: |g| / sqrt(factored v)
                sc = g.abs() / VF[i].clamp_min(1e-30)
            elif rule == "evidence":
                sc = g.abs() / h.clamp_min(1e-30).sqrt()
            elif rule == "newton":
                sc = g.abs() - 0.5 * h
            else:
                sc = torch.rand_like(g)
            sc = torch.where(legal, sc, torch.full_like(sc, -float("inf")))
            idx = sc.topk(k).indices
            if rule == "newton":
                idx = idx[sc[idx] > 0]
            sel[i] = (idx, d[idx].to(torch.int8))
            pred += float((-g[idx].abs() + 0.5 * h[idx]).sum())
    return sel, pred


for f in [float(x) for x in a.fracs.split(",")]:
    row = []
    for rule in RULES:
        sel, pred = choose(rule, f)
        n_m = sum(len(v[0]) for v in sel.values())
        row.append(f"{rule} {apply(sel):+.4f} (pred {pred:+.4f}, {n_m} moves)")
    print(f"  fraction {f:g}: " + " | ".join(row), flush=True)
if a.bytype:
    f = float(a.bytype)
    print(f"\nby matrix type at fraction {f:g} (only that type's moves applied)")
    for t in TYPES:
        row = []
        for rule in RULES:
            sel, pred = choose(rule, f, only=t)
            row.append(f"{rule} {apply(sel):+.4f}")
        print(f"  {t:7s} " + " | ".join(row), flush=True)
if a.locate:
    f = a.locate
    gen = torch.Generator(device=dev).manual_seed(5)

    def rnd_sel(i, colmask=None):
        g = G[i].view(-1); d = -torch.sign(g)
        legal = (T0[i].view(-1).float() + d).abs() <= 1
        if colmask is not None:
            legal &= colmask[None, :].expand(layers[i].N, -1).reshape(-1)
        k = max(1, int(f * g.numel()))
        sc = torch.where(legal, torch.rand(g.shape, device=dev, generator=gen), torch.full_like(g, -1.0))
        idx = sc.topk(k).indices
        return idx, d[idx].to(torch.int8)
    print(f"\nlocate: random legal moves, fraction {f:g} of each layer")
    per = []
    for i in T0:
        dl = apply({i: rnd_sel(i)})
        per.append((dl, names[i]))
    per.sort(reverse=True)
    print("  worst layers: " + ", ".join(f"{n} {d:+.4f}" for d, n in per[:10]))
    print(f"  sum over layers {sum(d for d, _ in per):+.4f}; layers above +0.01: {sum(d > 0.01 for d, _ in per)}")
    # input second moment per column (the GN diagonal summed over rows ~ x^2 mean x sum_i gy^2): use Hd's column sums
    for q in (0.001, 0.01, 0.05):
        sel_ex, sel_in = {}, {}
        for i in T0:
            cs = Hd[i].sum(0); kq = max(1, int(q * cs.numel()))
            top = torch.zeros_like(cs, dtype=torch.bool); top[cs.topk(kq).indices] = True
            sel_ex[i] = rnd_sel(i, ~top); sel_in[i] = rnd_sel(i, top)
        print(f"  top {100*q:g}% columns excluded: {apply(sel_ex):+.4f} | only those columns: {apply(sel_in):+.4f}", flush=True)
print(f"done in {time.time() - t0:.0f}s")

if a.bydepth:
    f = float(a.bydepth)
    print(f"\nby layer at fraction {f:g} (TTF's choice, this layer's moves alone): loss change, per 1k moves, consistency")
    print("  layer           moves   dloss    per1k   SNR(mean^2/var, mean)  top-1% SNR  |g|/sqrt(v) mean")
    sel, _ = choose("ttf", f, only=None)
    for i in T0:
        one = {i: sel[i]}
        dl = apply(one); nm = len(sel[i][0])
        snr = (G[i] * G[i] / SQ[i]).view(-1)
        top = snr.topk(max(1, snr.numel() // 100)).values.mean()
        ev = (G[i].abs() / VF[i].view_as(G[i]).clamp_min(1e-30)).mean() if i in VF else float("nan")
        print(f"  {names[i]:15s} {nm:6d} {dl:+.4f} {1000 * dl / max(nm, 1):+.4f}  {float(snr.mean()):.3e}  {float(top):.3e}  {float(ev):.3e}", flush=True)
