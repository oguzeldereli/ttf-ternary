"""At one state (master @STEP and its exact ternary copy), compare what master's own next trit changes cost with what our
rule's flips cost. Master is stepped K times with its saved AdamW state (1 and 10 steps); its trit changes D_master are
applied to the converted kernel model and their held-out loss change per 100k changes is measured (second order
included). Compared at the same count: our rule's flips proposed from (a) the true gradient T (16 batches; the best
signal our momentum could carry) and (b) one batch's gradient, both with the Adam step and the sign gate; and random
flips. For master's changes: uphill share on T, |T| quartile, and how far the latent sat from its rounding boundary.
  python -m scripts.analysis.master_vs_rule MASTER_CKPT CONVERTED_KERNEL_CKPT
"""
import sys, math, numpy as np, torch
from ttf.master import build_master_transformer
from ttf.train import split_params, get_batch
from ttf.kernel import unpack_rows
from scripts.analysis.testbench import Bench

mpath, kpath = sys.argv[1], sys.argv[2]
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
b = torch.load(mpath, map_location="cpu", weights_only=False)
mm = build_master_transformer(b["cfg"], grad_checkpoint=True); mm.load_state_dict(b["model"]); mm = mm.cuda().train()
master, emb, norms = split_params(mm)
opt = torch.optim.AdamW([{"params": master, "weight_decay": 0.1}, {"params": emb, "weight_decay": 0.1},
                         {"params": norms, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95))
opt.load_state_dict(b["opt"])
lin = [m for m in mm.modules() if hasattr(m, "ternary_weight")]
def trits():
    out = []
    for m in lin:
        t, g = m.ternary_weight(); out.append((t.clone(), g.clone(), (m.weight.detach().float() / g).clone()))
    return out
T0 = trits()
B = Bench(kpath)                      # converted kernel copy of the same state (dL/dT gradients, held-out loss)
assert len(B.Ls) == len(lin)
gen = torch.Generator().manual_seed(4242)
S1 = None
for _ in range(16):
    gd = B.grad([get_batch(tr, 16, 2048, "cuda", gen)]); g_ = [gd[n].float() for n in B.names]
    S1 = g_ if S1 is None else [a + c for a, c in zip(S1, g_)]
T = [a / 16 for a in S1]; del S1
gb = B.grad([get_batch(tr, 16, 2048, "cuda", torch.Generator().manual_seed(99))]); gb = [gb[n].float() for n in B.names]
W = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in B.Ls]
L0 = B.held_out()

def cost(D):
    n = sum(int((d != 0).sum()) for d in D)
    return (B.held_out(D) - L0) / max(n, 1) * 1e5, n

def describe(D, name):
    n = up = 0; q = np.zeros(4); bd = []
    for i, d in enumerate(D):
        nz = d != 0
        if not nz.any(): continue
        n += int(nz.sum()); up += int(((d.float() * T[i]) > 0)[nz].sum())
        a = T[i].abs(); qs = torch.quantile(a.flatten()[::97], torch.tensor([0.25, 0.5, 0.75], device=a.device))
        q += np.bincount(torch.bucketize(a[nz], qs).cpu().numpy(), minlength=4)[:4]
        bd.append((T0[i][2].abs() - 0.5).abs()[nz].flatten().cpu())
    bd = torch.cat(bd) if bd else torch.zeros(1)
    print(f"{name}: {n / 1e3:.0f}k changes, uphill {100 * up / max(n, 1):.1f}%, |T| quartile shares "
          f"{' / '.join(f'{100 * x / max(n, 1):.0f}' for x in q)}%, latent distance to the rounding boundary: median "
          f"{float(bd.median()):.3f}, share within 0.05: {100 * float((bd < 0.05).float().mean()):.0f}%")

# master's own changes after 1 and 10 steps
Dm = {}
for k in range(1, 11):
    x, y = get_batch(tr, 16, 2048, "cuda", torch.Generator().manual_seed(7000 + k))
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = mm(x, y)[1]
    loss.backward(); opt.step()
    if k in (1, 10):
        Dm[k] = [(m.ternary_weight()[0] - t0).clamp(-1, 1).to(torch.int8).cuda() for m, (t0, _, _) in zip(lin, T0)]
for i in range(len(T0)): T0[i] = (T0[i][0], T0[i][1], T0[i][2])
gw = sum(float((d != 0).float().mean()) for d in Dm[1]) / len(Dm[1])
print(f"held-out base {L0:.4f}; true gradient from 16 batches")

def rule(signal, count, gated=True, seed=0, shape="prop"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    P, MV = [], []
    for i, s in enumerate(signal):
        R, C = (s * s).mean(1), (s * s).mean(0)
        S = s / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
        gm = S.abs().mean().clamp_min(1e-12)
        if gated: S = S * (S.sign() == gb[i].sign())
        mv = -S.sign(); ok = ((W[i].float() + mv).abs() <= 1) & (S != 0)
        if shape == "prop": w = (S.abs() / (3 * gm)).clamp(max=1)        # the rule: p ~ |S|, saturating at 3 mean|S|
        elif shape == "flat": w = torch.ones_like(S)                      # sign only: every agreeing weight equally likely
        elif shape == "sat1": w = (S.abs() / gm).clamp(max=1)             # saturate already at mean|S|
        else: w = (1 - (S.abs() / (3 * gm)).clamp(max=1))                  # prefer the small |S|
        P.append(w * ok); MV.append(mv)
    sc = count / sum(float(p.sum()) for p in P)
    return [torch.where(torch.rand(p.shape, generator=g, device="cuda") < (p * sc).clamp(max=1),
                        (W[i] + MV[i].to(torch.int8)).clamp(-1, 1) - W[i], torch.zeros_like(W[i])).to(torch.int8)
            for i, p in enumerate(P)]

def random_flips(count, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tot = sum(w.numel() for w in W); pr = count / tot
    out = []
    for w in W:
        fire = torch.rand(w.shape, generator=g, device="cuda") < pr
        mv = torch.where(torch.rand(w.shape, generator=g, device="cuda") < 0.5, 1, -1).to(torch.int8)
        out.append(torch.where(fire, (w + mv).clamp(-1, 1) - w, torch.zeros_like(w)).to(torch.int8))
    return out

for k in (1, 10):
    D = Dm[k]; n = sum(int((d != 0).sum()) for d in D)
    describe(D, f"master, {k} step(s)")
    c_m, _ = cost(D)
    c_t, _ = cost(rule(T, n)); c_b, _ = cost(rule(gb, n)); c_r, _ = cost(random_flips(n))
    c_tn, _ = cost(rule(T, n, gated=False))
    print(f"  cost per 100k changes at this count ({n / 1e3:.0f}k): master {c_m:+.5f} | our rule on the true gradient "
          f"{c_t:+.5f} (no gate {c_tn:+.5f}) | our rule on one batch {c_b:+.5f} | random {c_r:+.5f}", flush=True)
    describe(rule(T, n), f"  our rule (true gradient), {k} step(s) count")
    for sh in ("flat", "sat1", "inv"):
        for nm, sig in (("true gradient", T), ("one batch", gb)):
            c, _ = cost(rule(sig, n, shape=sh)); print(f"  shape {sh:5s} on {nm:13s}: {c:+.5f}", flush=True)
