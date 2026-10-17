"""Sequential selection bench at master's state (master @3000 and its exact ternary copy): 10 steps of trit changes,
each step choosing ~as many flips as master changes per step from a signal measured at the *current* state (mean
gradient of NS batches; a fresh batch for the sign gate), then the held-out loss after the 10 steps. Compared with
master's own 10 steps of trit changes (from its saved AdamW state). Selection rules (direction = -sign(signal), only where
the gate batch agrees, blocked at +-1):
  prop   p ~ min(|s| / (3 mean|s|), 1)                 the current rule
  flat   p const                                        sign only
  inv    p ~ 1 - min(|s| / (3 mean|s|), 1)              prefer the small push
  cheap  p ~ 1 / (1 + v / median v)                     prefer weights with small gradient second moment v
  gainK  p ~ max(|s| - K * c * v, 0), c = median|s| / median v   expected gain: first-order push minus a curvature cost
         taken proportional to v (v = factored row x column EMA of g^2, as the Adam step uses), K = 0.5 / 1 / 2
  python -m scripts.analysis.selection_bench MASTER_CKPT CONVERTED_KERNEL_CKPT
  python -m scripts.analysis.selection_bench - KERNEL_CKPT [PER_STEP]   (no master: from one of our own checkpoints,
        PER_STEP flips per step, default 180000; also reports one step's first-order gain, real cost and |T| shares)
"""
import sys, numpy as np, torch
from ttf.master import build_master_transformer
from ttf.train import split_params, get_batch
from ttf.kernel import unpack_rows
from scripts.analysis.testbench import Bench

mpath, kpath = sys.argv[1], sys.argv[2]
import os
NS, K = int(os.environ.get("NS", "4")), int(os.environ.get("K", "10"))
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
OWN = mpath == "-"
if not OWN:
  b = torch.load(mpath, map_location="cpu", weights_only=False)
  mm = build_master_transformer(b["cfg"], grad_checkpoint=True); mm.load_state_dict(b["model"]); mm = mm.cuda().train()
  master, emb, norms = split_params(mm)
  opt = torch.optim.AdamW([{"params": master, "weight_decay": 0.1}, {"params": emb, "weight_decay": 0.1},
                           {"params": norms, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95))
  opt.load_state_dict(b["opt"])
  lin = [m for m in mm.modules() if hasattr(m, "ternary_weight")]
  T0m = [m.ternary_weight()[0].clone() for m in lin]
  for k in range(K):
      x, y = get_batch(tr, 16, 2048, "cuda", torch.Generator().manual_seed(7000 + k))
      opt.zero_grad(set_to_none=True)
      with torch.autocast("cuda", dtype=torch.bfloat16):
          loss = mm(x, y)[1]
      loss.backward(); opt.step()
  Dm = [(m.ternary_weight()[0] - t0).to(torch.int8).cuda() for m, t0 in zip(lin, T0m)]
  del mm, opt; torch.cuda.empty_cache()
B = Bench(kpath)
L0 = B.held_out()
if OWN:
    PER = int(sys.argv[3]) if len(sys.argv) > 3 else 180_000
    print(f"held-out base {L0:.4f}; own checkpoint {kpath}; {PER / 1e3:.0f}k flips per step")
else:
    nm = sum(int((d != 0).sum()) for d in Dm)
    print(f"held-out base {L0:.4f}; master's 10 steps: {nm / 1e3:.0f}k net trit changes, held-out {B.held_out(Dm) - L0:+.5f}")
    PER = nm // K


def run(shape, Kc=1.0, seed=0):
    D = [torch.zeros_like(t) for t in B.T0]
    g = torch.Generator(device="cuda").manual_seed(seed)
    bg = torch.Generator().manual_seed(9000 + seed)
    vs = None
    for step in range(K):
        B.set_trits(D)
        gs = [B.grad([get_batch(tr, 16, 2048, "cuda", bg)]) for _ in range(NS + 1)]
        B.set_trits([torch.zeros_like(t) for t in B.T0])
        sig = [sum(gd[n].float() for gd in gs[:NS]) / NS for n in B.names]
        gate = [gs[NS][n].float() for n in B.names]
        g2 = [sum(gd[n].float() ** 2 for gd in gs[:NS]) / NS for n in B.names]
        if vs is None: vs = [(x.mean(1), x.mean(0)) for x in g2]
        else: vs = [(0.9 * R + 0.1 * x.mean(1), 0.9 * C + 0.1 * x.mean(0)) for (R, C), x in zip(vs, g2)]
        P, MV = [], []
        for i, s in enumerate(sig):
            R, C = vs[i]; v = R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)
            a = s.abs(); m = a.mean().clamp_min(1e-12)
            if shape == "prop": w = (a / (3 * m)).clamp(max=1)
            elif shape == "flat": w = torch.ones_like(a)
            elif shape == "inv": w = 1 - (a / (3 * m)).clamp(max=1)
            elif shape == "cheap": w = 1 / (1 + v / v.median())
            else:
                c = a.median() / v.median().clamp_min(1e-30)
                w = (a - Kc * c * v).clamp(min=0)
            cur = B.T0[i] + D[i]
            mv = -s.sign(); ok = ((cur.float() + mv).abs() <= 1) & (s != 0) & (s.sign() == gate[i].sign())
            P.append(w * ok); MV.append(mv)
        sc = PER / max(sum(float(p.sum()) for p in P), 1e-12)
        for i, p in enumerate(P):
            fire = torch.rand(p.shape, generator=g, device="cuda") < (p * sc).clamp(max=1)
            cur = B.T0[i] + D[i]
            D[i] = torch.where(fire, (cur + MV[i].to(torch.int8)).clamp(-1, 1) - B.T0[i], D[i]).to(torch.int8)
    n = sum(int((d != 0).sum()) for d in D)
    return B.held_out(D) - L0, n


for shape, kc in (("prop", 0), ("flat", 0), ("inv", 0), ("cheap", 0), ("gain", 0.5), ("gain", 1.0), ("gain", 2.0)):
    d, n = run(shape, kc)
    print(f"{shape + (f' K={kc}' if shape == 'gain' else ''):10s} 10 sequential steps of ~{PER / 1e3:.0f}k flips: held-out {d:+.5f} "
          f"({n / 1e3:.0f}k net changes)", flush=True)
