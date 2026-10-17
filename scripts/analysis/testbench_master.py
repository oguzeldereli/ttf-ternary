"""The step-4000 test bench for master (latent weights + AdamW): which trits does master change in one real
step (and in 10), and how do those moves relate to the true gradient at master's own point?

True gradient gbar: mean over NB training batches (same seeds as the ternary bench) of the gradient w.r.t. the
latent weights (STE: same sign per weight as dL/dT). A step: set every parameter's grad to one batch's
gradient, clip to 1.0 like the trainer, AdamW step with master's optimizer state and LR at that step.
D = change of the absmean ternarization. Scores: flips, precision (share with D * gbar < 0), D . gbar,
cos(D, -gbar), held-out loss change of the whole step (latent weights and float tail both updated).

  python -m scripts.analysis.testbench_master [CKPT] [NB]
"""
import sys, os, json, math, copy, numpy as np, torch
import torch.nn.functional as F
from ttf.master import build_master_transformer, split_params, MasterTernaryLinear
from ttf.train import get_batch

dev = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/curve_master/ckpt_4000.pt"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 256
OUT = "checkpoints/testbench_4000_master"
BS, MICRO, SEQ, NVAL = 16, 8, 2048, 16
LR, MIN_LR, WARM, TOTAL, WD, CLIP = 1.5e-3, 1.5e-4, 305, 9155, 0.1, 1.0
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
os.makedirs(OUT, exist_ok=True)

b = torch.load(CKPT, map_location="cpu", weights_only=False)
m = build_master_transformer(b["cfg"], grad_checkpoint=True).to(dev)
m.load_state_dict(b["model"]); m.train()
master, emb, norms = split_params(m)
opt = torch.optim.AdamW([{"params": master, "weight_decay": WD}, {"params": emb, "weight_decay": WD},
                         {"params": norms, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))
opt.load_state_dict(b["opt"])
step = b["step"] + 1
lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * (step - WARM) / (TOTAL - WARM)))
for g in opt.param_groups: g["lr"] = lr
Ls = [l for l in m.modules() if isinstance(l, MasterTernaryLinear)]
params = [p for p in m.parameters() if p.requires_grad]
VB = [get_batch(val, BS // 2, SEQ, dev, torch.Generator().manual_seed(555 + i)) for i in range(NVAL)]
print(f"master checkpoint step {b['step']}, lr {lr:.3e}, truth {NB} batches", flush=True)


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, MICRO, SEQ, dev, g) for _ in range(BS // MICRO)]


def grad(batch):
    acc = [torch.zeros_like(p) for p in params]
    for x, y in batch:
        for p in params: p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x, y)[1].backward()
        for a, p in zip(acc, params):
            if p.grad is not None: a += p.grad.float()
    for p in params: p.grad = None
    return [a / len(batch) for a in acc]


def held_out():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([m(x, y)[1].item() for x, y in VB]))


trits = lambda: [l.ternary_weight()[0].clone() for l in Ls]
idx = [next(i for i, p in enumerate(params) if p is l.weight) for l in Ls]
s = batches(4242)
gbar = None
if os.path.exists(f"{OUT}/gbar_latent.pt"):            # reuse a saved truth
    G = torch.load(f"{OUT}/gbar_latent.pt"); gbar = [G[f"layer{i}"].to(dev) for i in range(len(Ls))]
    NB = 0
for i in range(NB):
    g = grad(next(s))
    gl = [g[j] for j in idx]
    gbar = gl if gbar is None else [a + c for a, c in zip(gbar, gl)]
    if (i + 1) % 64 == 0: print(f"  true gradient {i + 1}/{NB}", flush=True)
if NB:
    gbar = [a / NB for a in gbar]
    torch.save({f"layer{i}": a.cpu() for i, a in enumerate(gbar)}, f"{OUT}/gbar_latent.pt")
T0 = trits(); L0 = held_out()
state0 = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
ostate0 = copy.deepcopy(opt.state_dict())
for st in ostate0["state"].values():
    for k, v in st.items():
        if torch.is_tensor(v): st[k] = v.cpu()


def cos(a, c):
    num = sum(float((x.float() * y).sum()) for x, y in zip(a, c))
    na = sum(float((x.float() ** 2).sum()) for x in a) ** 0.5; nc = sum(float((y ** 2).sum()) for y in c) ** 0.5
    return num / max(na * nc, 1e-30)


def score(D, name):
    n = sum(int((d != 0).sum()) for d in D)
    dot = sum(float((d.float() * g).sum()) for d, g in zip(D, gbar))
    down = sum(int(((d != 0) & (d.float() * g < 0)).sum()) for d, g in zip(D, gbar))
    c = cos([d.float() for d in D], [-g for g in gbar])
    print(f"  {name:34s} flips {n:8d}  precision {down / max(n, 1):.3f}  D.gbar {dot:+.3e}  cos(D,-gbar) {c:+.4f}  "
          f"held-out dL {held_out() - L0:+.4f}", flush=True)
    return {"flips": n, "precision": down / max(n, 1), "dot": dot, "cos": c}


rows = {}
sb = batches(101)
for k in (1, 10):
    m.load_state_dict(state0); opt.load_state_dict(ostate0)
    for g in opt.param_groups: g["lr"] = lr
    sb = batches(101)
    for _ in range(k):
        g = grad(next(sb))
        for p, a in zip(params, g): p.grad = a
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
    D = [(t - t0).to(torch.int8) for t, t0 in zip(trits(), T0)]
    rows[f"{k} step"] = score(D, f"master, {k} real AdamW step{'s' if k > 1 else ''}")
json.dump(rows, open(f"{OUT}/scores.json", "w"), indent=1)
