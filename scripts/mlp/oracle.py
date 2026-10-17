"""Master-weights oracle: how well does the gradient predict the discrete moves a working
optimizer actually makes?

Train the ternary MLP LM with master weights (BitNet STE + AdamW). At a checkpoint, record
the ternary weights T0, keep training master for N more steps and record T1: the trits
that changed are the "oracle moves". At the checkpoint (same network) score candidate
signals on predicting them, per layer, then averaged:
  grad1   gradient on one batch                       (what the flip rule uses)
  gradK   gradient averaged over K batches            (noise removed: is it noisy or wrong?)
  mom     master's AdamW first moment (gradient history)
  thresh  closeness of the latent weight to its rounding boundary, in the move direction
Scores: direction accuracy of -sign(signal) on moved weights; AUC of |signal| for "moves
within N steps" among weights that can move in the signal's direction; precision@k with
k = number of oracle moves.

  python -m scripts.mlp.oracle
"""
import copy, math, numpy as np, torch, torch.nn.functional as F
from ttf.master import MasterTernaryLinear
from scripts.mlp.mlp_lab import MLPLM, windows

dev = "cuda"
CKPTS, HORIZONS, K, BATCH, LR, WARM, TOTAL = (300, 2000), (10, 50, 200), 32, 4096, 1.5e-3, 100, 6000


class FixedTern(torch.nn.Module):
    """The master's ternary network at a fixed point: W = t * gamma (a float parameter only
    so autograd returns dL/dW, identical to master's STE gradient at that point)."""
    def __init__(self, t, gamma):
        super().__init__()
        self.W = torch.nn.Parameter((t.float() * gamma).clone()); self.gamma = gamma
    def forward(self, x):
        from ttf.bitlinear import _act_quant_ste
        return F.linear(_act_quant_ste(x, 8), self.W)


def ternarize(w):
    g = w.abs().mean().clamp_min(1e-5)
    return (w / g).round().clamp(-1, 1).to(torch.int8), w / g


def auc(score, label):
    s = score[label.bool()]; n = score[~label.bool()]
    if s.numel() == 0 or n.numel() == 0: return float("nan")
    allv = torch.cat([s, n]); r = torch.empty_like(allv); r[allv.argsort()] = torch.arange(1, allv.numel() + 1, device=allv.device, dtype=allv.dtype)
    return ((r[:s.numel()].sum() - s.numel() * (s.numel() + 1) / 2) / (s.numel() * n.numel())).item()


def lookahead_sets(master, T0, g1, rate=0.02, g_ref=3.0, passes=2):
    """Propose flips from the one-batch gradient with the flip rule, then filter them with
    cross-batch look-ahead on a copy of the ternary network fixed at T0. Returns per layer
    (proposal directions, kept directions) as int tensors in {-1,0,1}."""
    probe = copy.deepcopy(master)
    ms = [m for m in master.modules() if isinstance(m, MasterTernaryLinear)]
    fx = []
    for (name, mod), t0 in zip([(n, m) for n, m in probe.named_modules() if isinstance(m, MasterTernaryLinear)], T0):
        gamma = [mm for mm in ms][len(fx)].weight.detach().float().abs().mean().clamp_min(1e-5)
        f = FixedTern(t0, gamma).to(dev)
        parent = probe.get_submodule(name.rsplit(".", 1)[0]) if "." in name else probe
        setattr(parent, name.rsplit(".", 1)[-1], f); fx.append(f)
    train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
    gb = torch.Generator().manual_seed(4242)
    gen = torch.Generator(device=dev).manual_seed(7)
    P, cur = [], []
    for f, t0, g in zip(fx, T0, g1):
        gn = g / g.abs().mean().clamp_min(1e-12)
        prob = (gn.abs() / g_ref).clamp(max=1) * rate
        fire = torch.rand(g.shape, device=dev, generator=gen) < prob
        d = (-gn.sign()).int() * fire.int()
        d = torch.where((t0.int() + d).abs() <= 1, d, torch.zeros_like(d))
        P.append(d); cur.append(d.clone())
    for _ in range(passes):
        for f, t0, d in zip(fx, T0, cur):
            f.W.data = (t0.int() + d).float() * f.gamma
        x, y = windows(train, BATCH, 8, gb, dev)
        probe.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probe(x, y)[1].backward()
        for i, (f, g) in enumerate(zip(fx, g1)):
            keep = (cur[i].float() * (g + f.W.grad.float())) < 0
            cur[i] = cur[i] * keep.int()
    return list(zip(P, cur))


def main():
    torch.manual_seed(0)
    model = MLPLM(lambda i, o: MasterTernaryLinear(i, o)).to(dev).train()
    decay = [p for n, p in model.named_parameters() if "norm" not in n]
    nodecay = [p for n, p in model.named_parameters() if "norm" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))
    train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
    gtr = torch.Generator().manual_seed(1234)
    Ls = [m for m in model.modules() if isinstance(m, MasterTernaryLinear)]

    def lr_at(s):
        if s < WARM: return LR * (s + 1) / WARM
        return 1.5e-4 + 0.5 * (LR - 1.5e-4) * (1 + math.cos(math.pi * (s - WARM) / (TOTAL - WARM)))

    def step(m, o, s, gen):
        for g in o.param_groups: g["lr"] = lr_at(s)
        x, y = windows(train, BATCH, 8, gen, dev)
        o.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, y)[1]
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
        return loss.item()

    def grads(m, nb, gen):
        acc = [torch.zeros_like(l.weight) for l in Ls_of(m)]
        for _ in range(nb):
            x, y = windows(train, BATCH, 8, gen, dev)
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(x, y)[1].backward()
            for a, l in zip(acc, Ls_of(m)): a += l.weight.grad.float()
        return [a / nb for a in acc]

    Ls_of = lambda m: [mm for mm in m.modules() if isinstance(mm, MasterTernaryLinear)]
    s = 0
    for ck in CKPTS:
        while s < ck:
            step(model, opt, s, gtr); s += 1
        T0, U0 = zip(*(ternarize(l.weight.detach().float()) for l in Ls))
        mom = [opt.state[l.weight]["exp_avg"].float().clone() for l in Ls]
        g1 = grads(model, 1, torch.Generator().manual_seed(777))
        gK = grads(model, K, torch.Generator().manual_seed(778))
        print(f"\n=== checkpoint step {ck} ({ck * BATCH / 1e6:.1f}M examples) ===")
        LA = lookahead_sets(model, T0, g1)
        # continue a copy of master for the horizons (same data stream as the real run)
        m2, o2 = copy.deepcopy(model), None
        o2 = torch.optim.AdamW([{"params": [p for n, p in m2.named_parameters() if "norm" not in n], "weight_decay": 0.1},
                                {"params": [p for n, p in m2.named_parameters() if "norm" in n], "weight_decay": 0.0}],
                               lr=LR, betas=(0.9, 0.95))
        o2.load_state_dict(opt.state_dict())
        g2 = copy.deepcopy(gtr); s2 = ck
        for H in HORIZONS:
            while s2 < ck + H:
                step(m2, o2, s2, g2); s2 += 1
            T1 = [ternarize(l.weight.detach().float())[0] for l in Ls_of(m2)]
            rows = {k: [] for k in ("grad1", "gradK", "mom", "thresh")}
            moved_frac = []
            for li in range(len(Ls)):
                t0, t1, u0 = T0[li].int(), T1[li].int(), U0[li]
                mv = (t1 - t0).sign()                                   # oracle move direction
                moved = mv != 0
                moved_frac.append(moved.float().mean().item())
                sig = {"grad1": g1[li], "gradK": gK[li], "mom": mom[li]}
                for name, g in sig.items():
                    d = -g.sign().int()                                  # proposed direction
                    can = (t0 + d).abs() <= 1                            # not pushing into the wall
                    dir_acc = (d[moved] == mv[moved]).float().mean().item()
                    label = moved & (d == mv)
                    a = auc(g.abs()[can].flatten(), label[can].flatten())
                    k = int(label.sum())
                    top = g.abs().masked_fill(~can, -1).flatten().topk(max(k, 1)).indices
                    prec = label.flatten()[top].float().mean().item()
                    rows[name].append((dir_acc, a, prec))
                # threshold closeness in master's own history direction (mom), as a predictor
                d = -mom[li].sign().int()
                can = (t0 + d).abs() <= 1
                frac = u0 - t0.float()                                   # position inside the cell
                close = (frac * d.float())                               # >0 = already leaning toward d
                label = moved & (d == mv)
                k = int(label.sum())
                top = close.masked_fill(~can, -9).flatten().topk(max(k, 1)).indices
                rows["thresh"].append(((d[moved] == mv[moved]).float().mean().item(),
                                       auc(close[can].flatten(), label[can].flatten()),
                                       label.flatten()[top].float().mean().item()))
            print(f"  horizon {H:3d} steps: master changed {np.mean(moved_frac) * 100:.2f}% of trits")
            # look-ahead (rule r=0.02, cross-batch x2) as a signal: precision of its kept flips
            tot = {"proposal": [0, 0], "look-ahead kept": [0, 0], "top-|g| same count": [0, 0]}
            for li in range(len(Ls)):
                mvl = (T1[li].int() - T0[li].int()).sign()
                P, Kp = LA[li]
                for name, D in (("proposal", P), ("look-ahead kept", Kp)):
                    sel = D != 0
                    tot[name][0] += int((sel & (D == mvl)).sum()); tot[name][1] += int(sel.sum())
                d = -g1[li].sign().int(); can = (T0[li].int() + d).abs() <= 1
                k = int((Kp != 0).sum())
                if k:
                    top = g1[li].abs().masked_fill(~can, -1).flatten().topk(k).indices
                    tot["top-|g| same count"][0] += int((d.flatten()[top] == mvl.flatten()[top]).sum())
                    tot["top-|g| same count"][1] += k
            base = np.mean(moved_frac)
            print("    " + " | ".join(f"{n}: {a / max(b, 1):.3f} of {b}" for n, (a, b) in tot.items())
                  + f"  (chance of a random move matching: ~{base / 2:.3f})")
            print(f"    {'signal':8s} {'dir. accuracy':>14s} {'AUC':>7s} {'precision@k':>12s}")
            for name, r in rows.items():
                r = np.array(r)
                print(f"    {name:8s} {np.nanmean(r[:, 0]):14.3f} {np.nanmean(r[:, 1]):7.3f} {np.nanmean(r[:, 2]):12.3f}")


if __name__ == "__main__":
    main()
