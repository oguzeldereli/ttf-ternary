"""A small ViT on CIFAR-100: TTF (no master weights) vs master weights (STE) vs full precision, the linear layers from
ttf-core and the TTF update wired as ttf.train wires it (on_grad -> ts_layer inside the backward, ts_step after).
Everything else (patch embedding, position embedding, class token, norms, head, row / column scales, rank-16 float
adapters) is float and trains with AdamW in every mode.

  python examples/vit/vit_cifar.py --mode ttf --lr 1e-3 --out runs/vit_ttf_lr1e-3
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttf.layers import KernelTernaryLinear, DX_MODE, FUSED_LAYER, FACTORED
from ttf.master import MasterTernaryLinear
from ttf.model import RMSNorm
from ttf.schedule import theta_at, THETA_START_X, THETA_END_X
from ttf.update import ts_layer, ts_step


class Attn(nn.Module):
    def __init__(self, d, h, lin):
        super().__init__()
        self.h = h
        self.wq, self.wk, self.wv, self.wo = lin(d, d), lin(d, d), lin(d, d), lin(d, d)

    def forward(self, x):
        B, T, D = x.shape
        sp = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(self.wq(x)), sp(self.wk(x)), sp(self.wv(x)))   # bidirectional
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h, hid, lin):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attn(d, h, lin)
        self.w_gate, self.w_up, self.w_down = lin(d, hid), lin(d, hid), lin(hid, d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class ViT(nn.Module):
    def __init__(self, lin, d=384, depth=8, heads=6, hid=1024, patch=4, n_cls=100, resid="fp32"):
        super().__init__()
        self.patch, self.resid = patch, resid
        n = (32 // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads, hid, lin) for _ in range(depth)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, img):
        B, p = img.shape[0], self.patch
        x = img.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * p * p)
        x = torch.cat([self.cls.expand(B, -1, -1), self.embed(x)], 1) + self.pos
        if self.resid == "bf16" and torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))   # (as ttf.model: the layers' inputs, and so the TTF
            # update's factors, in bf16; fp32 inputs send the update through fp32 GEMMs on the CUDA cores)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x[:, 0]))


def cifar100(root, dev):
    """root/cifar100.npz: train_x / test_x uint8 [n, 3, 32, 32], train_y / test_y fine labels (made from
    the Hugging Face copy uoft-cs/cifar100, md5 82a24ff67be71334af2304e34406a8cc; no torchvision dependency)"""
    import numpy as np
    z = np.load(os.path.join(root, "cifar100.npz"))
    return [(torch.from_numpy(z[p + "_x"]).to(dev), torch.from_numpy(z[p + "_y"]).to(dev)) for p in ("train", "test")]


MEAN = torch.tensor([0.507, 0.487, 0.441]).view(1, 3, 1, 1)
STD = torch.tensor([0.267, 0.256, 0.276]).view(1, 3, 1, 1)


def prep(x, train):
    """uint8 [B, 3, 32, 32] on the GPU -> normalised float; train: random crop (pad 4) and horizontal flip"""
    x = x.float().div_(255)
    if train:
        B = x.shape[0]
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i, j = torch.randint(0, 9, (2,))
        x = x[:, :, i:i + 32, j:j + 32]                     # one offset per batch (cheap); flips per image
        fl = torch.rand(B, device=x.device) < 0.5
        x = torch.where(fl[:, None, None, None], x.flip(3), x)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ttf", "master", "fp32"], required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta", type=float, default=16.0, help="the base threshold (ttf.train --ttf-theta)")
    ap.add_argument("--theta-start", type=float, default=None, help="theta at step 0, down to --theta over the warmup "
                    "(default 4 x --theta)")
    ap.add_argument("--theta-end", type=float, default=None, help="theta at the last step (default 5 x --theta; start = "
                    "end = theta: constant)")
    ap.add_argument("--rank", type=int, default=512)        # >= min(N, K) here: every layer full rank (dense state)
    ap.add_argument("--adapter-rank", type=int, default=16)
    ap.add_argument("--resid", choices=["fp32", "bf16"], default="fp32",
                    help="residual stream dtype (bf16: as ttf.model; fp32: the 28 Oct wave 1 runs)")
    ap.add_argument("--tau", type=float, default=1000.0, help="TTF leak time constant: A <- (1 - 1 / TAU) A")
    ap.add_argument("--lowrank", action="store_true", help="TTF: the low-rank accumulator at full rank (update.DENSE_ACC off)")
    ap.add_argument("--diag", type=int, default=0, help="examples/vit/ttf_diag.py checks every N steps (0: off)")
    ap.add_argument("--stop-epoch", type=int, default=0, help="stop after this epoch (the schedule stays --epochs long)")
    ap.add_argument("--freeze-trits", action="store_true", help="ttf: no TTF update (the trits stay at their init)")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (smoke test)")
    ap.add_argument("--data", default="data/cifar")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.theta_start = THETA_START_X * a.theta if a.theta_start is None else a.theta_start
    a.theta_end = THETA_END_X * a.theta if a.theta_end is None else a.theta_end
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    DX_MODE[0] = "int8"; FUSED_LAYER[0] = True                 # as ttf.train's defaults
    lin = {"ttf": lambda i, o: KernelTernaryLinear(i, o, 8, beta=True),
           "master": lambda i, o: MasterTernaryLinear(i, o, 8, dtype=torch.float32),
           "fp32": lambda i, o: nn.Linear(i, o, bias=False)}[a.mode]
    model = ViT(lin, resid=a.resid).to(dev)
    tern = [m for m in model.modules() if isinstance(m, (KernelTernaryLinear, MasterTernaryLinear))]
    extra = []
    for m in tern:                                             # the float extras of both ternary modes
        m.enable_rc_scale()
        if a.adapter_rank:
            extra += m.enable_lowrank_mag("add", a.adapter_rank)
    model = model.to(dev)
    (xtr, ytr), (xte, yte) = cifar100(a.data, dev)
    spe = xtr.shape[0] // a.bs
    total = spe * a.epochs
    warm = 5 * spe
    lr_at = lambda s: a.lr * s / warm if s < warm else \
        a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    # AdamW groups: matrices (embed, head, master latents, adapters) decayed; norms, scales, pos, cls not
    nod = {id(p) for n, p in model.named_parameters()
           if n.endswith(("row_scale", "col_scale", "pos", "cls", ".bias")) or "norm" in n or n.startswith(("n1", "n2"))
           or ".n1." in n or ".n2." in n}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in nod], "weight_decay": a.wd},
                             {"params": [p for p in params if id(p) in nod], "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    lr_state, cfg = {}, None
    if a.mode == "ttf" and not a.freeze_trits:
        cfg = dict(r=a.rank, lr_ratio=1.0, theta=a.theta, tau=a.tau, b2=0.95, spend=1.0, store_a="", offload=False,
                   theta_sched=None, progress=0.0, factored=256, band=0.0, band_lo=0.5, hspend=0.0)
        if a.lowrank:
            import ttf.update as U_
            U_.DENSE_ACC[0] = False
        for i, m in enumerate(tern):
            m._ts_idx = i
            m.on_grad = lambda layer, g, _c=cfg: ts_layer(layer, g, lr_state, _c)
    n_t = sum(m.N * m.K for m in tern) if tern else sum(p.numel() for n, p in model.named_parameters() if ".w" in n)
    n_f = sum(p.numel() for p in model.parameters()) - (sum(m.weight.numel() for m in tern) if a.mode == "master" else 0)
    print(f"{a.mode}: linear weights {n_t / 1e6:.2f}M, float params {n_f / 1e6:.2f}M, {total} steps", flush=True)
    log = open(os.path.join(a.out, "metrics.jsonl"), "a")

    def evaluate():
        model.eval(); ok = 0; tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, xte.shape[0], 1000):
                z = model(prep(xte[i:i + 1000], False)).float()
                ok += (z.argmax(-1) == yte[i:i + 1000]).sum().item(); tot += F.cross_entropy(z, yte[i:i + 1000], reduction="sum").item()
        model.train()
        return ok / xte.shape[0], tot / xte.shape[0]

    diag = None
    if a.diag:
        from ttf_diag import Diag
        diag = Diag(model, tern, a.mode, lr_state, cfg, xtr[:10000], ytr[:10000], prep, a.out, a.diag, total, opt)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(xtr.shape[0], device=dev)
        for b in range(spe):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            if cfg is not None:
                # the core theta schedule in rescale form, as ttf.train (before 7 Nov: the step (lr / peak)^0.75)
                cfg["lr_ratio"] = a.theta / theta_at(step / total, a.theta, a.theta_start, a.theta_end, warm / total)
                cfg["progress"] = step / total
                cfg["step"], cfg["start"], cfg["micro"], cfg["vdiag"] = step, 0, 1, False   # as ttf.train sets them
                FACTORED[0] = True     # --dw factored (ttf.train's default): (gy, x) to the update, dense accumulator at full rank
                for m in tern:
                    m.capture = True
            if diag is not None:
                diag.before_step(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(prep(xtr[idx], True)).float(), ytr[idx], label_smoothing=0.1)
            loss.backward()
            info = ts_step(model, lr_state, cfg) if cfg is not None else {}
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if diag is not None:
                diag.after_step(step)
            if step % 50 == 0:
                rec = {"step": step, "epoch": ep, "loss": loss.item(), "lr": lr, "s": round(time.time() - t0, 1), **info}
                log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 500 == 0:
                print(f"step {step} ep {ep} loss {loss.item():.3f} {info.get('ts_frac', 0):.2e} {time.time() - t0:.0f}s", flush=True)
            if a.steps and step >= a.steps:
                break
        if a.steps and step >= a.steps:
            break
        if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
            acc, ce = evaluate()
            log.write(json.dumps({"step": step, "epoch": ep + 1, "test_acc": acc, "test_ce": ce}) + "\n"); log.flush()
            print(f"epoch {ep + 1}: test acc {acc:.4f} ce {ce:.4f}", flush=True)
        if a.stop_epoch and ep + 1 >= a.stop_epoch:
            break
    acc, ce = evaluate()
    json.dump({"mode": a.mode, "lr": a.lr, "seed": a.seed, "epochs": a.epochs, "steps": step, "test_acc": acc,
               "test_ce": ce, "secs": round(time.time() - t0)}, open(os.path.join(a.out, "final.json"), "w"))
    print(f"FINAL test acc {acc:.4f} ce {ce:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
