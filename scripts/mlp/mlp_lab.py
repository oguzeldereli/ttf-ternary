"""Ternary MLP testbed: does a small MLP reproduce the transformer's gap between
stateless flips and BitNet-style master weights? If so, rules can be iterated here in
minutes instead of hours.

Task: next-token prediction on the wiki32k stream from the previous `ctx` tokens
(a Bengio-style neural LM): token embeddings -> concatenate -> ternary MLP -> tied head.
The ternary layers are the same classes the transformer uses:
  --mode master   latent fp32 weights, ternary forward via STE, AdamW (BitNet b1.58)
  --mode flip     stateless packed-ternary flips (KernelTernaryLinear), cosine rate
                  (+ --lookahead N [--xbatch] for the look-ahead filter)
Embeddings and norm gains are fp32 + AdamW in every mode.

  python -m scripts.mlp.mlp_lab --mode master --name mlp_master
  python -m scripts.mlp.mlp_lab --mode flip --name mlp_flip
  python -m scripts.mlp.mlp_lab --mode flip --lookahead 2 --xbatch --name mlp_xb2
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttf.model import RMSNorm
from ttf.master import MasterTernaryLinear
from ttf.flip import KernelTernaryLinear, set_flip_rate
from ttf.train import lookahead_step, gpu_temp


class MLPLM(nn.Module):
    def __init__(self, make_linear, vocab=32000, ctx=8, emb=256, hidden=1024, depth=3):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, emb)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        dims = [ctx * emb] + [hidden] * depth
        self.inp = make_linear(dims[0], hidden)
        self.norms = nn.ModuleList(RMSNorm(hidden) for _ in range(depth))
        self.hid = nn.ModuleList(make_linear(hidden, hidden) for _ in range(depth - 1))
        self.out = make_linear(hidden, emb)
        self.norm_out = RMSNorm(emb)

    def forward(self, idx, targets=None):
        h = self.tok_emb(idx).flatten(1)                          # [B, ctx*emb]
        h = self.inp(h)
        for i, lin in enumerate(self.hid):
            h = h + lin(F.silu(self.norms[i](h)))                 # residual MLP blocks
        h = self.out(F.silu(self.norms[-1](h)))
        logits = F.linear(self.norm_out(h), self.tok_emb.weight)  # tied head
        if targets is None:
            return logits, None
        return None, F.cross_entropy(logits.float(), targets)


def windows(data, n, ctx, gen, device):
    p = torch.randint(0, len(data) - ctx - 1, (n,), generator=gen).numpy()
    w = torch.from_numpy(data[p[:, None] + np.arange(ctx + 1)].astype(np.int64)).to(device)
    return w[:, :ctx], w[:, ctx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["master", "flip"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min_lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rate", type=float, default=0.02, help="peak flip rate (cosine to 0)")
    ap.add_argument("--rate_warmup", type=int, default=10)
    ap.add_argument("--g_ref", type=float, default=3.0)
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--xbatch", action="store_true")
    ap.add_argument("--lowrank", type=int, default=0,
                    help="flip signal = rank-r momentum of the gradient (0 = current gradient)")
    ap.add_argument("--lr_beta", type=float, default=0.97, help="decay of the low-rank momentum")
    ap.add_argument("--sketch", type=int, default=0,
                    help="flip signal = hashed (count-sketch) momentum with sketch*(N+K) buckets "
                         "per layer, i.e. the memory of rank-`sketch` momentum")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--max_temp", type=int, default=86)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs("checkpoints/mlp", exist_ok=True)
    out = open(f"checkpoints/mlp/{a.name}.jsonl", "w")

    torch.manual_seed(a.seed)
    if a.mode == "master":
        mk = lambda i, o: MasterTernaryLinear(i, o)
    else:
        mk = lambda i, o: KernelTernaryLinear(i, o, rate=a.rate, g_ref=a.g_ref, beta=True,
                                              int8=True, dw_mode="dense")
    model = MLPLM(mk, ctx=a.ctx, emb=a.emb, hidden=a.hidden, depth=a.depth).to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and "norm" not in n]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and "norm" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    Ls = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    n_tern = sum(l.N * l.K for l in Ls) or sum(m.weight.numel() for m in model.modules()
                                               if isinstance(m, MasterTernaryLinear))
    print(f"{a.name}: {a.mode}, ternary weights {n_tern / 1e6:.1f}M, "
          f"{a.batch} examples/step x {a.steps} steps", flush=True)

    train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
    val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
    gtr = torch.Generator().manual_seed(1234 + a.seed)
    gla = torch.Generator().manual_seed(4242 + a.seed)
    VAL = [windows(val, a.batch, a.ctx, torch.Generator().manual_seed(99 + i), dev) for i in range(8)]
    extra = (lambda: windows(train, a.batch, a.ctx, gla, dev)) if a.xbatch else None

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        r = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * r))

    def rate_at(s):
        if s < a.rate_warmup:
            return a.rate * s / a.rate_warmup
        r = (s - a.rate_warmup) / max(1, a.steps - a.rate_warmup)
        return a.rate * 0.5 * (1 + math.cos(math.pi * r))

    @torch.no_grad()
    def evaluate():
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = float(np.mean([model(x, y)[1].item() for x, y in VAL]))
        model.train()
        return v

    # rank-r momentum per layer, M ~ U V^T: one subspace-iteration step per update
    # (M <- beta*M + g, re-compressed to rank r without an SVD), memory r*(N+K) per layer
    LR_U = {id(l): torch.linalg.qr(torch.randn(l.N, a.lowrank, device=dev))[0] for l in Ls} if a.lowrank else {}
    LR_V = {id(l): torch.zeros(l.K, a.lowrank, device=dev) for l in Ls} if a.lowrank else {}

    def lowrank_flip(s, x=None, y=None):
        """Flip from the rank-r momentum; with --lookahead, filter those proposals by
        look-ahead (the keep test uses this step's gradient g_A, as in lookahead_step)."""
        from ttf.kernel import fused_flip, lookahead_filter
        saved = [l.wpacked.clone() for l in Ls] if a.lookahead else None
        gA = [l.gw.float().clone() for l in Ls] if a.lookahead else None
        propose(s)
        if not a.lookahead:
            return
        tail_g = [None if p.grad is None else p.grad.clone() for p in params]
        for _ in range(a.lookahead):
            for l in Ls: l.capture = True
            for p in params: p.grad = None
            xc, yc = extra() if extra is not None else (x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(xc, yc)[1].backward()
            for i, l in enumerate(Ls):
                lookahead_filter(saved[i], l.wpacked, gA[i], l.gw)
                l.gw = None
        for l in Ls: l.capture = False
        for p, g in zip(params, tail_g): p.grad = g

    @torch.no_grad()
    def propose(s):
        from ttf.kernel import fused_flip
        for i, l in enumerate(Ls):
            g = l.gw.float(); l.gw = None; l.capture = False
            U, V = LR_U[id(l)], LR_V[id(l)]
            Vn = torch.linalg.qr(a.lr_beta * V @ (U.T @ U) + g.T @ U)[0] if V.abs().sum() > 0 \
                else torch.linalg.qr(g.T @ U)[0]                         # K x r basis
            Un = a.lr_beta * U @ (V.T @ Vn) + g @ Vn                     # N x r coefficients
            LR_U[id(l)], LR_V[id(l)] = Un, Vn
            M = Un @ Vn.T                                                # transient N x K
            fused_flip(l.wpacked, M, l.rate, l.g_ref, 50_000 + s * 131 + i,
                       gmean=M.abs().mean().clamp_min(1e-12).item())

    # count-sketch momentum: weight (i,j) -> bucket h(i,j) with a fixed random sign; the hash is
    # recomputed from the flat index each step, so the only state is the bucket vector
    SK = {id(l): torch.zeros(a.sketch * (l.N + l.K), device=dev) for l in Ls} if a.sketch else {}

    def _hash(n, B, salt):
        i = torch.arange(n, device=dev, dtype=torch.int64)
        h = (i * 2654435761 + salt * 40503) % 4294967291
        return h % B, ((h // B) % 2) * 2 - 1

    @torch.no_grad()
    def sketch_flip(s):
        from ttf.kernel import fused_flip
        for i, l in enumerate(Ls):
            g = l.gw.float(); l.gw = None; l.capture = False
            S = SK[id(l)]
            b, sg = _hash(l.N * l.K, S.numel(), 1 + i)
            S.mul_(a.lr_beta).index_add_(0, b, g.flatten() * sg)
            M = (S[b] * sg).view(l.N, l.K)
            fused_flip(l.wpacked, M, l.rate, l.g_ref, 70_000 + s * 131 + i,
                       gmean=M.abs().mean().clamp_min(1e-12).item())

    t0 = time.time()
    for s in range(a.steps):
        if s % 4 == 0:
            t = gpu_temp()
            while t is not None and t >= a.max_temp:
                time.sleep(5); t = gpu_temp()
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        if Ls:
            set_flip_rate(model, rate_at(s))
            for l in Ls: l.capture = bool(a.lookahead or a.lowrank or a.sketch)
        x, y = windows(train, a.batch, a.ctx, gtr, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)[1]
        loss.backward()
        info = {}
        if a.sketch:
            sketch_flip(s)
        elif a.lowrank:
            lowrank_flip(s, x, y)
        elif a.lookahead:
            info = lookahead_step(model, x, y, params, dev, s, a.lookahead, 0, extra)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        rec = {"step": s, "examples": (s + 1) * a.batch, "loss": loss.item(), **info}
        if s % a.eval_every == 0 or s == a.steps - 1:
            rec["val_loss"] = evaluate()
            print(f"step {s:5d} | loss {rec['loss']:.3f} | val {rec['val_loss']:.3f} | "
                  f"{time.time() - t0:6.0f}s", flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
