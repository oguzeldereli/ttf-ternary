"""Causal diagnostics of the ternary moves in examples/vit/vit_cifar.py (--diag N). Every step, on every 4th row (counts on
the GPU: no host sync per step): the moves (trit changes) and how many undo an earlier move within 100 / 1000 steps. Every N steps, on a fixed set of training images (no
augmentation, plain cross-entropy), with the float parameters as they are:
  undo:   the loss with this step's moves reverted minus the loss with them (positive: the moves helped)
  undo_win: the same for all the moves since the previous check (trits back to that check's, floats as they are)
  random: the same number of random legal moves (the noise reference)
  band b (TTF): would-be moves sign(A) at entries with |A| / theta in [lo, hi), as many as the step's moves
  dist:   evidence still needed to move, in units of one step's normalised update: TTF theta - |A|; master
          |(|w| / gamma) - 0.5| gamma / lr (Adam's step is ~lr per entry)
and over two windows of steps the normalised update u of 2048 sampled entries of four layers (for the leak).
"""
import json
import os

import torch
import torch.nn.functional as F

import ttf.update as U
from ttf.kernel import pack_rows, unpack_rows
from ttf.layers import KernelTernaryLinear

BANDS = ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def trits(m):
    if isinstance(m, KernelTernaryLinear):
        return unpack_rows(m.wpacked, m.K).to(torch.int8)
    w = m.weight.detach()
    return (w / w.abs().mean().clamp_min(1e-5)).round().clamp_(-1, 1).to(torch.int8)


class Diag:
    def __init__(self, model, tern, mode, lr_state, cfg, x_eval, y_eval, prep, out, every, total, opt=None):
        self.model, self.tern, self.mode, self.st, self.cfg = model, tern, mode, lr_state, cfg
        self.xe, self.ye, self.prep, self.every, self.opt = x_eval, y_eval, prep, every, opt
        self.log = open(os.path.join(out, "diag.jsonl"), "a")
        self.out = out
        self.prev = [trits(m) for m in tern]
        self.last_sign = [torch.zeros_like(p[::4]) for p in self.prev]
        self.last_step = [torch.full(p[::4].shape, -10 ** 9, dtype=torch.int32, device=p.device) for p in self.prev]
        self.WIN = ("moves", "rev100", "rev1000", "zero_to", "to_zero")    # (on the sampled rows)
        self.win = torch.zeros(len(self.WIN), dtype=torch.int64, device=self.prev[0].device)
        self.lat0 = None
        # u traces: 4 layers (block 3: wq, wo, w_gate, w_down), 2048 entries, steps [W0, W0 + L) and [W1, W1 + L)
        self.L = min(1500, total // 4)
        self.windows = (int(0.15 * total), int(0.55 * total))
        g = torch.Generator(device="cpu").manual_seed(7)
        b = model.blocks[3]
        self.tr_layers = {id(b.attn.wq): "wq", id(b.attn.wo): "wo", id(b.w_gate): "w_gate", id(b.w_down): "w_down"}
        self.tr_idx = {}
        for m in (b.attn.wq, b.attn.wo, b.w_gate, b.w_down):
            I = torch.randint(0, m.N, (2048,), generator=g); J = torch.randint(0, m.K, (2048,), generator=g)
            self.tr_idx[id(m)] = (I.cuda(), J.cuda())
        self.trace, self.step = {}, 0
        self.snap = [p.clone() for p in self.prev]          # the trits at the previous check (undo_win)
        if mode == "ttf":
            orig = U._ts_dense_acc
            me = self

            def wrapped(l, fgb, ia, state, st, key, t, M_, final_, lk, lr_ratio, theta, c, sa, off, dev):
                w = me._window()
                if w is not None and id(l) in me.tr_idx:
                    gy, x, ib = fgb
                    I, J = me.tr_idx[id(l)]
                    u = -lr_ratio * ia[I] * ib[J] * (gy[:, I].float() * x[:, J].float()).sum(0)
                    me.trace.setdefault((w, me.tr_layers[id(l)]), []).append(u.half())
                    A = load_A(state, key, l)
                    if A is not None:
                        me.trace.setdefault((w, me.tr_layers[id(l)] + "_A"), []).append(A[I, J].half())
                return orig(l, fgb, ia, state, st, key, t, M_, final_, lk, lr_ratio, theta, c, sa, off, dev)
            U._ts_dense_acc = wrapped

    def _window(self):
        for k, w0 in enumerate(self.windows):
            if w0 <= self.step < w0 + self.L:
                return k
        return None

    def before_step(self, step):
        self.step = step
        if self.mode == "master" and (step + 1) % self.every == 0:   # (after_step(step + 1) runs the check)
            self.lat0 = [m.weight.detach().clone() for m in self.tern]

    @torch.no_grad()
    def eval_loss(self):
        self.model.eval(); tot = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, self.xe.shape[0], 1024):
                z = self.model(self.prep(self.xe[i:i + 1024], False)).float()
                tot += F.cross_entropy(z, self.ye[i:i + 1024], reduction="sum").item()
        self.model.train()
        return tot / self.xe.shape[0]

    def _set(self, m, q):
        if isinstance(m, KernelTernaryLinear):
            m.wpacked.copy_(pack_rows(q)); m._invalidate()

    @torch.no_grad()
    def after_step(self, step):
        """step: the number of completed steps"""
        cur = [trits(m) for m in self.tern]
        for i, (p, c) in enumerate(zip(self.prev, cur)):
            p, c = p[::4], c[::4]
            s = c - p
            moved = s != 0
            opp = moved & (self.last_sign[i] == -s)
            age = step - self.last_step[i]
            self.win += torch.stack([moved.sum(), (opp & (age <= 100)).sum(), (opp & (age <= 1000)).sum(),
                                     (moved & (p == 0)).sum(), (moved & (c == 0)).sum()])
            self.last_sign[i] = torch.where(moved, s, self.last_sign[i])
            self.last_step[i].masked_fill_(moved, step)
        if step % self.every == 0:
            self.causal(step, cur, [c != p for p, c in zip(self.prev, cur)])
        self.prev = cur
        for w in {k[0] for k in self.trace}:
            if self._window() != w and self.step >= self.windows[w] + self.L - 1:
                self._save_trace(w)

    def _save_trace(self, w):
        keys = [k for k in self.trace if k[0] == w]
        if not keys:
            return
        torch.save({k[1]: torch.stack(self.trace.pop(k)).cpu() for k in keys} | {"start": self.windows[w],
                   "idx": {self.tr_layers[i]: (I.cpu(), J.cpu()) for i, (I, J) in self.tr_idx.items()},
                   "theta": self.cfg["theta"] if self.cfg else None}, os.path.join(self.out, f"utrace_{w}.pt"))

    @torch.no_grad()
    def causal(self, step, cur, mv):
        rec = {"step": step, **dict(zip(self.WIN, self.win.tolist()))}
        self.win.zero_()
        n_mv = sum(int(m.sum()) for m in mv)
        rec["step_moves"] = n_mv
        base = self.eval_loss(); rec["loss"] = base
        if n_mv and self.mode == "ttf":                # undo this step's moves (floats as they are)
            for m, p in zip(self.tern, self.prev):
                self._set(m, p)
            rec["undo"] = self.eval_loss() - base
            for m, c in zip(self.tern, cur):
                self._set(m, c)
        if self.mode == "ttf":                          # all the moves since the previous check
            rec["win_moves"] = sum(int((c != p).sum()) for c, p in zip(cur, self.snap))
            for m, p in zip(self.tern, self.snap):
                self._set(m, p)
            rec["undo_win"] = self.eval_loss() - base
            for m, c in zip(self.tern, cur):
                self._set(m, c)
        self.snap = [c.clone() for c in cur]
        if n_mv and self.mode == "master" and self.lat0 is not None:
            new = [m.weight.detach().clone() for m in self.tern]
            for m, o, mk in zip(self.tern, self.lat0, mv):
                m.weight.data = torch.where(mk, o, m.weight.data)
            rec["undo"] = self.eval_loss() - base
            for m, w_ in zip(self.tern, new):
                m.weight.data = w_
        k = max(n_mv, 2000)
        rec["k"] = k
        if self.mode == "ttf":
            rec["random"] = self._apply_eval(lambda m, q, A: torch.ones_like(q, dtype=torch.bool), k, base, rand_sign=True)
            th = self.cfg["theta"]
            for lo, hi in BANDS:
                rec[f"band_{lo}_{hi}"] = self._apply_eval(
                    lambda m, q, A, lo=lo, hi=hi: (A.abs() >= lo * th) & (A.abs() < hi * th), k, base)
            ds = []
            for m in self.tern:
                A = load_A(self.st, id(m), m)
                if A is not None:          # (entries whose move would be legal)
                    q = unpack_rows(m.wpacked, m.K)
                    ok = (q + A.sign().to(q.dtype)).abs() <= 1
                    ds.append((th - A.abs())[ok][::97])
            if ds:
                d = torch.cat(ds).float()
                rec["dist_q"] = [round(float(x), 2) for x in torch.quantile(d, torch.tensor([0.01, 0.1, 0.5, 0.9], device=d.device))]
        elif self.mode == "master" and self.opt is not None:
            ds = []
            lr = self.opt.param_groups[0]["lr"]
            for m in self.tern:
                w = m.weight.detach(); gm = w.abs().mean()
                ds.append(((w.abs() / gm - 0.5).abs() * gm / max(lr, 1e-12)).flatten()[::97])
            d = torch.cat(ds).float()
            rec["dist_q"] = [round(float(x), 2) for x in torch.quantile(d, torch.tensor([0.01, 0.1, 0.5, 0.9], device=d.device))]
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()

    def _apply_eval(self, sel, k, base, rand_sign=False):
        """apply k moves at the entries sel picks (legal ones, sign(A) or random), spread over the layers by their
        candidate counts; the loss change, then restore"""
        cands, tot = [], 0
        for m in self.tern:
            q = unpack_rows(m.wpacked, m.K).to(torch.int8)
            A = load_A(self.st, id(m), m)
            if A is None:
                A = torch.zeros(q.shape, device=q.device)
            d = (torch.randint(0, 2, q.shape, device=q.device, dtype=torch.int8) * 2 - 1) if rand_sign else A.sign().to(torch.int8)
            ok = sel(m, q, A) & ((q + d).abs() <= 1) & (d != 0)
            cands.append((m, q, d, ok)); tot += int(ok.sum())
        if tot == 0:
            return None
        for m, q, d, ok in cands:
            n = int(ok.sum())
            take = min(n, round(k * n / tot))
            if take == 0:
                continue
            idx = ok.flatten().nonzero().squeeze(1)
            pick = idx[torch.randperm(idx.numel(), device=idx.device)[:take]]
            qn = q.flatten().clone(); qn[pick] += d.flatten()[pick]
            self._set(m, qn.view_as(q))
        r = self.eval_loss() - base
        for m, q, d, ok in cands:
            self._set(m, q)
        return {"dloss": r, "cands": tot, "per_1k": r / k * 1000}


def load_A(state, key, l):
    if key not in state:
        return None
    t = state[key]
    if U.is_dense_state(t):
        return U.load_dense(t, l)
    Uf, Vf = U._load_pair(t)
    return Uf @ Vf.T
