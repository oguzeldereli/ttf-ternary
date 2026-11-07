"""ttf.q38.train: layer-major TTF training of the ternary Qwen3.8-Flash-Next (from the ttfconv conversion).

One step, layer by layer (only one layer's weights and TTF state on the GPU at a time):
  1. forward, no grad: every micro-batch through layer i; layer i's inputs h_i kept on the host (bf16)
  2. head: the KD loss (top 64 exact + 64 tail samples, core data._kd_sampled weights; qwen._kd_topk_chunk 'renorm'
     with those log-weights is -sum_j w_j log q_j) and its gradient at h_L, per micro-batch, chunked over tokens
  3. backward, i = L-1 .. 0: recompute layer i from h_i with grad, backpropagate the micro-batch's gradient; the
     ternary modules collect (beta gy, x) for the whole step; then layer i's updates: ExpertTTF.update_b (the core's
     one-batch factored rule, all experts batched) per ternary module, AdapterTTF on the adapter trits, AdamW on
     layer i's trainable floats (fp32 masters), and the layer goes back to the host
The update of layer i comes after its input gradient (as the core's on_grad hook), so every micro-batch's backward
sees the trits its forward used.

Trainable: row / column scales, centring biases, the adapter's scales (cb, sa), norms, attention / DeltaNet floats
that the conversion's param_groups leaves trainable. Frozen: embed, lm_head, router, shared_expert_gate, hyper-connections, PLE,
DeltaNet a / b / conv / A_log / dt_bias.

  python -m ttf.q38.train --model DIR --conv DIR --kd-cache DIR --data TOKENS.bin --out DIR [--steps 1000 ...]
  python tests/checks/q38_train.py   (the end-to-end test on a random tiny config)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

from ttf.q38 import model as QM
from ttf.q38.moe import TernaryExperts, TernLinear, ExpertTTF, AdapterTTF, pack3
from ttf.q38 import moe as MT
from ttf.qwen import _kd_topk_chunk

DENSE = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
         "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj")
FROZEN = ("mlp.gate.", "shared_expert_gate", "hyper_connection", "ple.", "in_proj_a", "in_proj_b", "conv1d", "A_log",
          "dt_bias", "indexer")


def _get(m, path):
    for p in path.split("."):
        m = getattr(m, p, None)
        if m is None:
            return None
    return m


def _set(m, path, v):
    *a, b = path.split(".")
    for p in a:
        m = getattr(m, p)
    setattr(m, b, v)


def tern_modules(L):
    return [(n, m) for n, m in L.named_modules() if isinstance(m, TernaryExperts)]


def swap_ternary(L, adapter_rank=16, centring=True, quick=False):
    """layer L's routed experts and dense projections replaced by TernaryExperts / TernLinear (empty: the conversion
    loads them). quick: a rounding conversion from the float weights (tests only)"""
    for path in ("mlp.gu", "mlp.dn") + DENSE:
        f = _get(L, path)
        if f is None:
            continue
        if isinstance(f, QM.DenseMat):
            E, N, K = f.E, f.N, f.K
            W = f.weight.data
            t = TernaryExperts(E, K, N, centring=centring, adapter_rank=adapter_rank)
        else:
            assert isinstance(f, nn.Linear) and f.bias is None, path
            E, (N, K) = 1, f.weight.shape
            W = f.weight.data[None]
            t = TernLinear(K, N, centring=centring, adapter_rank=adapter_rank)
        if quick:
            with torch.no_grad():
                for e in range(E):
                    w = W[e].float()
                    s = w.abs().mean().clamp_min(1e-8)
                    T = (w / s).round().clamp(-1, 1).to(torch.int8)
                    t.load_expert(e, T, None, torch.ones(N), torch.ones(K))
                    t.row_scale[e] = s * (T.float() * w).sum(1).div((T.float() ** 2).sum(1).clamp_min(1)) / s / t.beta[e]
        _set(L, path, t)


def layer_floats(L):
    tr = []
    for n, p in L.named_parameters():
        if any(k in n for k in FROZEN):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True); tr.append((n, p))
    return tr


FOREACH_ADAMW = [True]


class AdamW:
    """AdamW over named parameters with fp32 masters, the state kept on the host between uses (one layer at a time)"""
    def __init__(self, b1=0.9, b2=0.95, eps=1e-8, wd=0.0):
        self.b1, self.b2, self.eps, self.wd = b1, b2, eps, wd
        self.st = {}

    @torch.no_grad()
    def step(self, key, named, lr):
        if FOREACH_ADAMW[0]:
            return self._step_foreach(key, named, lr)
        for n, p in named:
            if p.grad is None:
                continue
            k = (key, n)
            if k not in self.st:
                self.st[k] = [p.detach().float().cpu(), torch.zeros(p.shape), torch.zeros(p.shape), 0]
            m0, m1, m2, t = self.st[k]
            dev = p.device
            w, m, v = m0.to(dev), m1.to(dev), m2.to(dev)
            g = p.grad.float()
            t += 1
            m.mul_(self.b1).add_(g, alpha=1 - self.b1); v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mh = m / (1 - self.b1 ** t); vh = v / (1 - self.b2 ** t)
            w.mul_(1 - lr * self.wd).add_(-lr * mh / (vh.sqrt() + self.eps))
            p.data.copy_(w.to(p.dtype))
            self.st[k] = [w, m, v, t]            # (kept on the device: no host round trip per step)
            p.grad = None

    @torch.no_grad()
    def _step_foreach(self, key, named, lr):
        """step() with torch._foreach ops over the layer's parameters, grouped by their step count t: the same ops in
        the same order per tensor (one launch per op and group instead of per parameter)"""
        grp = {}
        for n, p in named:
            if p.grad is None:
                continue
            k = (key, n)
            if k not in self.st:
                self.st[k] = [p.detach().float().to(p.device), torch.zeros(p.shape, device=p.device),
                              torch.zeros(p.shape, device=p.device), 0]
            st = self.st[k]
            st[3] += 1
            grp.setdefault(st[3], []).append((p, st))
        for t, items in grp.items():
            ps = [p for p, _ in items]
            ws, ms, vs = ([st[i] for _, st in items] for i in range(3))
            ws = [w.to(p.device) for w, p in zip(ws, ps)]; ms = [m.to(p.device) for m, p in zip(ms, ps)]
            vs = [v.to(p.device) for v, p in zip(vs, ps)]
            gs = [p.grad.float() for p in ps]
            torch._foreach_mul_(ms, self.b1); torch._foreach_add_(ms, gs, alpha=1 - self.b1)
            torch._foreach_mul_(vs, self.b2); torch._foreach_addcmul_(vs, gs, gs, value=1 - self.b2)
            mh = torch._foreach_div(ms, 1 - self.b1 ** t); vh = torch._foreach_div(vs, 1 - self.b2 ** t)
            den = torch._foreach_sqrt(vh); torch._foreach_add_(den, self.eps)
            up = torch._foreach_mul(mh, -lr); torch._foreach_div_(up, den)
            torch._foreach_mul_(ws, 1 - lr * self.wd); torch._foreach_add_(ws, up)
            torch._foreach_copy_([p.data for p in ps], ws)
            for (p, st), w, m, v in zip(items, ws, ms, vs):
                st[0], st[1], st[2] = w, m, v
                p.grad = None

    def state_dict(self, key):
        return {n: [x.cpu() if torch.is_tensor(x) else x for x in v] for (k, n), v in self.st.items() if k == key}

    def load(self, key, d):
        for n, v in d.items():
            self.st[(key, n)] = v


def theta_at(step, steps, th0, th1, frm, thw=None, wsteps=0):
    """theta(t): a cosine warmup from thw down to th0 over the first wsteps steps (thw None: none), th0, then the late
    rise, a cosine from th0 at frm * steps to th1 at the last step (th1 None: none)"""
    if thw is not None and step < wsteps:
        return th0 + (thw - th0) * 0.5 * (1 + math.cos(math.pi * step / wsteps))
    s0 = frm * steps
    if th1 is None or step <= s0:
        return th0
    x = min(1.0, (step - s0) / max(1.0, steps - s0))
    return th0 + (th1 - th0) * 0.5 * (1 - math.cos(math.pi * x))


def lr_at(step, steps, peak, lo, warm):
    if step < warm:
        return peak * (step + 1) / warm
    x = (step - warm) / max(1, steps - warm)
    return lo + (peak - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, x)))


class _Ph:
    """--timing: CUDA-synced wall time per phase of the step (adds a sync per phase)"""
    def __init__(self, on):
        self.on, self.t = on, {}

    def __call__(self, k):
        ph = self
        class _C:
            def __enter__(s_):
                if ph.on:
                    torch.cuda.synchronize(); s_.t0 = time.time()
            def __exit__(s_, *a):
                if ph.on:
                    torch.cuda.synchronize(); ph.t[k] = ph.t.get(k, 0.0) + time.time() - s_.t0
        return _C()


class Trainer:
    def __init__(self, model, args, dev):
        self.m, self.a, self.dev = model, args, dev
        self.nL = len(model.layers)
        self.upd = ExpertTTF(seeds="layer", chunk=args.expert_chunk)
        MT.ORTH_B[0] = getattr(args, "expert_orth", "qr")
        MT.FIRE_B[0] = getattr(args, "fire", "kernel")
        MT.DIAG[0] = getattr(args, "diag", False)
        if getattr(args, "compile_hc", False):
            QM.compile_hc()
        MT.GROUPED[0] = not getattr(args, "no_grouped", False)
        MT.TF32_TURN[0] = bool(getattr(args, "tf32", False))
        print(f"expert orth {MT.ORTH_B[0]}, grouped_mm {MT.GROUPED[0]}", flush=True)
        self.aupd = AdapterTTF()
        self.opt = AdamW()
        self.c = dict(lr_ratio=1.0, theta=args.theta, tau=args.tau, b2=0.95, spend=1.0, factored=args.sketch)
        self.ts, self.ad = {}, {}        # (layer, module) -> ExpertTTF / AdapterTTF state, on the host
        for i, L in enumerate(model.layers):
            for n, t in tern_modules(L):
                r = max(1, int(args.rank_ratio * min(t.N, t.K)))
                self.ts[(i, n)] = ExpertTTF.new_state(t.E, t.N, t.K, r, device="cpu", pin=dev != "cpu" and args.pin,
                                                     dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.ttf_store])
                if t.ra:
                    self.ad[(i, n)] = AdapterTTF.new_state(t.E, t.N, t.K, t.ra)
                    if dev != "cpu" and args.pin:      # (pinned: the adapter state moves on the copy streams too)
                        self.ad[(i, n)] = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in self.ad[(i, n)].items()}
        for L in model.layers:
            layer_floats(L)
        for p in model.parameters() if False else []:
            pass
        for mod in (model.embed_tokens, model.lm_head, model.hyper_connection_mixer):
            for p in mod.parameters():
                p.requires_grad_(False)
        self.step = 0
        if getattr(args, "resident", False) and dev != "cpu":
            for L in model.layers:
                L.to(dev)

    # ---- one layer on / off the device (--resident: all layers stay on the device; only the unpacked trit caches go)
    def _on(self, i):
        L = self.m.layers[i]
        if not getattr(self.a, "resident", False):
            L.to(self.dev)
        for n, t in tern_modules(L):
            t.invalidate()
        return L

    def _off(self, i):
        L = self.m.layers[i]
        if self.dev != "cpu":
            for n, t in tern_modules(L):
                t._wc = None
            if not getattr(self.a, "resident", False):
                L.to("cpu")

    def _st_on(self, st):
        return {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in st.items()}

    def _st_off(self, st, dst):
        for k, v in st.items():
            if torch.is_tensor(v):
                dst[k].copy_(v, non_blocking=False) if dst[k].shape == v.shape else None

    # ---- --prefetch: layer i's TTF state moves on a copy stream (to the device while the layer above computes, back
    # to the host while the layer below computes); the copy stream is FIFO, so a layer's write-back of step s lands
    # before its prefetch of step s + 1; the step ends with a synchronize (the host state is then current for a save)
    def _pf_start(self, i):
        if i < 0:
            return None
        with torch.cuda.stream(self.cs):
            d = {n: {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) and k != "t" else v) for k, v in s_.items()}
                 for (k_, n), s_ in self.ts.items() if k_ == i}      # (the TTF step counter t stays on the host)
            d.update({("ad", n): {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in s_.items()}
                      for (k_, n), s_ in self.ad.items() if k_ == i})
            ev = torch.cuda.Event(); ev.record(self.cs)
        return d, ev

    def _pf_get(self, pf):
        d, ev = pf
        cur = torch.cuda.current_stream()
        cur.wait_event(ev)
        for s_ in d.values():
            for v in s_.values():
                if torch.is_tensor(v) and v.is_cuda:
                    v.record_stream(cur)
        return d

    def _wb(self, st, dst):
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event(); ev.record(cur)
        with torch.cuda.stream(self.cw):
            self.cw.wait_event(ev)
            for k, v in st.items():
                if torch.is_tensor(v) and v is not dst[k]:
                    dst[k].copy_(v, non_blocking=True); v.record_stream(self.cw)

    # ---- the head: loss and dL/dh per micro-batch
    def head_loss(self, h, kd, grad=True):
        """h [1, T, hc d]; kd (tgt [T], ids [T, k+S], log-weights [T, k+S]); returns (loss sum, dL/dh or None)"""
        a = self.a
        tgt, ids, lw = kd
        h = h.detach().requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            x = self.m.head_in(h)[0]                                       # [T, d]
            xd = x.detach().requires_grad_(grad)
            W = self.m.lm_head.weight
            tot = 0.0
            for c0 in range(0, x.shape[0], a.head_chunk):
                sl = slice(c0, c0 + a.head_chunk)
                l = _kd_topk_chunk(xd[sl], W, tgt[sl], ids[sl], lw[sl], a.kd_weight, mode="renorm")
                if grad:
                    (l / a.norm).backward()
                tot += float(l.detach())
            if grad:
                x.backward(xd.grad)
        return tot, (h.grad if grad else None)

    @torch.no_grad()
    def forward_stash(self, mbs, stash=True):
        if stash:
            self._ohint = {}
        pe = self.m.rope(mbs[0]["ids"].shape[1], self.dev)
        hs = [self.m.embed(b["ids"].to(self.dev)) for b in mbs]
        H = []
        if stash:      # the layer inputs on the host, one buffer for the run (pinned: fast copies both ways)
            shp = (self.nL, len(mbs)) + tuple(hs[0].shape)
            if getattr(self, "_stash", None) is None or self._stash.shape != shp:
                pin = self.a.pin and self.dev != "cpu"
                self._stash = torch.empty(shp, dtype=hs[0].dtype, pin_memory=pin)
        for i in range(self.nL):
            L = self._on(i)
            if stash:
                for m_, h in enumerate(hs):
                    self._stash[i, m_].copy_(h)
                H.append([self._stash[i, m_] for m_ in range(len(hs))])
            rec = (stash and self.dev != "cpu" and getattr(L, "mlp", None) is not None and hasattr(L.mlp, "rec")
                   and not getattr(self.a, "no_count_hint", False))
            if rec:
                L.mlp.rec = []
            hs = self._layer_fwd(L, hs, pe, mbs)
            if rec:      # (the step's expert counts of this layer to the host now, read by its update: no sync there)
                o = torch.stack(L.mlp.rec).diff(dim=1).sum(0)
                o = torch.cat([o.new_zeros(1), o.cumsum(0)])
                h_ = torch.empty(o.shape, dtype=o.dtype, pin_memory=self.a.pin)
                h_.copy_(o, non_blocking=True)
                ev = torch.cuda.Event(); ev.record()
                self._ohint[i] = (h_, ev, o)
                L.mlp.rec = None
            self._off(i)
        return hs, H, pe

    def _groups(self, n):
        w = max(1, getattr(self.a, "wbatch", 1))
        return [list(range(j, min(n, j + w))) for j in range(0, n, w)]

    def _layer_fwd(self, L, hs, pe, mbs):
        """the layer on the micro-batches, --wbatch windows per call (one batch [w, T, .]: fewer, larger launches)"""
        out = []
        for g in self._groups(len(hs)):
            if len(g) == 1:
                out.append(L(hs[g[0]], pe, mbs[g[0]]["ids"].to(self.dev)))
                continue
            y = L(torch.cat([hs[j] for j in g]), pe, torch.cat([mbs[j]["ids"] for j in g]).to(self.dev))
            out += list(y.split(1))
        return out

    def _upd_layer(self, i, L, tms, facs, cur_st, moves, lr, ph, pf, ev):
        """layer i's TTF updates (experts, dense, adapters) and AdamW, after its backward; ev not None: on the side
        stream self.su after ev (the tensors from the default stream recorded on it)"""
        a = self.a
        if ev is None:
            return self._upd_layer_(i, L, tms, facs, cur_st, moves, lr, ph, pf)
        with torch.cuda.stream(self.su):
            self.su.wait_event(ev)
            for fac in facs:
                for v in (fac or ()):
                    if torch.is_tensor(v) and v.is_cuda:
                        v.record_stream(self.su)
            for _, p in layer_floats(L):
                if p.grad is not None:
                    p.grad.record_stream(self.su)
            for _, t in tms:
                lf = getattr(t, "_ad_leaf", None)
                for v in (lf or ()):
                    if v.grad is not None:
                        v.grad.record_stream(self.su)
            hint = getattr(self, "_ohint", {}).get(i)
            if hint is not None:
                hint[2].record_stream(self.su)
            return self._upd_layer_(i, L, tms, facs, cur_st, moves, lr, ph, pf)

    def _upd_layer_(self, i, L, tms, facs, cur_st, moves, lr, ph, pf):
        a = self.a
        frz = self.step < getattr(a, "ttf_start", 0)    # (--ttf-start: trits frozen, no TTF update; the floats train)
        MT.DEFER_LAYER[0] = True        # (broken-Cholesky experts: one host check for the layer, before the write-backs)
        wbs = []
        for j, (n, t) in enumerate(tms):
            fac = facs[j]
            with ph("state_wait"):
                st = self._pf_get(cur_st)[n] if pf else self._st_on(self.ts[(i, n)])
            c_ = self.c
            dm = getattr(a, "dense_theta_mult", 1.0)
            if t.E == 1 and dm != 1.0:      # (the dense projections' theta x dm, rescale form: the bar unchanged)
                c_ = dict(self.c, lr_ratio=self.c["lr_ratio"] / dm)
            em = getattr(a, "expert_theta_mult", 1.0)
            if t.E > 1 and em != 1.0:       # (the routed experts' theta x em, rescale form; adapters keep theta)
                c_ = dict(self.c, lr_ratio=self.c["lr_ratio"] / em)
            ob = MT.ORTH_B[0]
            if t.E == 1 and getattr(a, "dense_orth", None):
                MT.ORTH_B[0] = a.dense_orth            # (the dense projections' bases by another method)
            hint = getattr(self, "_ohint", {}).get(i) if t.E > 1 else None
            if hint is not None and fac is not None:     # (the forward's counts; checked against the take's at the step's end)
                t._o_hint = hint[0], hint[1]
                bad = (fac[0] != hint[2]).any()
                self._obad = bad if getattr(self, "_obad", None) is None else self._obad | bad
            with ph("upd_exp" if t.E > 1 else "upd_dense"):
                nf = 0 if frz else self.upd.update_b(t, fac, st, c_, idx0=1000 * i + j, step_seed=self.step, ret_t=True)
            if frz:
                t._o_hint = None; t._ad_leaf = None     # (what the updates would have consumed)
            MT.ORTH_B[0] = ob
            wbs.append((st, self.ts[(i, n)]))
            moves["exp" if t.E > 1 else "dense"] += nf
            mm = moves.setdefault("by", {})
            mm[n] = mm.get(n, 0) + nf               # (per module name, summed over the layers)
            if MT.DIAG[0]:
                dg = getattr(t, "_diag", None) or {}
                t._diag = {}
                for k_, v_ in dg.items():
                    if not k_.endswith("_acc"):
                        dd = self._dg.setdefault(n, {}); dd[k_] = max(dd.get(k_, float("-inf")), v_)
                        dl = self._dgl.setdefault(i, {}).setdefault(n, {}); dl[k_] = v_
            if t.ra:
                with ph("upd_adapter"):
                    ast = self._pf_get(cur_st)[("ad", n)] if pf else self._st_on(self.ad[(i, n)])
                    moves["adapter"] += 0 if frz else self.aupd.update(t, ast, self.c, ret_t=True)
                    (self._wb if pf else self._st_off)(ast, self.ad[(i, n)])
            del fac, st
        MT.DEFER_LAYER[0] = False
        with ph("upd_exp"):
            for t_, nf_ in MT.finish_repairs():
                moves["exp" if t_.E > 1 else "dense"] += nf_
        with ph("state_wb"):
            for st_, dst_ in wbs:
                (self._wb if pf else self._st_off)(st_, dst_)
        del wbs
        with ph("adamw"):
            self.opt.step(i, layer_floats(L), lr)
        self._off(i)

    def train_step(self, mbs):
        a, dev = self.a, self.dev
        t0 = time.time()
        lr = lr_at(self.step, a.steps, a.lr, a.lr_min, a.warmup)
        th0 = getattr(a, "theta_base", None) or a.theta     # (sched.json theta_base: a new flat level; the bar stays a.theta)
        th = theta_at(self.step, a.steps, th0, getattr(a, "theta_end", None), getattr(a, "theta_rise_from", 0.0),
                      getattr(a, "theta_warm", None), getattr(a, "theta_warm_steps", 0))
        self.c["lr_ratio"] = a.theta / th     # (rescale form: the bar stays a.theta; the fixed leak 1 - 1 / tau)
        a.norm = sum(int((b["tgt"] != -1).sum()) for b in mbs)
        pf = getattr(a, "prefetch", False) and dev != "cpu"
        if pf:
            if getattr(self, "cs", None) is None:
                self.cs = torch.cuda.Stream()      # host -> device (prefetch)
                self.cw = torch.cuda.Stream()      # device -> host (write-back; the two copy engines run in parallel;
                                                   # a layer is fetched and written once per step, the step ends synced)
            nxt = self._pf_start(self.nL - 1)          # (overlaps the forward)
        ph = _Ph(getattr(a, "timing", False))
        with ph("fwd"):
            hs, H, pe = self.forward_stash(mbs)
        tf = time.time()
        gs, loss = [], 0.0
        with ph("head"):
            for h, b in zip(hs, mbs):
                l, g = self.head_loss(h, (b["tgt"].to(dev), b["kd_ids"].to(dev), b["kd_lw"].to(dev)))
                loss += l; gs.append(g)
        del hs
        moves = {"exp": 0, "dense": 0, "adapter": 0}
        self._dg, self._dgl = {}, {}
        # the updates of layer i run on a side stream from a worker thread while the backward of layer i - 1 runs
        # (--overlap-update, opt-in, unvalidated: no gain on an RTX PRO 6000 (37.0-38.2 s vs 36.6 serial; both threads
        # are CPU launch-bound under the GIL) and the moves differed from the serial run at step 3; at most one update in
        # flight; timing / diag run them inline)
        ov = pf and not ph.on and not MT.DIAG[0] and getattr(a, "overlap_update", False)
        if ov and getattr(self, "su", None) is None:
            import concurrent.futures as cf_
            self.su = torch.cuda.Stream()
            self.pool = cf_.ThreadPoolExecutor(max_workers=1)
        fut = None
        for i in reversed(range(self.nL)):
            L = self._on(i)
            tms = tern_modules(L)
            for _, t in tms:
                t.capture = True
            with ph("bwd"):
                for g in self._groups(len(mbs)):
                    h = torch.cat([H[i][j].to(dev) for j in g]).requires_grad_(True)
                    y = L(h, pe, torch.cat([mbs[j]["ids"] for j in g]).to(dev))
                    y.backward(torch.cat([gs[j] for j in g]))
                    for j, gj in zip(g, h.grad.split(1)):
                        gs[j] = gj
                        H[i][j] = None
            facs = []
            for _, t in tms:
                t.capture = False
                with ph("take"):
                    facs.append(t.take_grouped())
            if fut is not None:
                fut.result(); fut = None
            cur_st = None
            if pf:
                cur_st, nxt = nxt, self._pf_start(i - 1)
            if ov:
                ev = torch.cuda.Event(); ev.record()
                fut = self.pool.submit(self._upd_layer, i, L, tms, facs, cur_st, moves, lr, ph, pf, ev)
            else:
                self._upd_layer(i, L, tms, facs, cur_st, moves, lr, ph, pf, None)
            del facs
        if fut is not None:
            fut.result()
        if pf:
            torch.cuda.synchronize()
        if getattr(self, "_obad", None) is not None:
            bad, self._obad = bool(self._obad), None
            if bad:
                raise RuntimeError("the recompute's expert routing differs from the forward's: the host counts of the "
                                   "update were wrong this step (run with --no-count-hint)")
        self._ohint = {}
        # (the move counts were device tensors: one transfer here, no sync per module)
        moves = {k: ({n: int(x) for n, x in v.items()} if isinstance(v, dict) else int(v)) for k, v in moves.items()}
        self.step += 1
        return {"step": self.step, "loss": loss / a.norm, "lr": lr, "theta": th, "moves": moves,
                "rchol_fb": MT.RCHOL_FALLBACK[0],      # (cumulative: matrices whose Cholesky broke down, Householder QR)
                **({"diag": self._dg, "diag_layer": self._dgl} if MT.DIAG[0] else {}),
                "t_fwd": round(tf - t0, 2), "t": round(time.time() - t0, 2),
                **({"ph": {k: round(v, 2) for k, v in ph.t.items()}} if ph.on else {})}

    @torch.no_grad()
    def eval_loss(self, mbs):
        a = self.a
        a.norm = sum(int((b["tgt"] != -1).sum()) for b in mbs)
        hs, _, _ = self.forward_stash(mbs, stash=False)
        tot, totc, kw = 0.0, 0.0, a.kd_weight
        for h, b in zip(hs, mbs):
            kd = (b["tgt"].to(self.dev), b["kd_ids"].to(self.dev), b["kd_lw"].to(self.dev))
            tot += self.head_loss(h, kd, grad=False)[0]
            a.kd_weight = 0.0      # (the next-token cross-entropy on the same windows, logged beside the KD val)
            try:
                totc += self.head_loss(h, kd, grad=False)[0]
            finally:
                a.kd_weight = kw
        self.val_ce = float(totc / a.norm)
        return tot / a.norm

    # ---- checkpoint: one file per layer (ternary modules, floats, TTF / adapter / AdamW state) + meta
    def save(self, d, slim=False):
        """slim: the model only (trits, scales, adapters, floats), no TTF / AdamW state (~25 GB instead of ~230 at r 0.3:
        the stop save, a usable model but not a resumable run)"""
        os.makedirs(d, exist_ok=True)
        for i, L in enumerate(self.m.layers):
            blob = {"layer": {k: v.cpu() for k, v in L.state_dict().items()}}
            if not slim:
                blob.update(ts={n: s for (k, n), s in self.ts.items() if k == i},
                            ad={n: s for (k, n), s in self.ad.items() if k == i}, opt=self.opt.state_dict(i))
            torch.save(blob, os.path.join(d, f"L{i}.pt.tmp")); os.replace(os.path.join(d, f"L{i}.pt.tmp"), os.path.join(d, f"L{i}.pt"))
        json.dump({"step": self.step, "slim": slim}, open(os.path.join(d, "meta.json.tmp"), "w")); os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))

    def load(self, d):
        meta = json.load(open(os.path.join(d, "meta.json")))
        assert not meta.get("slim"), f"{d}: a slim (model only) save, not resumable"
        self.step = meta["step"]
        for i, L in enumerate(self.m.layers):
            blob = torch.load(os.path.join(d, f"L{i}.pt"), weights_only=False)
            L.load_state_dict(blob["layer"])
            for n, s in blob["ts"].items():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        self.ts[(i, n)][k].copy_(v)
            for n, s in blob["ad"].items():      # (into the existing, pinned buffers: a loaded tensor is pageable, and
                dst = self.ad.get((i, n))         # the prefetch from pageable memory cost +2.5 s/step, 6 Nov)
                if dst is None:
                    self.ad[(i, n)] = s; continue
                for k, v in s.items():
                    if torch.is_tensor(v) and torch.is_tensor(dst.get(k)) and dst[k].shape == v.shape:
                        dst[k].copy_(v)
                    else:
                        dst[k] = v
            self.opt.load(i, blob["opt"])


# ---- batches: the KD cache (core teacher_topk format, --kd-cache-mode sample) over the token stream
def kd_batches(args, step, M, T, val=False):
    from ttf.data import _kd_window, _kd_sampled  # noqa: F401  (_kd_window applies _kd_sampled in sample mode)
    data = np.memmap(args.data, dtype=np.uint32 if args.data_u32 else np.uint16, mode="r")
    out = []
    for j in range(M):
        idx = (args.val_offset + j) if val else (step * M + j)
        s0, ids, lw = _kd_window(args, idx)
        w = torch.from_numpy(data[s0:s0 + T + 1].astype(np.int64))
        out.append({"ids": w[None, :-1], "tgt": w[1:], "kd_ids": torch.from_numpy(ids.astype(np.int64)),
                    "kd_lw": torch.from_numpy(lw)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--conv"); ap.add_argument("--kd-cache"); ap.add_argument("--data")
    ap.add_argument("--data-u32", action="store_true"); ap.add_argument("--out", default="q38_run")
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--theta", type=float, default=16.0); ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--theta-end", type=float, default=None,
                    help="late theta rise: theta(t) from --theta to this by the last step, cosine, "
                    "starting at --theta-rise-from x steps; in rescale form (the bar stays --theta, lr_ratio scaled by "
                    "theta / theta(t), the leak 1 - 1 / tau: no burst, no state change)")
    ap.add_argument("--theta-rise-from", type=float, default=0.0)
    ap.add_argument("--theta-warm", type=float, default=None, help="theta warmup start (cosine down to --theta over "
                    "--theta-warm-steps; rescale form as --theta-end)")
    ap.add_argument("--theta-warm-steps", type=int, default=300)
    ap.add_argument("--tf32", action="store_true", help="the core's --tf32 (0.8B / 1.3B: level with fp32): the "
                    "accumulator's basis turn on TF32; the orthonormalisation and the fire stay in full fp32")
    ap.add_argument("--ttf-store", default="fp32", choices=["fp32", "bf16"], help="storage of the accumulator's U and V "
                    "(the core's --ttf-store; bf16 as the 27B recipe: half the host <-> device traffic)")
    ap.add_argument("--expert-theta-mult", type=float, default=1.0, help="the routed experts' (E > 1) theta x this "
                    "(rescale form, as --dense-theta-mult; adapters keep --theta)")
    ap.add_argument("--dense-theta-mult", type=float, default=1.0, help="the dense (E = 1) projections' theta x this "
                    "(rescale form)")
    ap.add_argument("--fire", choices=["kernel", "torch"], default="kernel", help="update_b's firing (moe.FIRE_B)")
    ap.add_argument("--dense-orth", choices=["qr", "rchol"], default=None, help="the dense (E = 1) projections' "
                    "orth (default: --expert-orth)")
    ap.add_argument("--diag", action="store_true", help="per-module update diagnostics in metrics (gradient spike, "
                    "update / accumulator, max update entry, basis orthogonality)")
    ap.add_argument("--sketch", type=int, default=256); ap.add_argument("--rank-ratio", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--ple-pq", default=None, help="the PLE n-gram table from ple_pq.py's compressed store (DIR) "
                    "instead of the FP8 table")
    ap.add_argument("--ttf-start", type=int, default=0, help="trits frozen (no TTF update: experts, dense, adapters) for "
                    "the first N steps; the floats train")
    ap.add_argument("--head-chunk", type=int, default=512); ap.add_argument("--expert-chunk", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=250); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-offset", type=int, default=16128); ap.add_argument("--val-micro", type=int, default=8)
    ap.add_argument("--adapter-rank", type=int, default=-1); ap.add_argument("--no-pin", dest="pin", action="store_false")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="move the TTF state on a copy stream, overlapped with compute")
    ap.add_argument("--expert-orth", default="qr", choices=["qr", "rchol"],
                    help="the expert update's bases: qr (Householder) or rchol (spanqr's robust Cholesky-QR, batched)")
    ap.add_argument("--overlap-update", action="store_true", help="unvalidated, no gain measured: each layer's update on "
                    "a side stream from a worker thread, overlapping the next layer's backward (default: inline)")
    ap.add_argument("--no-overlap-update", action="store_true", help=argparse.SUPPRESS)      # (the old default's opt-out)
    ap.add_argument("--no-count-hint", action="store_true", help="the expert update's host counts by a sync per module "
                    "(default: copied to the host in the forward, checked against the backward's at the step's end)")
    ap.add_argument("--no-grouped", action="store_true", help="padded bmm instead of moe_gg's grouped GEMMs for the experts")
    ap.add_argument("--resident", action="store_true", help="keep all layers on the device (no per-layer moves)")
    ap.add_argument("--compile-hc", action="store_true", help="torch.compile the hyper-connection / SwiGLU glue "
                    "(model.compile_hc: fused, fp32 inside; rounding level)")
    ap.add_argument("--wbatch", type=int, default=1, help="windows per layer call (forward and backward)")
    ap.add_argument("--timing", action="store_true", help="synced wall time per phase in the step's record")
    ap.add_argument("--stop-save", choices=("slim", "full", "none"), default="slim",
                    help="on SIGINT / SIGTERM: finish the step, save (slim: the model only, to OUT/stop; full: the "
                         "resumable checkpoint, to OUT/ckpt), exit 0")
    args = ap.parse_args()
    args.kd_cache_mode = "sample"
    dev = "cuda"
    cfg = QM.text_config(args.model)
    m = QM.build(cfg, device="meta")
    rd = QM.Reader(args.model)
    QM.load_globals(m, rd, "cpu")      # (set_param replaces the meta parameters; no to_empty, which would wipe the
                                        # n-gram hashing buffers; the cache: 64 shards x 256 windows, training 0..15999,
                                        # validation from 16128 = shard 63, never trained)
    if args.adapter_rank < 0:      # (from the conversion: the adapter rank of conv/L0.pt, so it cannot mismatch)
        c0 = torch.load(os.path.join(args.conv, "L0.pt"), weights_only=False, map_location="cpu")
        args.adapter_rank = int(next(iter(c0.values()))["ad_cb"].shape[1]) if any("ad_cb" in v for v in c0.values()) else 0
        del c0
        print(f"adapter rank {args.adapter_rank} (conv/L0.pt)", flush=True)
    for i, L in enumerate(m.layers):
        QM.load_layer(L, i, rd, "cpu", experts=False)
        swap_ternary(L, adapter_rank=args.adapter_rank)
        L.to("cpu")
        conv = torch.load(os.path.join(args.conv, f"L{i}.pt"), weights_only=False)
        for n, sd in conv.items():
            _get(L, n).load_state_dict(sd)
        if L.ple is not None:
            L.ple.ple_embedding.table, L.ple.ple_embedding.scale = QM.ngram_table_memmap(args.model, i)
            if getattr(args, "ple_pq", None):     # (the compressed table in the loop: prune1 + pq4.256, ple_pq)
                from ttf.q38.ple_pq import PQTable
                L.ple.ple_embedding.table, L.ple.ple_embedding.scale = PQTable(args.ple_pq, dev), None
                print(f"PLE table: {args.ple_pq} (codes, decoded on {dev})", flush=True)
    for mod in (m.embed_tokens, m.lm_head, m.hyper_connection_mixer):
        mod.to(dev)
    m.rotary.to(dev)
    tr = Trainer(m, args, dev)
    if args.resume and os.path.exists(os.path.join(args.out, "ckpt", "meta.json")):
        tr.load(os.path.join(args.out, "ckpt"))
    run(tr, args, lambda s: kd_batches(args, s, args.micro, args.seq), lambda: kd_batches(args, 0, args.val_micro, args.seq, val=True))


def run(tr, args, batches, val_batches):
    import signal
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    stop = []

    def _sig(num, frame):          # (a budget watchdog): end after the current step, save, exit 0
        if not stop:
            print(f"signal {num}: stopping after step {tr.step + 1}", flush=True)
        stop.append(num)
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    sched_p, sched_m = os.path.join(args.out, "sched.json"), [None]

    def _sched():      # (the theta schedule hot-swapped from OUT/sched.json between steps, no restart; kept on resume)
        try:
            m = os.path.getmtime(sched_p)
        except OSError:
            return
        if m == sched_m[0]:
            return
        sched_m[0] = m
        try:
            d = json.load(open(sched_p))
        except Exception as e:
            print(f"sched.json unreadable ({e}): unchanged", flush=True); return
        ok = {k: d[k] for k in ("theta_end", "theta_rise_from", "theta_warm", "theta_warm_steps", "theta_base", "expert_theta_mult", "dense_theta_mult") if k in d}
        for k, v in ok.items():
            setattr(args, k, v)
        print(json.dumps({"step": tr.step, "sched": ok}), file=log, flush=True); print(f"step {tr.step} sched {ok}", flush=True)
    while tr.step < args.steps:
        if stop:
            break
        _sched()
        if tr.step % args.eval_every == 0:
            v = tr.eval_loss(val_batches())
            print(json.dumps({"step": tr.step, "val": v, "val_ce": tr.val_ce}), file=log, flush=True); print(f"step {tr.step} val {v:.4f} val_ce {tr.val_ce:.4f}", flush=True)
        r = tr.train_step(batches(tr.step))
        print(json.dumps(r), file=log, flush=True)
        mv = {k: v for k, v in r['moves'].items() if k != "by"}
        print(f"step {r['step']} loss {r['loss']:.4f} moves {mv} {r['t']}s rchol_fb {r.get('rchol_fb', 0)} {r.get('ph', '')}", flush=True)
        assert math.isfinite(r["loss"]), "loss not finite"
        if stop:
            break
        if tr.step % args.save_every == 0 or tr.step == args.steps:
            tr.save(os.path.join(args.out, "ckpt"))
    if stop:
        t0 = time.time()
        mode = getattr(args, "stop_save", "slim")
        if mode != "none":
            tr.save(os.path.join(args.out, "stop" if mode == "slim" else "ckpt"), slim=mode == "slim")
        print(json.dumps({"step": tr.step, "stopped": stop[0], "save": mode, "save_s": round(time.time() - t0, 1)}),
              file=log, flush=True)
        print(f"stopped at step {tr.step} (signal {stop[0]}), {mode} save in {time.time() - t0:.0f} s", flush=True)
        return
    v = tr.eval_loss(val_batches())
    print(json.dumps({"step": tr.step, "val": v, "final": True}), file=log, flush=True); print(f"final val {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
