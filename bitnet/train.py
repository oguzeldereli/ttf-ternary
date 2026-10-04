"""Train the master-free ternary BitNet on a flat uint16 token stream.

Data format (nanoGPT-style): a single binary file of uint16 token ids for train
and another for val. Point TrainConfig.data_path / val_path at them.

The BitLinear matrices update themselves inside their backward hooks (no master
weights, no optimizer state tensors). Only the small float tail (embeddings +
norms) uses a real optimizer here.
"""
from __future__ import annotations
import os
import shutil
import json
import math
import time
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gpu_temp():
    """Current GPU temperature in Celsius via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None

from .config import ModelConfig, TrainConfig, PRESETS, DEFAULT_PRESET
from .model import BitTransformer
from .bitlinear import STATE
from .master import build_master_transformer, split_params, MasterTernaryLinear
from .kernel import fused_flip, unpack_rows, pack_rows, lookahead_filter
from .flip import KernelTernaryLinear
from .probe import probe, parse_schedule
from .flip import (build_flip_transformer, build_stateless_transformer,
                   build_kernel_transformer, apply_flips, enable_flip_tracking,
                   collect_flip_stats, flip_accumulated, set_flip_accum,
                   set_flip_rate, set_abs_scale, set_err_feedback, set_lockout,
                   set_norm_mode, set_inv_prob, set_hump,
                   reset_lockout, lockout_stats)


def get_batch(data, bs, seq_len, device, gen=None):
    # gen: dedicated RNG so the data order is identical across runs regardless of
    # what else consumes the global RNG (different modes draw differently).
    ix = torch.randint(len(data) - seq_len - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


TWO_STAGE = [0.0]
TS_SHADOWS = [(), ()]             # --ts_shadows: ranks of the shadow states (0 = full rank); --ts_shadow_layers
TS_DUMP = ["", (), ()]            # --ts_dump_steps / --ts_dump_layers: directory, steps, layer indices   # --two_stage F: BitNet-b1.58-style two-stage schedule (0: plain cosine)


def two_stage_mid(tc: TrainConfig):
    return tc.warmup_steps + (tc.max_steps - tc.warmup_steps) // 2


def lr_at(step, tc: TrainConfig):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if TWO_STAGE[0] and step <= tc.max_steps:
        # stage 1 (first half after warmup): cosine from the peak to half the peak; stage 2: restart at F x peak and
        # cosine down to min_lr (a lower-peak cool-down after a drop at the midpoint; weight decay is set to 0 in
        # stage 2 in the training loop). An approximation of the published BitNet b1.58 recipe, for the master baseline.
        mid = two_stage_mid(tc)
        if step < mid:
            r = (step - tc.warmup_steps) / max(1, mid - tc.warmup_steps)
            return 0.5 * tc.lr + 0.25 * tc.lr * (1 + math.cos(math.pi * r))
        r = (step - mid) / max(1, tc.max_steps - mid)
        p2 = TWO_STAGE[0] * tc.lr
        return tc.min_lr + 0.5 * (p2 - tc.min_lr) * (1 + math.cos(math.pi * r))
    if step > tc.max_steps:
        return tc.min_lr
    r = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def evaluate(model, data, tc, device):
    model.eval()
    STATE.updates_enabled = False
    losses = []
    # fixed windows: same val batches every eval and across runs (comparable).
    gen = torch.Generator().manual_seed(tc.seed + 12345)
    for _ in range(tc.eval_iters):
        x, y = get_batch(data, tc.batch_size, tc.seq_len, device, gen)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def rate_search_step(model, rate_now, rs_state, args, tc, data, gen, device, step):
    """Try rate x factor for each factor on the captured gradient, keep the best.

    Every candidate uses the SAME random numbers, so the flip sets are nested (a
    higher rate flips a superset) and the comparison is paired. Candidates are
    scored on a held-out train batch with the flips applied and nothing else
    changed, then the winner's flips are committed for real."""
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    saved = [l.wpacked.clone() for l in layers]
    grads = []
    for l in layers:
        grads.append(l.gw.contiguous()); l.gw = None; l.capture = False
    xt, yt = get_batch(data, args.rs_batch, tc.seq_len, device, gen)
    factors = [float(f) for f in args.rs_factors.split(",")]
    seed0 = 7_000_000 + step * 131
    was_training = model.training
    model.eval()

    def apply(rate):
        for i, (l, g) in enumerate(zip(layers, grads)):
            fused_flip(l.wpacked, g, rate, l.g_ref, seed0 + i,
                       gmean=g.abs().mean().clamp_min(1e-8).item())

    def restore():
        for l, w in zip(layers, saved): l.wpacked.copy_(w)

    def score():
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            return model(xt, yt)[1].item()

    base = score()
    losses = {}
    for f in factors:
        r = rate_now * f
        apply(r); losses[f] = score(); restore()
    best = min(losses, key=losses.get)
    rs_state["mult"] = float(min(max(rs_state["mult"] * best, args.rs_min), args.rs_max))
    apply(rate_now * best)                                  # commit the winner
    for l, w in zip(layers, saved):
        if l.track: l._record_flips(w, l.wpacked)
    if was_training: model.train()
    return {"rs_base": base, "rs_best": best,
            **{f"rs_{f:g}": losses[f] - base for f in factors}}


def lookahead_step(model, x, y, tail, device, step, iters=1, flip_seed=0, extra_batch=None,
                   signals=None, diag=False, gacc=None, sig_gmeans=None, want_delta=False):
    """Propose flips with the normal rule, then keep only the ones that are still
    downhill given all the others.

    A lone flip is never right-sized (median |g|/H ~ 60, curv_single.py); flips
    overshoot because they interact. So each flip is judged against the step the
    rest are taking: with g at W and g' at W+Delta (same batch), flip i is kept iff
    Delta_i * (g_i + g'_i) < 0 -- the midpoint slope along its own coordinate still
    descends, which is its exact contribution for a quadratic. Stateless: g is held
    only within the step. The tail (embedding, norms) keeps its first-pass gradient.
    iters > 1 re-checks the surviving flips at the filtered point.
    signals: per-layer tensors to propose from instead of g (e.g. the low-rank momentum);
    the keep test still uses the true gradients g and g'."""
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    saved = [l.wpacked.clone() for l in layers]            # packed, 1.6 bit/weight
    grads = []
    for l in layers:
        grads.append(l.gw.contiguous()); l.gw = None
    if gacc is not None:                      # --move_window: sum of the gradients the run sees
        for i, g in enumerate(grads):
            gacc.add(i, g)
    tail_g = [None if p.grad is None else p.grad.clone() for p in tail]
    seed0 = 9_000_000 + step * 131 + flip_seed * 1_000_003
    for i, (l, g) in enumerate(zip(layers, grads)):
        sig = g if signals is None else signals[i]
        gm = sig_gmeans[i] if sig_gmeans is not None else sig.abs().mean().clamp_min(1e-8)
        fused_flip(l.wpacked, sig, l.rate, l.g_ref, seed0 + i, gmean=gm)
    n_prop = n_keep = None
    gg = []
    for it in range(iters):
        for p in tail: p.grad = None
        # cross-batch look-ahead: judge each flip on a *different* batch, so a flip
        # survives only if it is downhill on both (removes interaction and much of the
        # batch noise at the same cost)
        xc, yc = extra_batch() if extra_batch is not None else (x, y)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            _, loss = model(xc, yc)
        loss.backward()                               # capture is still on: no flips
        cnt = torch.zeros(2, dtype=torch.int32, device=grads[0].device)
        if diag and it == 0:
            # cos(g, g'): g' is on another batch at the proposed point; with independent
            # batch noise this is |signal|^2 / |g|^2, the per-step signal fraction
            gg = [F.cosine_similarity(grads[i].flatten().float(), l.gw.flatten().float(), 0).item()
                  for i, l in enumerate(layers)]
        for i, l in enumerate(layers):
            # keep/revert straight on the packed bytes (bitnet.kernel.lookahead_filter)
            lookahead_filter(saved[i], l.wpacked, grads[i], l.gw, cnt)
            l.gw = None
        if n_prop is None:
            n_prop = cnt[0].clone()
        n_keep = cnt[1]
    for p, g in zip(tail, tail_g): p.grad = g
    for l, w in zip(layers, saved):
        l.capture = False
        if l.track: l._record_flips(w, l.wpacked)
    out = {"la_proposed": n_prop, "la_kept": n_keep}
    if want_delta:                            # kept trit changes per layer (int8), for --lr_spend
        from .kernel import unpack_rows
        out["_D"] = [(unpack_rows(l.wpacked, l.K).to(torch.int8) - unpack_rows(w, l.K).to(torch.int8))
                     for l, w in zip(layers, saved)]
    if gg:
        out.update(_diag_agg("d_gg", gg))
    return out


@torch.no_grad()
def _diag_agg(name, vals, per_block=7):
    """{name: mean over layers, name_types: mean per layer type (position in the block)}"""
    t = torch.tensor(vals)
    return {name: t.mean().item(),
            name + "_types": [t[j::per_block].mean().item() for j in range(per_block)]}


def _qfmt(name):
    """--ts_qfmt: how a stored low-rank matrix is rounded after each update (simulated storage formats)"""
    if name in ("", "none"): return None
    if name == "int8": return _q8                                                     # per column, stochastic
    if name == "int8det":                                                             # per column, nearest
        return lambda X: (X / (s := X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 127)).round().clamp_(-127, 127) * s
    if name == "int8row":                                                             # per row, stochastic
        return lambda X: (X / (s := X.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127) + torch.rand_like(X)).floor().clamp_(-127, 127) * s
    if name == "bf16": return lambda X: X.to(torch.bfloat16).float()                  # 8 exp, 7 mantissa bits
    if name == "fp16": return lambda X: X.to(torch.float16).float()                   # 5 exp, 10 mantissa bits
    if name == "fp16s":                                       # fp16 with a per-column scale (column absmax -> 2^15)
        return lambda X: (X / (s := X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 32768)).to(torch.float16).float() * s
    if name == "int4":                                                                # per column, stochastic, 15 levels
        return lambda X: (X / (s := X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 7) + torch.rand_like(X)).floor().clamp_(-7, 7) * s
    if name.startswith("fpm"):        # fpmK: floating point with K explicit mantissa bits (fp32 exponent range), nearest
        k_ = int(name[3:])
        def _fpm(X, k_=k_):
            m_, e_ = torch.frexp(X)
            return torch.ldexp(torch.round(m_ * 2 ** (k_ + 1)) / 2 ** (k_ + 1), e_)
        return _fpm
    if name == "fp8":                                                                 # e4m3 with a per-column scale
        return lambda X: (X / (s := X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 448)).to(torch.float8_e4m3fn).float() * s
    raise ValueError(name)


def _store_codec(name):
    """--ts_store_m / --ts_store_a: real compressed storage of a low-rank pair, as opposed to --ts_qfmt*, which rounds fp32
    copies. Returns enc(X) -> tuple of stored tensors: int8 / int8det / fp8 / fp16s keep codes + a per-column fp32 scale
    (1 x r), fp16 / bf16 keep the tensor in that dtype. Decoded values equal the simulated --ts_qfmt formats exactly
    (same rounding, same random numbers)."""
    if name in ("", "none", "fp32"):
        return None
    cs = lambda X, qmax: X.abs().amax(0, keepdim=True).clamp_min(1e-30) / qmax
    if name == "int8":
        def enc(X):
            sc = cs(X, 127); return ((X / sc + torch.rand_like(X)).floor().clamp_(-127, 127).to(torch.int8), sc)
    elif name == "int8det":
        def enc(X):
            sc = cs(X, 127); return ((X / sc).round().clamp_(-127, 127).to(torch.int8), sc)
    elif name == "fp8":
        def enc(X):
            sc = cs(X, 448); return ((X / sc).to(torch.float8_e4m3fn), sc)
    elif name == "fp16s":
        def enc(X):
            sc = cs(X, 32768); return ((X / sc).to(torch.float16), sc)
    elif name == "fp12s":     # fp16s with the low 4 mantissa bits rounded off: sign, 5 exponent, 6 mantissa bits
        def enc(X):
            sc = cs(X, 32768); return (_pack12((X / sc).to(torch.float16)), sc)
    elif name in ("fp16", "bf16"):
        dt = torch.float16 if name == "fp16" else torch.bfloat16
        def enc(X):
            return (X.to(dt),)
    else:
        raise ValueError(name)
    return enc


def _pack12(Y):
    """fp16 (..., n) -> uint8 (..., 3 * ceil(n / 2)): each value rounded (nearest, ties to even) to 12 bits (sign, 5
    exponent, 6 mantissa bits) and two values packed into 3 bytes. |Y| <= 32768 (the fp16s scaling), so rounding up
    never reaches the infinity code."""
    if Y.shape[-1] % 2:
        Y = torch.cat([Y, torch.zeros_like(Y[..., :1])], -1)
    h = Y.view(torch.int16).to(torch.int32) & 0xFFFF
    m = h & 0x7FFF
    m = (m + 7 + ((m >> 4) & 1)) >> 4
    v = (((h >> 15) << 11) | m).reshape(*Y.shape[:-1], Y.shape[-1] // 2, 2)
    a, b = v[..., 0], v[..., 1]
    return torch.stack([a & 0xFF, (a >> 8) | ((b & 0xF) << 4), b >> 4], -1).reshape(*Y.shape[:-1], -1).to(torch.uint8)


def _unpack12(P, n):
    """inverse of _pack12 -> fp16 (..., n)"""
    p = P.reshape(*P.shape[:-1], -1, 3).to(torch.int32)
    a = p[..., 0] | ((p[..., 1] & 0xF) << 8)
    b = (p[..., 1] >> 4) | (p[..., 2] << 4)
    v = torch.stack([a, b], -1).reshape(*P.shape[:-1], -1)[..., :n]
    h = ((v >> 11) << 15) | ((v & 0x7FF) << 4)
    return torch.where(h >= 32768, h - 65536, h).to(torch.int16).view(torch.float16)


def _decode(c, s):
    """stored codes (int8 / fp8 / fp16 / packed 12-bit) times their per-column scale"""
    return (_unpack12(c, s.shape[-1]) if c.dtype == torch.uint8 else c).float() * s


def _store_pair(enc, U, V):
    """(U, V) -> flat tuple of stored tensors (U codes[, U scale], V codes[, V scale]), so checkpoint / resume code that
    moves tuples of tensors works unchanged"""
    return enc(U) + enc(V)


def _load_pair(t):
    """inverse of _store_pair (and a no-op on a plain fp32 (U, V))"""
    if len(t) == 4:
        return _decode(t[0], t[1]), _decode(t[2], t[3])
    return t[0].float(), t[1].float()


class _PinnedPool:
    """Page-locked host memory of exactly the requested size for the offloaded --ts state. torch's pinned allocator
    rounds every block up to a power of two (measured on the lab PCs, torch 2.13: 100 x 9 MiB took 1600 MiB, one 900 MiB
    block 1024 MiB), which at 27B ranks 1024 turns 21.3 GiB of state into 29.7 GiB. Here every tensor of at least
    BIG bytes gets its own ordinary host buffer, padded so that a page-aligned slice of the rounded size fits inside it,
    and that slice is page-locked with cudaHostRegister (so two registrations never share a page, and every tensor has its
    own storage, as torch.save requires). Smaller tensors (scales, v's rows / columns) use torch's pinned allocator, where
    the rounding costs little. Buffers are never freed: the state lives for the whole run."""
    BIG = 1 << 20

    def __init__(self):
        self.keep, self.ptrs, self.reg = [], set(), 0

    def empty(self, shape, dtype):
        n = math.prod(shape) * dtype.itemsize
        if n < self.BIG:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        m = (n + 4095) // 4096 * 4096
        raw = torch.empty(m + 8192, dtype=torch.uint8)
        a = (-raw.data_ptr()) % 4096
        err = torch.cuda.cudart().cudaHostRegister(raw.data_ptr() + a, m, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed ({err}) for {m} bytes")
        t = raw[a:a + n].view(dtype).view(shape)
        self.keep.append(raw); self.ptrs.add(t.data_ptr()); self.reg += m
        return t

    def owns(self, t):
        return t.is_pinned() or t.data_ptr() in self.ptrs


_PINNED = _PinnedPool()


def _host_get(d, k, dev, off):
    """--ts_offload: a state tuple kept in pinned host memory, copied to the GPU for this layer's update (the copy is
    queued on the current stream, so it is ordered before the kernels that read it; no host sync)"""
    if off == "async":
        return _OFFLOAD.get(d, k, dev)
    v = d[k]
    return tuple(x.to(dev, non_blocking=True) for x in v) if off else v


def _host_put(d, k, v, off):
    """--ts_offload: write a state tuple back to its pinned host buffers (reused across steps; the device-to-host copy
    is queued on the same stream after this step's kernels and before next step's host-to-device copy)"""
    if not off:
        d[k] = v
        return
    if off == "async":
        return _OFFLOAD.put(d, k, v)
    old = d.get(k)
    if (old is not None and len(old) == len(v) and all(o.device.type == "cpu" and o.shape == x.shape and o.dtype == x.dtype
                                                       and _PINNED.owns(o) for o, x in zip(old, v))):
        for o, x in zip(old, v):
            o.copy_(x, non_blocking=True)
    else:
        d[k] = tuple(_PINNED.empty(x.shape, x.dtype).copy_(x, non_blocking=True) for x in v)


class _AsyncOffload:
    """--ts_offload_async: the --ts_offload copies on two side streams (host-to-device prefetch, device-to-host
    write-back), so the PCIe traffic overlaps with the backward pass instead of stalling it. The order in which the
    update reads its states (v, m, accumulator per layer) is recorded during the first step that reads them and repeats
    every step; from then on, reading state j prefetches states j+1 .. j+AHEAD (wrapping into the next step). Each
    write-back records an event that the next fetch of the same state waits for; tensors used on another stream than the
    one they were allocated on are marked with record_stream, so the caching allocator cannot hand them out early.
    Host buffers are read by the host only in save_ckpt, which synchronizes first."""
    AHEAD = 3                                     # one layer's states

    def __init__(self):
        self.up = self.down = None
        self.order, self.idx, self.pre, self.done, self.fixed = [], {}, {}, {}, False

    def _fetch(self, d, k, dev):
        tag = (id(d), k)
        with torch.cuda.stream(self.up):
            if tag in self.done:
                self.up.wait_event(self.done[tag])
            y = tuple(x.to(dev, non_blocking=True) for x in d[k])
            ev = self.up.record_event()
        return y, ev

    def get(self, d, k, dev):
        if self.up is None:
            self.up, self.down = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
        cur, tag = torch.cuda.current_stream(dev), (id(d), k)
        if not self.fixed:
            if tag in self.idx:
                self.fixed = True
            else:
                self.idx[tag] = len(self.order); self.order.append((d, k))
        y, ev = self.pre.pop(tag, None) or self._fetch(d, k, dev)
        cur.wait_event(ev)
        for x in y:
            x.record_stream(cur)
        if self.fixed and tag in self.idx:
            n, j = len(self.order), self.idx[tag]
            for a in range(1, min(self.AHEAD, n - 1) + 1):
                d2, k2 = self.order[(j + a) % n]
                if (id(d2), k2) not in self.pre and k2 in d2:
                    self.pre[(id(d2), k2)] = self._fetch(d2, k2, dev)
        return y

    def put(self, d, k, v):
        dev = v[0].device
        if self.up is None:
            self.up, self.down = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
        cur, tag = torch.cuda.current_stream(dev), (id(d), k)
        self.pre.pop(tag, None)                   # a prefetch issued before this write would be stale
        self.down.wait_stream(cur)
        old = d.get(k)
        with torch.cuda.stream(self.down):
            if (old is not None and len(old) == len(v) and all(o.device.type == "cpu" and o.shape == x.shape and
                                                               o.dtype == x.dtype and _PINNED.owns(o) for o, x in zip(old, v))):
                for o, x in zip(old, v):
                    o.copy_(x, non_blocking=True)
            else:
                d[k] = tuple(_PINNED.empty(x.shape, x.dtype).copy_(x, non_blocking=True) for x in v)
            for x in v:
                x.record_stream(self.down)
            self.done[tag] = self.down.record_event()


_OFFLOAD = _AsyncOffload()


def mem_report(model, tail_opt, lr_state):
    """--mem_report: MiB held by each part of training (tensors counted by their own size; CUDA context excluded)"""
    MB = 2 ** 20
    nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts if torch.is_tensor(t))
    trits = nbytes([b for n, b in model.named_buffers() if n.endswith("wpacked")])
    params = list(model.parameters())
    opt = nbytes([v for st in tail_opt.state.values() for v in st.values()])
    gpu, host = 0, 0
    def walk(x):
        nonlocal gpu, host
        if torch.is_tensor(x):
            if x.is_cuda: gpu += x.numel() * x.element_size()
            else: host += x.numel() * x.element_size()
        elif isinstance(x, (tuple, list)):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for y in x.values(): walk(y)
    walk(lr_state)
    return {"mem_trits_MB": trits / MB, "mem_tail_params_MB": nbytes(params) / MB,
            "mem_tail_grads_MB": nbytes([p.grad for p in params if p.grad is not None]) / MB,
            "mem_tail_opt_MB": opt / MB, "mem_ts_gpu_MB": gpu / MB, "mem_ts_host_MB": host / MB,
            "mem_alloc_MB": torch.cuda.memory_allocated() / MB, "mem_peak_MB": torch.cuda.max_memory_allocated() / MB}


def _ts_state_bytes(lr_state):
    """bytes held by the --ts state (accumulator pairs, short momentum, factored v), counted per stored tensor"""
    n = 0
    for k, v in lr_state.items():
        if k == "_ts":
            for kk, vv in v.items():
                if isinstance(vv, tuple):
                    n += sum(x.numel() * x.element_size() for x in vv if torch.is_tensor(x))
        elif isinstance(v, tuple):
            n += sum(x.numel() * x.element_size() for x in v if torch.is_tensor(x))
    return n


def _q8(X):
    """Stochastic rounding of each column to 255 levels of its absmax (what an int8 store + fp32 column scale holds)."""
    s = X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 127
    return (X / s + torch.rand_like(X)).floor().clamp_(-127, 127) * s


def _orth(X):
    """orthonormal basis of the columns of X (tall, K x r): two-pass Cholesky-QR (GEMMs plus an r x r Cholesky in
    fp64, no host sync) instead of a Householder QR, which was 1.6 s per 1.3B step. Only the span matters to --ts
    (A = U V^T does not depend on the basis chosen)."""
    if not ORTH_CHOL[0]:
        Q = torch.linalg.qr(X)[0]
        if ORTH_DBG[0]:
            _orth_stats(X, Q)
        return Q
    Q = X.double() if ORTH_CHOL[0] == "chol64" else X   # chol64: both passes in fp64 (the fp32 Gram fails at cond ~1e6)
    if ORTH_CHOL[0] == "chol_cs":   # unit columns first: same span, without the column-scale part of the conditioning
        Q = X / X.norm(dim=0, keepdim=True).clamp_min(1e-30)
    eye = None
    for _ in range(2):
        G = (Q.T @ Q).double()
        G.diagonal().add_(G.diagonal().mean().clamp_min(1e-300) * 1e-12)
        L = torch.linalg.cholesky_ex(G)[0]
        if eye is None:
            eye = torch.eye(G.shape[0], dtype=torch.float64, device=G.device)
        Linv = torch.linalg.solve_triangular(L, eye, upper=False)
        Q = Q @ Linv.T.to(Q.dtype)
    Q = torch.nan_to_num(Q).to(X.dtype)
    if ORTH_DBG[0]:
        _orth_stats(X, Q)
    return Q


def _orth_stats(X, Q):
    """TERN_ORTHDBG=1 (one step in 10): compare Q with Householder on the same input; count all-zero inputs / outputs"""
    Qh = torch.linalg.qr(X)[0]
    Qo = torch.linalg.qr(Q)[0]      # Q's own span, orthonormal, so the span error does not mix in the orthogonality error
    sv = torch.linalg.svdvals(X.double())
    ORTH_STATS.append(((Q.T @ Q - torch.eye(Q.shape[1], device=Q.device)).abs().max().item(),
                       ((Qh - Qo @ (Qo.T @ Qh)).norm() / Qh.norm()).item(),
                       (sv[0] / sv[-1].clamp_min(1e-300)).item(),
                       float(X.abs().max().item() == 0), float(Q.abs().max().item() == 0)))


ORTH_CHOL = [False]               # False (Householder) | "chol" | "chol_cs" | "chol64" (--ts_orth)
ORTH_DBG = [False]                # on for one step in 10 when TERN_ORTHDBG=1
ORTH_STATS = []                   # (max |Q^T Q - I|, span error vs Householder, condition number of the input) per call


def ts_layer(l, g, state, c):
    """one layer of --ts (see ts_step): g is this layer's gradient (fp32); c holds the settings and lr_ratio.
    Called from ts_step after the backward, or from the layer's backward itself (--ts_fused), right after its input
    gradient is computed, so only one layer's fp32 gradient exists at a time."""
    from .kernel import unpack_rows, pack_rows
    if c.get("shared_v"):
        return ts_layer_shared(l, g, state, c)
    r, lr_ratio, theta, tau, b1, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b1"], c["b2"]
    r = getattr(l, "_ts_rank", 0) or r                                                  # --ts_rank_kv: per-layer rank
    rk_sched = None
    if c.get("rank_sched"):   # --ts_rank_sched R0:R1: both ranks move linearly from R0 to R1 over training (multiples of 8)
        r0_, r1_ = c["rank_sched"]
        rk_sched = max(8, int(round((r0_ + (r1_ - r0_) * min(max(c.get("progress", 0.0), 0.0), 1.0)) / 8)) * 8)
        r = rk_sched
    diag = c.get("diag", False)                                                        # --ts_diag: this step measures
    ident = lambda X: X
    qf = _qfmt(c.get("qfmt") or ("int8" if c.get("int8") else ""))
    qm = qf if qf is not None and c.get("qwhich", "all") in ("all", "m") else ident    # the short momentum
    qa = qf if qf is not None and c.get("qwhich", "all") in ("all", "a") else ident    # the accumulator
    if c.get("qfmt_m"): qm = _qfmt(c["qfmt_m"]) or ident                               # --ts_qfmt_m: own format
    if c.get("qfmt_a"): qa = _qfmt(c["qfmt_a"]) or ident                               # --ts_qfmt_a: own format
    qVm, qV = (ident, ident) if c.get("qnoV") else (qm, qa)                             # --ts_qnoV: bases kept fp32
    if c.get("qfmt_aV"): qV = _qfmt(c["qfmt_aV"]) or ident                             # --ts_qfmt_aV: accumulator's V
    sm, sa = _store_codec(c.get("store_m", "")), _store_codec(c.get("store_a", ""))    # --ts_store_*: real storage
    st = state.setdefault("_ts", {})
    off, dev = c.get("offload", False), g.device                                       # --ts_offload: state on the host
    key = id(l); t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    g2 = g * g
    if ("v", key) in st:
        R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * g2.mean(1); C = b2 * C + (1 - b2) * g2.mean(0)
    else:
        R, C = (1 - b2) * g2.mean(1), (1 - b2) * g2.mean(0)
    _host_put(st, ("v", key), (R, C), off)
    v = R[:, None] * C[None, :] / R.mean().clamp_min(1e-30) / (1 - b2 ** t)
    del g2
    rs = min(rk_sched or getattr(l, "_ts_rank_s", 0) or getattr(l, "_ts_rank", 0) or c["rank_s"], l.N, l.K)
    if ("m", key) not in st:
        gen = torch.Generator(device=g.device).manual_seed(4_000_000 + 2 * getattr(l, "_ts_idx", 0))
        Vm = _orth(g.T @ torch.randn(l.N, rs, device=g.device, generator=gen))
        Um = (1 - b1) * g @ Vm
    else:
        Um, Vm = _load_pair(_host_get(st, ("m", key), dev, off))
        Vn = _orth(b1 * Vm @ (Um.T @ Um) + (1 - b1) * g.T @ Um)
        Md = (b1 * Um @ Vm.T + (1 - b1) * g) if diag else None    # the momentum before its rank-r projection
        Um = b1 * Um @ (Vm.T @ Vn) + (1 - b1) * g @ Vn; Vm = Vn
        if diag:      # share of the momentum's energy the projection drops (Um_new = Md Vn, Vn orthonormal)
            x_ = 1 - (Um.norm() ** 2 / Md.norm().clamp_min(1e-30) ** 2).item(); del Md
            _ts_rec(st, "m_drop", x_); _ts_rec(st, "m_drop_" + _TS_TYPES[getattr(l, "_ts_idx", 0) % 7], x_)
        if Vm.shape[1] != rs:     # --ts_rank_sched: grow / shrink the momentum to this step's rank
            Um, Vm = _resize_lowrank(Um, Vm, rs, g, 1 - b1, 7_000_000 + 2 * getattr(l, "_ts_idx", 0) + t)
    if c.get("qdiag"): _qdiag(st, key, "m", Um, Vm, t)
    _host_put(st, ("m", key), _store_pair(sm, Um, Vm) if sm else (qm(Um), qVm(Vm)), off)  # --ts_qfmt sim., --ts_store_m real
    u = -(Um @ Vm.T) / (1 - b1 ** t) / (v.sqrt() + 1e-8) * lr_ratio
    shadow_here = c.get("shadows") and (not c.get("shadow_layers") or getattr(l, "_ts_idx", -1) in c["shadow_layers"])
    if shadow_here:           # --ts_shadows: the same gradients into shadow states of other ranks (and full rank)
        _ts_shadows_step(l, g, v, st, c, t, lr_ratio, key)
    del v
    ra = min(r, l.N, l.K)
    if c.get("tiles", 1) > 1:     # --ts_tiles T: the accumulator as T x T tiles, each of rank ra / T (same memory)
        return _ts_tiled_rest(l, u, state, st, c, key, ra)
    if key not in state:
        gen = torch.Generator(device=g.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        V = _orth(u.T @ torch.randn(l.N, ra, device=g.device, generator=gen))
        U = u @ V
    else:
        U, V = _load_pair(_host_get(state, key, dev, off))
        U = U * (1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau)   # --ts_tau_anneal: memory in distance, not steps
        if c.get("resid"):        # --ts_resid: what the last projection discarded comes back with this step (dense)
            R_ = st.get(("resid", key))
            if R_ is not None:
                u = u + R_ * (1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau)
            Afull = U @ V.T + u
        X_ = V @ (U.T @ U) + u.T @ U
        if c.get("sketch"):       # --ts_sketch RHO: a random sketch of this step's u joins the basis update, so new
            S_ = u.T @ torch.randn(l.N, X_.shape[1], device=u.device)   # directions can enter (RHO of the norm)
            X_ = X_ + c["sketch"] * X_.norm() / S_.norm().clamp_min(1e-30) * S_; del S_
        Vn = _orth(X_); del X_
        Ad = (U @ V.T + u) if diag else None                     # the accumulator before its rank-r projection
        uVn = u @ Vn
        U = U @ (V.T @ Vn) + uVn; V = Vn
        if c.get("rescale"):      # --ts_rescale: share of each step's energy the basis keeps (EMA over ~100 steps)
            k_ = (uVn.norm() ** 2 / u.norm().clamp_min(1e-30) ** 2).clamp(0, 1)
            kp = st.get(("kept", key))
            st[("kept", key)] = k_ if kp is None else 0.99 * kp + 0.01 * k_
        if c.get("refresh") and t % c["refresh"] == 0:   # --ts_refresh R: every R steps the weakest columns of the
            # basis are replaced by the top directions of what this step's u has outside it (randomized range finder,
            # one power iteration); the new columns start with this step's evidence along them
            kr = max(1, int(round(V.shape[1] * c.get("refresh_frac", 0.125))))
            gen = torch.Generator(device=u.device).manual_seed(5_000_000 + 2 * getattr(l, "_ts_idx", 0) + t)
            Om = torch.randn(l.N, kr, device=u.device, generator=gen)
            Y_ = u.T @ Om - V @ (uVn.T @ Om)                     # residual^T Om, residual = u - u V V^T
            Z_ = u @ Y_ - uVn @ (V.T @ Y_)                       # residual Y
            Y_ = u.T @ Z_ - V @ (uVn.T @ Z_)
            Q_ = _orth(Y_ - V @ (V.T @ Y_)); del Y_, Z_, Om
            idx = (U * U).sum(0).argsort()[:kr]                  # columns holding the least evidence
            V = V.clone(); U = U.clone()
            V[:, idx] = Q_; U[:, idx] = u @ Q_
            del Q_
        del uVn
        if V.shape[1] != ra:      # --ts_rank_sched: grow / shrink the accumulator to this step's rank
            U, V = _resize_lowrank(U, V, ra, u, 1.0, 8_000_000 + 2 * getattr(l, "_ts_idx", 0) + t)
        if c.get("resid"):
            st[("resid", key)] = Afull - U @ V.T; del Afull
        if diag:      # evidence dropped by the projection (U_new = Ad Vn) and share of this step's u that is kept
            _ts_rec(st, "a_drop", 1 - (U.norm() ** 2 / Ad.norm().clamp_min(1e-30) ** 2).item())
            x_ = ((u @ Vn).norm() ** 2 / u.norm().clamp_min(1e-30) ** 2).item(); del Ad
            _ts_rec(st, "u_kept", x_); _ts_rec(st, "u_kept_" + _TS_TYPES[getattr(l, "_ts_idx", 0) % 7], x_)
    if c.get("shadow"):   # --ts_shadow: a full-rank accumulator fed the same steps and spent by the same moves
        S = st.get(("shadow", key))
        S = u.clone() if S is None else S.mul_(1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau).add_(u)
        st[("shadow", key)] = S
    use_tab = (c.get("table") or 0) > 0 or c.get("exact_spend")
    if use_tab:   # --ts_table F / --ts_exact_spend: a sparse table of exact corrections on top of the low-rank A
        E = torch.zeros(l.N, l.K, device=dev)
        T0 = st.get(("tab", key))
        if T0 is not None:
            E.view(-1)[T0[0]] = T0[1] * (1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau)
        if (c.get("table") or 0) > 0:     # this step's evidence that the rank-r basis does not hold
            E += u - (u @ V) @ V.T
    del u
    A = U @ V.T
    if use_tab:
        A = A + E
    w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
    d = A.sign()
    th = theta
    if c.get("rescale") and ("kept", key) in st:   # --ts_rescale: compare A / s with theta, s = sqrt(share kept)
        th = theta * st[("kept", key)].sqrt().clamp(0.25, 1.0)
    fire = (A.abs() >= th) & ((w0.float() + d).abs() <= 1)
    if diag and c.get("shadow"):   # distance of the rank-r accumulator from the full-rank one
        _ts_rec(st, "a_vs_shadow", ((A - st[("shadow", key)]).norm() / st[("shadow", key)].norm().clamp_min(1e-30)).item())
    del A
    if c["gate"]:
        fire &= (d == -g.sign())
    before = l.wpacked.clone() if l.track else None
    D = torch.where(fire, d, torch.zeros_like(d))
    nf = fire.sum()                           # a device tensor: no host sync per layer
    l.wpacked.copy_(pack_rows((w0.float() + D).to(torch.int8)))
    U = U - c["spend"] * th * (D @ V)
    if use_tab:   # exact spend: the part of the move the basis does not hold goes into the table
        DV = D @ V
        Er = c["spend"] * th * (D - DV @ V.T)
        E -= Er if (c.get("table") or 0) > 0 else Er * (D != 0)
        if (c.get("table") or 0) > 0:     # keep the F * N * K entries that decide moves: largest |low-rank + table|
            kk = max(1, int(c["table"] * E.numel()))
            sc_ = (U @ V.T + E).abs().view(-1)
            idx = sc_.topk(kk, sorted=False).indices
        else:                             # exact spend only: corrections of recent moves until they fade below 0.5
            ae = E.abs().view(-1)
            idx = (ae >= 0.5).nonzero().squeeze(1)
            kmax = max(1, int(0.02 * E.numel()))
            if idx.numel() > kmax:
                idx = idx[ae[idx].topk(kmax, sorted=False).indices]
        st[("tab", key)] = (idx, E.view(-1)[idx].clone())
        del E, Er, DV
    if shadow_here:
        _ts_shadows_after(l, st, c, key, D, w0, theta, diag, (U + c["spend"] * theta * (D @ V)) @ V.T, t)
    if c.get("shadow"):
        S = st[("shadow", key)]
        if diag:      # which moves the full-rank accumulator would make, against the moves made
            fs = (S.abs() >= theta) & ((w0.float() + S.sign()).abs() <= 1)
            both = (fs & fire).sum().item()
            _ts_rec(st, "shadow_recall", both / max(fs.sum().item(), 1)); _ts_rec(st, "shadow_precision", both / max(fire.sum().item(), 1))
            _ts_rec(st, "shadow_moves", fs.float().mean().item()); _ts_rec(st, "moves", fire.float().mean().item())
            del fs
        S.sub_(c["spend"] * theta * D)
    if c.get("qdiag"): _qdiag(st, key, "a", U, V, t)
    _host_put(state, key, _store_pair(sa, U, V) if sa else (qa(U), qV(V)), off)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + fire.numel()
    if before is not None:
        l._record_flips(before, l.wpacked)


def _rank_sched(spec):
    """--ts_rank_sched "R0:R1" -> (R0, R1), or None"""
    if not spec:
        return None
    a_, b_ = spec.split(":")
    return (int(a_), int(b_))


def _resize_lowrank(U, V, r_new, X, coef, seed):
    """--ts_rank_sched: a low-rank pair U V^T resized to rank r_new. Shrink: the best rank-r_new part of U V^T (SVD of
    the N x r factor U, since V is orthonormal). Grow: the top directions of X (this step's gradient or step) outside
    span(V), found by a randomized range finder with one power iteration, with coefficients coef * X Q."""
    r_old = V.shape[1]
    if r_new == r_old:
        return U, V
    if r_new < r_old:
        P, S_, Vh = torch.linalg.svd(U, full_matrices=False)
        return P[:, :r_new] * S_[:r_new], V @ Vh[:r_new].T
    k = r_new - r_old
    gen = torch.Generator(device=X.device).manual_seed(seed)
    Om = torch.randn(X.shape[0], k, device=X.device, generator=gen)
    XV = X @ V
    Y = X.T @ Om - V @ (XV.T @ Om)
    Z = X @ Y - XV @ (V.T @ Y)
    Y = X.T @ Z - V @ (XV.T @ Z)
    Q = _orth(Y - V @ (V.T @ Y))
    return torch.cat([U, coef * (X @ Q)], 1), torch.cat([V, Q], 1)


def _ts_tiled_rest(l, u, state, st, c, key, ra):
    """--ts_tiles T: the accumulator split into T x T tiles of (N / T) x (K / T), each with its own rank ra // T basis
    (T * (ra // T) * (N + K) numbers: the memory of one rank-ra pair), updated by batched subspace iteration; firing
    and spend as in ts_layer. Momentum and v stay global."""
    from .kernel import unpack_rows, pack_rows
    T_ = c["tiles"]; n_, k_ = l.N // T_, l.K // T_; rt = max(1, ra // T_)
    assert n_ * T_ == l.N and k_ * T_ == l.K, "--ts_tiles: N and K must divide by T"
    theta, tau, lr_ratio = c["theta"], c["tau"], c["lr_ratio"]
    til = lambda X: X.reshape(T_, n_, T_, k_).permute(0, 2, 1, 3).reshape(T_ * T_, n_, k_)
    until = lambda X: X.reshape(T_, T_, n_, k_).permute(0, 2, 1, 3).reshape(l.N, l.K)
    ub = til(u)
    if key not in state:
        gen = torch.Generator(device=u.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        Om = torch.randn(T_ * T_, n_, rt, device=u.device, generator=gen)
        V = torch.linalg.qr(ub.transpose(1, 2) @ Om)[0]
        U = ub @ V
    else:
        U, V = state[key]
        U = U * (1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau)
        Vn = torch.linalg.qr(V @ (U.transpose(1, 2) @ U) + ub.transpose(1, 2) @ U)[0]
        U = U @ (V.transpose(1, 2) @ Vn) + ub @ Vn; V = Vn
    del ub, u
    A = until(U @ V.transpose(1, 2))
    w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
    d = A.sign()
    fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
    del A
    before = l.wpacked.clone() if l.track else None
    D = torch.where(fire, d, torch.zeros_like(d))
    nf = fire.sum()
    l.wpacked.copy_(pack_rows((w0.float() + D).to(torch.int8)))
    U = U - c["spend"] * theta * (til(D) @ V)
    state[key] = (U, V)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + fire.numel()
    if before is not None:
        l._record_flips(before, l.wpacked)


_TS_TYPES = ("q", "k", "v", "o", "gate", "up", "down")   # order of the ternary layers in a block


@torch.no_grad()
def _ts_shadows_step(l, g, v, st, c, t, lr_ratio, key):
    """--ts_shadows R1,R2,...: for each rank (0 = full rank) a shadow short momentum and accumulator, fed this layer's
    gradient with the same v and schedule as the run; spent later by the run's actual moves (_ts_shadows_after)"""
    b1, tau = c["b1"], c["tau"]
    decay = 1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau
    sq = v.sqrt() + 1e-8
    sh = st.setdefault(("shd", key), {})
    for rr in c["shadows"]:
        d = sh.get(rr)
        if rr == 0:
            M = (1 - b1) * g if d is None else d[0].mul_(b1).add_(g, alpha=1 - b1)
            uu = -M / (1 - b1 ** t) / sq * lr_ratio
            A = uu.clone() if d is None else d[1].mul_(decay).add_(uu)
            sh[0] = (M, A)
            continue
        k = min(rr, l.N, l.K)
        if d is None:
            gen = torch.Generator(device=g.device).manual_seed(5_000_000 + 7 * getattr(l, "_ts_idx", 0) + rr)
            Vm = _orth(g.T @ torch.randn(l.N, k, device=g.device, generator=gen)); Um = (1 - b1) * g @ Vm
        else:
            Um, Vm, U, V = d
            Vn = _orth(b1 * Vm @ (Um.T @ Um) + (1 - b1) * g.T @ Um)
            Um = b1 * Um @ (Vm.T @ Vn) + (1 - b1) * g @ Vn; Vm = Vn
        uu = -(Um @ Vm.T) / (1 - b1 ** t) / sq * lr_ratio
        if d is None:
            gen = torch.Generator(device=g.device).manual_seed(6_000_000 + 7 * getattr(l, "_ts_idx", 0) + rr)
            V = _orth(uu.T @ torch.randn(l.N, k, device=g.device, generator=gen)); U = uu @ V
        else:
            U = U * decay
            Vn = _orth(V @ (U.T @ U) + uu.T @ U)
            U = U @ (V.T @ Vn) + uu @ Vn; V = Vn
        sh[rr] = (Um, Vm, U, V)


@torch.no_grad()
def _ts_shadows_after(l, st, c, key, D, w0, theta, diag, A_run, t):
    """compare each shadow accumulator (and the run's own, A_run, before its spend) with the full-rank shadow on the
    measuring steps, dump matrices of chosen layers on chosen steps, then spend every shadow by the run's moves D"""
    sh = st[("shd", key)]
    ty = _TS_TYPES[getattr(l, "_ts_idx", 0) % 7]
    full = sh.get(0)
    if diag and full is not None:
        M_f, A_f = full
        nf = A_f.norm().clamp_min(1e-30)
        big = A_f.abs() >= theta / 2
        fire_f = (A_f.abs() >= theta) & ((w0.float() + A_f.sign()).abs() <= 1)
        sv2 = torch.linalg.svdvals(A_f) ** 2; tot = sv2.sum().clamp_min(1e-30)
        for kk in (64, 128, 512):
            _ts_rec(st, f"full_top{kk}", (sv2[:kk].sum() / tot).item())
        cands = [("run", A_run, None)] + [(f"r{rr}", sh[rr][2] @ sh[rr][3].T, sh[rr][0] @ sh[rr][1].T) for rr in c["shadows"] if rr]
        for name, X, Mx in cands:
            fire_x = (X.abs() >= theta) & ((w0.float() + X.sign()).abs() <= 1)
            both = (fire_x & fire_f).sum().item()
            vals = {"cos": ((X * A_f).sum() / (X.norm().clamp_min(1e-30) * nf)).item(), "energy": (X.norm() / nf).item(),
                    "sign_agree_big": (X.sign() == A_f.sign())[big].float().mean().item() if big.any() else float("nan"),
                    "fire_recall": both / max(fire_f.sum().item(), 1), "fire_precision": both / max(fire_x.sum().item(), 1),
                    "fire_frac": fire_x.float().mean().item()}
            if Mx is not None:
                vals["mom_cos"] = ((Mx * M_f).sum() / (Mx.norm().clamp_min(1e-30) * M_f.norm().clamp_min(1e-30))).item()
            for k_, x_ in vals.items():
                if x_ == x_:
                    _ts_rec(st, f"{name}_{k_}", x_)
                    if ty == "down":
                        _ts_rec(st, f"{name}_{k_}_down", x_)
        _ts_rec(st, "full_fire_frac", fire_f.float().mean().item())
    if c.get("dump_dir") and t in c.get("dump_steps", ()) and getattr(l, "_ts_idx", -1) in c.get("dump_layers", ()):
        import numpy as _np
        os.makedirs(c["dump_dir"], exist_ok=True)
        out = {"w": w0.cpu().numpy(), "A_run": A_run.half().cpu().numpy()}
        for rr in c["shadows"]:
            if rr == 0:
                out["A_full"] = sh[0][1].half().cpu().numpy(); out["M_full"] = sh[0][0].float().cpu().numpy().astype("float32")
            else:
                out[f"A_r{rr}"] = (sh[rr][2] @ sh[rr][3].T).half().cpu().numpy()
                out[f"M_r{rr}"] = (sh[rr][0] @ sh[rr][1].T).float().cpu().numpy()
        _np.savez_compressed(os.path.join(c["dump_dir"], f"t{t}_layer{l._ts_idx}_{ty}.npz"), **out)
    cs = c["spend"] * theta
    for rr in c["shadows"]:
        if rr == 0:
            sh[0][1].sub_(cs * D)
        else:
            Um, Vm, U, V = sh[rr]
            sh[rr] = (Um, Vm, U - cs * (D @ V), V)


def _ts_rec(st, name, x):
    d = st.setdefault("_dg", {}); n = d.setdefault(name, [0.0, 0]); n[0] += x; n[1] += 1


@torch.no_grad()
def ts_layer_shared(l, g, state, c):
    """--ts_shared_v: the short momentum and the accumulator share ONE orthonormal basis V (K x r): stored Um (N x r),
    U (N x r), V (K x r) instead of two (U, V) pairs, a quarter less state when N ~ K. The basis follows both matrices:
    one subspace-iteration step on [M / |M|; A / |A|] (each normalised, so neither dominates), using the decayed
    accumulator before this step's u (u needs the projected momentum, which needs the basis). Everything else as
    ts_layer: factored v, bias corrections, threshold firing within [-1, 1], spend theta D V."""
    from .kernel import unpack_rows, pack_rows
    r, lr_ratio, theta, tau, b1, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b1"], c["b2"]
    assert not (c.get("store_m") or c.get("store_a") or c.get("offload") or c.get("qfmt") or c.get("int8")), \
        "--ts_shared_v: fp32 state only"
    ra = min(getattr(l, "_ts_rank", 0) or r, l.N, l.K)
    st = state.setdefault("_ts", {})
    key = id(l); t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    g2 = g * g
    if ("v", key) in st:
        R, C = st[("v", key)]; R = b2 * R + (1 - b2) * g2.mean(1); C = b2 * C + (1 - b2) * g2.mean(0)
    else:
        R, C = (1 - b2) * g2.mean(1), (1 - b2) * g2.mean(0)
    st[("v", key)] = (R, C); del g2
    sq = (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt_().add_(1e-8)
    decay = 1 - (lr_ratio if c["tau_anneal"] else 1.0) / tau
    if key not in state:
        M = (1 - b1) * g; Ao = None
        gen = torch.Generator(device=g.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        G = torch.randn(l.N, ra, device=g.device, generator=gen)
        Vn = _orth(M.T @ G)
    else:
        Um, U, V = st[("m", key)][0], state[key][0], state[key][1]
        M = b1 * Um @ V.T + (1 - b1) * g                       # dense, this step
        Ao = decay * U @ V.T                                   # dense, decayed, before this step's u
        X = M.T @ (M @ V) / M.norm().clamp_min(1e-30) ** 2 + Ao.T @ (Ao @ V) / Ao.norm().clamp_min(1e-30) ** 2
        Vn = _orth(X); del X
    Um = M @ Vn
    if c.get("diag"):
        _ts_rec(st, "m_drop", 1 - (Um.norm() ** 2 / M.norm().clamp_min(1e-30) ** 2).item())
    del M
    u = -(Um @ Vn.T) / (1 - b1 ** t) / sq * lr_ratio; del sq
    if c.get("diag"):
        _ts_rec(st, "u_kept", ((u @ Vn).norm() ** 2 / u.norm().clamp_min(1e-30) ** 2).item())
        if Ao is not None:
            Ad_ = Ao + u; _ts_rec(st, "a_drop", 1 - ((Ad_ @ Vn).norm() ** 2 / Ad_.norm().clamp_min(1e-30) ** 2).item()); del Ad_
    U = (u @ Vn) if Ao is None else (Ao @ Vn + u @ Vn); del Ao, u
    A = U @ Vn.T
    w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
    d = A.sign()
    fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1); del A
    if c["gate"]:
        fire &= (d == -g.sign())
    before = l.wpacked.clone() if l.track else None
    D = torch.where(fire, d, torch.zeros_like(d))
    l.wpacked.copy_(pack_rows((w0.float() + D).to(torch.int8)))
    U = U - c["spend"] * theta * (D @ Vn)
    st[("m", key)] = (Um,)
    state[key] = (U, Vn)
    st["_nf"] = st.get("_nf", 0) + fire.sum(); st["_nt"] = st.get("_nt", 0) + fire.numel()
    if before is not None:
        l._record_flips(before, l.wpacked)


def _qdiag(st, key, which, U, V, t):
    """--ts_qdiag: what the stored numbers look like (accumulated per step, read and reset at logging):
    crest = column absmax / column rms (int8 per column spends log2(127 / crest) bits on a typical entry);
    sub = share of this step's changes of U smaller than half an int8 step (absmax / 254), i.e. below resolution;
    rel = median |change| / (absmax / 127); range = log2(99.9th / 0.1th percentile of |U|) (exponent bits needed)."""
    D = st.setdefault("_qd", {})
    pt, prev = st.get(("prevU", which, key), (None, None))
    a = U.abs(); amax = a.amax(0, keepdim=True).clamp_min(1e-30)
    rec = D.setdefault(which, {"crest": 0.0, "sub": 0.0, "rel": 0.0, "range": 0.0, "n": 0})
    rec["crest"] += float((amax / a.pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-30)).median())
    smp = a.flatten()[:: max(1, a.numel() // 200_000)].float()
    q = torch.quantile(smp[smp > 0], torch.tensor([0.001, 0.999], device=smp.device)) if (smp > 0).any() else None
    if q is not None: rec["range"] += float(torch.log2(q[1] / q[0].clamp_min(1e-30)))
    if prev is not None and pt == t - 1 and prev.shape == U.shape:
        dU = (U - prev).abs()
        rec["sub"] += float((dU < amax / 254).float().mean()); rec["rel"] += float((dU / (amax / 127)).median())
    rec["n"] += 1
    st[("prevU", which, key)] = (t, U.detach().clone())


def ts_step(model, state, r, step, lr_ratio, theta=16.0, tau=300.0, rank_s=128, b1=0.9, b2=0.95, gate=False, spend=1.0,
            tau_anneal=False, int8=False, fused=False, qfmt="", qwhich="all", qnoV=False, qdiag=False,
            qfmt_m="", qfmt_a="", qfmt_aV="", store_m="", store_a="", offload=False, diag=False, shadow=False,
            shared_v=False, sketch=0.0, resid=False, shadows=(), dump_dir="", dump_steps=(), dump_layers=(),
            shadow_layers=(), rescale=False, refresh=0, refresh_frac=0.125, table=0.0, exact_spend=False, tiles=1,
            rank_sched=None, progress=0.0):
    """--ts: master's two timescales with sublinear state (the mechanism the 1 Oct master branches point to).
    Per layer:
      v      factored row x column EMA of g^2 (decay b2), v_ij ~ R_i C_j / mean(R)          (N + K floats)
      m      short momentum, rank rank_s, decay b1 (~10 steps): the direction, like Adam's m   (rank_s (N + K))
      u      = -(m / (1 - b1^t)) / sqrt(v / (1 - b2^t)): Adam's normalized step
      A      long accumulator, rank r: A <- (1 - 1/tau) A + lr_ratio * u  (the latent's sub-threshold position;
             lr_ratio = lr_t / lr_peak, so steps shrink as the schedule decays, as master's latent steps do)
      fire   a trit moves by d = sign(A_ij) the step |A_ij| >= theta (deterministic, no rate), unless blocked at +-1
             (optionally only where this batch's gradient agrees: gate); the move spends spend * theta:
             A -= spend * theta * D (kept low-rank: U -= ... (D V)). spend 1 puts the weight at its new trit's centre
             (master with --m_snap); spend 2 leaves it at the boundary it just crossed, as master's latent does.
             theta ~ half a bin in latent units over the peak lr (~16)."""
    c = dict(r=r, lr_ratio=lr_ratio, theta=theta, tau=tau, rank_s=rank_s, b1=b1, b2=b2, gate=gate, spend=spend,
             tau_anneal=tau_anneal, int8=int8, qfmt=qfmt, qwhich=qwhich, qnoV=qnoV, qdiag=qdiag,
             qfmt_m=qfmt_m, qfmt_a=qfmt_a, qfmt_aV=qfmt_aV, store_m=store_m, store_a=store_a, offload=offload,
             diag=diag, shadow=shadow, shared_v=shared_v, sketch=sketch, resid=resid, shadows=shadows,
             rescale=rescale, refresh=refresh, refresh_frac=refresh_frac, table=table, exact_spend=exact_spend,
             tiles=tiles, rank_sched=rank_sched, progress=progress,
             dump_dir=dump_dir, dump_steps=dump_steps, dump_layers=dump_layers, shadow_layers=shadow_layers)
    st = state.setdefault("_ts", {})
    if not fused:
        for i, l in enumerate([l for l in model.modules() if isinstance(l, KernelTernaryLinear)]):
            l._ts_idx = i
            g = l.gw.float(); l.gw = None; l.capture = False
            ts_layer(l, g, state, c)
            del g
    nf, nt = st.pop("_nf", 0), st.pop("_nt", 0)
    info = {"ts_frac": float(nf) / max(nt, 1)}
    for k_, (tot, n_) in st.pop("_dg", {}).items():      # --ts_diag: means over layers this step
        info["dg_" + k_] = tot / max(n_, 1)
    for w, rec in st.pop("_qd", {}).items():
        n = max(rec["n"], 1)
        info.update({f"qd_{w}_{k}": rec[k] / n for k in ("crest", "sub", "rel", "range")})
    return info


@torch.no_grad()
def bop_step(model, state, gamma, tau, b2=0.95, norm=True, bias_correct=False):
    """--bop: the closest prior method, as a baseline (not part of --ts). Latent-free flips with ONE timescale:
    Bop (Helwegen et al. 2019) keeps per weight m <- (1 - gamma) m + gamma g and flips a binary weight when |m| > tau and
    sign(m) = sign(w); Bop2ndOrder normalises m by the second moment. Ternary version: a trit moves one level against
    sign(m) when |m_hat / sqrt(v)| > tau (--bop_norm, v the factored second moment used by --ts) or |m| > tau, and the
    move stays in [-1, 1]. m is per weight in fp32 (Bop's own state, not sublinear); no accumulator, no reset after a
    move, no short momentum, no rank limit, no schedule. As in Bop, m starts at 0 without bias correction (with
    --bop_bc, m_hat = m / (1 - (1 - gamma)^t), whose noise ~ 1 / sqrt(t) would cross any small tau for ~1 / tau^2 steps).
    At equilibrium the noise of m / sqrt(v) is ~ sqrt(gamma / 2) (0.022 at gamma 1e-3), which sets the useful tau."""
    from .kernel import unpack_rows, pack_rows
    nf, nt = 0, 0
    for l in [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]:
        g = l.gw.float(); l.gw = None; l.capture = False
        k = id(l); t = state.get(("bop_t", k), 0) + 1; state[("bop_t", k)] = t
        m = state.get(("bop_m", k))
        m = gamma * g if m is None else m.mul_(1 - gamma).add_(g, alpha=gamma)
        state[("bop_m", k)] = m
        sig = m / (1 - (1 - gamma) ** t) if bias_correct else m
        if norm:
            g2 = g * g
            if ("bop_v", k) in state:
                R, C = state[("bop_v", k)]; R = b2 * R + (1 - b2) * g2.mean(1); C = b2 * C + (1 - b2) * g2.mean(0)
            else:
                R, C = (1 - b2) * g2.mean(1), (1 - b2) * g2.mean(0)
            state[("bop_v", k)] = (R, C); del g2
            v = R[:, None] * C[None, :] / R.mean().clamp_min(1e-30) / (1 - b2 ** t)
            sig = sig / (v.sqrt() + 1e-8); del v
        w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
        d = -sig.sign()
        fire = (sig.abs() > tau) & ((w0.float() + d).abs() <= 1)
        before = l.wpacked.clone() if l.track else None
        l.wpacked.copy_(pack_rows((w0.float() + torch.where(fire, d, torch.zeros_like(d))).to(torch.int8)))
        nf += fire.sum(); nt += fire.numel()
        if before is not None:
            l._record_flips(before, l.wpacked)
        del g, sig, d, fire, w0
    return {"ts_frac": float(nf) / max(nt, 1)}


def lowrank_step(model, state, r, beta, step, flip_seed=0, adapt=False, propose_only=False,
                 diag=False, gate=False, refresh=0, refresh_every=10, vnorm=0.0, mask_stuck=False,
                 qk_protect=0.0, qk_map=None, speed_ref=0.0, speed_row=0.0, ncap=0.0, pfun_tanh=0.0, grav_up=0.0,
                 undo=False, slow_rank=0, slow_beta=0.999, dither=False, dry_vec=0.0, spend=0.0, dry_w=0.0, mom_int8=False, vfull=False,
                 rare_rank=0, rare_dry=0.01, rare_mode="sum", rare_rate=0.5, tiers=(), rare_weight=0.0,
                 pshape="prop", pshape_T=0, dry_start=0.0, dry_warm=0):
    """Flip from a rank-r momentum of each layer's gradient instead of the current gradient.

    Per layer M ~ U V^T with V (K x r) orthonormal and U (N x r): M <- beta*M + g, kept at rank
    r by one subspace-iteration step (two thin GEMMs + a QR, no SVD). The flip rule and rate
    are unchanged; only its signal is the accumulated direction. State: r*(N+K) floats/layer.

    adapt: the decay follows the turn of the gradient, beta_t = beta * cos(g_t, M_{t-1}) per
    layer. Aligned: plain momentum; 90 deg: reset (M = g); 180 deg: -beta, so the old momentum
    is reflected onto the new direction and adds to it. Returns the per-layer cosines.

    speed_ref: (--speed_ref B) divide M by a slow EMA (decay B) of its mean |M| instead of by this step's mean |M|,
            so the number of flips follows the momentum's size (it drops when the new gradients cancel it while
            climbing a wall) instead of staying fixed. One float per layer.

    ncap / pfun_tanh (the user's rule, with beta = 1: momentum as one velocity vector, gravity = the gradient):
            ncap C caps the whole momentum (all layers as one vector) at C x a slow EMA of the total gradient norm;
            pfun_tanh S sets the flip chance to rate * tanh(|M_ij| / (g_ref v0)), v0 = S x a slow EMA of the layer's
            mean |g| (absolute gradient units, not the momentum's own size). Two floats + one per layer.

    undo: (--undo) a corrective move: last step's flips that this batch's gradient and the updated momentum now both
            call uphill are flipped back before this step's flips (the last step's moves are kept for one step,
            2 bits per weight packed, like the gradient itself a one-step buffer).
    grav_up: (--grav_up D) asymmetric gravity: where this batch's gradient opposes a weight's momentum (climbing), that
            weight's momentum decays with D instead of beta; on the full M, then one subspace step back to rank r.

    slow_rank / slow_beta: (--slow_gate R) a second, slow momentum (rank R, decay slow_beta) kept like M; flip only
            where the fast and the slow momentum agree in sign (a long-horizon consistency gate). R*(N+K) floats.
    dither: (--dither_ld) the flip draw is a low-discrepancy sequence per weight, u_ij = frac(h_ij + step * phi) with
            h_ij a fixed hash of the weight (regenerated each step from a seed, not stored), instead of a fresh random
            number: each weight flips its expected number of times with little sampling noise. No state.
    dry_start / dry_warm: (--dry_start D0 --dry_warm N) a memory that starts short and lengthens: the friction goes from D0
            (strong: short memory) to dry_vec over the first N steps on a cosine, then stays at dry_vec.
    dry_vec: (--dry_vec D) dry friction on the whole momentum (all layers as one vector): after each step
            ||M|| -= D x a slow EMA of the total gradient norm (to 0), use with beta 1. Two floats.
    dry_w: (--dry_w D) dry friction per weight: |M_ij| -= D x the layer's mean |g| each step (to 0), on the full M,
            then one subspace step back to rank r (like grav_up); use with beta 1.
    rare_rank / rare_dry / rare_mode: (--rare_rank R) a second momentum for rare patterns: rank R, fed only the part of
            each gradient outside the main momentum's column space, g2 = g - (g V) V^T, with its own, weaker dry
            friction (--rare_dry, so a longer memory; decay 1). rare_mode "sum": the flip signal is M + M2; "flip": M2
            proposes its own flips after the main ones, at rare_rate x the flip rate, with the same Adam step and gate.
    tiers: (--tiers "R:D,R:D,...") generalises rare_rank to a chain: tier t is fed what the main momentum and tiers
            0..t-1 miss (the residual after projecting out each one's column space in turn), has rank R and its own dry
            friction D (each later tier usually weaker = longer memory); combined as rare_mode (sum / own flips).
    rare_weight: (--rare_weight w, with rare_mode sum) add each extra momentum at w x the main one's mean size instead of
            at its own size: M + w (mean|M| / mean|M_t|) M_t, so a long-memory extra momentum cannot outgrow and swamp the
            main one. Logs the raw size ratio mean|M_t| / mean|M| (first tier) as rare_ratio.
    pshape: (--pshape) which weights the flips go to, at the same expected number of flips as the usual rule (p ~
            min(|S| / (3 mean|S|), 1)), same direction (-sign S) and the same gate: "flat" every eligible weight equally
            likely (sign only); "inv" preferring the small |S|: p ~ 1 - min(|S| / (3 mean|S|), 1); "cheap" preferring
            weights with a small gradient second moment v = R_i C_j / mean R (the Adam step's factors): p ~ 1 / (1 + v /
            median v). With pshape_T > 0 (--pshape_sched cos) the flips are a blend that moves from the usual rule to the
            chosen shape over training: p = (1 - lam) p_prop + lam p_shape, lam = (1 - cos(pi t / T)) / 2, both at the
            same expected count (steep weights first, cheap ones at the end). Master changes the weights nearest a rounding boundary, evenly across gradient sizes; the usual
            rule piles its flips on the steepest weights, where many flips together overshoot (1 Oct bench).
    mom_int8: (--mom_int8) store the momentum factors U, V at 8 bits: after each step every column is rounded
            stochastically to 255 levels of its own absmax (per-column fp32 scale), so the next step reads what an int8
            store would hold. Measures the precision cost; the memory itself is not yet packed.
    spend: (--spend c, plain path) a flip consumes the push that caused it: U += c * mean|M| * (D V), D = the trits'
            change (-sign(M) where they flipped), so those entries of M shrink by c * mean|M| (kept low-rank).

    propose_only: update M but do not flip; leave l.gw and capture on and return the M's,
    so lookahead_step proposes from M and filters with the true gradients.

    diag: also log, per step, how M_{t-1} relates to this step's gradient g: mean |g|, mean |M|,
    sign agreement (all weights and the top 1% |M|, the ones the rule proposes) and the share
    of |g| inside M's rank-r row x column subspace."""
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    cs, Ms = [], []
    state["_gms"] = []
    D = {k: [] for k in ("d_gabs", "d_mabs", "d_agree", "d_agree_top", "d_insub")}
    gn2 = 0.0
    TIERS = list(tiers) if tiers else ([(rare_rank, rare_dry)] if rare_rank else [])
    tier_gn2 = [0.0] * len(TIERS)
    for i, l in enumerate(layers):
        g = l.gw.float()
        if ncap or dry_vec: gn2 += float(g.pow(2).sum())
        if pfun_tanh:
            ga = state.setdefault("_gabs", {}); gam = g.abs().mean()
            ga[id(l)] = gam if id(l) not in ga else 0.99 * ga[id(l)] + 0.01 * gam
        if not propose_only:
            l.gw = None; l.capture = False
        if mask_stuck:
            # --lr_mask_stuck: drop the gradient at weights it pushes past +-1 (already at the bound in
            # the direction -sign(g)) before it enters M, so M holds only pressure a flip can act on
            from .kernel import unpack_rows
            t = unpack_rows(l.wpacked, l.K).to(torch.int8)
            g = torch.where(-g.sign() * t == 1, torch.zeros_like(g), g)
        key = id(l)
        if key not in state:
            U0 = torch.linalg.qr(torch.randn(l.N, r, device=g.device))[0]
            V = torch.linalg.qr(g.T @ U0)[0]
            state[key] = (g @ V, V)
        else:
            U, V = state[key]
            # cos(g, M) with M = U V^T, V orthonormal: <g, M> = sum((g V) * U), |M| = |U|
            c = ((g @ V) * U).sum() / (g.norm() * U.norm()).clamp_min(1e-30)
            cs.append(c.item())
            if diag:
                Mo = U @ V.T
                a = Mo.abs().flatten()
                thr = a.kthvalue(max(1, int(a.numel() * 0.99))).values
                ag = (Mo.sign() == g.sign()).flatten()
                D["d_gabs"].append(g.abs().mean().item()); D["d_mabs"].append(a.mean().item())
                D["d_agree"].append(ag.float().mean().item())
                D["d_agree_top"].append(ag[a >= thr].float().mean().item())
                Qu = torch.linalg.qr(U)[0]
                D["d_insub"].append(((Qu.T @ g @ V).norm() / g.norm().clamp_min(1e-30)).item())
                del Mo, a, ag
            b = beta * c if adapt else beta
            if dry_w:
                Mf = b * (U @ V.T) + g
                Mf = Mf.sign() * (Mf.abs() - dry_w * g.abs().mean()).clamp_min(0)
                Vn = torch.linalg.qr(Mf.T @ U)[0]
                state[key] = (Mf @ Vn, Vn); del Mf
            elif grav_up:
                Mf = U @ V.T
                Mf = torch.where(Mf.sign() * g.sign() < 0, grav_up * Mf, b * Mf) + g
                Vn = torch.linalg.qr(Mf.T @ U)[0]
                state[key] = (Mf @ Vn, Vn); del Mf
            else:
                Vn = torch.linalg.qr(b * V @ (U.T @ U) + g.T @ U)[0]
                state[key] = (b * U @ (V.T @ Vn) + g @ Vn, Vn)
        U, V = state[key]
        if slow_rank:
            sk = ("slow", key)
            if sk not in state:
                U0 = torch.linalg.qr(torch.randn(l.N, slow_rank, device=g.device))[0]
                Vs0 = torch.linalg.qr(g.T @ U0)[0]
                state[sk] = (g @ Vs0, Vs0)
            else:
                Us, Vs_ = state[sk]
                Vn = torch.linalg.qr(slow_beta * Vs_ @ (Us.T @ Us) + g.T @ Us)[0]
                state[sk] = (slow_beta * Us @ (Vs_.T @ Vn) + g @ Vn, Vn)
        if refresh:
            # --lr_refresh: per direction, a moving average of the share of the gradient it catches;
            # every refresh_every steps the `refresh` weakest directions are replaced by the top
            # directions of the gradient outside the subspace (randomized range finder, one power step)
            E = state.setdefault("_energy", {})
            share = (g @ V).pow(2).sum(0) / g.pow(2).sum().clamp_min(1e-30)
            E[key] = share if key not in E else 0.9 * E[key] + 0.1 * share
            if step % refresh_every == 0:
                k = min(refresh, V.shape[1] - 1)
                keep = E[key].argsort(descending=True)[:V.shape[1] - k]
                Vk, Uk = V[:, keep], U[:, keep]
                R = g - (g @ Vk) @ Vk.T
                Om = torch.randn(R.shape[1], k + 4, device=g.device)
                Q = torch.linalg.qr(R.T @ (R @ Om))[0][:, :k]
                Q = torch.linalg.qr(Q - Vk @ (Vk.T @ Q))[0]
                V = torch.cat([Vk, Q], 1); U = torch.cat([Uk, g @ Q], 1)
                E[key] = torch.cat([E[key][keep], E[key][keep].mean().expand(k)])
                state[key] = (U, V)
        g2, Vprev = g, V
        for t, (tr, _) in enumerate(TIERS):              # each tier: what the momenta before it miss
            rk = ("rare", key) if t == 0 else ("rare", t, key)
            g2 = g2 - (g2 @ Vprev) @ Vprev.T
            tier_gn2[t] += float(g2.pow(2).sum())
            if rk not in state:
                U0 = torch.linalg.qr(torch.randn(l.N, tr, device=g.device))[0]
                V2 = torch.linalg.qr(g2.T @ U0)[0]
                state[rk] = (g2 @ V2, V2)
            else:
                U2, V2 = state[rk]
                Vn = torch.linalg.qr(V2 @ (U2.T @ U2) + g2.T @ U2)[0]
                state[rk] = (U2 @ (V2.T @ Vn) + g2 @ Vn, Vn)
            Vprev = state[rk][1]
        del g2
        TK = [("rare", key) if t == 0 else ("rare", t, key) for t in range(len(TIERS))]
        M = U @ V.T
        if TIERS and rare_mode == "sum":
            m_main = M.abs().mean().clamp_min(1e-12)
            for t, rk in enumerate(TK):
                Mt = state[rk][0] @ state[rk][1].T
                m_t = Mt.abs().mean().clamp_min(1e-12)
                if t == 0: state.setdefault("_rare_ratio", []).append(float(m_t / m_main))
                M = M + (rare_weight * m_main / m_t * Mt if rare_weight else Mt)
                del Mt
        # spend's scale: mean |M| in gradient units (U's units), taken before vnorm rescales M
        state.setdefault("_gm", {})[key] = M.abs().mean().clamp_min(1e-12)
        if vnorm:
            # --lr_vnorm: factored second moment of the gradient (Adafactor-style row and column EMAs of
            # g^2, N+K numbers per layer); propose from M / sqrt(v), v_ij ~ R_i C_j / mean(R)
            Vs = state.setdefault("_v", {})
            g2 = g.pow(2)
            if vfull:       # --lr_vfull: one EMA of g^2 per weight (analysis only: N x K floats)
                Vf = state.setdefault("_vf", {})
                Vf[key] = g2 if key not in Vf else vnorm * Vf[key] + (1 - vnorm) * g2
                Vs[key] = (Vf[key].mean(1), Vf[key].mean(0))
                M = M / Vf[key].sqrt().clamp_min(1e-30)
            R, C = g2.mean(1), g2.mean(0)
            if key in Vs:
                R = vnorm * Vs[key][0] + (1 - vnorm) * R; C = vnorm * Vs[key][1] + (1 - vnorm) * C
            if not vfull:
                Vs[key] = (R, C)
                M = M / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
        if propose_only:
            gm = M.abs().mean().clamp_min(1e-12)
            if gate:        # --lr_gate: propose only where the current batch gradient agrees in sign
                M = M * (M.sign() == g.sign())
            if qk_protect and qk_map is not None and key in qk_map:
                # --qk_protect: the rows of head h in wq / wk get their proposals scaled by T_h^-alpha (T_h =
                # learned temperature), so heads the model has sharpened are rewritten less; gm (the threshold
                # scale) stays that of the unscaled M
                attn = qk_map[key]
                T = attn.qk_logscale.detach().exp().float()
                M = M * T.repeat_interleave(attn.head_dim).pow(-qk_protect)[:, None]
            Ms.append(M); state.setdefault("_gms", []).append(gm); continue
        before = l.wpacked.clone() if l.track else None
        if undo:
            from .kernel import unpack_rows, pack_rows
            pm = state.setdefault("_pmove", {})
            w_start = unpack_rows(l.wpacked, l.K).to(torch.int8)
            if key in pm:
                d = unpack_rows(pm[key], l.K).to(torch.int8).float()      # last step's move, -1 / 0 / +1
                back = (d != 0) & (d * g > 0)
                if undo != "g": back &= M.sign() == d       # --undo_g: this batch's gradient alone decides
                if back.any():
                    l.wpacked.copy_(pack_rows(torch.where(back, (w_start.float() - d).clamp(-1, 1).to(torch.int8), w_start)))
                state["_nundo"] = state.get("_nundo", 0) + int(back.sum())
        gm = M.abs().mean().clamp_min(1e-12)
        if gate:            # --lr_gate without look-ahead: flip only where the current batch gradient agrees in sign
            M = M * (M.sign() == g.sign())
        if slow_rank:       # --slow_gate: and only where the slow momentum agrees
            Us, Vs_ = state[("slow", key)]
            M = M * (M.sign() == (Us @ Vs_.T).sign())
        if speed_row:       # --speed_row B: per-row speed reference, slow EMA (decay B) of each row's mean |M|
            rs = state.setdefault("_srow", {}); rm = M.abs().mean(1).clamp_min(1e-12)
            rs[key] = rm if key not in rs else speed_row * rs[key] + (1 - speed_row) * rm
            M = M * (gm / rs[key])[:, None]
        if speed_ref:
            sr = state.setdefault("_sref", {})
            sr[key] = gm if key not in sr else speed_ref * sr[key] + (1 - speed_ref) * gm
            gm = sr[key]
        if pfun_tanh:       # absolute units: p = rate * tanh(|M_ij| / (g_ref v0)), no division by the current size
            gm = (pfun_tanh * state["_gabs"][key]).clamp_min(1e-12)
            M = M.sign() * (l.g_ref * gm) * torch.tanh(M.abs() / (l.g_ref * gm))
        if dither or spend or pshape != "prop":
            from .kernel import unpack_rows, pack_rows
            w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
        if pshape != "prop" and not dither:
            mv = -M.sign()
            ok = ((w0.float() + mv).abs() <= 1) & (M != 0)
            p0 = l.rate * (M.abs() / (l.g_ref * gm)).clamp(max=1) * ok            # the usual rule's flip chances
            if pshape == "flat": w_ = ok.float()
            elif pshape == "inv": w_ = (1 - (M.abs() / (l.g_ref * gm)).clamp(max=1)) * ok
            else:
                R_, C_ = state["_v"][key]
                v_ = R_[:, None] * C_[None, :] / R_.mean().clamp_min(1e-30)
                w_ = ok / (1 + v_ / v_.median().clamp_min(1e-30))
            p = (w_ * (p0.sum() / w_.sum().clamp_min(1e-12))).clamp(max=1)         # same expected number of flips
            if pshape_T:                                                           # cosine blend: rule early, shape late
                lam = 0.5 * (1 - math.cos(math.pi * min(step / pshape_T, 1.0)))
                p = (1 - lam) * p0 + lam * p
            fire = torch.rand(p.shape, device=p.device, generator=torch.Generator(device=p.device).manual_seed(
                8_000_000 + step * 131 + i + flip_seed * 1_000_003)) < p
            l.wpacked.copy_(pack_rows(torch.where(fire, (w0 + mv.to(torch.int8)).clamp(-1, 1), w0).to(torch.int8)))
            del p0, w_, p, fire
        elif dither:
            p = l.rate * (M.abs() / (l.g_ref * gm)).clamp(max=1)
            gen = torch.Generator(device=M.device).manual_seed(7_000_003 + i + flip_seed * 1_000_003)
            u = (torch.rand(M.shape, generator=gen, device=M.device) + (step * 0.6180339887498949) % 1.0) % 1.0
            d = -M.sign().to(torch.int8)
            l.wpacked.copy_(pack_rows(torch.where(u < p, (w0 + d).clamp(-1, 1), w0).to(torch.int8)))
            del p, u, d
        else:
            fused_flip(l.wpacked, M, l.rate, l.g_ref, 8_000_000 + step * 131 + i + flip_seed * 1_000_003, gmean=gm)
        if spend:
            D = (unpack_rows(l.wpacked, l.K).to(torch.int8) - w0).float()
            U, V = state[key]
            state[key] = (U + spend * state["_gm"][key] * (D @ V), V)
            if TIERS and rare_mode == "sum":          # the summed signal: every part pays for the flip
                for rk in TK:
                    U2, V2 = state[rk]
                    state[rk] = (U2 + spend * state["_gm"][key] * (D @ V2), V2)
            del D
        for t, rk in enumerate(TK if rare_mode == "flip" else []):   # each tier proposes its own flips
            from .kernel import unpack_rows
            U2, V2 = state[rk]
            M2 = U2 @ V2.T
            gm2_raw = M2.abs().mean().clamp_min(1e-12)
            if vnorm:
                R, C = state["_v"][key]
                M2 = M2 / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
            gm2 = M2.abs().mean().clamp_min(1e-12)
            if gate:
                M2 = M2 * (M2.sign() == g.sign())
            w1 = unpack_rows(l.wpacked, l.K).to(torch.int8)
            fused_flip(l.wpacked, M2, l.rate * rare_rate, l.g_ref,
                       9_000_000 + 7_777 * t + step * 131 + i + flip_seed * 1_000_003, gmean=gm2)
            if spend:
                D2 = (unpack_rows(l.wpacked, l.K).to(torch.int8) - w1).float()
                state[rk] = (U2 + spend * gm2_raw * (D2 @ V2), V2)
                del D2
            del M2, w1
        if dither or spend or pshape != "prop":
            del w0
        if mom_int8:
            U, V = state[key]
            state[key] = (_q8(U), _q8(V))
        if undo:
            from .kernel import unpack_rows, pack_rows
            pm[key] = pack_rows((unpack_rows(l.wpacked, l.K).to(torch.int8) - w_start).clamp(-1, 1).to(torch.int8))
            del w_start
        if before is not None:
            l._record_flips(before, l.wpacked)
    if dry_vec:             # dry friction on the momentum as one vector: ||M|| -= D x EMA of the gradient norm
        gn = gn2 ** 0.5
        state["_gnorm_d"] = gn if "_gnorm_d" not in state else 0.99 * state["_gnorm_d"] + 0.01 * gn
        keys = [id(l) for l in layers if id(l) in state]
        tot = sum(float(state[k][0].pow(2).sum()) for k in keys) ** 0.5
        D_t = dry_vec
        if dry_warm and dry_start:
            D_t = dry_vec + (dry_start - dry_vec) * 0.5 * (1 + math.cos(math.pi * min(step / dry_warm, 1.0)))
        f = max(tot - D_t * state["_gnorm_d"], 0.0) / max(tot, 1e-30)
        for k in keys:
            U, V = state[k]; state[k] = (U * f, V)
        state["_dry_f"] = f
    for t, (_, td) in enumerate(TIERS):   # each tier's own dry friction (weaker: a longer memory)
        if not td: continue
        gn = tier_gn2[t] ** 0.5
        nk = "_gnorm_r" if t == 0 else f"_gnorm_r{t}"
        state[nk] = gn if nk not in state else 0.99 * state[nk] + 0.01 * gn
        rks = [(("rare", id(l)) if t == 0 else ("rare", t, id(l))) for l in layers]
        rks = [k for k in rks if k in state]
        tot = sum(float(state[k][0].pow(2).sum()) for k in rks) ** 0.5
        f = max(tot - td * state[nk], 0.0) / max(tot, 1e-30)
        for k in rks:
            U2, V2 = state[k]; state[k] = (U2 * f, V2)
    if ncap:                # cap the momentum as one vector: ||M||^2 = sum over layers of ||U||^2 (V orthonormal)
        gn = gn2 ** 0.5
        state["_gnorm"] = gn if "_gnorm" not in state else 0.99 * state["_gnorm"] + 0.01 * gn
        keys = [id(l) for l in layers if id(l) in state]
        tot = sum(float(state[k][0].pow(2).sum()) for k in keys) ** 0.5
        lim = ncap * state["_gnorm"]
        if tot > lim:
            for k in keys:
                U, V = state[k]; state[k] = (U * (lim / tot), V)
        state["_ncap_frac"] = min(1.0, lim / max(tot, 1e-30))
    info = {"lr_cos": sum(cs) / len(cs), "lr_cos_layers": cs} if cs else {}
    rr = state.pop("_rare_ratio", None)
    if rr: info["rare_ratio"] = sum(rr) / len(rr)
    nu = state.pop("_nundo", None)
    if nu is not None: info["undo_n"] = nu      # --undo: last step's moves flipped back this step
    if diag and cs:
        for k, v in D.items():
            info.update(_diag_agg(k, v))
        info.update(_diag_agg("d_cos", cs))
    return (info, Ms) if propose_only else (info or None)


def evidence_step(model, lr_state, ev, r, beta, step, bits, flip_seed=0):
    """--evidence_bits B: a signed per-weight counter (B bits, range +-L with L = 2^(B-1) - 1). Momentum's flip rule
    proposes at L x the rate; a proposal does not flip, it ticks the counter in its direction (opposite ticks
    cancel); a weight flips only when its counter reaches +-L, and the counter then resets. Master's
    threshold-with-memory at B bits per weight (a reference: it is state, not stateless)."""
    from .kernel import unpack_rows
    info, Ms = lowrank_step(model, lr_state, r, beta, step, flip_seed, propose_only=True)
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    for l in layers:
        l.gw = None; l.capture = False
    L = 2 ** (bits - 1) - 1
    n_tick = n_flip = 0
    for i, (l, M) in enumerate(zip(layers, Ms)):
        w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
        c = ev.setdefault(i, torch.zeros_like(w0))
        tmp = l.wpacked.clone()
        fused_flip(tmp, M.contiguous(), l.rate * L, l.g_ref, 8_000_000 + step * 131 + i + flip_seed * 1_000_003,
                   gmean=M.abs().mean().clamp_min(1e-12))
        tick = unpack_rows(tmp, l.K).to(torch.int8) - w0                 # proposed moves (0 where blocked by +-1)
        c += tick; c.clamp_(-L, L)
        fire = c.abs() >= L
        new = torch.where(fire, (w0 + c.sign()).clamp(-1, 1), w0)
        c[fire] = 0
        before = l.wpacked.clone() if l.track else None
        l.wpacked.copy_(pack_rows(new.to(torch.int8)))
        if before is not None: l._record_flips(before, l.wpacked)
        n_tick += int((tick != 0).sum()); n_flip += int((new != w0).sum())
    info["ev_ticks"] = n_tick; info["ev_flips"] = n_flip
    return info


def _sel_features(M, g, w0, idx_mask, i):
    """flip-time features of the proposals of layer i (idx_mask: proposed entries), as in the offline selector:
    |M|/mean|M|, |g|/mean|g|, sign(M)==sign(g), trit -1/0/+1, move goes to 0, row and column rms of M relative
    to the layer's, depth, layer type (7)"""
    f, gg, t = M[idx_mask], g[idx_mask], w0[idx_mask].float()
    fm, gm = M.abs().mean().clamp_min(1e-12), g.abs().mean().clamp_min(1e-12)
    rr = M.pow(2).mean(1).sqrt(); cr = M.pow(2).mean(0).sqrt(); lr_ = M.pow(2).mean().sqrt().clamp_min(1e-12)
    rows, cols = idx_mask.nonzero(as_tuple=True)
    typ = torch.zeros(len(f), 7, device=f.device); typ[:, i % 7] = 1
    X = torch.stack([f.abs() / fm, gg.abs() / gm, ((f > 0) == (gg > 0)).float(), (t == -1).float(), (t == 0).float(),
                     (t == 1).float(), (t != 0).float(), rr[rows] / lr_, cr[cols] / lr_,
                     torch.full_like(f, (i // 7) / 11.0)], 1)
    return torch.cat([X, typ], 1)


def select_step(model, lr_state, sel, r, beta, step, args, flip_seed=0):
    """--select: momentum proposes flips at sel_prop x the rate; a small MLP scores each proposal from flip-time
    features and the top sel_keep fraction is applied. The MLP learns online: every proposal (kept or not) is
    remembered for sel_k steps while each step's batch gradient at its weight is added up; its label is 1 if the
    later gradients still push in the proposed direction (move * sum < 0), 0 if they push back. Memory scales
    with the number of proposals, not with the number of weights."""
    from .kernel import unpack_rows
    info, Ms = lowrank_step(model, lr_state, r, beta, step, flip_seed, propose_only=True)
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    g = [l.gw.float() for l in layers]
    for l in layers:
        l.gw = None; l.capture = False
    dev = g[0].device
    if "mlp" not in sel:
        sel["mlp"] = torch.nn.Sequential(torch.nn.Linear(17, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                         torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(dev)
        sel["opt"] = torch.optim.Adam(sel["mlp"].parameters(), 1e-3)
        sel["buf"] = []; sel["trained"] = 0; sel["mu"] = None
    # 1) this step's gradient goes into every remembered proposal's running sum; mature entries become labels
    for e in sel["buf"]:
        for i in range(len(layers)):
            if e["idx"][i].numel():
                e["acc"][i] += g[i].flatten()[e["idx"][i]]
        e["age"] += 1
    mature = [e for e in sel["buf"] if e["age"] >= args.sel_k]
    sel["buf"] = [e for e in sel["buf"] if e["age"] < args.sel_k]
    if mature:
        X = torch.cat([e["X"][i] for e in mature for i in range(len(layers)) if e["idx"][i].numel()]).float()
        y = torch.cat([((e["mv"][i] * e["acc"][i]) < 0).float() for e in mature for i in range(len(layers))
                       if e["idx"][i].numel()])
        sel["mu"], sel["sd"] = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = (X - sel["mu"]) / sel["sd"]
        with torch.enable_grad():
            for _ in range(args.sel_train_steps):
                bi = torch.randint(0, len(Xn), (min(8192, len(Xn)),), device=dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(sel["mlp"](Xn[bi]).squeeze(1), y[bi])
                sel["opt"].zero_grad(); loss.backward(); sel["opt"].step()
        sel["trained"] += 1
        info["sel_label_good"] = float(y.mean()); info["sel_bce"] = float(loss)
    # 2) propose at sel_prop x the rate, keep the best sel_keep fraction per layer
    entry = {"idx": [], "mv": [], "acc": [], "X": [], "age": 0}
    n_prop = n_keep = 0
    for i, (l, M) in enumerate(zip(layers, Ms)):
        w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
        tmp = l.wpacked.clone()
        fused_flip(tmp, M.contiguous(), l.rate * args.sel_prop, l.g_ref,
                   8_000_000 + step * 131 + i + flip_seed * 1_000_003, gmean=M.abs().mean().clamp_min(1e-12))
        mv_full = unpack_rows(tmp, l.K).to(torch.int8) - w0
        prop = mv_full != 0
        n = int(prop.sum())
        if n == 0:
            for k_ in ("idx", "mv", "acc", "X"): entry[k_].append(torch.empty(0, device=dev))
            continue
        X = _sel_features(M, g[i], w0, prop, i)
        if sel["trained"] >= args.sel_warm and sel["mu"] is not None:
            with torch.no_grad():
                score = sel["mlp"]((X - sel["mu"]) / sel["sd"]).squeeze(1)
        else:
            score = torch.rand(n, device=dev)                  # until the selector has learned: a random subset
        kk = max(1, int(round(n * args.sel_keep)))
        keep = torch.zeros(n, dtype=torch.bool, device=dev); keep[score.topk(kk).indices] = True
        flat = prop.flatten().nonzero(as_tuple=True)[0]
        mv = mv_full.flatten()[flat].float()
        new = w0.flatten().clone(); new[flat[keep]] += mv[keep].to(torch.int8)
        before = l.wpacked.clone() if l.track else None
        l.wpacked.copy_(pack_rows(new.view_as(w0).clamp_(-1, 1)))
        if before is not None: l._record_flips(before, l.wpacked)
        entry["idx"].append(flat); entry["mv"].append(mv); entry["acc"].append(torch.zeros_like(mv))
        entry["X"].append(X.half())
        n_prop += n; n_keep += int(keep.sum())
    sel["buf"].append(entry)
    info["sel_prop"] = n_prop; info["sel_kept"] = n_keep; info["sel_active"] = int(sel["trained"] >= args.sel_warm)
    return info


def multibeta_step(model, states, betas, scores, r, step, flip_seed=0):
    """--multibeta: several low-rank momenta with different memories; per layer the one whose previous value
    predicted this step's gradient best (running average of cos(g_t, M_b,t-1)) proposes the flips."""
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    cos, Mss = [], []
    for b, st in zip(betas, states):
        info_b, Ms = lowrank_step(model, st, r, b, step, flip_seed, propose_only=True)
        cos.append(info_b.get("lr_cos_layers")); Mss.append(Ms)
    for l in layers:
        l.gw = None; l.capture = False
    chosen, cchosen = [0] * len(betas), []
    for i, l in enumerate(layers):
        sc = scores.setdefault(i, [0.0] * len(betas))
        if cos[0] is not None:
            for j in range(len(betas)):
                sc[j] = 0.9 * sc[j] + 0.1 * cos[j][i]
        j = max(range(len(betas)), key=lambda q: sc[q])
        chosen[j] += 1
        if cos[j] is not None: cchosen.append(cos[j][i])
        M = Mss[j][i]
        before = l.wpacked.clone() if l.track else None
        fused_flip(l.wpacked, M.contiguous(), l.rate, l.g_ref, 8_000_000 + step * 131 + i + flip_seed * 1_000_003,
                   gmean=M.abs().mean().clamp_min(1e-12))
        if before is not None: l._record_flips(before, l.wpacked)
    info = {"mb_choice": chosen}
    if cchosen: info["lr_cos"] = sum(cchosen) / len(cchosen)
    return info


def accum_step(model, state, r, beta, step, K, z, flip_seed=0, fresh=None):
    """--accum_flip K: accumulate, then flip once. Momentum is updated every step (rank-r, as lowrank_step) but no
    trit moves; every K-th step each layer flips the weights whose accumulated push stands out from the noise of
    the sum: |M_ij| > z * rms(M) of the layer (pure noise would pass for ~0.3% at z = 3; a consistent signal lifts
    more entries past it, so the number of flips follows the accumulated evidence). Then the momentum restarts
    from zero (the landscape has moved)."""
    from .kernel import unpack_rows
    info, Ms = lowrank_step(model, state, r, beta, step, flip_seed, propose_only=True)
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    for l in layers:
        l.gw = None; l.capture = False
    if (step + 1) % K != 0:
        info["accum_flips"] = 0
        return info
    # candidates: entries whose accumulated push stands out (|M| > z rms per layer); how many of them to flip is
    # chosen by loss: random subsets of 0, 1/1024, 1/256, 1/64 .. 1/2, all of the candidates, each scored on the same 2 fresh
    # batches (forward only); the best is applied. Random subsets spread the flips over rows and columns
    W0, MV = [], []
    for l, M in zip(layers, Ms):
        thr = z * M.pow(2).mean().sqrt()
        w = unpack_rows(l.wpacked, l.K).to(torch.int8)
        move = torch.where(M.abs() > thr, -torch.sign(M), torch.zeros_like(M)).to(torch.int8)
        W0.append(w); MV.append(move)
    gen = torch.Generator(device=W0[0].device).manual_seed(step * 7 + flip_seed)
    U = [torch.rand(w.shape, device=w.device, generator=gen) for w in W0]
    batches_ = [fresh() for _ in range(2)] if fresh is not None else []

    def apply(frac):
        for l, w, mv, u in zip(layers, W0, MV, U):
            sel = (u < frac) & (mv != 0)
            l.wpacked.copy_(pack_rows((w + torch.where(sel, mv, torch.zeros_like(mv))).clamp_(-1, 1)))

    def loss_now():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return sum(model(xb, yb)[1].item() for xb, yb in batches_) / max(len(batches_), 1)

    fracs = [0.0, 1 / 1024, 1 / 256, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
    scores = []
    if batches_:
        for f in fracs:
            apply(f); scores.append(loss_now())
        best = fracs[min(range(len(fracs)), key=lambda i: scores[i])]
    else:
        best = 1.0
    before = [l.wpacked.clone() for l in layers]
    for l, w in zip(layers, W0): l.wpacked.copy_(pack_rows(w))
    apply(best)
    n = 0
    for l, w, b in zip(layers, W0, before):
        n += int((unpack_rows(l.wpacked, l.K).to(torch.int8) != w).sum())
        if l.track: l._record_flips(pack_rows(w), l.wpacked)
        U_, V_ = state[id(l)]
        state[id(l)] = (torch.zeros_like(U_), V_)                   # restart the sum, keep the subspace
    info["accum_candidates"] = int(sum(int((m != 0).sum()) for m in MV))
    info["accum_frac"] = best
    info["accum_flips"] = n
    return info


def mech_step(model, state, step, args, x, y, tail, flip_seed=0):
    """--mech: momentum mechanisms for a landscape that moves under its own flips (no look-ahead). State is kept
    full-size per weight (fp32) to judge the mechanisms; a low-rank version comes after. Gradient convention: a
    flip moves against the sign of the signal. Per layer:

    v1    target point: D = estimated displacement to the minimum (trit units); each step D <- b_D (D - move) +
          (1 - b_D)(-g / h), h one curvature number per layer measured once (same-batch secant after the first
          step); flip toward D (signal -D). Logs the size of the carried part b_D (D - move) vs the new part.
    user  target point from the most recent gradients (short memory, can reverse) + remembered direction F
          corrected by the measured effect of each move + rotation of F onto the target when they disagree:
            delta = g_prev_batch(now) - g_prev_batch(then)   (the change our last move caused; batch noise cancels)
            h <- 0.8 h + 0.2 max(delta.move / |move|^2, eps) (geometry re-measured every step)
            F <- F + gain * delta ; F <- beta F + g           (direction corrected by the move, then new evidence)
            gs <- b_s gs + (1 - b_s) g                        (recent gradients)
            D <- b_D (D - move) + (1 - b_D)(-gs / h)          (target point, reversible)
            if cos(F, -D) < 0: F <- |F| (-D / |D|)            (geometry changed: rotate the memory onto the target)
          flip from F.
    Both start warm from the saved low-rank momentum when there is one (F = U V^T, S = 1/(1 - beta))."""
    from .kernel import unpack_rows
    layers = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    kind, beta = args.mech, args.lr_beta
    b_D = args.mech_beta_d if args.mech_beta_d else (0.97 if kind == "v1" else 0.9)
    need_delta = kind == "user" or any("h" not in state.get(i, {}) for i in range(len(layers)))
    g_now = []
    for l in layers:
        g_now.append(l.gw.float()); l.gw = None
    delta = None
    if need_delta and state.get("prev_xy") is not None:
        tail_g = [None if p.grad is None else p.grad.clone() for p in tail]
        xp, yp = state["prev_xy"]
        for l in layers: l.capture = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(xp, yp)[1].backward()
        delta = [l.gw.float() - gp for l, gp in zip(layers, state["prev_g"])]
        for l in layers: l.gw = None
        for p, g in zip(tail, tail_g): p.grad = g
    for l in layers: l.capture = False
    info = {"mech_flips": 0}
    carried, fresh, resets, cosFD = [], [], 0, []
    new_h = []
    for i, (l, g) in enumerate(zip(layers, g_now)):
        st = state.setdefault(i, {})
        if "F" not in st:
            uv = state.get("warm", {}).get(i)
            st["F"] = (uv[0] @ uv[1].T) if uv is not None else g.clone()
            st["S"] = 1.0 / (1 - beta) if uv is not None else 1.0
            st["gs"] = g.clone()
        mv = st.get("move")
        if delta is not None and mv is not None:
            n = float((mv * mv).sum())
            if n > 0 and kind == "v1":
                st["h"] = max(float((delta[i] * mv).sum()) / n, 1e-12)      # as on the bench: measured once
            elif n > 0:
                # user: curvature from running averages of (delta . move) and |move|^2 (one step's secant can be
                # ~0 or negative), floored so a typical target distance |gs| / h stays within ~2 trit steps
                st["hnum"] = 0.8 * st.get("hnum", 0.0) + 0.2 * float((delta[i] * mv).sum())
                st["hden"] = 0.8 * st.get("hden", 0.0) + 0.2 * n
                floor = 0.5 * float(st["gs"].abs().mean())
                st["h"] = max(st["hnum"] / st["hden"], floor, 1e-12)
            if kind == "user":
                st["F"] += args.mech_gain * delta[i]
        st["F"] = beta * st["F"] + g; st["S"] = beta * st["S"] + 1.0
        st["gs"] = args.mech_beta_s * st["gs"] + (1 - args.mech_beta_s) * g
        if "h" in st:
            h = st["h"]
            if "D" not in st:
                st["D"] = -(st["F"] / st["S"]) / h
                if kind == "user": st["D"].clamp_(-2.0, 2.0)
            else:
                src = g if kind == "v1" else st["gs"]
                c_part = b_D * (st["D"] - (mv if mv is not None else 0.0))
                n_part = (1 - b_D) * (-src / h)
                carried.append(float(c_part.norm())); fresh.append(float(n_part.norm()))
                st["D"] = c_part + n_part
                if kind == "user": st["D"].clamp_(-2.0, 2.0)                # a trit cannot move further
            if kind == "user":
                c = float((st["F"] * (-st["D"])).sum() / (st["F"].norm() * st["D"].norm()).clamp_min(1e-30))
                cosFD.append(c)
                if c < 0:
                    st["F"] = st["F"].norm() * (-st["D"]) / st["D"].norm().clamp_min(1e-30); resets += 1
        sig = (-st["D"]) if (kind == "v1" and "D" in st) else st["F"]
        before = unpack_rows(l.wpacked, l.K).to(torch.int8)
        fused_flip(l.wpacked, sig.contiguous(), l.rate, l.g_ref, 9_000_000 + step * 131 + i + flip_seed * 1_000_003,
                   gmean=sig.abs().mean().clamp_min(1e-12))
        after = unpack_rows(l.wpacked, l.K).to(torch.int8)
        st["move"] = (after - before).float()
        info["mech_flips"] += int(st["move"].abs().sum())
        if l.track: l._record_flips(pack_rows(before), l.wpacked)
        if "h" in st: new_h.append(st["h"])
    still = kind == "user" or any("h" not in state.get(i, {}) for i in range(len(layers)))
    state["prev_xy"] = (x, y) if still else None      # v1 needs the same-batch pass only until h is measured
    state["prev_g"] = g_now if still else None
    if carried:
        info["mech_carried"] = sum(carried) / len(carried); info["mech_fresh"] = sum(fresh) / len(fresh)
    if cosFD:
        info["mech_cosFD"] = sum(cosFD) / len(cosFD); info["mech_resets"] = resets
    if new_h:
        info["mech_h"] = sum(new_h) / len(new_h)
    return info


class TritTracker:
    """Master mode: per step, trits changed (absmean ternarization of the latent weights), flips that
    undo the weight's previous change, and trits never changed since tracking began."""

    def __init__(self, model):
        from .master import MasterTernaryLinear
        self.Ls = [l for l in model.modules() if isinstance(l, MasterTernaryLinear)]
        self.T = [l.ternary_weight()[0].clone() for l in self.Ls]
        self.last = [torch.zeros_like(t) for t in self.T]
        self.touched = [torch.zeros_like(t, dtype=torch.bool) for t in self.T]
        self.n = sum(t.numel() for t in self.T)

    @torch.no_grad()
    def step(self):
        fl = rv = nv = 0
        for i, l in enumerate(self.Ls):
            t = l.ternary_weight()[0]
            d = t - self.T[i]
            ch = d != 0
            fl = fl + ch.sum(); rv = rv + (ch & (d == -self.last[i])).sum()
            self.last[i] = torch.where(ch, d, self.last[i])
            self.touched[i] |= ch
            nv = nv + (~self.touched[i]).sum()
            self.T[i] = t
        fl, rv, nv = torch.stack([fl, rv, nv]).cpu().tolist()
        return {"flip_frac": fl / self.n, "rev_flips": rv, "never_frac": nv / self.n}


class Recorder:
    """--record_sample: every step, for a fixed random sample of the ternary weights (a share FRAC of
    every layer), record the gradient the run saw (bf16), the trit after the step (int8) and the
    momentum after the step (ours: M = U V^T; master: Adam's first moment; bf16). Any window size
    can then be computed offline (scripts/analysis/windows_offline.py). Written in chunks to
    OUT/record/chunk_*.npz; OUT/record/index.npz holds the sample."""

    def __init__(self, model, frac, out_dir, opt=None, seed=12345, chunk=50):
        from .master import MasterTernaryLinear
        self.Ls = [l for l in model.modules() if isinstance(l, (KernelTernaryLinear, MasterTernaryLinear))]
        self.opt, self.chunk = opt, chunk
        self.dir = os.path.join(out_dir, "record"); os.makedirs(self.dir, exist_ok=True)
        gen = torch.Generator().manual_seed(seed)
        self.rows, self.cols, sizes = [], [], []
        for l in self.Ls:
            n = max(1, int(round(frac * l.N * l.K)))
            idx = torch.randperm(l.N * l.K, generator=gen)[:n].sort().values
            dev = next(model.parameters()).device
            self.rows.append((idx // l.K).to(dev)); self.cols.append((idx % l.K).to(dev)); sizes.append(n)
        np.savez(os.path.join(self.dir, "index.npz"), sizes=np.array(sizes),
                 shapes=np.array([[l.N, l.K] for l in self.Ls]),
                 rows=torch.cat(self.rows).cpu().numpy(), cols=torch.cat(self.cols).cpu().numpy())
        self.g = [None] * len(self.Ls)
        self.buf = {"step": [], "g": [], "t": [], "m": []}

    def add(self, i, g):
        self.g[i] = g[self.rows[i], self.cols[i]].float()

    @torch.no_grad()
    def end_step(self, step, model, lr_state):
        from .kernel import unpack_rows
        t, mm = [], []
        for i, l in enumerate(self.Ls):
            r, c = self.rows[i], self.cols[i]
            if self.opt is not None:
                t.append(l.ternary_weight()[0][r, c])
                st = self.opt.state.get(l.weight, {})
                mm.append(st["exp_avg"][r, c].float() if "exp_avg" in st else torch.zeros_like(r, dtype=torch.float32))
            else:
                t.append(unpack_rows(l.wpacked, l.K).to(torch.int8)[r, c])
                if id(l) in lr_state:
                    U, V = lr_state[id(l)]
                    mm.append((U[r].float() * V[c].float()).sum(1))
                else:
                    mm.append(torch.zeros_like(r, dtype=torch.float32))
        g = torch.cat([x if x is not None else torch.zeros_like(self.rows[i], dtype=torch.float32)
                       for i, x in enumerate(self.g)])
        self.buf["step"].append(step)
        self.buf["g"].append(g.to(torch.bfloat16).view(torch.int16).cpu().numpy())
        self.buf["t"].append(torch.cat(t).to(torch.int8).cpu().numpy())
        self.buf["m"].append(torch.cat(mm).to(torch.bfloat16).view(torch.int16).cpu().numpy())
        self.g = [None] * len(self.Ls)
        if len(self.buf["step"]) >= self.chunk:
            self.flush()

    def flush(self):
        if not self.buf["step"]:
            return
        np.savez(os.path.join(self.dir, f"chunk_{self.buf['step'][0]:07d}.npz"),
                 step=np.array(self.buf["step"]), g=np.stack(self.buf["g"]),
                 t=np.stack(self.buf["t"]), m=np.stack(self.buf["m"]))
        self.buf = {"step": [], "g": [], "t": [], "m": []}


class NestedWindow:
    """--window_curve: one window opened at a chosen step and measured (MoveWindow metrics) after
    each of several lengths from that same start, e.g. 1, 2, 5, ..., 100 steps: agreement of the net
    move with the summed gradient as a function of the window length. Metrics c<W>_...; the start
    step is logged as c_start."""

    def __init__(self, sizes, opt=None):
        self.sizes = sorted(set(sizes)); self.opt = opt
        self.w = None; self.start = None

    def open(self, model, lr_state, step):
        self.w = MoveWindow(max(self.sizes), self.opt)
        self.w.open(model, lr_state)
        self.start = step

    def add(self, i, g):
        if self.w is not None:
            self.w.add(i, g)

    @torch.no_grad()
    def tick(self, model, lr_state):
        if self.w is None:
            return {}
        w = self.w
        if w.k == 1:
            self.Dk, self.Sk = {}, {}
        if w.k in self.sizes and w.k < self.sizes[-1]:
            # keep the net move and summed gradient of the first k steps: at the end of the longest
            # window they are scored against its summed gradient (the best "true gradient" on the path)
            self.Dk[w.k] = [(w.trits(l) - t0).to(torch.int8) for l, t0 in zip(w.layers(model), w.T0)]
            self.Sk[w.k] = [x.to(torch.bfloat16) for x in w.S]
        if w.k == 1:
            self.D1 = self.Dk[1]; self.S1 = [x.float() for x in self.Sk[1]]
        if w.k not in self.sizes:
            return {}
        w.S_prev = None
        extra = {k: [] for k in ("agree1", "cos1", "agree1_later", "cos1_later")}
        if w.k > 1:
            cos = lambda a, b: F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0)
            for d, S, S1 in zip(self.D1, w.S, self.S1):
                d = d.float(); nz = d != 0
                if not nz.any():
                    continue
                later = S - S1
                extra["agree1"].append(((d * S)[nz] < 0).float().mean())
                extra["cos1"].append(cos(d, -S))
                extra["agree1_later"].append(((d * later)[nz] < 0).float().mean())
                extra["cos1_later"].append(cos(d, -later))
        if w.k == self.sizes[-1]:
            cos = lambda a, b: F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0)
            for kk, D in self.Dk.items():
                ag, cs, agl, csl = [], [], [], []
                for d, S, Sk in zip(D, w.S, self.Sk[kk]):
                    d = d.float(); nz = d != 0
                    if not nz.any():
                        continue
                    later = S - Sk.float()
                    ag.append(((d * S)[nz] < 0).float().mean()); cs.append(cos(d, -S))
                    agl.append(((d * later)[nz] < 0).float().mean()); csl.append(cos(d, -later))
                for name, v in (("agree", ag), ("cos", cs), ("agree_later", agl), ("cos_later", csl)):
                    if v:
                        extra.setdefault(f"d{kk}_{name}", []).extend(v)
        r = w.close(model, lr_state)
        out = {("c%d_" % w.k) + k[2:]: v for k, v in r.items() if k != "w_steps"}
        for k, v in extra.items():
            if v:
                out.update({("c%d_" % w.k) + kk: vv for kk, vv in
                            _diag_agg(k, torch.stack(v).cpu().tolist()).items()})
        out["c_start"] = self.start
        if self.w.k >= self.sizes[-1]:
            self.w = None                      # free the window's buffers until the next start
            self.Dk, self.Sk = {}, {}
        return out


class MultiWindow:
    """Several MoveWindows of different lengths over the same run (and/or a Recorder). One size keeps
    the plain w_ metric names; several prefix them w<N>_."""

    def __init__(self, sizes, opt=None, recorder=None, curve=None, curve_at=None):
        self.ws = [MoveWindow(n, opt) for n in sizes]
        self.prefix = len(sizes) > 1
        self.rec = recorder
        self.curve = NestedWindow(curve, opt) if curve else None
        self.curve_at = curve_at               # steps after which a curve window opens; None = at start

    def layers(self, model):
        return MoveWindow(1).layers(model)

    def open(self, model, lr_state, step=None):
        for w in self.ws:
            w.open(model, lr_state)
        if self.curve is not None and self.curve_at is None:
            self.curve.open(model, lr_state, step)

    def add(self, i, g):
        for w in self.ws:
            w.add(i, g)
        if self.rec is not None:
            self.rec.add(i, g)
        if self.curve is not None:
            self.curve.add(i, g)

    def tick(self, model, lr_state, step=None):
        if self.rec is not None:
            self.rec.end_step(step, model, lr_state)
        rec = {}
        if self.curve is not None:
            rec.update(self.curve.tick(model, lr_state))
            if self.curve_at is not None and step in self.curve_at:
                self.curve.open(model, lr_state, step)
        for w in self.ws:
            if w.due(None):
                r = w.close(model, lr_state)
                if self.prefix:
                    r = {("w%d_" % w.n) + k[2:]: v for k, v in r.items()}
                rec.update(r)
                w.open(model, lr_state)
        return rec


class MoveWindow:
    """Per window of N steps, per ternary layer: D = net trit move, S = sum of the first-pass
    gradients the run saw, M0 = momentum at the window start. Logs (mean over layers, and per layer
    type) the moved share; sign agreement of the moved trits with -S and -M0 and cos(D, -S),
    cos(D, -M0); cos(S, M0); coherence |S|^2 / sum |g_t|^2 (1 = pure noise, N = one consistent
    direction); cos(S, S of the previous window).
    Master mode (opt = its AdamW): trits are the absmean ternarization of the latent weights, M0 is
    Adam's first moment, and cos(latent move, -S) is logged too."""

    def __init__(self, n, opt=None):
        self.n = n; self.S_prev = None; self.opt = opt

    def layers(self, model):
        from .master import MasterTernaryLinear
        return [l for l in model.modules() if isinstance(l, (KernelTernaryLinear, MasterTernaryLinear))]

    def trits(self, l):
        from .kernel import unpack_rows
        if self.opt is not None:
            return l.ternary_weight()[0]
        return unpack_rows(l.wpacked, l.K).to(torch.int8)

    def mom(self, l, lr_state):
        if self.opt is not None:
            st = self.opt.state.get(l.weight, {})
            return st["exp_avg"].to(torch.bfloat16).clone() if "exp_avg" in st else None
        return ((lr_state[id(l)][0] @ lr_state[id(l)][1].T).to(torch.bfloat16)
                if id(l) in lr_state else None)

    @torch.no_grad()
    def open(self, model, lr_state):
        Ls = self.layers(model)
        dev = next(model.parameters()).device
        self.T0 = [self.trits(l).clone() for l in Ls]
        self.M0 = [self.mom(l, lr_state) for l in Ls]
        self.W0 = [l.weight.detach().float().clone() for l in Ls] if self.opt is not None else None
        self.S = [torch.zeros(l.N, l.K, device=dev) for l in Ls]
        self.sq = [torch.zeros((), device=dev) for l in Ls]
        self.k = 0

    def add(self, i, g):
        self.S[i] += g
        self.sq[i] += (g.float() ** 2).sum()
        if i == 0:
            self.k += 1

    def due(self, step):
        return self.k >= self.n

    @torch.no_grad()
    def close(self, model, lr_state):
        cos = lambda a, b: F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0)
        out = {k: [] for k in ("w_moved", "w_agree_S", "w_cos_S", "w_agree_M", "w_cos_M",
                               "w_cos_SM", "w_coh", "w_cos_Sprev", "w_cos_lat_S")}
        for i, l in enumerate(self.layers(model)):
            D = (self.trits(l) - self.T0[i]).float()
            if self.W0 is not None:
                out["w_cos_lat_S"].append(cos(l.weight.detach().float() - self.W0[i], -self.S[i]))
            nz = D != 0
            S = self.S[i]
            out["w_moved"].append(nz.float().mean())
            out["w_agree_S"].append(((D * S)[nz] < 0).float().mean())
            out["w_cos_S"].append(cos(D, -S))
            out["w_coh"].append((S ** 2).sum() / self.sq[i].clamp_min(1e-30))
            if self.M0[i] is not None:
                M = self.M0[i].float()
                out["w_agree_M"].append(((D * M)[nz] < 0).float().mean())
                out["w_cos_M"].append(cos(D, -M))
                out["w_cos_SM"].append(cos(S, M))
            if self.S_prev is not None:
                out["w_cos_Sprev"].append(cos(S, self.S_prev[i]))
        self.S_prev = self.S
        rec = {"w_steps": self.k}
        for k, v in out.items():
            if v:
                rec.update(_diag_agg(k, torch.stack(v).cpu().tolist()))
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="latent mode: 0=stateless SGD+SR; ~0.9=Lion-style int8 momentum")
    ap.add_argument("--mode", default="kernel",
                    choices=["kernel", "evidence", "flip", "stateless", "latent",
                             "master", "fp32"],
                    help="kernel=stateless flips, Triton GEMM; evidence=kernel + "
                         "2-bit per-weight counter; flip=predicted flips + int8 "
                         "evidence (torch); stateless=zero-accumulator (torch); "
                         "latent=int8 master-free latent; master=latent master "
                         "weights + STE + AdamW (the b->inf baseline)")
    ap.add_argument("--master_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="master mode: latent weight dtype (bf16 rounds away small "
                         "AdamW steps; fp32 is the honest ceiling)")
    ap.add_argument("--theta", type=float, default=24.0, help="flip fire threshold")
    ap.add_argument("--rate", type=float, default=2e-2, help="stateless flip rate")
    ap.add_argument("--lowrank", type=int, default=0,
                    help="flip signal = rank-r momentum of each layer's gradient (M ~ U V^T, "
                         "r*(N+K) floats per layer) instead of the current gradient")
    ap.add_argument("--lr_beta", type=float, default=0.97, help="decay of the low-rank momentum")
    ap.add_argument("--move_window", default="",
                    help="every N steps log how the net trit move of the window relates to the "
                         "summed gradient of the window and to the momentum at its start "
                         "(needs --lookahead; see MoveWindow). Several sizes: '1,2,10' "
                         "(metrics then prefixed w1_, w2_, ...)")
    ap.add_argument("--window_curve", default="",
                    help="window lengths, e.g. '1,2,5,10,20,50,100': one nested window measured "
                         "after each length from the same start (see NestedWindow)")
    ap.add_argument("--window_curve_at", default="",
                    help="steps after which a --window_curve window opens (default: at the start)")
    ap.add_argument("--record_sample", type=float, default=0.0,
                    help="record per step the gradient, trit and momentum of this share of the "
                         "ternary weights (fixed random sample), for any-window offline analysis")
    ap.add_argument("--profile", type=int, default=0,
                    help="profile this many steps (after 20 warm-up steps), print the tables, exit")
    ap.add_argument("--track_reversals", action="store_true",
                    help="log flips that undo a weight's previous change, and net displacement "
                         "from the (resumed) start")
    ap.add_argument("--lr_gate", action="store_true",
                    help="propose from M only where the current batch gradient has the same sign")
    ap.add_argument("--lr_refresh", type=int, default=0,
                    help="replace this many of the weakest momentum directions every "
                         "--lr_refresh_every steps with top directions of the gradient outside the subspace")
    ap.add_argument("--lr_refresh_every", type=int, default=10)
    ap.add_argument("--qk_protect", type=float, default=0.0,
                    help="with --qk_temp: scale wq/wk flip proposals of head h by T_h^-alpha (alpha = this)")
    ap.add_argument("--qk_temp", action="store_true",
                    help="learnable per-head attention temperature (log-scale per head, no weight decay)")
    ap.add_argument("--lowrank_mag", default="",
                    help="low-rank float magnitude beyond the trits: 'add:R' (W = bT + AB^T) or 'mul:R' "
                         "(W = bT o (1 + AB^T))")
    ap.add_argument("--mag_wd", type=float, default=0.0, help="weight decay of the --lowrank_mag parameters")
    ap.add_argument("--mag_lr_mult", type=float, default=1.0, help="LR multiplier of the --lowrank_mag parameters")
    ap.add_argument("--mag_cap", type=float, default=0.0,
                    help="additive magnitude: scale the adapter output down to at most this x the rms of the "
                         "trit path's output, per layer and forward (0 = off)")
    ap.add_argument("--lr_mask_stuck", action="store_true",
                    help="zero the gradient at weights already at +-1 in its push direction before it enters M")
    ap.add_argument("--lr_vnorm", type=float, default=0.0,
                    help="propose from M / sqrt(v) with v a factored (row x column) EMA of g^2, this decay "
                         "(0 = off)")
    ap.add_argument("--lr_spend", type=float, default=0.0,
                    help="kept flips consume c * mean|M| of the momentum at their entries (0 = off)")
    ap.add_argument("--lr_diag", action="store_true",
                    help="log per step how the low-rank momentum relates to the gradient "
                         "(|g|, |M|, sign agreement, subspace share, cos(g, g') across batches)")
    ap.add_argument("--lr_adapt", action="store_true",
                    help="low-rank momentum decay = lr_beta * cos(g, M): reset at 90 deg, "
                         "reflected at 180 deg")
    ap.add_argument("--master_bits", type=int, default=0,
                    help="master mode degraded: latent weights stored on a (2^K - 1)-level grid (stochastic "
                         "rounding) after every step; 0 = full precision")
    ap.add_argument("--evidence_bits", type=int, default=0,
                    help="per-weight signed evidence counter of this many bits: momentum's proposed flips tick it, "
                         "a flip happens only at +-(2^(bits-1) - 1) net ticks, then it resets (0 = off)")
    ap.add_argument("--select", action="store_true",
                    help="online-learned flip selector: propose at sel_prop x the rate, keep the top sel_keep by an MLP "
                         "trained on whether later gradients still push in the proposed direction")
    ap.add_argument("--sel_prop", type=float, default=2.0)
    ap.add_argument("--sel_keep", type=float, default=0.5)
    ap.add_argument("--sel_k", type=int, default=8, help="steps of later gradients summed for a proposal's label")
    ap.add_argument("--sel_warm", type=int, default=5, help="training rounds before the selector is used")
    ap.add_argument("--sel_train_steps", type=int, default=30)
    ap.add_argument("--speed_ref", type=float, default=0.0,
                    help="momentum flips: divide M by a slow EMA (this decay, e.g. 0.995) of its mean |M| instead of "
                         "by its current mean, so the flip count follows the momentum's size (0 = off)")
    ap.add_argument("--m_factv", action="store_true",
                    help="master: second moment factored (row x column, our rule's normalization) instead of per weight")
    ap.add_argument("--m_gate", action="store_true",
                    help="master: latent update only where this batch's gradient agrees in sign with m (our gate)")
    ap.add_argument("--m_rank", type=int, default=0, help="master: first moment kept at rank R (like our momentum)")
    ap.add_argument("--m_beta1", type=float, default=0.0, help="master: Adam beta1 for the latent weights (default 0.9)")
    ap.add_argument("--m_beta2", type=float, default=0.0, help="master: Adam beta2 for the latent weights (default 0.95)")
    ap.add_argument("--m_clamp", type=float, default=0.0, help="master: latent clamped to |W| <= C * gamma after each step")
    ap.add_argument("--m_leak", type=float, default=0.0,
                    help="master: latent's offset from its trit centre decays with a time constant of TAU steps")
    ap.add_argument("--m_snap", action="store_true", help="master: a latent whose trit changed is set to the new centre")
    ap.add_argument("--m_lag", type=float, default=0.0,
                    help="master: the trits follow the latent only with probability Q per step (rate-limited firing)")
    ap.add_argument("--m_gfix", type=int, default=0, help="master: freeze each matrix's absmean scale at step S")
    ap.add_argument("--lr_vfull", action="store_true",
                    help="momentum flips: Adam step normalized by a per-weight EMA of g^2 (N x K floats) instead of "
                         "the factored row x column one (analysis: is the factored normalization what we lack?)")
    ap.add_argument("--ts", action="store_true",
                    help="two timescales (ts_step): short low-rank momentum for the direction, long low-rank accumulator "
                         "of Adam-normalized steps, a trit moves when the accumulator crosses theta (no flip rate)")
    ap.add_argument("--ts_theta", type=float, default=16.0, help="--ts: firing threshold (latent half-bin / peak lr)")
    ap.add_argument("--ts_rescale", action="store_true", help="--ts: fire where |A| >= theta * s and spend theta * s, "
                    "s = sqrt(EMA of the share of each step's energy the rank-r basis keeps), clamped to [0.25, 1]: "
                    "undoes the shrinking of the accumulator by its projection")
    ap.add_argument("--ts_refresh", type=int, default=0, help="--ts: every R steps replace the weakest columns of the "
                    "accumulator's basis by the top directions of the step's evidence outside the basis")
    ap.add_argument("--ts_refresh_frac", type=float, default=0.125, help="--ts_refresh: share of the columns replaced")
    ap.add_argument("--ts_table", type=float, default=0.0, help="--ts: a sparse table of F * N * K exact entries on top of "
                    "the low-rank accumulator: each step's evidence outside the basis and the unprojected part of every "
                    "spend, kept where |low-rank + table| is largest (memory ~6 F bytes per weight)")
    ap.add_argument("--ts_exact_spend", action="store_true", help="--ts: the part of each move's spend that the basis "
                    "does not hold is kept in a sparse table on the moved entries until it fades below 0.5 (<= 2%% of entries)")
    ap.add_argument("--ts_tiles", type=int, default=1, help="--ts: the accumulator as T x T tiles, each of rank lowrank / T "
                    "(the same memory as one rank-lowrank pair); fp32 state only")
    ap.add_argument("--ts_rank_sched", default="", help="--ts: R0:R1, both ranks move linearly from R0 to R1 over training")
    ap.add_argument("--ts_spend", type=float, default=1.0,
                    help="--ts: a move spends this x theta (1: to the new trit's centre, like master --m_snap; 2: stays at "
                         "the boundary it crossed, like master)")
    ap.add_argument("--gemm", default="triton", choices=["triton", "cublas"],
                    help="ternary GEMMs: our Triton kernels, or cuBLAS bf16 on trits unpacked per call (faster on A100)")
    ap.add_argument("--ts_orth", default="qr", choices=["qr", "chol", "chol_cs", "chol64"],
                    help="--ts: subspace orthonormalisation by Householder QR or two-pass Cholesky-QR (faster)")
    ap.add_argument("--ts_qfmt", default="", help="--ts: storage format of the low-rank matrices: int8 | int8det | "
                    "int8row | int4 | bf16 | fp16 | fp16s (per-column scale) | fp8 | fpmK (K mantissa bits) (simulated; --ts_int8 = int8)")
    ap.add_argument("--ts_qfmt_m", default="", help="--ts: storage format of the short momentum (direction) alone")
    ap.add_argument("--ts_qfmt_a", default="", help="--ts: storage format of the accumulator alone")
    ap.add_argument("--ts_qfmt_aV", default="", help="--ts: storage format of the accumulator's basis V alone")
    ap.add_argument("--ts_store_m", default="", help="--ts: REAL storage of the short momentum (int8 | int8det | fp8 | "
                    "fp16 | fp16s | bf16), as opposed to the simulated --ts_qfmt_m")
    ap.add_argument("--ts_store_a", default="", help="--ts: REAL storage of the accumulator (same formats, and fp12s: "
                    "fp16s rounded to 6 mantissa bits, two values in 3 bytes)")
    ap.add_argument("--ts_rank_kv", type=int, default=0, help="--ts: rank of both states for the attention k / v "
                    "projections (0: --lowrank / --ts_rank_s); with grouped-query attention these are narrow, so a full "
                    "rank there costs more than dense storage")
    ap.add_argument("--ts_shadows", default="", help="--ts: shadow states at these ranks (0 = full rank), fed the same "
                    "gradients and spent by the run's moves; with --ts_diag N compared with the full-rank shadow every N "
                    "steps (correlation, energy, sign agreement, which weights would move, momentum agreement, spectrum)")
    ap.add_argument("--ts_shadow_layers", default="", help="--ts_shadows: only these layer indices (block b: q = 7b, "
                    "down = 7b + 6); default all")
    ap.add_argument("--ts_dump_steps", default="", help="--ts_shadows: save the matrices at these update counts")
    ap.add_argument("--ts_dump_layers", default="0,6", help="--ts_dump_steps: layer indices (block 0: q = 0, down = 6)")
    ap.add_argument("--ts_sketch", type=float, default=0.0, help="--ts: a random sketch of each step u joins the "
                    "accumulator's basis update with this share of its norm (lets new directions in at low rank)")
    ap.add_argument("--ts_resid", action="store_true", help="--ts: feed what the accumulator's projection discarded back "
                    "into the next step (dense N x K per layer: an upper-bound test, not sublinear)")
    ap.add_argument("--ts_rank_s_types", default="", help="--ts: per-layer-type rank of the short momentum only, e.g. "
                    "q:128,k:128,v:128,o:128,gate:384,up:384,down:384 (the momentum is lower-rank than the accumulator)")
    ap.add_argument("--ts_rank_types", default="", help="--ts: per-layer-type ranks, e.g. q:64,k:64,v:96,o:64,gate:160,"
                    "up:160,down:128 (both states; types not listed use --lowrank / --ts_rank_s)")
    ap.add_argument("--ts_shared_v", action="store_true", help="--ts: the short momentum and the accumulator share one "
                    "orthonormal basis (a quarter less state; fp32 state only)")
    ap.add_argument("--ts_diag", type=int, default=0, help="--ts: every N steps log what the rank-r projections drop "
                    "(dg_m_drop, dg_a_drop, dg_u_kept) and, with --ts_shadow, the full-rank accumulator's moves")
    ap.add_argument("--ts_shadow", action="store_true", help="--ts: keep a full-rank accumulator fed the same steps and "
                    "spent by the same moves (diagnostic: dense N x K per layer)")
    ap.add_argument("--two_stage", type=float, default=0.0, help="BitNet-b1.58-style schedule: stage 1 cosine from the "
                    "peak to half of it, at the midpoint restart at F x peak and cosine to min_lr; weight decay 0 in "
                    "stage 2 (F e.g. 0.4; 0: plain cosine)")
    ap.add_argument("--lean_tail", action="store_true", help="float tail in bf16 (embedding, adapter) / fp16 (gains, "
                    "scales) with LeanAdam: int8 first moment, factored second moment, stochastic-rounding updates")
    ap.add_argument("--mem_report", action="store_true", help="log the MiB of trits, float tail, its gradients and "
                    "optimizer state, TTF state on GPU / host, allocated and peak, at steps 10 / 50 / 100 / 1000")
    ap.add_argument("--ts_offload", action="store_true", help="--ts: keep the state in pinned host memory; each layer's "
                    "state is copied to the GPU for its update and back (same stream: ordered, no host sync)")
    ap.add_argument("--ts_offload_async", action="store_true", help="--ts_offload with the copies on side streams: the "
                    "next layer's state is prefetched while this layer updates, write-backs overlap with compute")
    ap.add_argument("--ts_qwhich", default="all", choices=["all", "m", "a"],
                    help="--ts_qfmt applies to both matrices / the short momentum only / the accumulator only")
    ap.add_argument("--ts_qnoV", action="store_true", help="--ts_qfmt: keep the orthonormal bases V in fp32")
    ap.add_argument("--ts_qdiag", type=int, default=0,
                    help="--ts: every N steps log the stored numbers' crest factor, sub-resolution change share, "
                         "change / int8 step and log2 dynamic range (qd_* in metrics.jsonl)")
    ap.add_argument("--ts_int8", action="store_true",
                    help="--ts: both low-rank matrices (short momentum and accumulator) kept on an int8 grid (per column)")
    ap.add_argument("--ts_fused", action="store_true",
                    help="--ts: each layer's update runs inside the backward, right after the layer's input gradient, so "
                         "the fp32 gradient of only one layer exists at a time (saves N*K*4 bytes over all layers)")
    ap.add_argument("--ts_anneal", type=float, default=1.0,
                    help="--ts: the accumulator's steps scale with (lr_t / lr_peak)^P (1: like master's latent; 0.5 milder)")
    ap.add_argument("--bop", action="store_true",
                    help="baseline: ternary Bop (one-timescale latent-free flips, per-weight EMA; needs --lowrank 1 for the "
                         "gradient capture, the rank is unused)")
    ap.add_argument("--bop_gamma", type=float, default=1e-3, help="--bop: EMA rate of m (memory 1 / gamma steps)")
    ap.add_argument("--bop_tau", type=float, default=0.016, help="--bop: threshold on |m_hat / sqrt(v)| (or |m| with --bop_raw)")
    ap.add_argument("--bop_raw", action="store_true", help="--bop: threshold the raw m (original Bop), no normalisation")
    ap.add_argument("--bop_bc", action="store_true", help="--bop: bias-correct m (off in Bop)")
    ap.add_argument("--ts_tau_anneal", action="store_true",
                    help="--ts: the leak shrinks with the accumulator's steps (decay 1 - ratio / tau): memory counted in "
                         "distance travelled, so the accumulator's level does not sink below theta as the lr decays")
    ap.add_argument("--ts_tau", type=float, default=300.0, help="--ts: the accumulator's leak time constant (steps)")
    ap.add_argument("--ts_rank_s", type=int, default=128, help="--ts: rank of the short momentum")
    ap.add_argument("--ts_b1", type=float, default=0.9, help="--ts: decay of the short momentum")
    ap.add_argument("--ts_b2", type=float, default=0.95, help="--ts: decay of the factored second moment")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for fp32 matmuls (the low-rank momentum's products); A100 fp32 is 19.5 TFLOPs, TF32 156")
    ap.add_argument("--undo_g", action="store_true",
                    help="undo on this batch's gradient alone (no momentum condition: with spend it mostly holds anyway)")
    ap.add_argument("--undo", action="store_true",
                    help="momentum flips: flip back last step's moves that this batch's gradient and the updated "
                         "momentum both call uphill (one-step buffer of the moves)")
    ap.add_argument("--slow_gate", type=int, default=0,
                    help="momentum flips: rank of a second, slow momentum; flip only where it agrees in sign (0 = off)")
    ap.add_argument("--slow_beta", type=float, default=0.999, help="decay of the --slow_gate momentum")
    ap.add_argument("--dither_ld", action="store_true",
                    help="momentum flips: low-discrepancy flip draw per weight (fixed hash + step * golden ratio)")
    ap.add_argument("--dry_vec", type=float, default=0.0,
                    help="momentum flips: dry friction on the whole momentum, ||M|| -= D x gradient norm per step")
    ap.add_argument("--rare_rank", type=int, default=0,
                    help="momentum flips: rank of a second momentum fed the gradient outside the main one's subspace (0 = off)")
    ap.add_argument("--rare_dry", type=float, default=0.01, help="dry friction of the --rare_rank momentum (decay 1)")
    ap.add_argument("--rare_mode", default="sum", choices=["sum", "flip"],
                    help="--rare_rank: add it to the flip signal, or let it propose its own flips")
    ap.add_argument("--tiers", default="",
                    help="momentum flips: chain of extra momenta 'R:D,R:D,...' (rank, dry friction), each fed what the ones before miss")
    ap.add_argument("--pshape", default="prop", choices=["prop", "flat", "inv", "cheap"],
                    help="momentum flips: which weights get the flips (same expected count, direction and gate as prop)")
    ap.add_argument("--pshape_sched", default="", choices=["", "cos"],
                    help="--pshape: blend from the usual rule (start) to the shape (end) on a cosine")
    ap.add_argument("--rare_weight", type=float, default=0.0,
                    help="--rare_mode sum: add the extra momenta at this x the main one's mean size (0 = their own size)")
    ap.add_argument("--rare_rate", type=float, default=0.5, help="--rare_mode flip: its flip rate as a share of the rate")
    ap.add_argument("--mom_int8", action="store_true",
                    help="momentum flips: round the momentum factors U, V to int8 (per-column scale) after every step")
    ap.add_argument("--dry_start", type=float, default=0.0,
                    help="--dry_vec: friction at step 0 (strong = short memory), easing to --dry_vec over --dry_warm steps")
    ap.add_argument("--dry_warm", type=int, default=0, help="steps over which --dry_start eases to --dry_vec (cosine)")
    ap.add_argument("--dry_w", type=float, default=0.0,
                    help="momentum flips: dry friction per weight, |M_ij| -= D x mean|g| per step (full M, then rank r)")
    ap.add_argument("--spend", type=float, default=0.0,
                    help="momentum flips (no look-ahead): a flip consumes c * mean|M| of the momentum at its entry")
    ap.add_argument("--grav_up", type=float, default=0.0,
                    help="asymmetric gravity: where the batch gradient opposes a weight's momentum, decay it with this "
                         "instead of --lr_beta (e.g. 0.5; 0 = off)")
    ap.add_argument("--mom_ncap", type=float, default=0.0,
                    help="cap the whole momentum (all layers as one vector) at this x a slow EMA of the total gradient "
                         "norm (0 = off); with --lr_beta 1: the user's frictionless velocity with a speed limit")
    ap.add_argument("--pfun_tanh", type=float, default=0.0,
                    help="flip chance rate * tanh(|M| / (g_ref v0)), v0 = this x a slow EMA of the layer's mean |g| "
                         "(absolute gradient units instead of dividing by the momentum's current size; 0 = off)")
    ap.add_argument("--speed_row", type=float, default=0.0,
                    help="momentum flips: per-row speed reference, divide each row of M by a slow EMA (this decay) of "
                         "that row's mean |M| (0 = off; N floats per layer)")
    ap.add_argument("--adapt_rate", action="store_true",
                    help="flip-rate controller keeping the running cos(g_t, M_t-1) near adapt_rate_target")
    ap.add_argument("--adapt_rate_target", type=float, default=0.03)
    ap.add_argument("--adapt_rate_gain", type=float, default=0.3)
    ap.add_argument("--multibeta", default="",
                    help="comma-separated momentum decays, e.g. 0.8,0.95,0.99: per layer the best predictor flips")
    ap.add_argument("--accum_flip", type=int, default=0,
                    help="accumulate momentum K steps without flips, then flip where |M| > accum_z * rms(M) per "
                         "layer, and restart the momentum (0 = off)")
    ap.add_argument("--accum_z", type=float, default=3.0)
    ap.add_argument("--mech", default="", choices=["", "v1", "user"],
                    help="momentum mechanism without look-ahead (full-size state, see mech_step): v1 = target "
                         "point; user = recent-gradient target + move-corrected direction + rotation on disagreement")
    ap.add_argument("--mech_beta_d", type=float, default=0.0, help="target-point decay (0 = 0.97 for v1, 0.9 user)")
    ap.add_argument("--mech_beta_s", type=float, default=0.8, help="user: memory of the recent-gradient average")
    ap.add_argument("--mech_gain", type=float, default=1.0,
                    help="user: gain of the move correction F += gain * delta (the bench's 33 overshot)")
    ap.add_argument("--lookahead_off", default="",
                    help="steps without look-ahead, 'A:B' = off for A <= step < B (flips from the proposals "
                         "directly); e.g. 915:4000 = on to 30M, off to 131M, on again")
    ap.add_argument("--lookahead_xbatch", action="store_true",
                    help="take the look-ahead pass(es) on fresh batches instead of the "
                         "training batch (each pass a new batch)")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="look-ahead flip filter: propose flips, keep those whose "
                         "midpoint slope (g + g at W+Delta) still descends; value = "
                         "number of re-check passes (each one extra fwd/bwd)")
    ap.add_argument("--rate_search", action="store_true",
                    help="line-search the flip rate: every --rs_every steps, try "
                         "rate x each of --rs_factors on the step's gradient, score each "
                         "on a held-out train batch, keep the best (hill-climbs a global "
                         "multiplier; no per-weight state)")
    ap.add_argument("--rs_every", type=int, default=50)
    ap.add_argument("--rs_factors", default="0.5,1,2")
    ap.add_argument("--rs_batch", type=int, default=4, help="tuning batch (sequences)")
    ap.add_argument("--rs_min", type=float, default=0.01)
    ap.add_argument("--rs_max", type=float, default=8.0)
    ap.add_argument("--hump", action="store_true",
                    help="flip probability ~ measured net gain per flip: "
                         "(g/g_ref)*(1-(g/g0)^alpha), rising through the peak band "
                         "and falling to 0 at g0 (g in units of mean|g|)")
    ap.add_argument("--hump_g0", type=float, default=37.0)
    ap.add_argument("--hump_alpha", type=float, default=1.8)
    ap.add_argument("--inv_prob", action="store_true",
                    help="reverse the probability ramp: the SMALLEST gradients flip "
                         "most often (p = (1 - min(|g|/g_ref,1)) * rate)")
    ap.add_argument("--norm", default="tensor", choices=["tensor", "row"],
                    help="normalise the flip threshold by the whole tensor's mean|g| "
                         "(default) or by each output row's own mean|g|")
    ap.add_argument("--g_ref", type=float, default=3.0,
                    help="flip-probability saturation point, in units of mean|g|: "
                         "p = min(|g|/(g_ref*mean|g|), 1) * rate. Below the knee the "
                         "expected step is proportional to the gradient (SGD); above "
                         "it, magnitude is discarded (signSGD)")
    ap.add_argument("--ev_bits", type=int, default=2,
                    help="evidence mode: counter bit-depth (2,3,4...)")
    ap.add_argument("--int8", action="store_true",
                    help="kernel/evidence: s8 tensor-core path (exact int32 forward, "
                         "8-bit gy in dx, int8 dw)")
    ap.add_argument("--int8_dx", action="store_true",
                    help="also run grad_x on the s8 kernel (default: bf16 dx kernel)")
    ap.add_argument("--dw_mode", default="dense", choices=["int8", "cublas", "sign", "dense"],
                    help="weight-gradient kernel: int8 tensor cores, sign XNOR outer "
                         "product, or dense torch matmul")
    ap.add_argument("--rate_schedule", default="const",
                    choices=["const", "cosine", "linear", "exp"],
                    help="Arm A: decay the flip rate over the run. exp = "
                         "rate * exp(-step/--rate_tau): halves repeatedly, "
                         "approaches zero asymptotically without reaching it")
    ap.add_argument("--rate_tau", type=float, default=1000.0,
                    help="exp schedule: steps per e-folding")
    ap.add_argument("--abs_scale", action="store_true",
                    help="Arm B: freeze the flip-threshold denominator after "
                         "--calib_steps (absolute scale; flip rate can then fall)")
    ap.add_argument("--calib_steps", type=int, default=200)
    ap.add_argument("--stop_after", type=int, default=None,
                    help="stop after this many steps WITHOUT changing the LR / flip "
                         "schedules (which still span --steps); evaluates on exit")
    ap.add_argument("--flip_lockout", type=int, default=0,
                    help="per-weight lockout: a weight may flip at most once per N "
                         "steps (1 bit/weight). 0 = off")
    ap.add_argument("--lockout_mode", default="once", choices=["once", "noreversal"],
                    help="once = one flip per epoch; noreversal = after the first "
                         "flip only the same direction is allowed")
    ap.add_argument("--err_feedback", action="store_true",
                    help="spatial error feedback: push each layer's unapplied flip "
                         "demand into grad_x (kernel mode, grad_accum 1 only)")
    ap.add_argument("--ef_alpha", type=float, default=1.0,
                    help="mixing weight of the error-feedback term in gy")
    ap.add_argument("--rate_peak", type=float, default=0.0,
                    help="with --rate_warmup: ramp linearly from --rate to this peak over "
                         "the warmup, then run --rate_schedule from the peak (0 = off)")
    ap.add_argument("--rate_warmup", type=int, default=0,
                    help="steps of flip-rate warmup toward --rate_peak")
    ap.add_argument("--snap_every", type=int, default=0,
                    help="also keep a copy of the checkpoint every N steps (ckpt_<step>.pt)")
    ap.add_argument("--tail_fp32", action="store_true",
                    help="float tail (embeddings, norm gains) in fp32 + AdamW, as in master "
                         "mode (default bf16 + 8-bit Adam freezes the norm gains at 1.0)")
    ap.add_argument("--probe", default="",
                    help="probe schedule, e.g. '0-40:5,40-160:20' (bitnet/probe.py)")
    ap.add_argument("--seed", type=int, default=None,
                    help="run seed (init, data order, look-ahead batches); default TrainConfig.seed = 1337")
    ap.add_argument("--flip_seed", type=int, default=0,
                    help="offset for the flip RNG only (data order unchanged): replicates")
    ap.add_argument("--rate_min", type=float, default=0.0,
                    help="floor for --rate_schedule: the schedule decays from --rate "
                         "to this value instead of to zero (keeps flips alive)")
    ap.add_argument("--track_flips", action="store_true",
                    help="log per-layer flip rate + never-changed fraction each step")
    ap.add_argument("--no_beta", action="store_true",
                    help="kernel/evidence: legacy raw-trit forward (no 1/sqrt(K*rho) "
                         "scale). Only for reproducing pre-beta checkpoints.")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--min_lr", type=float, default=None,
                    help="cosine floor (default: lr/10)")
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--eval_iters", type=int, default=None)
    ap.add_argument("--loss_chunk", type=int, default=2048,
                    help="tokens per output-head chunk (peak VRAM knob; math is identical)")
    ap.add_argument("--save_secs", type=float, default=900.0,
                    help="wall-clock seconds between checkpoint saves")
    ap.add_argument("--max_temp", type=int, default=86,
                    help="save + stop if GPU temp (C) reaches this (crash guard)")
    ap.add_argument("--resume_temp", type=int, default=85,
                    help="after a thermal pause, wait until GPU temp (C) is at or below "
                         "this before resuming")
    ap.add_argument("--rc_scale", action="store_true",
                    help="learned per-row and per-column scales on every ternary layer "
                         "(W_ij = beta * r_i * c_j * q_ij; N + K floats per layer, fp32, no decay)")
    ap.add_argument("--rc_lr", type=float, default=0.0,
                    help="constant LR for the row/column scales (no warmup/decay); 0 = use the "
                         "float-tail LR schedule")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the elementwise parts (RMSNorm, RoPE, SwiGLU)")
    ap.add_argument("--ckpt_skip", type=int, default=0,
                    help="leave the last K layers without gradient checkpointing (faster, "
                         "more VRAM; ~1.3 GiB per layer at 16x2048 tokens)")
    ap.add_argument("--soft_start", type=int, default=0,
                    help="power ramp: over the first N steps after (re)start, sleep between "
                         "steps so the GPU duty cycle rises from ~15%% to 100%% (a cold "
                         "idle -> full-power jump has reset the machine); 0 = off")
    ap.add_argument("--temp_check", type=int, default=4,
                    help="check GPU temp every N steps")
    args = ap.parse_args()
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.gemm == "cublas":
        from . import kernel as _kmod
        _kmod.GEMM_BACKEND[0] = "cublas"
    ORTH_CHOL[0] = args.ts_orth if args.ts_orth != "qr" else False
    TWO_STAGE[0] = args.two_stage
    TS_SHADOWS[0] = tuple(int(x) for x in args.ts_shadows.split(",")) if args.ts_shadows else ()
    TS_SHADOWS[1] = tuple(int(x) for x in args.ts_shadow_layers.split(",")) if args.ts_shadow_layers else ()
    if args.ts_dump_steps:
        TS_DUMP[:] = [os.path.join(args.out_dir or ".", "acc_dumps"), tuple(int(x) for x in args.ts_dump_steps.split(",")),
                      tuple(int(x) for x in args.ts_dump_layers.split(","))]
    ORTH_DEBUG = os.environ.get("TERN_ORTHDBG") == "1"
    TIME = os.environ.get("TERN_TIME") == "1"   # time the forward/backward vs the momentum step (synchronizes)

    mc: ModelConfig = PRESETS[args.preset]
    tc = TrainConfig()
    if args.seed is not None: tc.seed = args.seed
    if args.data: tc.data_path = args.data
    if args.val: tc.val_path = args.val
    if args.steps: tc.max_steps = args.steps
    if args.batch_size: tc.batch_size = args.batch_size
    if args.grad_accum is not None: tc.grad_accum = args.grad_accum
    if args.seq_len: mc.max_seq_len = tc.seq_len = args.seq_len
    if args.out_dir: tc.out_dir = args.out_dir
    if args.lr: tc.lr = args.lr
    if args.warmup is not None: tc.warmup_steps = args.warmup
    tc.min_lr = args.min_lr if args.min_lr is not None else tc.lr / 10
    if args.eval_interval: tc.eval_interval = args.eval_interval
    if args.eval_iters: tc.eval_iters = args.eval_iters
    if args.flip_seed:
        from . import flip as _flip_mod
        _flip_mod._FLIP_SEED[0] = args.flip_seed * 1_000_003

    BitTransformer.loss_chunk = args.loss_chunk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc.seed)
    print(mc.report())
    use_beta = args.mode in ("kernel", "evidence") and not args.no_beta

    if args.mode == "kernel":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                         rate=args.rate, beta=use_beta,
                                         int8=args.int8, dw_mode=args.dw_mode,
                                         int8_dx=args.int8_dx, g_ref=args.g_ref)
    elif args.mode == "evidence":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                         rate=args.rate, evidence=True,
                                         ev_bits=args.ev_bits, beta=use_beta,
                                         int8=args.int8, dw_mode=args.dw_mode,
                                         int8_dx=args.int8_dx)
    elif args.mode == "flip":
        model, _ = build_flip_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                          theta=args.theta)
    elif args.mode == "fp32":
        # full-precision reference (no ternary, no activation quantization): plain nn.Linear,
        # fp32 weights + AdamW, bf16 autocast in the forward like every other mode
        model = BitTransformer(mc, grad_checkpoint=tc.grad_checkpoint,
                               make_linear=lambda i, o: nn.Linear(i, o, bias=False))
    elif args.mode == "master":
        model = build_master_transformer(
            mc, grad_checkpoint=tc.grad_checkpoint,
            dtype=torch.float32 if args.master_dtype == "fp32" else torch.bfloat16)
    elif args.mode == "stateless":
        model = build_stateless_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                            rate=args.rate)
    else:
        model = BitTransformer(mc, grad_checkpoint=tc.grad_checkpoint)
    model = model.to(device)
    model.train()
    if args.ckpt_skip:
        model.ckpt_skip = args.ckpt_skip
    if args.ts_rank_types:
        rt = {k: int(v) for k, v in (x.split(":") for x in args.ts_rank_types.split(","))}
        lays_ = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
        for i, m in enumerate(lays_):
            if _TS_TYPES[i % 7] in rt:
                m._ts_rank = rt[_TS_TYPES[i % 7]]
        print(f"--ts_rank_types: {rt}", flush=True)
    if args.ts_rank_s_types:
        rt = {k: int(v) for k, v in (x.split(":") for x in args.ts_rank_s_types.split(","))}
        lays_ = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
        for i, m in enumerate(lays_):
            if _TS_TYPES[i % 7] in rt:
                m._ts_rank_s = rt[_TS_TYPES[i % 7]]
        print(f"--ts_rank_s_types: {rt}", flush=True)
    if args.ts_rank_kv:
        from .model import Attention
        for a in model.modules():
            if isinstance(a, Attention):
                a.wk._ts_rank = a.wv._ts_rank = args.ts_rank_kv
        print(f"--ts_rank_kv: k / v projections at rank {args.ts_rank_kv}", flush=True)
    if args.qk_temp:
        from .model import Attention
        for a in model.modules():
            if isinstance(a, Attention):
                a.qk_logscale = nn.Parameter(torch.zeros(a.n_heads, device=device))
        print("per-head attention temperature on (log-scale, init 0)", flush=True)
    if args.rc_scale:
        from .flip import enable_rc_scales
        if args.mode == "master":
            from .master import MasterTernaryLinear
            nrc = 0
            for m_ in model.modules():
                if isinstance(m_, MasterTernaryLinear): m_.enable_rc_scale(); nrc += m_.N + m_.K
            print(f"row/column scales (master): {nrc / 1e3:.1f}k floats", flush=True)
        else:
            print(f"row/column scales: {enable_rc_scales(model) / 1e3:.1f}k floats", flush=True)
    if args.compile:
        from .model import enable_compile
        enable_compile()

    if args.mode in ("master", "fp32"):
        # the baseline keeps everything in fp32 with a standard AdamW: no bf16
        # rounding anywhere, so the ceiling is not limited by storage precision.
        master, emb, norms = split_params(model)
        # the float extras (--rc_scale, --qk_temp) without weight decay, as in the kernel path's tail groups
        xid = {id(p) for n, p in model.named_parameters() if n.endswith(("row_scale", "col_scale", "qk_logscale"))}
        norms = norms + [p for p in emb if id(p) in xid]; emb = [p for p in emb if id(p) not in xid]
        tail = master + emb + norms
        if args.m_factv or args.m_rank or args.m_beta1 or args.m_gate or args.m_beta2:
            from .master_opt import AdamX
            tail_opt = AdamX(
                [{"params": master, "weight_decay": tc.weight_decay, "factv": args.m_factv, "rank": args.m_rank, "gate": args.m_gate,
                  "betas": (args.m_beta1 or tc.beta1, args.m_beta2 or tc.beta2)},
                 {"params": emb, "weight_decay": tc.weight_decay},
                 {"params": norms, "weight_decay": 0.0}],
                lr=tc.lr, betas=(tc.beta1, tc.beta2))
            print(f"master optimizer AdamX: factored v {args.m_factv}, first-moment rank {args.m_rank or 'full'}, "
                  f"beta1 {args.m_beta1 or tc.beta1}, gate {args.m_gate}", flush=True)
        else:
            tail_opt = torch.optim.AdamW(
                [{"params": master, "weight_decay": tc.weight_decay},
                 {"params": emb, "weight_decay": tc.weight_decay},
                 {"params": norms, "weight_decay": 0.0}],
                lr=tc.lr, betas=(tc.beta1, tc.beta2))
        print(f"master mode: {sum(p.numel() for p in master)/1e6:.1f}M latent "
              f"({args.master_dtype}) + {sum(p.numel() for p in emb+norms)/1e6:.1f}M tail, "
              f"AdamW fp32 states", flush=True)
    elif args.lean_tail:
        # --lean_tail: embedding (and the adapter, added later) in bf16, norm gains / row-column scales / attention
        # temperature in fp16, LeanAdam (int8 first moment, factored second moment, stochastic-rounding write-back)
        from .opt_lean import LeanAdam
        tail = model.float_tail_parameters()
        names = {id(p): n for n, p in model.named_parameters()}
        small = [p for p in tail if names[id(p)].endswith(("norm.weight", "qk_logscale", "row_scale", "col_scale"))]
        sid = {id(p) for p in small}
        emb = [p for p in tail if id(p) not in sid]
        for p in emb:
            p.data = p.data.to(torch.bfloat16)
        for p in small:
            p.data = p.data.to(torch.float16)
        rc = [p for p in small if names[id(p)].endswith(("row_scale", "col_scale"))]
        rcid = {id(p) for p in rc}
        norms = [p for p in small if id(p) not in rcid]
        tail_groups = [{"params": emb, "weight_decay": tc.weight_decay}, {"params": norms, "weight_decay": 0.0}]
        if rc:
            tail_groups.append({"params": rc, "weight_decay": 0.0, "rc_lr": args.rc_lr})
        tail_opt = LeanAdam(tail_groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))
        print(f"lean float tail: {sum(p.numel() for p in emb)/1e6:.1f}M bf16 + {sum(p.numel() for p in small)/1e3:.1f}k fp16, "
              f"LeanAdam (int8 m, factored v, stochastic rounding)", flush=True)
    elif args.tail_fp32:
        # float tail kept in fp32 with a standard AdamW, exactly like master mode. In
        # bf16 an Adam step (~lr <= 1.5e-3) on a norm gain of 1.0 is below half the
        # representable gap (2^-7) and always rounds away: the gains never train.
        tail = model.float_tail_parameters()
        for p in tail:
            p.data = p.data.float()
        tids = {id(p) for p in tail}
        rc = [p for n, p in model.named_parameters() if n.endswith(("row_scale", "col_scale"))]
        rcid = {id(p) for p in rc}
        norms = [p for n, p in model.named_parameters() if id(p) in tids and id(p) not in rcid
                 and (n.endswith("norm.weight") or n.endswith("qk_logscale"))]
        nids = {id(p) for p in norms} | rcid
        emb = [p for p in tail if id(p) not in nids]
        tail_groups = [{"params": emb, "weight_decay": tc.weight_decay},
                       {"params": norms, "weight_decay": 0.0}]
        if rc:
            tail_groups.append({"params": rc, "weight_decay": 0.0, "rc_lr": args.rc_lr})
        tail_opt = torch.optim.AdamW(tail_groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))
        print(f"fp32 float tail: {sum(p.numel() for p in emb)/1e6:.1f}M emb + "
              f"{sum(p.numel() for p in norms)/1e3:.1f}k norm gains, AdamW fp32 states", flush=True)
    else:
        # float tail (embeddings + norms + flip predictor): bf16 params + 8-bit Adam.
        # NOTE: norm gains cannot train in bf16 (see --tail_fp32).
        from .opt8 import Adam8bit
        rc = [p for n, p in model.named_parameters() if n.endswith(("row_scale", "col_scale"))]
        rcid = {id(p) for p in rc}
        for p in model.float_tail_parameters():
            if id(p) not in rcid:                 # scales stay fp32: bf16 would freeze them
                p.data = p.data.to(torch.bfloat16)
        tail = model.float_tail_parameters()
        groups = [{"params": [p for p in tail if id(p) not in rcid],
                   "weight_decay": tc.weight_decay}]
        if rc:
            groups.append({"params": rc, "weight_decay": 0.0, "rc_lr": args.rc_lr})
        tail_opt = Adam8bit(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                            weight_decay=tc.weight_decay)

    STATE.momentum = args.momentum
    STATE.weight_decay = tc.weight_decay

    if args.mode in ("kernel", "evidence") and tc.grad_accum > 1:
        set_flip_accum(model, tc.grad_accum)
        print(f"flip accumulation: one flip per {tc.grad_accum} micro-steps "
              f"({tc.grad_accum * tc.batch_size * tc.seq_len:,} tokens/step)", flush=True)

    if args.hump:
        n = set_hump(model, args.hump_g0, args.hump_alpha)
        print(f"hump flip probability g0={args.hump_g0} alpha={args.hump_alpha} "
              f"({n} layers)", flush=True)

    if args.inv_prob:
        n = set_inv_prob(model, True)
        print(f"reverse-magnitude flip probability ({n} layers)", flush=True)

    if args.norm == "row":
        n = set_norm_mode(model, "row")
        print(f"per-row flip normalisation ({n} layers)", flush=True)

    if args.flip_lockout > 0:
        n = set_lockout(model, args.flip_lockout, args.lockout_mode)
        print(f"flip lockout: {args.lockout_mode}, epoch {args.flip_lockout} steps "
              f"({n} layers, 2 bits/weight of mask)", flush=True)

    if args.err_feedback:
        if args.mode != "kernel" or tc.grad_accum != 1:
            raise SystemExit("--err_feedback requires --mode kernel and --grad_accum 1")
        n = set_err_feedback(model, True, args.ef_alpha)
        print(f"spatial error feedback on ({n} layers, alpha {args.ef_alpha})", flush=True)

    if args.abs_scale:
        n = set_abs_scale(model, True, args.calib_steps)
        print(f"Arm B: absolute flip scale, frozen after {args.calib_steps} steps "
              f"({n} layers)", flush=True)

    if args.track_flips:
        n = enable_flip_tracking(model)
        print(f"flip tracking on for {n} ternary layers", flush=True)

    sampler = torch.Generator().manual_seed(tc.seed)
    # separate stream for cross-batch look-ahead, so the training batch order is unchanged
    la_gen = torch.Generator().manual_seed(tc.seed + 4242)
    lr_state = {}
    la_extra = ((lambda: get_batch(train_data, tc.batch_size, tc.seq_len, device, la_gen))
                if args.lookahead_xbatch else None)
    rs_state = {"mult": 1.0}
    rs_gen = torch.Generator().manual_seed(tc.seed + 777)
    if (args.lookahead or args.lowrank) and (args.mode != "kernel" or args.rate_search
                                             or (tc.grad_accum != 1 and not args.ts)):
        # (--ts sums the captured micro-batch gradients, so it allows --grad_accum > 1)
        raise SystemExit("--lookahead / --lowrank require --mode kernel, --grad_accum 1 (except --ts), no --rate_search")
    if args.rate_search:
        if args.mode != "kernel" or tc.grad_accum != 1:
            raise SystemExit("--rate_search requires --mode kernel and --grad_accum 1")
        print(f"flip-rate line search: every {args.rs_every} steps, factors "
              f"{args.rs_factors}, tuning batch {args.rs_batch}x{tc.seq_len}", flush=True)
    train_data = np.memmap(tc.data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(tc.val_path, dtype=np.uint16, mode="r") if os.path.exists(tc.val_path) else train_data

    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    metrics_path = os.path.join(tc.out_dir, "metrics.jsonl")
    metrics_f = open(metrics_path, "a")

    def log_metrics(rec):
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    def save_ckpt(step):
        # atomic: write tmp then rename, so a kill mid-save never corrupts ckpt.pt
        if args.ts_offload_async and torch.cuda.is_available():
            torch.cuda.synchronize()      # the host copies of the state are written on a side stream
        tmp = ckpt_path + ".tmp"
        # flip-tracking masks are plain attributes (not state_dict buffers), so they
        # are saved alongside: otherwise "never flipped since init" silently resets
        # to 100% on resume.
        touched = {n: m.touched for n, m in model.named_modules()
                   if hasattr(m, "touched") and m.touched is not None}
        # step the mask started accumulating from: never-flipped is a "since init"
        # number only when this is 0.
        touched_origin = _touched_origin["v"]
        torch.save({"model": model.state_dict(), "opt": tail_opt.state_dict(),
                    "touched": touched, "touched_origin": touched_origin,
                    "cfg": mc, "step": step, "mode": args.mode, "beta": use_beta,
                    "int8": args.int8, "dw_mode": args.dw_mode, "rate_min": args.rate_min,
                    "int8_dx": args.int8_dx, "rate_schedule": args.rate_schedule,
                    "abs_scale": args.abs_scale, "err_feedback": args.err_feedback,
                    "g_ref": args.g_ref, "norm": args.norm,
                    "inv_prob": args.inv_prob, "hump": args.hump,
                    "hump_g0": args.hump_g0, "hump_alpha": args.hump_alpha,
                    "flip_lockout": args.flip_lockout,
                    "lockout_mode": args.lockout_mode,
                    "ef_alpha": args.ef_alpha,
                    # low-rank momentum (U, V) per ternary layer, in module order
                    "lowrank": [lr_state.get(id(m)) for m in model.modules()
                                if isinstance(m, KernelTernaryLinear)] if lr_state else None,
                    # --ts: the rest of the per-layer state (update count, factored v, short momentum), module order
                    "ts_state": [{k[0]: st_[k] for k in (("t", id(m)), ("v", id(m)), ("m", id(m))) if k in st_}
                                 for m in model.modules() if isinstance(m, KernelTernaryLinear)]
                                if (st_ := lr_state.get("_ts")) else None}, tmp)
        os.replace(tmp, ckpt_path)

    la_off = tuple(int(x) for x in args.lookahead_off.split(":")) if args.lookahead_off else (0, 0)
    la_on = lambda st: bool(args.lookahead) and not (la_off[0] <= st < la_off[1])
    mag_on = [False]
    mech_state = {}
    sel_state = {}
    ev_state = {}
    ar_state = {"mult": 1.0, "ema": 0.0}
    mb_betas = [float(b) for b in args.multibeta.split(",")] if args.multibeta else []
    mb_states = [{} for _ in mb_betas]; mb_scores = {}
    acc_gen = torch.Generator().manual_seed(tc.seed + 999)
    accum_fresh = (lambda: get_batch(train_data, tc.batch_size, tc.seq_len, device, acc_gen)) if args.accum_flip else None

    def enable_mag(step0):
        # the new parameters join the float tail (the look-ahead passes and the gradient clip treat them like the
        # tail) as their own AdamW group. Enabled after the resume when branching a checkpoint without them (so it
        # loads unchanged), before it when the checkpoint already has them (so they and their AdamW state load)
        kind, r = args.lowrank_mag.split(":")
        newp = []
        from .master import MasterTernaryLinear
        for m in model.modules():
            if isinstance(m, (KernelTernaryLinear, MasterTernaryLinear)):
                newp += m.enable_lowrank_mag(kind, int(r))
        if args.lean_tail:
            for p_ in newp:
                p_.data = p_.data.to(torch.bfloat16)
        tail.extend(newp)
        tail_opt.add_param_group({"params": newp, "weight_decay": args.mag_wd, "lr_mult": args.mag_lr_mult,
                                  "lr": lr_at(step0, tc) * args.mag_lr_mult})
        if args.mag_cap:
            for m in model.modules():
                if isinstance(m, KernelTernaryLinear):
                    m.mag_cap = args.mag_cap
        mag_on[0] = True
        print(f"low-rank magnitude '{kind}' rank {r}: {sum(p.numel() for p in newp) / 1e6:.2f}M floats "
              f"(wd {args.mag_wd}, lr x{args.mag_lr_mult}, cap {args.mag_cap or 'off'})", flush=True)

    start_step = 0
    _touched_origin = {"v": 0}
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        if blob.get("beta", False) != use_beta:
            raise SystemExit(f"checkpoint beta={blob.get('beta', False)} but run beta="
                             f"{use_beta}: forward differs, refusing to resume")
        if args.lowrank_mag and any(k.endswith("mag_A") for k in blob["model"]):
            enable_mag(int(blob.get("step", 0)) + 1)
        new_rc = args.rc_scale and not any(k.endswith("row_scale") for k in blob["model"])
        if new_rc:
            # branching a checkpoint trained without row/column scales: they start at 1 (the layer is unchanged)
            # and their optimizer group starts fresh; everything else loads as saved
            missing, unexpected = model.load_state_dict(blob["model"], strict=False)
            assert all(k.endswith(("row_scale", "col_scale")) for k in missing) and not unexpected, (missing, unexpected)
            rc_groups = [g for g in tail_opt.param_groups if "rc_lr" in g]
            tail_opt.param_groups[:] = [g for g in tail_opt.param_groups if "rc_lr" not in g]
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
            for g in rc_groups:
                tail_opt.add_param_group(g)
            print(f"row/column scales added to a checkpoint without them ({len(missing)} tensors, start at 1)", flush=True)
        else:
            model.load_state_dict(blob["model"])
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
        start_step = int(blob.get("step", 0)) + 1
        # fast-forward the data streams, so a resumed run sees the batches the uninterrupted
        # run would have (not the step-0 batches again)
        for _ in range(start_step * tc.grad_accum):
            torch.randint(len(train_data) - tc.seq_len - 1, (tc.batch_size,), generator=sampler)
        if args.lookahead_xbatch:
            for _ in range(sum(la_on(i) for i in range(start_step)) * args.lookahead):
                torch.randint(len(train_data) - tc.seq_len - 1, (tc.batch_size,), generator=la_gen)
        if blob.get("lowrank"):
            lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
            for m, uv in zip(lays, blob["lowrank"]):
                if uv is not None:
                    lr_state[id(m)] = tuple(t.to(device) for t in uv)
            print(f"resumed low-rank momentum for {len(lr_state)} layers", flush=True)
        if blob.get("ts_state"):
            lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
            st_ = lr_state.setdefault("_ts", {})
            for m, d_ in zip(lays, blob["ts_state"]):
                for k_, v_ in d_.items():
                    st_[(k_, id(m))] = tuple(x.to(device) for x in v_) if isinstance(v_, tuple) else v_
            print(f"resumed --ts state (short momentum, v, step counts) for {len(lays)} layers", flush=True)
        tmask = blob.get("touched") or {}
        nres = 0
        for n, m in model.named_modules():
            if n in tmask and getattr(m, "touched", None) is not None:
                m.touched.copy_(tmask[n].to(m.touched.device)); nres += 1
        if nres:
            _touched_origin["v"] = int(blob.get("touched_origin", 0))
            print(f"resumed from {ckpt_path} at step {start_step} (flip history for "
                  f"{nres} layers, accumulated since step {_touched_origin['v']})",
                  flush=True)
        else:
            _touched_origin["v"] = start_step
            print(f"resumed from {ckpt_path} at step {start_step}", flush=True)
            if args.track_flips:
                print(f"  WARNING: checkpoint has no flip history -> 'never flipped' "
                      f"counts only from step {start_step}, NOT since init. The true "
                      f"since-init value is <= the last value logged before this "
                      f"resume (it is monotone non-increasing).", flush=True)

    # save on SIGTERM/SIGINT so `kill` (or Ctrl-C) leaves a fresh resumable ckpt
    import signal
    # the checkpoint's "step" is the last COMPLETED step (resume starts at step + 1), so the
    # handler only sets a flag and the loop saves and exits at the end of the current step
    _last = {"stop": False}
    def _graceful(signum, frame):
        print(f"signal {signum}: saving ckpt after the current step ...", flush=True)
        _last["stop"] = True
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    # after any resume, so 'net' counts from the weights this run starts at
    if args.lowrank_mag and not mag_on[0]:
        enable_mag(start_step)
    if args.multibeta and lr_state:
        for st in mb_states:
            for k_, v_ in lr_state.items():
                if isinstance(k_, int): st[k_] = tuple(t.clone() for t in v_)
        print(f"--multibeta {mb_betas}: every momentum starts from the saved one", flush=True)
    if args.mech and lr_state:
        lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
        mech_state["warm"] = {i: lr_state[id(m)] for i, m in enumerate(lays) if id(m) in lr_state}
        print(f"--mech {args.mech}: warm start from the saved low-rank momentum ({len(mech_state['warm'])} layers)", flush=True)
    if args.track_reversals:
        for m in model.modules():
            if isinstance(m, KernelTernaryLinear):
                m.enable_reversal_tracking()
    t0 = time.time()
    last_save = time.time()
    end_step = tc.max_steps if args.stop_after is None else min(tc.max_steps, args.stop_after)
    probe_steps = parse_schedule(args.probe)
    ramp_t = time.time()
    prof = None
    qk_map = None
    if args.qk_protect:
        from .model import Attention
        qk_map = {id(p): a for a in model.modules() if isinstance(a, Attention) for p in (a.wq, a.wk)}
    mwin = None
    if args.move_window or args.record_sample or args.window_curve:
        recorder = (Recorder(model, args.record_sample, tc.out_dir,
                             tail_opt if args.mode == "master" else None) if args.record_sample else None)
        mwin = MultiWindow([int(v) for v in str(args.move_window).split(",") if v],
                           tail_opt if args.mode == "master" else None, recorder,
                           [int(v) for v in args.window_curve.split(",") if v],
                           {int(v) for v in args.window_curve_at.split(",") if v} or None)
        mwin.open(model, lr_state, start_step - 1)
    ttrack = TritTracker(model) if (args.mode == "master" and args.track_flips) else None
    _snap_state = {}                      # --m_snap: each master layer's trits after the last step
    ts_cfg = None
    if args.ts and args.ts_fused:         # --ts_fused: each layer's update inside its backward (ts_layer)
        ts_cfg = dict(r=args.lowrank, lr_ratio=1.0, theta=args.ts_theta, tau=args.ts_tau, rank_s=args.ts_rank_s,
                      b1=args.ts_b1, b2=args.ts_b2, gate=args.lr_gate, spend=args.ts_spend,
                      tau_anneal=args.ts_tau_anneal, int8=args.ts_int8, qfmt=args.ts_qfmt, qwhich=args.ts_qwhich,
                      qnoV=args.ts_qnoV, qdiag=False, qfmt_m=args.ts_qfmt_m, qfmt_a=args.ts_qfmt_a, qfmt_aV=args.ts_qfmt_aV,
                      store_m=args.ts_store_m, store_a=args.ts_store_a, offload="async" if args.ts_offload_async else args.ts_offload,
                      diag=False, shadow=args.ts_shadow, shared_v=args.ts_shared_v,
                      sketch=args.ts_sketch, resid=args.ts_resid, rescale=args.ts_rescale, table=args.ts_table,
                      exact_spend=args.ts_exact_spend, tiles=args.ts_tiles, rank_sched=_rank_sched(args.ts_rank_sched),
                      progress=0.0,
                      refresh=args.ts_refresh, refresh_frac=args.ts_refresh_frac, shadows=TS_SHADOWS[0],
                      dump_dir=TS_DUMP[0], dump_steps=TS_DUMP[1], dump_layers=TS_DUMP[2],
                      shadow_layers=TS_SHADOWS[1])
        for i_, m_ in enumerate([m_ for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)]):
            m_._ts_idx = i_
            m_.on_grad = lambda layer, g, _c=ts_cfg: ts_layer(layer, g, lr_state, _c)
        print("--ts_fused: the update runs inside each layer's backward", flush=True)
    for step in range(start_step, end_step):
        ORTH_DBG[0] = ORTH_DEBUG and step % 10 == 0
        if args.profile and step == start_step + 20:         # after autotune / warm-up
            from torch.profiler import profile, ProfilerActivity
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__(); t_prof = time.time()
        if prof is not None and step == start_step + 20 + args.profile:
            torch.cuda.synchronize(); prof.__exit__(None, None, None)
            wall = (time.time() - t_prof) / args.profile
            ka = prof.key_averages()
            gpu = sum(e.self_device_time_total for e in ka) / 1e6 / args.profile
            print(f"PROFILE {args.profile} steps: wall {wall:.3f} s/step, GPU kernel time {gpu:.3f} s/step "
                  f"({gpu / wall * 100:.0f}% busy)", flush=True)
            print(ka.table(sort_by="self_device_time_total", row_limit=30), flush=True)
            print(ka.table(sort_by="self_cpu_time_total", row_limit=30), flush=True)
            prof.export_chrome_trace(os.path.join(tc.out_dir, "profile_trace.json"))
            raise SystemExit(0)
        k = step - start_step
        if k < args.soft_start:
            # soft start: idle for a shrinking fraction of the previous step's time
            dt = time.time() - ramp_t
            duty = 0.15 + 0.85 * k / args.soft_start
            if k > 0:
                time.sleep(min(dt * (1.0 / duty - 1.0), 15.0))   # step 0 includes compiling
            ramp_t = time.time()
        # thermal guard: PAUSE (not stop) while GPU is at/above max_temp; resume
        # in place once it cools. No exit, no save — just wait it out.
        if step % args.temp_check == 0:
            temp = gpu_temp()
            if temp is not None and temp >= args.max_temp:
                print(f"GPU {temp}C >= {args.max_temp}C -> pausing to cool "
                      f"(step {step})", flush=True)
                resume_at = min(args.resume_temp, args.max_temp - 1)
                while temp is not None and temp > resume_at:
                    time.sleep(5)
                    temp = gpu_temp()
                print(f"cooled to {temp}C -> resuming", flush=True)
        if step in probe_steps:
            log_metrics({"step": step, "probe": True, **probe(model, val_data, device)})
        if args.flip_lockout > 0 and step % args.flip_lockout == 0:
            reset_lockout(model)
        lr = lr_at(step, tc)
        if ts_cfg is not None:
            ts_cfg["lr_ratio"] = (lr / tc.lr) ** args.ts_anneal
            ts_cfg["progress"] = step / max(1, tc.max_steps)
            ts_cfg["qdiag"] = bool(args.ts_qdiag) and step % args.ts_qdiag in (0, 1)   # two steps: a change to measure
            ts_cfg["diag"] = bool(args.ts_diag) and step % args.ts_diag == 0
        rate_now = args.rate
        if args.rate_schedule != "const":
            prog = min(1.0, step / max(1, tc.max_steps))
            if args.rate_schedule == "exp":
                # f(step) = rate * exp(-step / tau); never exactly zero
                rate_now = args.rate * math.exp(-step / args.rate_tau)
            elif args.rate_peak > 0 and step < args.rate_warmup:
                rate_now = args.rate + (args.rate_peak - args.rate) * step / args.rate_warmup
            else:
                top = args.rate
                if args.rate_peak > 0:          # decay from the peak over what is left
                    top = args.rate_peak
                    prog = min(1.0, (step - args.rate_warmup)
                               / max(1, tc.max_steps - args.rate_warmup))
                mult = (0.5 * (1 + math.cos(math.pi * prog))
                        if args.rate_schedule == "cosine" else 1.0 - prog)
                rate_now = args.rate_min + (top - args.rate_min) * mult
            set_flip_rate(model, rate_now)
        if args.rate_search:
            rate_now = rate_now * rs_state["mult"]
            set_flip_rate(model, rate_now)
        if args.adapt_rate:
            rate_now = rate_now * ar_state["mult"]
            set_flip_rate(model, rate_now)
        STATE.lr = lr
        for g in tail_opt.param_groups:
            g["lr"] = (g["rc_lr"] if g.get("rc_lr") else lr) * g.get("lr_mult", 1.0)
            if TWO_STAGE[0]:          # --two_stage: weight decay off in the second stage
                g.setdefault("wd0", g.get("weight_decay", 0.0))
                g["weight_decay"] = g["wd0"] if step < two_stage_mid(tc) else 0.0

        # gradient accumulation: BitLinear hooks apply lr*g/accum each micro-step
        # (SGD is linear, so summing micro-steps approximates one averaged step).
        STATE.updates_enabled = True
        STATE.grad_scale = tc.grad_accum
        tail_opt.zero_grad(set_to_none=True)
        last_loss = 0.0
        # ---- flip-rate line search: capture this step's gradient instead of flipping
        search = (args.rate_search and step > 0 and step % args.rs_every == 0)
        if search or args.lookahead or args.lowrank:
            for l in model.modules():
                if isinstance(l, KernelTernaryLinear): l.capture = True
        if TIME: torch.cuda.synchronize(); _t0 = time.time()
        for micro in range(tc.grad_accum):
            if args.ts and tc.grad_accum > 1:      # --ts with accumulation: sum the captured micro-batch gradients
                for l_ in model.modules():
                    if isinstance(l_, KernelTernaryLinear):
                        l_.cap_accum = True; l_.cap_final = micro == tc.grad_accum - 1
            x, y = get_batch(train_data, tc.batch_size, tc.seq_len, device, sampler)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / tc.grad_accum).backward()  # triggers BitLinear fused updates
            last_loss = loss.item()
        if args.mode in ("kernel", "evidence") and tc.grad_accum > 1:
            flip_accumulated(model)
        if TIME:
            torch.cuda.synchronize(); _t1 = time.time()
            _m1, _p1 = torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30
        if args.lowrank:
            if la_on(step):             # propose from M, keep by the look-ahead test
                rs_info, Ms = lowrank_step(model, lr_state, args.lowrank, args.lr_beta, step,
                                           args.flip_seed, args.lr_adapt, propose_only=True,
                                           diag=args.lr_diag, gate=args.lr_gate,
                                           refresh=args.lr_refresh, refresh_every=args.lr_refresh_every,
                                           vnorm=args.lr_vnorm, mask_stuck=args.lr_mask_stuck,
                                           qk_protect=args.qk_protect, qk_map=qk_map)
                la = lookahead_step(model, x, y, tail, device, step, args.lookahead,
                                    args.flip_seed, la_extra, signals=Ms,
                                    diag=args.lr_diag, gacc=mwin, sig_gmeans=lr_state["_gms"],
                                    want_delta=bool(args.lr_spend))
                if args.lr_spend:
                    # --lr_spend: a kept flip consumes the push that caused it (D = -sign(M) there, so
                    # adding c * mean|M| * D shrinks those entries), kept low-rank: U += c*gm*(D V)
                    lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
                    for m, d in zip(lays, la.pop("_D")):
                        U, V = lr_state[id(m)]
                        lr_state[id(m)] = (U + args.lr_spend * lr_state["_gm"][id(m)] * (d.float() @ V), V)
                rs_info.update(la)
                del Ms
            elif args.mech:
                rs_info = mech_step(model, mech_state, step, args, x, y, tail, args.flip_seed)
            elif args.evidence_bits:
                rs_info = evidence_step(model, lr_state, ev_state, args.lowrank, args.lr_beta, step,
                                        args.evidence_bits, args.flip_seed)
            elif args.select:
                rs_info = select_step(model, lr_state, sel_state, args.lowrank, args.lr_beta, step, args, args.flip_seed)
            elif args.multibeta:
                rs_info = multibeta_step(model, mb_states, mb_betas, mb_scores, args.lowrank, step, args.flip_seed)
            elif args.bop:
                rs_info = bop_step(model, lr_state, args.bop_gamma, args.bop_tau, args.ts_b2, norm=not args.bop_raw,
                                   bias_correct=args.bop_bc)
            elif args.ts:
                rs_info = ts_step(model, lr_state, args.lowrank, step, (lr / tc.lr) ** args.ts_anneal, args.ts_theta, args.ts_tau,
                                  args.ts_rank_s, args.ts_b1, args.ts_b2, gate=args.lr_gate, spend=args.ts_spend,
                                  tau_anneal=args.ts_tau_anneal, int8=args.ts_int8, fused=args.ts_fused,
                                  qfmt=args.ts_qfmt, qwhich=args.ts_qwhich, qnoV=args.ts_qnoV,
                                  qdiag=bool(args.ts_qdiag) and step % args.ts_qdiag in (0, 1),
                                  qfmt_m=args.ts_qfmt_m, qfmt_a=args.ts_qfmt_a, qfmt_aV=args.ts_qfmt_aV,
                      store_m=args.ts_store_m, store_a=args.ts_store_a, offload="async" if args.ts_offload_async else args.ts_offload,
                      diag=bool(args.ts_diag) and step % args.ts_diag == 0, shadow=args.ts_shadow, shared_v=args.ts_shared_v,
                      sketch=args.ts_sketch, resid=args.ts_resid, rescale=args.ts_rescale, table=args.ts_table,
                      exact_spend=args.ts_exact_spend, tiles=args.ts_tiles, rank_sched=_rank_sched(args.ts_rank_sched),
                      progress=step / max(1, tc.max_steps),
                      refresh=args.ts_refresh, refresh_frac=args.ts_refresh_frac, shadows=TS_SHADOWS[0],
                      dump_dir=TS_DUMP[0], dump_steps=TS_DUMP[1], dump_layers=TS_DUMP[2],
                      shadow_layers=TS_SHADOWS[1])
            elif args.accum_flip:
                rs_info = accum_step(model, lr_state, args.lowrank, args.lr_beta, step, args.accum_flip,
                                     args.accum_z, args.flip_seed, fresh=accum_fresh)
            else:
                rs_info = lowrank_step(model, lr_state, args.lowrank, args.lr_beta, step,
                                       args.flip_seed, args.lr_adapt, gate=args.lr_gate, vnorm=args.lr_vnorm,
                                       speed_ref=args.speed_ref, speed_row=args.speed_row,
                                       ncap=args.mom_ncap, pfun_tanh=args.pfun_tanh, grav_up=args.grav_up,
                                       undo=("g" if args.undo_g else args.undo), slow_rank=args.slow_gate, slow_beta=args.slow_beta,
                                       dither=args.dither_ld, dry_vec=args.dry_vec, spend=args.spend,
                                       dry_w=args.dry_w, mom_int8=args.mom_int8, rare_rank=args.rare_rank,
                                       rare_dry=args.rare_dry, rare_mode=args.rare_mode, rare_rate=args.rare_rate,
                                       tiers=[(int(a), float(b)) for a, b in (x.split(':') for x in args.tiers.split(','))] if args.tiers else (),
                                       rare_weight=args.rare_weight, pshape=args.pshape,
                                       pshape_T=args.steps if args.pshape_sched == "cos" else 0,
                                       dry_start=args.dry_start, dry_warm=args.dry_warm, vfull=args.lr_vfull)
        elif la_on(step):
            rs_info = lookahead_step(model, x, y, tail, device, step, args.lookahead,
                                     args.flip_seed, la_extra)
        elif search:
            rs_info = rate_search_step(model, rate_now, rs_state, args, tc, train_data,
                                       rs_gen, device, step)
        else:
            rs_info = None
        if mwin is not None and args.mode == "master":   # the gradients master's latent weights see
            for i, l in enumerate(mwin.layers(model)):
                mwin.add(i, l.weight.grad)
        if args.adapt_rate and rs_info is not None and "lr_cos" in rs_info:
            # --adapt_rate: keep momentum predictive of the next gradient; below target the flip rate shrinks
            # (the landscape moves less per step), above it grows back; bounded to [1/16, 2] x the schedule
            ar_state["ema"] = 0.9 * ar_state["ema"] + 0.1 * float(rs_info["lr_cos"])
            ar_state["mult"] = min(max(ar_state["mult"] * math.exp(args.adapt_rate_gain * (ar_state["ema"] - args.adapt_rate_target)),
                                       1 / 16), 2.0)
            rs_info["rate_mult"] = ar_state["mult"]; rs_info["cos_ema"] = ar_state["ema"]
        torch.nn.utils.clip_grad_norm_(tail, tc.grad_clip)
        tail_opt.step()
        if TIME:
            torch.cuda.synchronize(); _t2 = time.time()
            print(f"time step {step}: forward+backward {_t1 - _t0:.3f}s, update (momentum step / AdamW) {_t2 - _t1:.3f}s"
                  f" | GiB allocated after backward {_m1:.2f} (peak in fwd+bwd {_p1:.2f}), after update "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} (peak {torch.cuda.max_memory_allocated() / 2**30:.2f})",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()
        if args.mode == "master" and (args.m_clamp or args.m_leak or args.m_snap or args.m_gfix or args.m_lag):
            from .master import MasterTernaryLinear
            from .master_opt import latent_ops
            mlays = [m_ for m_ in model.modules() if isinstance(m_, MasterTernaryLinear)]
            if args.m_gfix and step == args.m_gfix or (args.m_gfix and step > args.m_gfix and mlays[0].gamma_fixed is None):
                for m_ in mlays: m_.gamma_fixed = m_.weight.detach().abs().mean().clamp_min(1e-5)
            latent_ops(mlays, _snap_state, args.m_clamp, args.m_leak, args.m_snap, args.m_lag, step)
        if args.master_bits and args.mode == "master":
            # --master_bits K: master degraded toward us: after every step each latent weight is stored on a
            # (2^K - 1)-level uniform grid over [-2 gamma, 2 gamma] (gamma = the matrix's absmean), with stochastic
            # rounding (unbiased, so small updates still count on average). K = 2 leaves 3 levels: a stateless master
            from .master import MasterTernaryLinear
            with torch.no_grad():
                for m_ in model.modules():
                    if isinstance(m_, MasterTernaryLinear):
                        w_ = m_.weight; gam = w_.abs().mean().clamp_min(1e-8); c_ = 2 * gam
                        lv = 2 ** args.master_bits - 1; d_ = 2 * c_ / (lv - 1)
                        u_ = (w_.clamp(-c_, c_) + c_) / d_
                        w_.copy_((torch.floor(u_ + torch.rand_like(u_))).clamp_(0, lv - 1) * d_ - c_)
        n_flips = apply_flips(model) if args.mode == "flip" else 0

        rec = {"step": step, "loss": last_loss, "lr": lr, "flip_rate_cfg": rate_now,
               "rs_mult": rs_state["mult"],
               "touched_origin": _touched_origin["v"],
               "tokens": (step + 1) * tc.grad_accum * tc.batch_size * tc.seq_len}
        fs = collect_flip_stats(model) if args.track_flips else {}
        if ttrack is not None:
            rec.update(ttrack.step())
        if rs_info is not None:
            # device counters (look-ahead) become numbers here: one read per step, at logging
            rs_info = {k: (v.item() if torch.is_tensor(v) else v) for k, v in rs_info.items()}
            rec.update(rs_info)
        if args.flip_lockout > 0 and step % tc.log_interval == 0:
            lf = lockout_stats(model)
            if lf is not None:
                rec["locked_frac"] = lf
        if fs:
            if "rev_flips" in fs:
                rec.update(rev_flips=fs["rev_flips"], net_frac=fs["net_frac"])
            rec.update(flip_frac=fs["flip_frac_total"], never_frac=fs["never_frac_total"],
                       flip_frac_layers=fs["flip_frac"], never_frac_layers=fs["never_frac"],
                       gmean_layers=fs["gmean_layers"],
                       frozen_scale_layers=fs["frozen_scale_layers"])
        if mwin is not None:
            rec.update(mwin.tick(model, lr_state, step))
        log_metrics(rec)

        if step % tc.log_interval == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            extra = f"| flips {n_flips:>7d} " if args.mode == "flip" else ""
            if rs_info and "lr_cos" in rs_info:
                extra += f"| cos(g,M) {rs_info['lr_cos']:+.3f} "
            if rs_info and "sel_prop" in rs_info:
                extra += f"| select {rs_info['sel_kept']}/{rs_info['sel_prop']} {'on' if rs_info['sel_active'] else 'warmup'} "
                if "sel_label_good" in rs_info: extra += f"good {rs_info['sel_label_good']:.2f} "
            if rs_info and "rate_mult" in rs_info:
                extra += f"| rate x{rs_info['rate_mult']:.3f} (cos ema {rs_info['cos_ema']:+.3f}) "
            if rs_info and "mb_choice" in rs_info:
                extra += f"| beta choice {rs_info['mb_choice']} "
            if rs_info and "accum_candidates" in rs_info:
                extra += (f"| accum flips {rs_info['accum_flips']} of {rs_info['accum_candidates']} candidates "
                          f"(fraction {rs_info['accum_frac']:.3g}) ")
            if rs_info and "mech_flips" in rs_info:
                extra += f"| mech flips {rs_info['mech_flips']} "
                if "mech_carried" in rs_info:
                    extra += f"carried/new {rs_info['mech_carried']:.3g}/{rs_info['mech_fresh']:.3g} "
                if "mech_cosFD" in rs_info:
                    extra += f"cos(F,target) {rs_info['mech_cosFD']:+.2f} resets {rs_info['mech_resets']} "
            if rs_info and "la_kept" in rs_info:
                extra += f"| la kept {rs_info['la_kept'] / max(rs_info['la_proposed'], 1) * 100:.0f}% "
            if fs:
                extra += (f"| flip {fs['flip_frac_total']*100:.3f}% "
                          f"| never {fs['never_frac_total']*100:.1f}% ")
            print(f"step {step:6d} | loss {last_loss:6.3f} | lr {lr:.2e} {extra}"
                  f"| {dt:6.1f}s | peakVRAM {mem:.2f}GiB", flush=True)
            if args.ts and step in (10, 1000):
                print(f"  --ts state: {_ts_state_bytes(lr_state) / 2**20:.1f} MiB stored", flush=True)
            if args.mem_report and step in (10, 50, 100, 1000):
                mr = mem_report(model, tail_opt, lr_state)
                log_metrics({"step": step, **mr})
                print("  memory (MiB): " + ", ".join(f"{k[4:-3]} {v:.1f}" for k, v in mr.items()), flush=True)
            if ORTH_STATS:
                e = torch.tensor(ORTH_STATS); ORTH_STATS.clear()
                print(f"  orthdbg ({len(e)} calls): |Q^T Q - I| max {e[:, 0].max():.1e} | span error vs Householder max "
                      f"{e[:, 1].max():.1e} median {e[:, 1].median():.1e} | input condition max {e[:, 2].max():.1e} "
                      f"median {e[:, 2].median():.1e} | zero inputs {int(e[:, 3].sum())} zero outputs "
                      f"{int(e[:, 4].sum())}", flush=True)
        if step > 0 and step % tc.eval_interval == 0:
            vl = evaluate(model, val_data, tc, device)
            log_metrics({"step": step, "val_loss": vl, "val_ppl": math.exp(vl)})
            print(f"  ---- val loss {vl:.3f} | ppl {math.exp(vl):.1f}", flush=True)
        if args.snap_every and step > 0 and step % args.snap_every == 0:
            save_ckpt(step)
            shutil.copyfile(ckpt_path, os.path.join(tc.out_dir, f"ckpt_{step}.pt"))
        if time.time() - last_save >= args.save_secs:
            save_ckpt(step)
            last_save = time.time()
            print(f"  ---- checkpoint saved at step {step}", flush=True)
        if _last["stop"]:
            if mwin is not None and mwin.rec is not None:
                mwin.rec.flush()
            save_ckpt(step); print(f"saved at step {step}. exiting.", flush=True)
            raise SystemExit(0)

    if mwin is not None and mwin.rec is not None:
        mwin.rec.flush()
    save_ckpt(end_step - 1)
    vl = evaluate(model, val_data, tc, device)
    log_metrics({"step": end_step, "val_loss": vl, "val_ppl": math.exp(vl),
                 "final": True})
    print(f"FINAL val loss {vl:.4f} | ppl {math.exp(vl):.2f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
