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
from .master import build_master_transformer, split_params
from .flip import KernelTernaryLinear, build_kernel_transformer, enable_flip_tracking, collect_flip_stats


_KD_SHARD = {}


def _kd_window(args, idx):
    """--kd_cache: (start, ids [seq, k], log-probs [seq, k]) of window idx of the precomputed teacher targets (shards
    kd_00000.npz, ... of B windows each); waits for a shard that is still being written, wraps around once the cache
    is marked done."""
    import glob
    if "B" not in _KD_SHARD:
        while not glob.glob(os.path.join(args.kd_cache, "kd_00000.npz")):
            time.sleep(30)
        _KD_SHARD["B"] = len(np.load(os.path.join(args.kd_cache, "kd_00000.npz"))["starts"])
    B = _KD_SHARD["B"]
    sh, j = divmod(max(idx, 0), B)
    fn = os.path.join(args.kd_cache, f"kd_{sh:05d}.npz")
    if _KD_SHARD.get("sh") != sh:
        waited = 0
        while not os.path.exists(fn):
            if os.path.exists(os.path.join(args.kd_cache, "done")):     # the cache ran out (or has a gap: a shard
                have = sorted(glob.glob(os.path.join(args.kd_cache, "kd_*.npz")))   # claimed but never written):
                fn = have[sh % len(have)]; break                               # the existing shards, in order
            if waited % 600 == 0:
                print(f"--kd_cache: waiting for {fn}", flush=True)
            time.sleep(30); waited += 30
        z = np.load(fn)
        _KD_SHARD.update(sh=sh, starts=z["starts"], ids=z["ids"], lp=z["lp"])
    j = j % len(_KD_SHARD["starts"])                                    # a shorter (last) shard: wrap within it
    return int(_KD_SHARD["starts"][j]), _KD_SHARD["ids"][j], _KD_SHARD["lp"][j]


def _kd_cache_batch(args, idx, data, seq_len, device, bs=1):
    """--kd_cache: micro-batch idx = windows idx * bs ... idx * bs + bs - 1 of the cache (bs 1: window idx, as before).
    Returns x, y [bs, seq] and (ids [bs * seq, k], log-probs [bs * seq, k]) on device, rows in the order of x.flatten."""
    ws, ids_, lps_ = [], [], []
    for b in range(bs):
        s0, ids, lp = _kd_window(args, idx * bs + b)
        ws.append(torch.from_numpy(data[s0:s0 + seq_len + 1].astype(np.int64)))
        ids_.append(torch.from_numpy(ids.astype(np.int64))); lps_.append(torch.from_numpy(lp))
    if str(device).startswith("cuda"):   # pinned + non-blocking: a copy from pageable memory waits for the GPU's queue
        put = lambda t_: t_.pin_memory().to(device, non_blocking=True)
    else:
        put = lambda t_: t_.to(device)
    w = put(torch.stack(ws))
    return w[:, :-1], w[:, 1:], (put(torch.cat(ids_)), put(torch.cat(lps_)))


def get_batch(data, bs, seq_len, device, gen=None):
    # gen: dedicated RNG so the data order is identical across runs regardless of
    # what else consumes the global RNG (different modes draw differently).
    ix = torch.randint(len(data) - seq_len - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


LR_LINEAR = [False]               # --lr_linear: linear decay after warmup instead of the cosine
LR_WSD = [0]                      # --lr_wsd D: warmup, constant peak lr until step D, then a cosine to min_lr at the end
LR_REWARM = [0, 0]                # --lr_rewarm S:W: from step S, linear warmup over W steps from min_lr to the peak, then cosine
TS_FUSED_STATS = [True]          # TERN_FUSED_STATS=0: the TTF row / column sums of g^2 by torch reductions (the old path)


def _ts_scale(args, step, tc, lr):
    """the TTF step scale (lr / peak lr)^p, p = --ts_anneal"""
    return max((lr / tc.lr) ** args.ts_anneal, 0.0)


def _tern_lays(model):
    from .tern_emb import tern_emb_layers
    return tern_emb_layers(model)


def _theta_at(sched, prog):
    """--ts_theta_sched T0:T1[:START]: T0 until fraction START of training (default 0), then linear to T1 at the end"""
    th0_, th1_, st_ = (tuple(sched) + (0.0,))[:3]
    prog = min(max(prog, 0.0), 1.0)
    return th0_ if prog <= st_ else th0_ + (th1_ - th0_) * (prog - st_) / max(1e-9, 1.0 - st_)


def lr_at(step, tc: TrainConfig):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step > tc.max_steps:
        return tc.min_lr
    if LR_REWARM[1] and step >= LR_REWARM[0]:
        # continued training of a decayed checkpoint: re-warm from min_lr to the peak, then a fresh cosine to the end
        s0, w = LR_REWARM[:2]
        pk = tc.lr * (LR_REWARM[2] if len(LR_REWARM) > 2 else 1.0)   # S:W:F re-warms to F x the peak (TTF's step too)
        if step < s0 + w:
            return tc.min_lr + (pk - tc.min_lr) * (step - s0 + 1) / w
        r = (step - s0 - w) / max(1, tc.max_steps - s0 - w)
        return tc.min_lr + 0.5 * (pk - tc.min_lr) * (1 + math.cos(math.pi * r))
    if LR_WSD[0]:
        # warmup-stable-decay: the peak lr until step D, then a cosine down to min_lr at the last step
        if step < LR_WSD[0]:
            return tc.lr
        r = (step - LR_WSD[0]) / max(1, tc.max_steps - LR_WSD[0])
        return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))
    r = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    if LR_LINEAR[0]:              # --lr_linear: a straight line from the peak to min_lr (to zero with --min_lr 0)
        return tc.min_lr + (tc.lr - tc.min_lr) * (1.0 - r)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def evaluate(model, data, tc, device):
    model.eval()
    STATE.updates_enabled = False
    losses = []
    # fixed windows: same val batches every eval and across runs (comparable). At least 16 windows per batch whatever
    # the training batch size (before 5 Oct a run at batch 4 was scored on the first quarter of a batch-16 run's windows,
    # so training-time val losses of different batch sizes were not comparable), run in chunks of the batch size
    gen = torch.Generator().manual_seed(tc.seed + 12345)
    bs = max(tc.batch_size, 16)
    for _ in range(tc.eval_iters):
        x, y = get_batch(data, bs, tc.seq_len, device, gen)
        for a in range(0, bs, tc.batch_size):
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x[a:a + tc.batch_size], y[a:a + tc.batch_size])
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _store_codec(name):
    """--ts_store_a: compressed storage of the accumulator's low-rank pair. Returns enc(X) -> tuple of stored tensors:
    int8 / int8det / fp8 / fp16s keep codes + a per-column fp32 scale (1 x r), fp16 / bf16 keep the tensor in that
    dtype."""
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
            sc = cs(X, 32768)
            if X.is_cuda and X.dim() == 2 and CODEC_FUSED[0]:   # one kernel, identical codes (kernel.pack12_scaled)
                from .kernel import pack12_scaled
                return (pack12_scaled(X.float(), sc), sc)
            return (_pack12((X / sc).to(torch.float16)), sc)
    elif name in ("fp16", "bf16"):
        dt = torch.float16 if name == "fp16" else torch.bfloat16
        def enc(X):
            return (X.to(dt),)
    else:
        raise ValueError(name)
    return enc


CODEC_FUSED = [os.environ.get("TERN_CODEC_FUSED", "1") != "0"]   # fp12s through kernel.pack12_scaled / unpack12_scaled


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
    if c.dtype == torch.uint8 and c.is_cuda and c.dim() == 2 and CODEC_FUSED[0]:   # one kernel, identical values
        from .kernel import unpack12_scaled
        return unpack12_scaled(c, s)
    return (_unpack12(c, s.shape[-1]) if c.dtype == torch.uint8 else c).float() * s


def _store_pair(enc, U, V):
    with torch.profiler.record_function("ttf.codec"):
        return _store_pair_(enc, U, V)


def _store_pair_(enc, U, V):
    """(U, V) -> flat tuple of stored tensors (U codes[, U scale], V codes[, V scale]), so checkpoint / resume code that
    moves tuples of tensors works unchanged"""
    return enc(U) + enc(V)


def _load_pair(t):
    with torch.profiler.record_function("ttf.codec"):
        return _load_pair_(t)


def _load_pair_(t):
    """inverse of _store_pair (and a no-op on a plain fp32 (U, V))"""
    if len(t) == 4:
        return _decode(t[0], t[1]), _decode(t[2], t[3])
    return t[0].float(), t[1].float()


def _ts_on_host(args, i, n):
    """--ts_offload with --ts_offload_keep F: layer i of n (module order) keeps its state in host memory unless it is
    among the first round(F n)"""
    return bool(args.ts_offload) and i >= round(args.ts_offload_keep * n)


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
    with torch.profiler.record_function("ttf.store"):
        return _host_get_(d, k, dev, off)


def _host_get_(d, k, dev, off):
    """--ts_offload: a state tuple kept in pinned host memory, copied to the GPU for this layer's update (the copy is
    queued on the current stream, so it is ordered before the kernels that read it; no host sync)"""
    v = d[k]
    return tuple(x.to(dev, non_blocking=True) for x in v) if off else v


def _host_put(d, k, v, off):
    with torch.profiler.record_function("ttf.store"):
        return _host_put_(d, k, v, off)


def _host_put_(d, k, v, off):
    """--ts_offload: write a state tuple back to its pinned host buffers (reused across steps; the device-to-host copy
    is queued on the same stream after this step's kernels and before next step's host-to-device copy)"""
    if not off:
        d[k] = v
        return
    old = d.get(k)
    if (old is not None and len(old) == len(v) and all(o.device.type == "cpu" and o.shape == x.shape and o.dtype == x.dtype
                                                       and _PINNED.owns(o) for o, x in zip(old, v))):
        for o, x in zip(old, v):
            o.copy_(x, non_blocking=True)
    else:
        d[k] = tuple(_PINNED.empty(x.shape, x.dtype).copy_(x, non_blocking=True) for x in v)


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


def _orth(X):
    with torch.profiler.record_function("ttf.orth"):
        return _orth_(X)


def _orth_(X):
    """orthonormal basis of the columns of X (tall, K x r), Householder QR. Only the span matters to TTF
    (A = U V^T does not depend on the basis chosen)."""
    return torch.linalg.qr(X)[0]


ORTH_DBG = [False]                # on for one step in 10 when TERN_ORTHDBG=1


_POW3 = {}


_TS_FAST_OFF = ("shared_v", "global", "global_", "fuse", "qfmt", "qfmt_m", "qfmt_a", "qfmt_aV", "int8",
                "store_m", "store_a", "shadows", "shadow", "resid", "sketch", "rescale", "refresh",
                "rank_sched", "locate", "table", "spend_table", "exact_spend", "rsketch", "decay", "gate", "sgate",
                "orth_defer", "diag", "qdiag")


def ts_layer(l, g, state, c):
    """one layer of TTF (the no-momentum update): g is this layer's gradient (fp32), or its factors (gy, x) with
    --ts_factored; c holds the settings and lr_ratio. Called from the layer's backward right after its input gradient
    is computed (--ts_fused), so only one layer's fp32 gradient exists at a time.
    Per layer: v, the factored row x column EMA of g^2 (decay b2); u = -lr_ratio g / sqrt(v); A, the accumulator, rank
    r: A <- (1 - lr_ratio^p / tau) A + u projected on its basis (V turns with the evidence); a trit moves by sign(A_ij)
    when |A_ij| >= theta (and stays in {-1, 0, 1}), and the move spends theta: A -= theta D (kept low-rank: U -= theta
    D V). With --ts_micro (grad_accum M) every micro-batch's u enters the accumulator, normalised by the v of the steps
    before; v, the basis turn, the firing and the spend once per step, on the last micro-batch."""
    from .kernel import unpack_rows, pack_rows
    if not torch.is_tensor(g):           # --ts_factored: the backward passed (gy, x) instead of the dense gradient
        return _ts_layer_factored(l, g, state, c)
    r, lr_ratio, theta, tau, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    if c.get("theta_sched"):   # --ts_theta_sched T0:T1[:START]: the threshold moves linearly from T0 to T1
        theta = _theta_at(c["theta_sched"], c.get("progress", 0.0))
    M_ = int(c.get("micro") or 1)   # --ts_micro with --grad_accum M: the accumulator takes every micro-batch's u
    final_ = getattr(l, "_ts_final", True) if M_ > 1 else True   # firing and the spend only on the last one
    b2o = b2
    if M_ > 1:
        b2 = b2 ** (1.0 / M_)
    sa = _store_codec(c.get("store_a", ""))                                            # --ts_store_a: real storage
    st = state.setdefault("_ts", {})
    off, dev = (c.get("offload", False) if getattr(l, "_ts_host", True) else False), g.device  # --ts_offload(_keep)
    key = id(l); t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    mic_nm = M_ > 1
    if mic_nm:    # --ts_micro: the accumulator takes every micro-batch's u, normalised by the v of the steps before
        # (this step's v needs all of its micro-batches); on the first step there is none, so each micro-batch uses its
        # own g^2 x M^2 (each g_i carries the loss / M). Firing and the spend once per step, on the last one. As Adam's
        # v includes the current gradient (which bounds a step at 1 / sqrt(1 - beta2) per weight), this micro-batch's
        # own square (x M^2, the batch's scale) joins with weight 1 - beta2; without it a gradient spike passes into the
        # accumulator whole (110M: a 6% flip burst at step ~55, 5.64 vs 4.42 at 600 steps)
        ts_ = (t - 1) // M_
        if g.is_cuda and TS_FUSED_STATS[0]:   # one pass for both sums of g^2 (rounding-level differences only)
            from .kernel import sq_sums as _sq_sums
            _rsq, _csq = _sq_sums(g)
            r0, c0 = _rsq * (M_ * M_ / l.K), _csq * (M_ * M_ / l.N)
        else:
            _rsq = None
            r0, c0 = torch.linalg.vector_norm(g, dim=1).square_().mul_(M_ * M_ / l.K), \
                torch.linalg.vector_norm(g, dim=0).square_().mul_(M_ * M_ / l.N)
        if ("v", key) in st and ts_ > 0:
            R0, C0 = _host_get(st, ("v", key), dev, off)
            bc_ = 1 - b2o ** ts_
            r0 = b2o * R0 / bc_ + (1 - b2o) * r0; c0 = b2o * C0 / bc_ + (1 - b2o) * c0; del R0, C0
        if g.is_cuda:     # the fused kernel: u in one pass, sqrt(v) never formed (bit-identical to the expression below)
            sq_prev = (r0, c0, r0.mean().clamp_min(1e-30))
        else:
            sq_prev = (r0[:, None] * c0[None, :] / r0.mean().clamp_min(1e-30)).sqrt_()
        del r0, c0
    fs_ = mic_nm and _rsq is not None and not c.get("vdiag")   # the fused sums stand in for g^2's means
    g2 = None if fs_ else g * g
    if M_ > 1:    # --ts_micro: v is the batch's, stepped once per step with the step's beta2 as without micro-batches:
        # row / column means of (sum_i g_i)^2 = the micro-batches' own squares + 2 <g_i, sum of earlier g_i'>, the
        # cross terms through a k-dim random sketch of the running sum (unbiased; k (N + K) floats, held within the step)
        Sr, Sc = _micro_sketch(l, dev, (t - 1) // M_)
        PS, SP, rs, cs, ds, es = st.pop(("mx", key), None) or (0, 0, 0, 0, 0, 0)
        gS, Sg = g @ Sr, Sc @ g
        d1, d0 = (_rsq / l.K, _csq / l.N) if fs_ else (g2.mean(1), g2.mean(0))
        rs = rs + d1 + (2.0 / l.K) * (gS * PS).sum(1) if torch.is_tensor(PS) else d1
        cs = cs + d0 + (2.0 / l.N) * (Sg * SP).sum(0) if torch.is_tensor(SP) else d0
        ds, es = ds + d1, es + d0
        PS, SP = PS + gS, SP + Sg
        del gS, Sg, d1, d0
        if final_:
            rs, cs = torch.maximum(rs, ds / M_), torch.maximum(cs, es / M_)   # sketch noise: never below pure noise
            if ("v", key) in st:
                R, C = _host_get(st, ("v", key), dev, off); R = b2o * R + (1 - b2o) * rs; C = b2o * C + (1 - b2o) * cs
            else:
                R, C = (1 - b2o) * rs, (1 - b2o) * cs
            _host_put(st, ("v", key), (R, C), off)
            # the full [N, K] v only where it is read: --ts_vdiag (the micro-batch u uses the factored v of the steps before)
            v = R[:, None] * C[None, :] / R.mean().clamp_min(1e-30) / (1 - b2o ** (t // M_)) if c.get("vdiag") else None
        else:
            st[("mx", key)] = (PS, SP, rs, cs, ds, es); v = None
        del PS, SP
    else:
        if ("v", key) in st:
            R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * g2.mean(1); C = b2 * C + (1 - b2) * g2.mean(0)
        else:
            R, C = (1 - b2) * g2.mean(1), (1 - b2) * g2.mean(0)
        _host_put(st, ("v", key), (R, C), off)
        v = R[:, None] * C[None, :] / R.mean().clamp_min(1e-30) / (1 - b2 ** t)
    if c.get("vdiag") and v is not None:   # --ts_vdiag: this step's g^2 against the second moment it is divided by
        sp_ = float(g2.mean() / v.mean().clamp_min(1e-30)); _ts_rec(st, "g2_over_v", sp_); _ts_recmax(st, "g2_over_v", sp_)
    del g2
    # u: the full-rank Adam-normalised gradient (no short momentum: the accumulator's own rank-r projection filters the
    # noise; its long-run sum is what a momentum would hold)
    if mic_nm and isinstance(sq_prev, tuple):
        from .kernel import ttf_u as _ttf_u
        u = _ttf_u(g, sq_prev[0], sq_prev[1], sq_prev[2], lr_ratio)
    else:
        u = -g / ((sq_prev if mic_nm else v.sqrt()) + 1e-8) * lr_ratio
    if mic_nm:
        del sq_prev
    if c.get("vdiag"):    # |u| / lr_ratio: rms, and the largest entry (a spike the leaky sum turns into a burst)
        ua_ = u.abs() / max(lr_ratio, 1e-12)
        _ts_rec(st, "u_rms", float(ua_.pow(2).mean().sqrt())); m_ = float(ua_.max())
        _ts_rec(st, "u_max", m_); _ts_recmax(st, "u_max", m_)
        _ts_rec(st, "u_gt4", float((ua_ > 4).float().mean())); del ua_
    if c.get("start") and c.get("step", 0) < c["start"]:   # --ts_start S: the trits stay frozen (no evidence kept,
        del u, v                                             # nothing fires) until step S; v keeps learning
        return
    del v
    ra = min(r, l.N, l.K)
    if mic_nm and key in state:   # --ts_micro: the basis stays fixed within the step (every micro-batch's u is projected
        # on it: U += u_i V), and turns once, on the last micro-batch, from X = V (U^T U) + M u_M^T U (the last
        # micro-batch's, scaled: no memory beyond the state); turning it per micro-batch lets each micro-batch's noisier
        # u rotate the evidence away (110M r512: 5.64 vs 4.42 at 600 steps)
        U, V = _load_pair(_host_get(state, key, dev, off))
        first_ = (t - 1) % M_ == 0
        if first_:
            U = U * (1 - (lr_ratio ** float(c["tau_anneal"]) if c["tau_anneal"] else 1.0) / tau)
        if final_:
            S_ = M_ * (u.T @ U)
            Vn = _orth(V @ (U.T @ U) + S_); del S_
            U = (U + u @ V) @ (V.T @ Vn); V = Vn
        else:
            U = U + u @ V
    elif key not in state:
        gen = torch.Generator(device=g.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        V = _orth(u.T @ torch.randn(l.N, ra, device=g.device, generator=gen))
        U = u @ V
    else:
        U, V = _load_pair(_host_get(state, key, dev, off))
        U = U * (1 - (lr_ratio ** float(c["tau_anneal"]) if c["tau_anneal"] else 1.0) / tau)   # --ts_tau_anneal: memory in distance, not steps
        X_ = V @ (U.T @ U) + u.T @ U
        Vn = _orth(X_); del X_
        uVn = u @ Vn
        U = U @ (V.T @ Vn) + uVn; V = Vn
        del uVn
    if mic_nm and not final_:   # --ts_micro: not the last micro-batch: keep the accumulator, no firing
        _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
        return
    del u
    A = U @ V.T
    w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
    d = A.sign()
    fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
    del A
    before = l.wpacked.clone() if l.track else None
    D = torch.where(fire, d, torch.zeros_like(d))
    nf = fire.sum()                           # a device tensor: no host sync per layer
    l.wpacked.copy_(pack_rows(w0 + D.to(torch.int8)))   # int8 add: exact, no [N, K] fp32 temporary
    U = U - c["spend"] * (theta * (D @ V))
    _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + fire.numel()
    if before is not None:
        l._record_flips(before, l.wpacked)


def _mmb(A, B):
    """A (bf16, the factor) @ B on the bf16 tensor cores with fp32 accumulation and output (B rounded to bf16, as the
    dense path rounds the weight gradient to bf16)"""
    return torch.mm(A, B.to(A.dtype), out_dtype=torch.float32) if A.dtype == torch.bfloat16 else A @ B


def _fg_mm_r(fg, a, W):
    """u @ W for u = -lr diag(a) g diag(b), g = gy^T x, without g: -lr a * (gy^T (x (b W))) (fp32; W is K x r)"""
    gy, x, b = fg
    return a[:, None] * _mmb(gy.T, _mmb(x, b[:, None] * W))


def _fg_mm_l(fg, a, W):
    """u^T @ W (W is N x r) without g: -lr b * (x^T (gy (a W))) (the -lr is applied by the caller)"""
    gy, x, b = fg
    return b[:, None] * _mmb(x.T, _mmb(gy, a[:, None] * W))


def _ts_layer_factored(l, fg, state, c):
    """--ts_factored K: the --ts --ts_nomom update (16 x 1 or --ts_micro) from the factors of the weight gradient,
    g = gy^T x, which is never formed. Everything the update reads from g is a product with a thin matrix: the
    accumulator's projections u V and u^T U (u = -lr g / sqrt(v) with the factored v = a_i b_j, so u = -lr diag(1/a) g
    diag(1/b)), and v's row / column sums of g^2, estimated from a K-column random sketch g S (E[S S^T] = I; with
    --ts_micro the same sketch the cross terms use, so the batch's sums are ||(sum_i g_i) S||^2). Differences from the
    dense path: the sums of g^2 are sketch estimates (relative noise sqrt(2 / K) per row and step, before v's EMA), and
    the 1e-8 in sqrt(v) + 1e-8 is left out (a scale factor needs a product form). Cost per call 2 T (N + K)(K_s + 2 r)
    instead of the dense gradient's 2 T N K."""
    gy, x = fg                                # [T, N], [T, K] (bf16; products accumulate in fp32: _mmb)
    r, lr_ratio, theta, tau, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    if c.get("theta_sched"):
        theta = _theta_at(c["theta_sched"], c.get("progress", 0.0))
    M_ = int(c.get("micro") or 1)
    final_ = getattr(l, "_ts_final", True) if M_ > 1 else True
    st = state.setdefault("_ts", {})
    dev = gy.device
    off = c.get("offload", False) if getattr(l, "_ts_host", True) else False
    sa = _store_codec(c.get("store_a", ""))
    key = id(l); t = st.get(("t", key), 0) + 1; st[("t", key)] = t
    lk = 1 - (lr_ratio ** float(c["tau_anneal"]) if c["tau_anneal"] else 1.0) / tau
    k_ = int(c["factored"])
    ts_ = (t - 1) // M_
    gen = torch.Generator(device=dev).manual_seed(11_000_000 + 2 * getattr(l, "_ts_idx", 0) + 1_000_003 * ts_)
    Sr = torch.randn(l.K, k_, device=dev, generator=gen) / k_ ** 0.5
    Sc = torch.randn(k_, l.N, device=dev, generator=gen) / k_ ** 0.5
    gS = _mmb(gy.T, _mmb(x, Sr))                                          # g Sr  [N, k]
    Sg = _mmb(x.T, _mmb(gy, Sc.T)).T                                      # Sc g  [k, K]
    rsq, csq = gS.square().sum(1), Sg.square().sum(0)                     # ~ row / column sums of g^2
    if M_ > 1:   # as ts_layer's --ts_micro --ts_nomom branch, with the sketched own squares
        r0, c0 = rsq * (M_ * M_ / l.K), csq * (M_ * M_ / l.N)
        if ("v", key) in st and ts_ > 0:
            R0, C0 = _host_get(st, ("v", key), dev, off)
            bc_ = 1 - b2 ** ts_
            r0 = b2 * R0 / bc_ + (1 - b2) * r0; c0 = b2 * C0 / bc_ + (1 - b2) * c0; del R0, C0
        m_ = r0.mean().clamp_min(1e-30)
        PS, SP, rs, cs, ds, es = st.pop(("mx", key), None) or (0, 0, 0, 0, 0, 0)
        if torch.is_tensor(PS):    # the running sketch sums are held in bf16 between micro-batches (k (N + K) x 2 bytes
            PS, SP = PS.float(), SP.float()   # per layer; their rounding, 2^-9, is far below the sketch's own noise)
        d1, d0 = rsq / l.K, csq / l.N
        rs = rs + d1 + (2.0 / l.K) * (gS * PS).sum(1) if torch.is_tensor(PS) else d1
        cs = cs + d0 + (2.0 / l.N) * (Sg * SP).sum(0) if torch.is_tensor(SP) else d0
        ds, es = ds + d1, es + d0
        PS, SP = PS + gS, SP + Sg
        if final_:
            rs, cs = torch.maximum(rs, ds / M_), torch.maximum(cs, es / M_)
            if ("v", key) in st:
                R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * rs; C = b2 * C + (1 - b2) * cs
            else:
                R, C = (1 - b2) * rs, (1 - b2) * cs
            _host_put(st, ("v", key), (R, C), off)
            if c.get("vdiag"):   # --ts_vdiag: this micro-batch's g^2 against v (the u extremes need the dense u: not kept)
                sp_ = float((rsq.sum() / (l.N * l.K)) / (C.mean() / (1 - b2 ** (t // M_))).clamp_min(1e-30))
                _ts_rec(st, "g2_over_v", sp_); _ts_recmax(st, "g2_over_v", sp_)
        else:
            st[("mx", key)] = (PS.to(torch.bfloat16), SP.to(torch.bfloat16), rs, cs, ds, es)
        del PS, SP
        a = (r0 / m_).sqrt_(); b = c0.sqrt()                              # sqrt(v_ij) = a_i b_j
    else:        # 16 x 1: v includes this step's g^2
        if ("v", key) in st:
            R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * rsq / l.K; C = b2 * C + (1 - b2) * csq / l.N
        else:
            R, C = (1 - b2) * rsq / l.K, (1 - b2) * csq / l.N
        _host_put(st, ("v", key), (R, C), off)
        if c.get("vdiag"):
            sp_ = float((rsq.sum() / (l.N * l.K)) / (C.mean() / (1 - b2 ** t)).clamp_min(1e-30))
            _ts_rec(st, "g2_over_v", sp_); _ts_recmax(st, "g2_over_v", sp_)
        a = (R / R.mean().clamp_min(1e-30) / (1 - b2 ** t)).sqrt(); b = C.sqrt()
    del gS, Sg
    ia = 1.0 / torch.maximum(a, a.max() * 1e-12 + 1e-30)                 # 1 / a_i, 1 / b_j (zero rows: g is zero there;
    ib = 1.0 / torch.maximum(b, b.max() * 1e-12 + 1e-30)                 # device-side bounds: no host sync per call)
    fgb = (gy, x, ib)
    if c.get("start") and c.get("step", 0) < c["start"]:                  # --ts_start: frozen, v keeps learning
        return
    ra = min(r, l.N, l.K)
    if key not in state:
        g_ = torch.Generator(device=dev).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        V = _orth(-lr_ratio * _fg_mm_l(fgb, ia, torch.randn(l.N, ra, device=dev, generator=g_)))
        U = -lr_ratio * _fg_mm_r(fgb, ia, V)
    elif M_ > 1:   # basis fixed within the step, turned on the last micro-batch from M u_M^T U
        U, V = _load_pair(_host_get(state, key, dev, off))
        if (t - 1) % M_ == 0:
            U = U * lk
        if final_:
            S_ = (-lr_ratio * M_) * _fg_mm_l(fgb, ia, U)
            Vn = _orth(V @ (U.T @ U) + S_); del S_
            U = (U - lr_ratio * _fg_mm_r(fgb, ia, V)) @ (V.T @ Vn); V = Vn
        else:
            U = U - lr_ratio * _fg_mm_r(fgb, ia, V)
    else:
        U, V = _load_pair(_host_get(state, key, dev, off))
        U = U * lk
        Vn = _orth(V @ (U.T @ U) - lr_ratio * _fg_mm_l(fgb, ia, U))
        U = U @ (V.T @ Vn) - lr_ratio * _fg_mm_r(fgb, ia, Vn); V = Vn
    if not final_:
        _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
        return
    from .kernel import unpack_rows, pack_rows
    before = l.wpacked.clone() if l.track else None
    if FIRE_FUSED[0] and dev.type == "cuda":   # --ts_fire_fused: A, the moves, the repack and D V in one kernel
        from .kernel import ttf_fire
        DV, nf = ttf_fire(U, V, l.wpacked, l.K, theta)
        U = U - c["spend"] * (theta * DV); del DV
    else:
        A = U @ V.T
        w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
        d = A.sign()
        fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
        del A
        D = torch.where(fire, d, torch.zeros_like(d))
        nf = fire.sum()
        l.wpacked.copy_(pack_rows(w0 + D.to(torch.int8)))
        U = U - c["spend"] * (theta * (D @ V))
    _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + l.N * l.K
    if before is not None:
        l._record_flips(before, l.wpacked)


MICRO_SKETCH = [int(os.environ.get("TERN_MICRO_K", "64"))]   # sketch size (TERN_MICRO_K: tests only)
FIRE_FUSED = [False]          # --ts_fire_fused (with --ts_factored): ttf.kernel.ttf_fire


def _micro_sketch(l, dev, step):
    """--ts_micro: a layer's random maps (K x k, k x N; E[S S^T] = I) for the cross terms of the batch's v; fixed within
    a step, new every step, so their errors are independent across steps and v's EMA averages them out"""
    k = MICRO_SKETCH[0]
    gen = torch.Generator(device=dev).manual_seed(11_000_000 + 2 * getattr(l, "_ts_idx", 0) + 1_000_003 * step)
    return (torch.randn(l.K, k, device=dev, generator=gen) / k ** 0.5,
            torch.randn(k, l.N, device=dev, generator=gen) / k ** 0.5)


MICRO_DBG = [os.environ.get("TERN_MICRO_DBG", "")]   # diagnostic of --ts_micro (see _micro_dbg_u)


_GLOBAL_OUT_SIDE = (3, 6)   # o and down write to the residual stream (output side); q k v gate up read it (input side)


def _ts_rec(st, name, x):
    d = st.setdefault("_dg", {}); n = d.setdefault(name, [0.0, 0]); n[0] += x; n[1] += 1


def _ts_recmax(st, name, x):
    d = st.setdefault("_dgmax", {}); d[name] = max(d.get(name, -float("inf")), x)


def ts_step(model, state, c, fused=True):
    """after the backward: with --ts_fused (every recipe) the layers have already updated inside their backward, so this
    only collects the step's statistics (share of trits moved, --ts_vdiag means / maxima); without it each layer's
    captured gradient goes through ts_layer here"""
    st = state.setdefault("_ts", {})
    if not fused:
        for i, l in enumerate([l for l in model.modules() if isinstance(l, KernelTernaryLinear)]):
            l._ts_idx = i
            g = l.gw.float(); l.gw = None; l.capture = False
            ts_layer(l, g, state, c)
            del g
    nf, nt = st.pop("_nf", 0), st.pop("_nt", 0)
    info = {"ts_frac": float(nf) / max(nt, 1)}
    for k_, (tot, n_) in st.pop("_dg", {}).items():      # --ts_vdiag: means over layers this step
        info["dg_" + k_] = tot / max(n_, 1)
    for k_, x_ in st.pop("_dgmax", {}).items():           # --ts_vdiag: maxima over layers this step
        info["dgmax_" + k_] = x_
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--mode", default="kernel",
                    choices=["kernel", "master", "fp32"],
                    help="kernel = packed trits on the Triton GEMM (trained by TTF with --ts); master = BitNet b1.58 "
                         "(latent weights + STE + AdamW); fp32 = the full-precision model")
    ap.add_argument("--master_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="master mode: latent weight dtype (bf16 rounds away small "
                         "AdamW steps; fp32 is the honest ceiling)")
    ap.add_argument("--lowrank", type=int, default=0,
                    help="flip signal = rank-r momentum of each layer's gradient (M ~ U V^T, "
                         "r*(N+K) floats per layer) instead of the current gradient")
    ap.add_argument("--profile", type=int, default=0,
                    help="profile this many steps (after 20 warm-up steps), print the tables, exit")
    ap.add_argument("--qk_temp", action="store_true",
                    help="learnable per-head attention temperature (log-scale per head, no weight decay)")
    ap.add_argument("--lowrank_mag", default="",
                    help="low-rank float magnitude beyond the trits: 'add:R' (W = bT + AB^T) or 'mul:R' "
                         "(W = bT o (1 + AB^T))")
    ap.add_argument("--mag_wd", type=float, default=0.0, help="weight decay of the --lowrank_mag parameters")
    ap.add_argument("--mag_lr_mult", type=float, default=1.0, help="LR multiplier of the --lowrank_mag parameters")
    ap.add_argument("--ts", action="store_true",
                    help="two timescales (ts_step): short low-rank momentum for the direction, long low-rank accumulator "
                         "of Adam-normalized steps, a trit moves when the accumulator crosses theta (no flip rate)")
    ap.add_argument("--ts_theta", type=float, default=16.0, help="--ts: firing threshold (latent half-bin / peak lr)")
    ap.add_argument("--mem_cap_gib", type=float, default=0.0, help="cap PyTorch's memory on the GPU at this many GiB "
                    "(memory-matched comparisons; 0: no cap)")
    ap.add_argument("--init_hf", default="", help="start from a pretrained Llama checkpoint in Hugging Face format (a "
                    "directory with config.json and safetensors); the model shape comes from its config (--preset is ignored). "
                    "See ttf/hf_init.py")
    ap.add_argument("--init_hf_method", default="als", choices=["absmean", "row", "als", "gptq", "gptqc", "gptqs", "zero", "scratch", "native", "random"],
                    help="--init_hf, kernel mode: how the weights become trits and scales (ttf/hf_init.py)")
    ap.add_argument("--distill", default="", help="a full-precision teacher (Hugging Face Llama dir): the loss is "
                    "(1 - W) CE + W KL(teacher || student) per token (--distill_w W)")
    ap.add_argument("--distill_w", type=float, default=1.0, help="--distill: weight of the KL term (1: KL only)")
    ap.add_argument("--kd_cache", default="", help="distil from precomputed teacher targets (scripts/quant/teacher_topk.py: fixed windows of --data with the teacher's top-k log-probabilities): each micro-batch takes the next window in order; the loss is (1 - --distill_w) CE + --distill_w x cross-entropy to the renormalised top k (Qwen models, batch_size 1)")
    ap.add_argument("--kd_mode", default="renorm", choices=["rest", "renorm"], help="--kd_cache: rest = the top k plus one bucket for the teacher's remaining mass (the KL on that distribution); renorm = the top k renormalised (drops the tail)")
    ap.add_argument("--kd_cache_start", type=int, default=0, help="--kd_cache: the training step that reads window 0 (the step a run is resumed at when the cache is introduced)")
    ap.add_argument("--moe", default="", help="E:K[:H]: mixture-of-experts feed-forward, E ternary SwiGLU experts (hidden H, default "
                    "the dense one), top-K routing (ttf/model.py MoEFeedForward)")
    ap.add_argument("--moe_aux", type=float, default=0.01, help="--moe: weight of the load-balancing loss")
    ap.add_argument("--init_stream", action="store_true", help="--init_hf (Qwen): convert layer by layer, sequentially, reading "
                    "each layer's weights from the safetensors (27B: no full-precision copy in memory)")
    ap.add_argument("--data_u32", action="store_true", help="--data / --val are uint32 token streams (vocabularies above 65536)")
    ap.add_argument("--hid_kd", type=float, default=0.0, help="--distill: W > 0 adds W x the error of every block's "
                    "update against the teacher block's (mean and varying parts, each relative), along the student's "
                    "own path, to the global loss (more targets per token than the logits)")
    ap.add_argument("--init_calib", type=int, default=64, help="--init_hf_method gptq: calibration windows "
                    "(of --seq_len tokens) for the input second moments")
    ap.add_argument("--ts_theta_sched", default="", help="--ts: T0:T1[:START], the threshold is T0 until fraction START of training (default 0), then moves "
                    "linearly to T1 at the end")
    ap.add_argument("--ts_start", type=int, default=0, help="--ts: the trits stay frozen until step S (no evidence is kept "
                    "and nothing fires; the floats and the second moment train): calibrate a converted model first")
    ap.add_argument("--ts_vdiag_every", type=int, default=1, help="--ts_vdiag on every N-th step only (it is "
                    "logging alone, but its full-size passes and host reads cost ~28k GPU stalls per 27B step)")
    ap.add_argument("--ts_vdiag", action="store_true", help="--ts: log per step the gradient's size against the second "
                    "moment it is divided by (g^2 / v, mean and max over layers) and the |u| tail")
    ap.add_argument("--ts_nomom", action="store_true", help="--ts: no short momentum; the accumulator takes the "
                    "full-rank g / sqrt(v) each step (the momentum's state freed)")
    ap.add_argument("--ts_spend", type=float, default=1.0,
                    help="--ts: a move spends this x theta (1: to the new trit's centre, like master --m_snap; 2: stays at "
                         "the boundary it crossed, like master)")
    ap.add_argument("--gemm", default="triton", choices=["triton", "cublas", "int8mm"],
                    help="ternary GEMMs: our Triton kernels, or cuBLAS bf16 on trits unpacked per call (faster on A100)")
    ap.add_argument("--ts_store_a", default="", help="--ts: REAL storage of the accumulator (same formats, and fp12s: "
                    "fp16s rounded to 6 mantissa bits, two values in 3 bytes)")
    ap.add_argument("--tern_emb", default="", help="tied | untied: ternary token embedding (and output head) with "
                    "per-row / per-column float scales; kernel mode trains its trits with TTF after the backward, master "
                    "mode keeps a float latent (STE) (ttf/tern_emb.py)")
    ap.add_argument("--lr_linear", action="store_true", help="after warmup, a linear decay from the peak lr to --min_lr "
                    "instead of the cosine (TTF's step scale follows it through the lr ratio)")
    ap.add_argument("--lr_wsd", type=int, default=0, help="warmup-stable-decay: the peak lr until step D (this), then a "
                    "cosine to --min_lr at the last step (0 = off); branch a decay off any stable-phase checkpoint")
    ap.add_argument("--lr_rewarm", default="", help="S:W, continue a decayed run: from step S a linear warmup over W steps "
                    "from --min_lr back to the peak, then a cosine to --min_lr at the last step")
    ap.add_argument("--lean_batched", action="store_true", help="--lean_tail: the low-precision 1-D parameters as one "
                    "flat pass and the 2-D ones stacked by shape (the same arithmetic; ~10x fewer kernel launches)")
    ap.add_argument("--lean_tail", action="store_true", help="float tail in bf16 (embedding, adapter) / fp16 (gains, "
                    "scales) with LeanAdam: int8 first moment, factored second moment, stochastic-rounding updates")
    ap.add_argument("--init_mag_svd", type=int, default=0, help="--init_hf gptq / gptqc with --lowrank_mag add:R: the "
                    "float adapters start from the best rank-R part of each layer's rounding error (in the input "
                    "covariance's norm) instead of 0: the lost precision moved into the float part (hf_init._mag_svd_init)")
    ap.add_argument("--ts_fire_fused", action="store_true", help="--ts_factored: the firing (A = U V^T, legal moves, "
                    "repack, the spend's D V) in one kernel (ttf.kernel.ttf_fire): no [N, K] temporaries")
    ap.add_argument("--ts_acc_scale", type=float, default=1.0, help="--resume: multiply every TTF accumulator by F (a "
                    "run continued at theta F x its old theta keeps each entry's distance to the bar: no burst of moves)")
    ap.add_argument("--fused_norm", action="store_true", help="Qwen: the RMSNorms (not the gated ones) through one fused "
                    "kernel forward and one backward (the same formula in fp32; reduction order differs)")
    ap.add_argument("--fused_scales", action="store_true", help="kernel layers with row / column scales and the 'add' "
                    "adapter: the column scale's backward and the adapter add + row scale (forward and backward) in fused "
                    "kernels (the same per-element roundings; the scale gradients' sums in another order)")
    ap.add_argument("--ts_factored", type=int, default=0, help="--ts --ts_nomom --ts_fused: K > 0: the TTF update from "
                    "the factors of the weight gradient (g = gy^T x is never formed; the sums of g^2 from a K-column "
                    "sketch; see _ts_layer_factored); --ts_vdiag keeps only g^2 / v")
    ap.add_argument("--ts_micro", action="store_true", help="--ts_fused with --grad_accum M: the TTF update sees each "
                    "micro-batch's gradient (momentum and v step per micro-batch with beta^(1/M), v / M; accumulator, firing "
                    "and spend once per step), so no dense gradient is kept between micro-batches: activations shrink M-fold")
    ap.add_argument("--ckpt_block", action="store_true", help="checkpoint each block once (its input) instead of attention "
                    "and MLP separately: half the saved activations (kernel mode; master builds dense weights per sublayer)")
    ap.add_argument("--no_final_save", action="store_true", help="no checkpoint at the end of the run (memory and "
                    "speed probes of large models)")
    ap.add_argument("--mem_snapshot", default="", help="PATH:STEP: record every CUDA allocation (with its Python stack) "
                    "from the start and dump the history to PATH (a pickle) at logged step STEP; scripts/analysis/mem_snapshot.py "
                    "attributes the peak")
    ap.add_argument("--eval_first_n", type=int, default=0, help="before training: the loss over the first N val windows in "
                    "order (qwen_check.py's windows), e.g. a conversion against the full-precision reference")
    ap.add_argument("--mem_report", action="store_true", help="log the MiB of trits, float tail, its gradients and "
                    "optimizer state, TTF state on GPU / host, allocated and peak, at steps 10 / 50 / 100 / 1000")
    ap.add_argument("--ts_offload", action="store_true", help="--ts: keep the state in pinned host memory; each layer's "
                    "state is copied to the GPU for its update and back (same stream: ordered, no host sync)")
    ap.add_argument("--ts_offload_keep", type=float, default=0.0, help="--ts_offload(_async): the state of the first "
                    "share of the ternary layers (module order) stays on the GPU, the rest in host memory (fill a GPU's "
                    "spare memory with state; 0 = all on the host)")
    ap.add_argument("--ts_fused", action="store_true",
                    help="--ts: each layer's update runs inside the backward, right after the layer's input gradient, so "
                         "the fp32 gradient of only one layer exists at a time (saves N*K*4 bytes over all layers)")
    ap.add_argument("--ts_anneal", type=float, default=1.0,
                    help="--ts: the accumulator's steps scale with (lr_t / lr_peak)^P (1: like master's latent; 0.5 milder)")
    ap.add_argument("--ts_tau_anneal_pow", type=float, default=1.0, help="--ts_tau_anneal: the leak per step is "
                    "a^Q / tau (Q = this; a = the step scale), so the memory grows as a^-Q")
    ap.add_argument("--ts_tau_anneal", action="store_true",
                    help="--ts: the leak shrinks with the accumulator's steps (decay 1 - ratio / tau): memory counted in "
                         "distance travelled, so the accumulator's level does not sink below theta as the lr decays")
    ap.add_argument("--ts_tau", type=float, default=300.0, help="--ts: the accumulator's leak time constant (steps)")
    ap.add_argument("--ts_b2", type=float, default=0.95, help="--ts: decay of the factored second moment")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for fp32 matmuls (the low-rank momentum's products); A100 fp32 is 19.5 TFLOPs, TF32 156")
    ap.add_argument("--int8", action="store_true",
                    help="kernel/evidence: s8 tensor-core path (exact int32 forward, "
                         "8-bit gy in dx, int8 dw)")
    ap.add_argument("--int8_dx", action="store_true",
                    help="also run grad_x on the s8 kernel (default: bf16 dx kernel)")
    ap.add_argument("--stop_after", type=int, default=None,
                    help="stop after this many steps WITHOUT changing the LR / flip "
                         "schedules (which still span --steps); evaluates on exit")
    ap.add_argument("--snap_every", type=int, default=0,
                    help="also keep a copy of the checkpoint every N steps (ckpt_<step>.pt)")
    ap.add_argument("--tail_fp32", action="store_true",
                    help="float tail (embeddings, norm gains) in fp32 + AdamW, as in master "
                         "mode (default bf16 + 8-bit Adam freezes the norm gains at 1.0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="run seed (init, data order, look-ahead batches); default TrainConfig.seed = 1337")
    ap.add_argument("--track_flips", action="store_true",
                    help="log per-layer flip rate + never-changed fraction each step")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--extend_to", type=int, default=0, help="train until this step: after --steps the schedule's lr stays "
                    "at --min_lr (continuing a finished run from its checkpoint with --resume)")
    ap.add_argument("--weight_decay", type=float, default=None, help="weight decay of the float tail's embedding / adapter group (default 0.1)")
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
    ap.add_argument("--pipe", action="store_true", help="Qwen, under torchrun --nproc_per_node G: pipeline parallel over G "
                    "GPUs (stage r runs a contiguous block of decoder layers; only hidden states cross stages, nothing is "
                    "all-reduced); every stage reads the same batches. Checkpoints: one per stage (OUT_DIR, OUT_DIR_rank1, "
                    "...: scripts/pp_merge.py joins them); a resume reads its own stage's file, else OUT_DIR/ckpt.pt (a "
                    "single-GPU checkpoint or stage 0's)")
    ap.add_argument("--pipe_split", default="", help="--pipe: decoder layers per stage, e.g. 17,16,16,15 (default even)")
    ap.add_argument("--ckpt_skip", type=int, default=0,
                    help="leave the last K layers without gradient checkpointing (faster, "
                         "more VRAM; ~1.3 GiB per layer at 16x2048 tokens)")
    ap.add_argument("--temp_check", type=int, default=4,
                    help="check GPU temp every N steps")
    args = ap.parse_args()
    if os.environ.get("TERN_SYNCDBG"):   # report every host sync (location in the warning)
        torch.cuda.set_sync_debug_mode(1)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.gemm in ("cublas", "int8mm"):
        from . import kernel as _kmod
        _kmod.GEMM_BACKEND[0] = args.gemm
    LR_WSD[0] = args.lr_wsd
    LR_LINEAR[0] = args.lr_linear
    if args.lr_rewarm:
        _rw = args.lr_rewarm.split(":")
        LR_REWARM[:] = [int(_rw[0]), int(_rw[1])] + ([float(_rw[2])] if len(_rw) > 2 else [])
    TS_FUSED_STATS[0] = os.environ.get("TERN_FUSED_STATS", "1") != "0"
    TIME = os.environ.get("TERN_TIME") == "1"   # time the forward/backward vs the momentum step (synchronizes)

    mc: ModelConfig = PRESETS[args.preset]
    _QWEN = False
    if args.init_hf:
        from .qwen import is_qwen35
        _QWEN = is_qwen35(args.init_hf)
        if _QWEN:              # Qwen3.5 / 3.8 (Gated DeltaNet + gated attention): transformers' model, ternary linears
            from .qwen import model_config as _qmc
            mc = _qmc(args.init_hf)
        else:
            from .hf_init import model_config_from_hf
            mc = model_config_from_hf(args.init_hf)
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
    if args.weight_decay is not None: tc.weight_decay = args.weight_decay   # float tail (embedding) decoupled weight decay
    tc.min_lr = args.min_lr if args.min_lr is not None else tc.lr / 10
    if args.eval_interval: tc.eval_interval = args.eval_interval
    if args.eval_iters: tc.eval_iters = args.eval_iters

    BitTransformer.loss_chunk = args.loss_chunk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _WS = int(os.environ.get("WORLD_SIZE", "1"))      # data-parallel proof of concept (torchrun --nproc_per_node G)
    _RK = int(os.environ.get("RANK", "0"))
    _PP = (_RK, _WS) if args.pipe else None           # --pipe: (stage, stages); the data is then that of one GPU
    assert not args.pipe or _WS > 1, "--pipe: run under torchrun with several processes"
    assert not args.pipe or not (args.distill or args.tern_emb), "--pipe: none of the per-micro-batch extras yet"
    _base_out = args.out_dir
    if _WS > 1:
        import torch.distributed as _dist
        _lr_ = int(os.environ.get("LOCAL_RANK", "0"))
        _ng_ = torch.cuda.device_count()
        _lws_ = int(os.environ.get("LOCAL_WORLD_SIZE", str(_WS)))   # ranks on this machine (several machines: < WORLD_SIZE)
        torch.cuda.set_device(_lr_ % max(_ng_, 1))
        _be_ = "nccl" if _ng_ >= _lws_ else "gloo"     # gloo: several ranks share one GPU (a correctness test)
        _dist.init_process_group(_be_)
        print(f"{'pipeline stage' if _PP else 'data parallel: rank'} {_RK} of {_WS} ({_lws_} on this machine), GPU "
              f"{_lr_ % max(_ng_, 1)} of {_ng_}, {_be_}", flush=True)
        if _RK > 0 and args.out_dir:
            args.out_dir = args.out_dir.rstrip("/") + f"_rank{_RK}"
            tc.out_dir = args.out_dir
    if _PP:
        _WS, _RK = 1, 0       # --pipe: no data-parallel reduction, every stage reads the same windows
    if args.mem_cap_gib and device == "cuda":   # --mem_cap_gib G: PyTorch may use at most G GiB of this GPU (a 96 GB card
        # standing in for a 24 GB one in memory-matched comparisons; allocations beyond it raise out-of-memory)
        tot_ = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib / tot_))
        print(f"--mem_cap_gib: PyTorch capped at {args.mem_cap_gib:.1f} of {tot_:.1f} GiB", flush=True)
    if args.mem_snapshot and device == "cuda":   # --mem_snapshot PATH:STEP: every allocation with its Python stack
        torch.cuda.memory._record_memory_history(enabled="all", stacks="python", max_entries=2_000_000)
    torch.manual_seed(tc.seed)
    print(mc.report())
    use_beta = args.mode == "kernel"

    _qsw = {}
    if _QWEN:
        assert args.mode == "kernel" and not args.tern_emb, "Qwen: kernel mode, no --tern_emb yet"
        from .qwen import build_qwen, QwenTern
        QwenTern.loss_chunk = args.loss_chunk
        _own = None
        if _PP:
            from .qwen import text_config, pp_split
            _rng = pp_split(text_config(args.init_hf).num_hidden_layers, _PP[1], args.pipe_split)
            _own = (*_rng[_PP[0]], _PP[0] == 0, _PP[0] == _PP[1] - 1)
            print(f"--pipe: stage {_PP[0]} of {_PP[1]}: decoder layers {_own[0]}..{_own[1] - 1} of {_rng[-1][1]}"
                  f"{' + embedding' if _own[2] else ''}{' + norm, head and loss' if _own[3] else ''}", flush=True)
        model, _qsw = build_qwen(args.init_hf, "kernel", grad_checkpoint=tc.grad_checkpoint, device=device, own=_own,
                                 make_linear=lambda i_, o_: KernelTernaryLinear(i_, o_, mc.act_bits, beta=use_beta,
                                                                                 int8=args.int8, int8_dx=args.int8_dx))
        if _PP:
            model.pp = {"rank": _PP[0], "n": _PP[1], "lo": _own[0], "hi": _own[1]}
    elif args.mode == "kernel":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint, beta=use_beta, int8=args.int8,
                                         int8_dx=args.int8_dx)
    elif args.mode == "fp32":
        # full-precision reference (no ternary, no activation quantization): plain nn.Linear,
        # fp32 weights + AdamW, bf16 autocast in the forward like every other mode
        model = BitTransformer(mc, grad_checkpoint=tc.grad_checkpoint,
                               make_linear=lambda i, o: nn.Linear(i, o, bias=False))
    elif args.mode == "master":
        model = build_master_transformer(
            mc, grad_checkpoint=tc.grad_checkpoint,
            dtype=torch.float32 if args.master_dtype == "fp32" else torch.bfloat16)
    model = model.to(device) if not _PP else model._apply(lambda t_: t_ if t_.is_meta else t_.to(device))   # --pipe: others' stay meta
    try:   # counted from the modules (mc.report() above is a formula: approximate for hybrid / HF models)
        n_t_ = sum(m_.N * m_.K for m_ in model.modules() if getattr(m_, "wpacked", None) is not None and hasattr(m_, "N"))
        n_f_ = sum(p_.numel() for p_ in model.parameters())
        print(f"counted       : ternary {n_t_ / 1e6:,.1f} M + float {n_f_ / 1e6:,.1f} M", flush=True)
    except Exception:
        pass
    if args.moe:      # --moe E:K[:H]: every feed-forward becomes a mixture of E ternary experts, top-K routing
        from .model import enable_moe
        from .master import MasterTernaryLinear as _MTL   # main() imports the name locally later on
        _mk = (lambda i_, o_: KernelTernaryLinear(i_, o_, mc.act_bits, beta=use_beta, int8=args.int8,
                                                  int8_dx=args.int8_dx).to(device)) \
            if args.mode == "kernel" else \
            (lambda i_, o_: _MTL(i_, o_, mc.act_bits, dtype=torch.float32 if args.master_dtype == "fp32"
                                                else torch.bfloat16).to(device)) if args.mode == "master" else \
            (lambda i_, o_: nn.Linear(i_, o_, bias=False).to(device))   # master: ternary experts with latents, as its layers
        E_, k_, H_ = enable_moe(model, args.moe, _mk)
        BitTransformer.moe_aux_w = args.moe_aux
        print(f"--moe: {E_} experts of hidden {H_}, top-{k_} routing, load-balancing weight {args.moe_aux}", flush=True)
    model.train()
    if args.tern_emb:
        from .tern_emb import enable_tern_emb
        n_te = enable_tern_emb(model, args.tern_emb, "master" if args.mode == "master" else "kernel")
        assert args.mode == "master" or (args.ts and args.ts_fused and tc.grad_accum == 1), \
            "--tern_emb in kernel mode: --ts --ts_fused, grad_accum 1"
        print(f"--tern_emb {args.tern_emb}: {n_te / 1e6:.1f}M ternary embedding weights (+ row / column scales)", flush=True)
    if args.ckpt_skip:
        model.ckpt_skip = args.ckpt_skip
    if args.ckpt_block:
        for b_ in model.layers:
            b_.ckpt_block = True
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
    _resume_path = os.path.join(tc.out_dir, "ckpt.pt")
    if _PP and not os.path.exists(_resume_path) and os.path.exists(os.path.join(_base_out, "ckpt.pt")):
        _resume_path = os.path.join(_base_out, "ckpt.pt")    # --pipe: a single-GPU checkpoint (or stage 0's) for all
    _resuming = args.resume and os.path.exists(_resume_path)
    if args.init_hf and _resuming:   # the checkpoint holds the converted model (trits, scales, float parts)
        print(f"--init_hf: resuming from {tc.out_dir}/ckpt.pt, conversion skipped", flush=True)
        if _QWEN:   # as the conversion leaves them: embedding and head frozen (the optimizer is built below)
            model.hf.model.embed_tokens.weight.requires_grad_(False); model.hf.lm_head.weight.requires_grad_(False)
        if args.init_hf_method == "gptqc":   # centred layers: their buffers and bias
            _ck = torch.load(_resume_path, map_location="cpu", weights_only=False, mmap=True)["model"]
            _nc = 0                                             # exist before the optimizer is built
            for n_, m_ in model.named_modules():
                _dv = "meta" if isinstance(m_, KernelTernaryLinear) and m_.wpacked.is_meta else device   # --pipe
                if isinstance(m_, KernelTernaryLinear) and f"{n_}.in_mean" in _ck:
                    m_.enable_centering(_ck[f"{n_}.in_mean"].to(_dv), _ck[f"{n_}.out_bias"].to(_dv)); _nc += 1
            print(f"resume: {_nc} centred layers restored", flush=True)
            del _ck
    if args.init_hf and not _resuming:
        from .hf_init import init_from_hf, GPTQ_OPTS as _GO
        _GO["mag_svd"] = args.init_mag_svd
        assert args.mode in ("kernel", "master", "fp32"), "--init_hf: kernel, master or fp32 mode"
        _hess = None
        if _QWEN and args.init_hf_method == "random":   # speed measurements without the weights (no download):
            with torch.no_grad():                         # the trits as built (random), float parts N(0, 0.02)
                for p_ in model.parameters():
                    if not p_.is_meta:
                        p_.normal_(0, 0.02)
                model.hf.model.embed_tokens.weight.requires_grad_(False); model.hf.lm_head.weight.requires_grad_(False)
            print("--init_hf_method random: random trits and float parts (speed tests only)", flush=True)
        elif _QWEN and args.init_stream:   # --init_stream: layer by layer, sequential (models too big for one fp copy)
            from .qwen import convert_qwen_streaming
            convert_qwen_streaming(model, args.init_hf, _qsw, tc.data_path, method=args.init_hf_method,
                                   n_seq=args.init_calib, seq=tc.seq_len, dev=device, dtype_u32=args.data_u32,
                                   log=lambda m_: print(m_, flush=True))
        elif _QWEN:
            from .qwen import load_qwen, calib_hessians_qwen
            if args.init_hf_method in ("gptq", "gptqc"):
                _hess = calib_hessians_qwen(args.init_hf, tc.data_path, n_seq=args.init_calib, seq=tc.seq_len,
                                            dev=device, dtype_u32=args.data_u32, log=lambda m_: print(m_, flush=True))
            load_qwen(model, args.init_hf, _qsw, method=args.init_hf_method, rc=args.rc_scale, hess=_hess,
                      log=lambda m_: print(m_, flush=True))
            _hess = None
        elif args.init_hf_method == "gptqs":    # sequential, error-aware, centred GPTQ (training-free conversion)
            from .hf_init import convert_seq_asym
            _hess = convert_seq_asym(args.init_hf, tc.data_path, n_seq=min(args.init_calib, 32), dev=device,
                                     log=lambda m_: print(m_, flush=True))
        elif args.init_hf_method in ("gptq", "gptqc"):     # input second moments of the full-precision model (not the run's windows)
            from .hf_init import calib_hessians
            _hess = calib_hessians(args.init_hf, tc.data_path, n_seq=args.init_calib, dev=device,
                                   log=lambda m_: print(m_, flush=True))
        if not _QWEN:
            init_from_hf(model, args.init_hf, args.mode, args.init_hf_method, rc=args.rc_scale, hess=_hess,
                         log=lambda m_: print(m_, flush=True))
    _teacher = None
    if args.kd_cache:  # precomputed teacher targets (scripts/quant/teacher_topk.py)
        assert hasattr(model, "kd_topk"), "--kd_cache: Qwen models"
        model.kd_w = args.distill_w
        model.kd_mode = args.kd_mode
        print(f"--kd_cache ({args.kd_mode})", flush=True)
        print(f"--kd_cache {args.kd_cache}: from window 0 at step {args.kd_cache_start}, loss (1 - {args.distill_w}) CE "
              f"+ {args.distill_w} KD to the teacher's top k", flush=True)
    if args.distill and _QWEN:   # Qwen: transformers' own model in bf16 as the teacher (it must fit next to the student)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as _QT
        _qt = _QT.from_pretrained(args.distill, dtype=torch.bfloat16).to(device).eval()
        for p_ in _qt.parameters():
            p_.requires_grad_(False)
        _teacher = lambda x_: (_qt(input_ids=x_).logits, None)
        model.kd_w = args.distill_w
        print(f"--distill {args.distill} (Qwen): teacher in bf16, loss (1 - {args.distill_w}) CE + {args.distill_w} KL", flush=True)
    elif args.distill:   # --distill HF_DIR: the run learns the full-precision model's distribution (KL), not only the labels
        from .hf_init import model_config_from_hf as _mch, init_from_hf as _ifh
        from .model import BitTransformer as _BT
        _teacher = _BT(_mch(args.distill, max_seq_len=tc.seq_len), grad_checkpoint=False,
                       make_linear=lambda i_, o_: nn.Linear(i_, o_, bias=False)).to(device)
        _ifh(_teacher, args.distill, "fp32", log=lambda *_: None)
        _teacher = _teacher.eval()
        for p_ in _teacher.parameters():   # parameters only: Module.to(bf16) would also cast the complex RoPE table
            p_.data = p_.data.to(torch.bfloat16); p_.requires_grad_(False)   # to real (teacher CE 5.20 instead of 2.34)
        model.kd_w = args.distill_w
        print(f"--distill {args.distill}: teacher in bf16, loss (1 - {args.distill_w}) CE + {args.distill_w} KL", flush=True)
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
        LeanAdam.BATCHED = bool(args.lean_batched)
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

    STATE.weight_decay = tc.weight_decay

    if args.track_flips:
        n = enable_flip_tracking(model)
        print(f"flip tracking on for {n} ternary layers", flush=True)

    sampler = torch.Generator().manual_seed(tc.seed + 1_000_003 * _RK)   # data parallel: each rank its own batches
    lr_state = {}
    _ddt = np.uint32 if args.data_u32 else np.uint16      # --data_u32: token ids above 65535 (Qwen's 248k vocabulary)
    train_data = np.memmap(tc.data_path, dtype=_ddt, mode="r")
    val_data = np.memmap(tc.val_path, dtype=_ddt, mode="r") if os.path.exists(tc.val_path) else train_data

    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    metrics_path = os.path.join(tc.out_dir, "metrics.jsonl")
    metrics_f = open(metrics_path, "a")

    def log_metrics(rec):
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    def _opt_sd():    # --pipe: without the other stages' optimizer state (on meta after a resume from a full checkpoint)
        sd_ = tail_opt.state_dict()
        if _PP:
            sd_["state"] = {k_: v_ for k_, v_ in sd_["state"].items()
                            if not any(torch.is_tensor(t_) and t_.is_meta for t_ in v_.values())}
        return sd_

    def save_ckpt(step):
        # atomic: write tmp then rename, so a kill mid-save never corrupts ckpt.pt
        tmp = ckpt_path + ".tmp"
        # flip-tracking masks are plain attributes (not state_dict buffers), so they
        # are saved alongside: otherwise "never flipped since init" silently resets
        # to 100% on resume.
        touched = {n: m.touched for n, m in model.named_modules()
                   if hasattr(m, "touched") and m.touched is not None}
        # step the mask started accumulating from: never-flipped is a "since init"
        # number only when this is 0.
        touched_origin = _touched_origin["v"]
        torch.save({"model": {k_: v_ for k_, v_ in model.state_dict().items() if not v_.is_meta},   # --pipe: own stage's
                    "opt": _opt_sd(), "pipe": (model.pp if _PP else None),
                    "touched": touched, "touched_origin": touched_origin,
                    "cfg": mc, "step": step, "mode": args.mode, "beta": use_beta,
                    "int8": args.int8, "int8_dx": args.int8_dx,
                    "dw_mode": "dense",    # (read by the evaluation scripts of checkpoints before 17 Oct 2026)
                    # low-rank momentum (U, V) per ternary layer, in module order
                    "lowrank": [lr_state.get(id(m)) for m in model.modules()
                                if isinstance(m, KernelTernaryLinear)] if lr_state else None,
                    # --ts: the rest of the per-layer state (update count, factored v, short momentum), module order
                    "ts_state": [{k[0]: st_[k] for k in (("t", id(m)), ("v", id(m)), ("m", id(m))) if k in st_}
                                 for m in model.modules() if isinstance(m, KernelTernaryLinear)]
                                if (st_ := lr_state.get("_ts")) else None,
                    # --tern_emb: the ternary embedding matrices' TTF state (U V, update count, factored v)
                    "tern_emb_state": [(lr_state.get(id(m)), {k[0]: lr_state.get("_ts", {})[k] for k in
                                        (("t", id(m)), ("v", id(m))) if k in lr_state.get("_ts", {})})
                                       for m in _tern_lays(model)] if args.tern_emb and lr_state else None}, tmp)
        os.replace(tmp, ckpt_path)

    mag_on = [False]
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
        mag_on[0] = True
        print(f"low-rank magnitude '{kind}' rank {r}: {sum(p.numel() for p in newp) / 1e6:.2f}M floats "
              f"(wd {args.mag_wd}, lr x{args.mag_lr_mult})", flush=True)

    start_step = 0
    _touched_origin = {"v": 0}
    _load_path = _resume_path
    if _PP and args.resume:
        print(f"--pipe: stage {_PP[0]} resumes from {_load_path}", flush=True)
    if args.resume and os.path.exists(_load_path):
        # to host memory: every piece is copied to its device below (on the GPU the whole checkpoint would sit next
        # to the model it is loaded into: the 27B's 10.5 GB did not fit)
        blob = torch.load(_load_path, map_location="cpu", weights_only=False, mmap=True)
        if blob.get("beta", False) != use_beta:
            raise SystemExit(f"checkpoint beta={blob.get('beta', False)} but run beta="
                             f"{use_beta}: forward differs, refusing to resume")
        if args.lowrank_mag and any(k.endswith("mag_A") or ".mag_At." in k for k in blob["model"]):
            _R = int(args.lowrank_mag.split(":")[1]); _grown = 0
            for k_ in [k_ for k_ in blob["model"] if k_.endswith("mag_A")]:   # a larger 'add' rank than the checkpoint's:
                A_, B_ = blob["model"][k_], blob["model"][k_[:-1] + "B"]      # new columns of A at 0 (the layer is
                if A_.shape[1] < _R:                                           # unchanged), of B random as at init
                    g_ = torch.Generator().manual_seed(17_000_000 + _grown)
                    blob["model"][k_] = torch.cat([A_, torch.zeros(A_.shape[0], _R - A_.shape[1], dtype=A_.dtype)], 1)
                    blob["model"][k_[:-1] + "B"] = torch.cat([B_, (torch.randn(B_.shape[0], _R - B_.shape[1], generator=g_)
                                                                  * (1.0 / B_.shape[0]) ** 0.5).to(B_.dtype)], 1)
                    _grown += 1
            enable_mag(int(blob.get("step", 0)) + 1)
            if _grown and "opt" in blob:   # the adapters' optimizer state starts fresh (its shapes are the old rank's)
                _gi = len(tail_opt.param_groups) - 1
                for i_ in blob["opt"]["param_groups"][_gi]["params"]:
                    blob["opt"]["state"].pop(i_, None)
                print(f"--lowrank_mag {args.lowrank_mag}: {_grown} adapters grown from the checkpoint's rank "
                      f"(new columns of A at 0; their optimizer state restarts)", flush=True)
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
        elif _PP:    # --pipe: this stage's tensors must all be there; the other stages' (meta) are skipped
            missing, unexpected = model.load_state_dict(blob["model"], strict=False)
            _md = dict(model.state_dict())
            _miss = [k_ for k_ in missing if not _md[k_].is_meta]
            assert not _miss and not unexpected, (f"--pipe: {_load_path} lacks this stage's {_miss[:4]}", unexpected[:4])
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
        else:
            model.load_state_dict(blob["model"])
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
        start_step = int(blob.get("step", 0)) + 1
        # fast-forward the data streams, so a resumed run sees the batches the uninterrupted
        # run would have (not the step-0 batches again)
        for _ in range(start_step * tc.grad_accum):
            torch.randint(len(train_data) - tc.seq_len - 1, (tc.batch_size,), generator=sampler)
        _lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
        _sdev = {id(m): ("cpu" if _ts_on_host(args, i, len(_lays)) else device) for i, m in enumerate(_lays)}
        if blob.get("lowrank"):
            lays = _lays
            for m, uv in zip(lays, blob["lowrank"]):
                if uv is not None and not m.wpacked.is_meta:
                    lr_state[id(m)] = tuple(t.to(_sdev[id(m)]) for t in uv)
            print(f"resumed low-rank momentum for {len(lr_state)} layers", flush=True)
            if args.ts and args.lowrank:   # a larger --lowrank than the checkpoint's: grow each accumulator (the --ts_micro
                # path never resizes): V gains orthonormal columns outside its span, U zeros there, so A = U V^T is
                # unchanged and the new directions start empty
                _sa = _store_codec(args.ts_store_a or "")
                _ng = 0
                for i_, m in enumerate(lays):
                    if id(m) not in lr_state or m.wpacked.is_meta:
                        continue
                    U_, V_ = _load_pair(lr_state[id(m)])
                    ra_ = min(args.lowrank, m.N, m.K)
                    if V_.shape[1] >= ra_:
                        continue
                    dv_ = V_.device; U_, V_ = U_.to(device).float(), V_.to(device).float()
                    g_ = torch.Generator(device=device).manual_seed(9_000_001 + 2 * i_)
                    X_ = torch.randn(m.K, ra_ - V_.shape[1], device=device, generator=g_)
                    X_ = X_ - V_ @ (V_.T @ X_); X_ = X_ - V_ @ (V_.T @ X_)
                    V2 = torch.cat([V_, _orth(X_)], 1); U2 = torch.cat([U_, torch.zeros(m.N, ra_ - V_.shape[1], device=device)], 1)
                    pr_ = _store_pair(_sa, U2, V2) if _sa else (U2, V2)
                    lr_state[id(m)] = tuple(t.to(dv_) for t in pr_); _ng += 1
                if _ng:
                    print(f"--lowrank {args.lowrank}: grew the accumulators of {_ng} layers from the checkpoint's rank",
                          flush=True)
            if args.ts and args.ts_acc_scale != 1.0:   # --ts_acc_scale F (a new theta = F x the run's): every accumulator x F,
                _sa = _store_codec(args.ts_store_a or "")   # so each entry keeps its distance to the bar (no burst of moves)
                for m in lays:
                    if id(m) in lr_state and not m.wpacked.is_meta:
                        U_, V_ = _load_pair(lr_state[id(m)]); dv_ = U_.device
                        U_ = U_.to(device).float() * args.ts_acc_scale; V_ = V_.to(device).float()
                        pr_ = _store_pair(_sa, U_, V_) if _sa else (U_, V_)
                        lr_state[id(m)] = tuple(t.to(dv_) for t in pr_)
                print(f"--ts_acc_scale {args.ts_acc_scale}: accumulators rescaled (theta now {args.ts_theta})", flush=True)
        if blob.get("tern_emb_state"):
            st_ = lr_state.setdefault("_ts", {})
            for m, (uv, d_) in zip(_tern_lays(model), blob["tern_emb_state"]):
                if uv is not None:
                    lr_state[id(m)] = tuple(t.to(device) for t in uv)
                for k_, v_ in d_.items():
                    st_[(k_, id(m))] = tuple(x.to(device) for x in v_) if isinstance(v_, tuple) else v_
            print(f"resumed --tern_emb TTF state for {len(blob['tern_emb_state'])} matrices", flush=True)
        if blob.get("ts_state"):
            lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
            st_ = lr_state.setdefault("_ts", {})
            for m, d_ in zip(lays, blob["ts_state"]):
                for k_, v_ in (d_.items() if not m.wpacked.is_meta else ()):
                    st_[(k_, id(m))] = tuple(x.to(_sdev[id(m)]) for x in v_) if isinstance(v_, tuple) else v_
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
        del blob
        torch.cuda.empty_cache()

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
    if _RK == 0 and (not _PP or _PP[0] == 0):   # the results database: this run's records in <out_dir>/records.jsonl
        from .results import RunLog
        RUNLOG[0] = RunLog(tc.out_dir, os.path.basename(os.path.normpath(tc.out_dir)), vars(args), data=tc.data_path,
                           steps=tc.max_steps, parent=(ckpt_path if start_step > 0 else None))
        RUNLOG[0].event("running", start_step)

    # after any resume, so 'net' counts from the weights this run starts at
    if args.lowrank_mag and not mag_on[0]:
        enable_mag(start_step)
    t0 = time.time()
    last_save = time.time()
    last_ = max(tc.max_steps, args.extend_to or 0)   # --extend_to N: keep training past the schedule (lr held at min_lr)
    end_step = last_ if args.stop_after is None else min(last_, args.stop_after)
    prof = None
    ts_cfg = None
    if args.ts:
        assert args.ts_nomom, "TTF with a short momentum lives in ttf.legacy (python -m ttf.legacy.train)"
        _lays = [m_ for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)]
        for i_, m_ in enumerate(_lays):   # --ts_offload(_keep): which layers' state lives in host memory
            m_._ts_host = _ts_on_host(args, i_, len(_lays))
        if args.ts_offload_keep and args.ts_offload:
            print(f"--ts_offload_keep: state of {sum(not m_._ts_host for m_ in _lays)} of {len(_lays)} layers on the GPU",
                  flush=True)
        ts_cfg = dict(r=args.lowrank, lr_ratio=1.0, theta=args.ts_theta, tau=args.ts_tau, b2=args.ts_b2, spend=args.ts_spend,
                      tau_anneal=(args.ts_tau_anneal_pow if args.ts_tau_anneal else False), store_a=args.ts_store_a,
                      offload=args.ts_offload,
                      theta_sched=(tuple(float(x) for x in args.ts_theta_sched.split(":")) if args.ts_theta_sched else None),
                      progress=0.0, factored=args.ts_factored)
    if args.ts and args.ts_fused:         # --ts_fused: each layer's update inside its backward (ts_layer)
        for i_, m_ in enumerate([m_ for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)]):
            m_._ts_idx = i_
            if _WS > 1:   # data parallel: every rank's TTF update sees the mean gradient, so all ranks move the same trits
                def _og(layer, g, _c=ts_cfg):
                    _dist.all_reduce(g); g.div_(_WS); ts_layer(layer, g, lr_state, _c)
                m_.on_grad = _og
            else:
                def _og1(layer, g, _c=ts_cfg):
                    with torch.profiler.record_function("ttf.update"):
                        ts_layer(layer, g, lr_state, _c)
                m_.on_grad = _og1
            m_.ts_micro = bool(args.ts_micro)
        print("--ts_fused: the update runs inside each layer's backward", flush=True)
        FIRE_FUSED[0] = bool(args.ts_fire_fused)
        if args.ts_factored:
            assert args.ts_micro or tc.grad_accum == 1, "--ts_factored: 16 x 1 or --ts_micro"
            print(f"--ts_factored {args.ts_factored}: TTF from the gradient's factors (no dense weight gradient)", flush=True)
    if args.fused_scales:
        from .flip import FUSED_SCALES as _FSC
        _FSC[0] = True
    if args.fused_norm and _QWEN:
        from .qwen import enable_fused_norm
        enable_fused_norm()
        print("--fused_norm: Qwen RMSNorm in one kernel each way", flush=True)
    if args.eval_first_n:   # --eval_first_n N: the model as built or resumed (e.g. right after --init_hf), scored on
        model.eval(); STATE.updates_enabled = False   # the first N val windows in order (qwen_check.py's windows)
        ls_ = []
        with torch.no_grad():
            for i_ in range(args.eval_first_n):
                w_ = torch.from_numpy(val_data[i_ * tc.seq_len:(i_ + 1) * tc.seq_len + 1].astype(np.int64)).to(device)[None]
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    ls_.append(float(model(w_[:, :-1], w_[:, 1:])[1]))
        print(f"eval_first_n: loss {np.mean(ls_):.4f} over the first {args.eval_first_n} val windows ("
              + " ".join(f"{x:.4f}" for x in ls_) + ")", flush=True)
        log_metrics({"step": 0, "first_n_loss": float(np.mean(ls_))})
        model.train(); STATE.updates_enabled = True
    for step in range(start_step, end_step):
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
        lr = lr_at(step, tc)
        if ts_cfg is not None:
            ts_cfg["lr_ratio"] = _ts_scale(args, step, tc, lr)
            ts_cfg["progress"] = step / max(1, tc.max_steps)
            ts_cfg["step"] = step; ts_cfg["start"] = args.ts_start
            ts_cfg["vdiag"] = args.ts_vdiag and step % max(args.ts_vdiag_every, 1) == 0   # diagnostics only
            ts_cfg["micro"] = tc.grad_accum if args.ts_micro else 1
            if args.ts_factored:
                from .flip import FACTORED as _FCT
                _FCT[0] = True          # (--ts_vdiag reads its g^2 / v from the sketch)
        STATE.lr = lr
        for g in tail_opt.param_groups:
            g["lr"] = (g["rc_lr"] if g.get("rc_lr") else lr) * g.get("lr_mult", 1.0)
        STATE.updates_enabled = True
        STATE.grad_scale = tc.grad_accum
        tail_opt.zero_grad(set_to_none=True)
        last_loss = 0.0
        if args.lowrank:          # TTF: every ternary layer hands its gradient to the update (ts_layer)
            for l in model.modules():
                if isinstance(l, KernelTernaryLinear): l.capture = True
        if TIME: torch.cuda.synchronize(); _t0 = time.time()
        _prof = None      # TERN_PROF=DIR: one profiled step (the 3rd of the run), GPU time by operation into DIR
        if os.environ.get("TERN_PROF") and step == start_step + 2:
            from torch.profiler import profile, ProfilerActivity
            _prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True)
            _prof.__enter__(); torch.cuda.synchronize(); _pt0 = time.time()
        if _PP:   # --pipe: the step's micro-batches (the windows the single-GPU loop reads) through the pipeline
            _bt = []
            for micro in range(tc.grad_accum):
                if args.kd_cache:
                    x, y, kd_ = _kd_cache_batch(args, (step - args.kd_cache_start) * tc.grad_accum + micro, train_data,
                                                tc.seq_len, device, bs=tc.batch_size)
                else:
                    (x, y), kd_ = get_batch(train_data, tc.batch_size, tc.seq_len, device, sampler), None
                _bt.append((x, y, kd_))

            def _pre_bwd(i_):
                if args.ts and tc.grad_accum > 1:
                    for l_ in model.modules():
                        if isinstance(l_, KernelTernaryLinear):
                            l_.cap_accum = True; l_.cap_final = i_ == tc.grad_accum - 1
            last_loss = model.pp_train(_bt, _pre_bwd, lambda: torch.autocast(device_type=device.split(":")[0],
                                                                             dtype=torch.bfloat16)).item()
            del _bt
        for micro in range(0 if _PP else tc.grad_accum):
            if args.ts and tc.grad_accum > 1:      # --ts with accumulation: sum the captured micro-batch gradients
                for l_ in model.modules():
                    if isinstance(l_, KernelTernaryLinear):
                        l_.cap_accum = True; l_.cap_final = micro == tc.grad_accum - 1
            if args.kd_cache:          # --kd_cache: the next precomputed window, its tokens and the teacher's top k
                # data parallel: each rank its own windows (rank 0 of 1 reads what a single run always read)
                x, y, model.kd_topk = _kd_cache_batch(args, ((step - args.kd_cache_start) * tc.grad_accum + micro) * _WS + _RK,
                                                      train_data, tc.seq_len, device, bs=tc.batch_size)
            else:
                x, y = get_batch(train_data, tc.batch_size, tc.seq_len, device, sampler)
            if _teacher is not None:   # --distill: the full-precision model's next-token distribution for this batch
                with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    if args.hid_kd:      # --hid_kd: the teacher's stream at every block boundary too (one forward)
                        model.hid_t = _teacher.hiddens(x); model.hid_w = args.hid_kd
                        tz_ = F.linear(_teacher.norm(model.hid_t[-1]),
                                       _teacher.tok_emb.weight if _teacher.lm_head is None else _teacher.lm_head.weight)
                    else:
                        tz_, _ = _teacher(x)
                tz_ = tz_.reshape(-1, tz_.shape[-1])   # in row chunks: the fp32 copy of all logits is ~4 GB at 32k tokens
                model.kd_logp = torch.cat([F.log_softmax(c_.float(), -1).to(torch.bfloat16) for c_ in tz_.split(4096)]); del tz_
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16), \
                    torch.profiler.record_function("model.fwd"):
                _, loss = model(x, y)
            with torch.profiler.record_function("model.bwd"):   # includes the fused TTF updates (ttf.update)
                (loss / tc.grad_accum).backward()
            model.kd_logp = None
            if args.hid_kd:
                model.hid_t = None
                if micro == 0 and step % 50 == 0:
                    print(f"hid_kd: step {step} block-update error {model.hid_last:.4f}", flush=True)
            if args.kd_cache:
                model.kd_topk = None
            if args.tern_emb and args.mode != "master":   # the ternary embedding's TTF update (both of its gradients)
                from .tern_emb import tern_emb_step
                tern_emb_step(model, lr_state, ts_cfg, ts_layer)
            last_loss = loss.detach()          # read once after the last micro-batch (a .item() here synced every one)
        if torch.is_tensor(last_loss):
            last_loss = last_loss.item()
        if TIME:
            torch.cuda.synchronize(); _t1 = time.time()
            _m1, _p1 = torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30
        _rfu = torch.profiler.record_function("step.update"); _rfu.__enter__()   # TERN_PROF: after the backward
        if args.ts:
            with torch.profiler.record_function("ttf.step"):
                rs_info = ts_step(model, lr_state, ts_cfg, fused=args.ts_fused)
        else:
            rs_info = None
        if _WS > 1:       # data parallel: the float tail's gradients are averaged before the clip and the step
            for p_ in tail:
                if p_.grad is not None:
                    _dist.all_reduce(p_.grad); p_.grad.div_(_WS)
        _clip_ps = tail
        _rfc = torch.profiler.record_function("tail.clip"); _rfc.__enter__()
        if _PP:   # --pipe: the clip's norm over all stages' float tails (each stage holds its own gradients)
            _gs = [p_.grad for p_ in _clip_ps if p_.grad is not None]
            _n2 = (torch.stack([torch.linalg.vector_norm(g_, 2).float() for g_ in _gs]) ** 2).sum() if _gs else \
                torch.zeros((), device=device)
            _n2 = _n2.reshape(1).cpu() if _dist.get_backend() == "gloo" else _n2.reshape(1)
            _dist.all_reduce(_n2)
            _cf = (tc.grad_clip / (_n2.to(device).sqrt() + 1e-6)).clamp(max=1.0)[0]
            for g_ in _gs:
                g_.mul_(_cf.to(g_.dtype))
        else:
            torch.nn.utils.clip_grad_norm_(_clip_ps, tc.grad_clip)
        _rfc.__exit__(None, None, None)
        if _WS > 1 and step % 100 == 0:   # data parallel: the ranks' trits must stay identical (same g, same rule)
            _ck = torch.tensor([float(sum(int(m_.wpacked.sum()) for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)))],
                               device=device, dtype=torch.float64)
            _cks = [torch.zeros_like(_ck) for _ in range(_WS)]; _dist.all_gather(_cks, _ck)
            print(f"  data parallel step {step}: packed-trit checksums {[int(x.item()) for x in _cks]} "
                  f"({'in sync' if len({int(x.item()) for x in _cks}) == 1 else 'OUT OF SYNC'})", flush=True)
        with torch.profiler.record_function("tail.step"):
            tail_opt.step()
        _rfu.__exit__(None, None, None)
        if _prof is not None:
            torch.cuda.synchronize(); _pdt = time.time() - _pt0; _prof.__exit__(None, None, None)
            _pd = os.environ["TERN_PROF"]; os.makedirs(_pd, exist_ok=True)
            _ka = _prof.key_averages()
            with open(os.path.join(_pd, f"prof_step{step}.txt"), "w") as _f:
                _f.write(f"step {step} wall {_pdt:.2f}s\n")
                _f.write(_ka.table(sort_by="self_cuda_time_total", row_limit=60, max_name_column_width=70))
                _f.write("\n\nby CPU time\n" + _ka.table(sort_by="self_cpu_time_total", row_limit=30, max_name_column_width=70))
                _rng = [e for e in _ka if e.key.split(".")[0] in ("tern", "ttf", "model", "loss")]   # the named ranges
                _f.write("\n\nnamed ranges (CUDA total = everything inside them)\n")
                _ct = lambda e: max(getattr(e, "device_time_total", 0) or 0, getattr(e, "self_device_time_total", 0) or 0)
                for e in sorted(_rng, key=lambda e: -_ct(e)):   # (the GPU-side annotation of a range carries its time)
                    ct = _ct(e)
                    _f.write(f"  {e.key:28s} calls {e.count:7d}  CUDA {ct / 1e6:9.3f} s  CPU {e.cpu_time_total / 1e6:9.3f} s\n")
            with open(os.path.join(_pd, f"prof_step{step}_stack.txt"), "w") as _f:   # by the Python line that launched it
                if os.environ.get("TERN_PROF_NOSTACK"):   # (that table takes ~20 min on a 27B step)
                    _f.write("skipped (TERN_PROF_NOSTACK)\n")
                else:
                    _f.write(_prof.key_averages(group_by_stack_n=6).table(sort_by="self_cuda_time_total", row_limit=50,
                                                                           max_name_column_width=60))
            if os.environ.get("TERN_PROF_TRACE"):
                _prof.export_chrome_trace(os.path.join(_pd, f"trace_step{step}.json"))
            print(f"TERN_PROF: step {step} profiled ({_pdt:.1f}s wall) -> {_pd}", flush=True)
        if TIME:
            torch.cuda.synchronize(); _t2 = time.time()
            print(f"time step {step}: forward+backward {_t1 - _t0:.3f}s, update (momentum step / AdamW) {_t2 - _t1:.3f}s"
                  f" | GiB allocated after backward {_m1:.2f} (peak in fwd+bwd {_p1:.2f}), after update "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} (peak {torch.cuda.max_memory_allocated() / 2**30:.2f})",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()

        rec = {"step": step, "loss": last_loss, "lr": lr, "touched_origin": _touched_origin["v"],
               "tokens": (step + 1) * tc.grad_accum * tc.batch_size * tc.seq_len}
        fs = collect_flip_stats(model) if args.track_flips else {}
        if rs_info is not None:
            # device counters (look-ahead) become numbers here: one read per step, at logging
            rs_info = {k: (v.item() if torch.is_tensor(v) else v) for k, v in rs_info.items()}
            rec.update(rs_info)
        if fs:
            rec.update(flip_frac=fs["flip_frac_total"], never_frac=fs["never_frac_total"],
                       flip_frac_layers=fs["flip_frac"], never_frac_layers=fs["never_frac"])
        log_metrics(rec)

        if step % tc.log_interval == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            extra = ""
            if fs:
                extra += (f"| flip {fs['flip_frac_total']*100:.3f}% "
                          f"| never {fs['never_frac_total']*100:.1f}% ")
            print(f"step {step:6d} | loss {last_loss:6.3f} | lr {lr:.2e} {extra}"
                  f"| {dt:6.1f}s | peakVRAM {mem:.2f}GiB", flush=True)
            if args.ts and step in (10, 1000):
                print(f"  --ts state: {_ts_state_bytes(lr_state) / 2**20:.1f} MiB stored", flush=True)
            if args.mem_snapshot and device == "cuda" and step == int(args.mem_snapshot.rsplit(":", 1)[1]):
                torch.cuda.memory._dump_snapshot(args.mem_snapshot.rsplit(":", 1)[0])    # scripts/analysis/mem_snapshot.py
                torch.cuda.memory._record_memory_history(enabled=None)
                print(f"--mem_snapshot: allocation history to step {step} in {args.mem_snapshot.rsplit(':', 1)[0]}", flush=True)
            if args.mem_report and step in (10, 50, 100, 1000):
                mr = mem_report(model, tail_opt, lr_state)
                log_metrics({"step": step, **mr})
                print("  memory (MiB): " + ", ".join(f"{k[4:-3]} {v:.1f}" for k, v in mr.items()), flush=True)
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
            save_ckpt(step); print(f"saved at step {step}. exiting.", flush=True)
            if RUNLOG[0] is not None:
                RUNLOG[0].event("stopped", step, "signal")
            raise SystemExit(0)

    if not args.no_final_save:
        save_ckpt(end_step - 1)
    vl = evaluate(model, val_data, tc, device)
    log_metrics({"step": end_step, "val_loss": vl, "val_ppl": math.exp(vl),
                 "final": True})
    print(f"FINAL val loss {vl:.4f} | ppl {math.exp(vl):.2f}", flush=True)
    if RUNLOG[0] is not None:
        RUNLOG[0].result("val_train", vl, n=tc.eval_iters * max(tc.batch_size, 16), step=end_step)
        RUNLOG[0].event("finished", end_step)
    print("done", flush=True)


RUNLOG = [None]          # the run's results.RunLog (rank 0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        if RUNLOG[0] is not None:
            RUNLOG[0].event("failed", reason=f"{type(e).__name__}: {str(e)[:300]}")
        raise
