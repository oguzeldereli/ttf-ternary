"""TTF, the update of the ternary layers (no momentum): the low-rank accumulator A = U V^T per layer, its
firings into trit moves, the storage codecs of U and V, host offload, and the step statistics."""
from __future__ import annotations
import os
import math
import torch

from .layers import KernelTernaryLinear
from .schedule import _theta_at


TS_FUSED_STATS = [True]          # TERN_FUSED_STATS=0: the TTF row / column sums of g^2 by torch reductions (the old path)


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
