"""TTF, the update of the ternary layers (no momentum): the low-rank accumulator A = U V^T per layer, its
firings into trit moves, the storage codecs of U and V, host offload, and the step statistics."""
from __future__ import annotations
import os
import math
import torch

from .layers import _rf
from .schedule import _theta_at


def _store_codec(name):
    """--ttf-store: compressed storage of the accumulator's low-rank pair. Returns enc(X) -> tuple of stored tensors:
    fp12s keeps 12-bit codes (X scaled per column to |X| <= 32768, then fp16 with the low 4 mantissa bits rounded off:
    sign, 5 exponent, 6 mantissa bits) + the per-column fp32 scale (1 x r); bf16 keeps the tensor in bf16. (The removed
    codecs int8 / int8det / fp8 / fp16s / fp16 still decode: _decode, _load_pair.)"""
    if name in ("", "none", "fp32"):
        return None
    if name == "fp12s":
        def enc(X):
            sc = X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 32768
            if X.is_cuda and X.dim() == 2:   # one kernel, identical codes (kernel.pack12_scaled)
                from .kernel import pack12_scaled
                return (pack12_scaled(X.float(), sc), sc)
            return (_pack12((X / sc).to(torch.float16)), sc)
    elif name == "bf16":
        def enc(X):
            return (X.to(torch.bfloat16),)
    else:
        raise ValueError(f"--ttf-store {name}: fp32, bf16 or fp12s")
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
    if c.dtype == torch.uint8 and c.is_cuda and c.dim() == 2:   # one kernel, identical values
        from .kernel import unpack12_scaled
        return unpack12_scaled(c, s)
    return (_unpack12(c, s.shape[-1]) if c.dtype == torch.uint8 else c).float() * s


def _store_pair(enc, U, V):
    with _rf("ttf.codec"):
        return _store_pair_(enc, U, V)


def _store_pair_(enc, U, V):
    """(U, V) -> flat tuple of stored tensors (U codes[, U scale], V codes[, V scale]), so checkpoint / resume code that
    moves tuples of tensors works unchanged"""
    return enc(U) + enc(V)


def _load_pair(t):
    with _rf("ttf.codec"):
        return _load_pair_(t)


def _load_pair_(t):
    """inverse of _store_pair (and a no-op on a plain fp32 (U, V)); (U fp32, V codes[, V scale]): --ttf-step-fp32"""
    if len(t) == 4:
        return _decode(t[0], t[1]), _decode(t[2], t[3])
    if len(t) == 3:
        return t[0], _decode(t[1], t[2])
    return t[0].float(), t[1].float()


def _ts_on_host(args, i, n):
    """--ttf-offload async with --ttf-offload-keep F: layer i of n (module order) keeps its state in host memory
    unless it is among the first round(F n)"""
    return args.ttf_offload != "none" and i >= round(args.ttf_offload_keep * n)


class _PinnedPool:
    """Page-locked host memory of exactly the requested size for the offloaded TTF state. torch's pinned allocator
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


class _AsyncOffload:
    """--ttf-offload async: the TTF state in pinned host memory, its copies on two side streams (host-to-device prefetch, device-to-host
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


def _host_get(d, k, dev, off):
    with _rf("ttf.store"):
        return _host_get_(d, k, dev, off)


def _host_get_(d, k, dev, off):
    """--ttf-offload async: a state tuple kept in pinned host memory, prefetched to the GPU for this layer's update"""
    return _OFFLOAD.get(d, k, dev) if off else d[k]


def _host_put(d, k, v, off):
    with _rf("ttf.store"):
        return _host_put_(d, k, v, off)


def _host_put_(d, k, v, off):
    """--ttf-offload async: write a state tuple back to its pinned host buffers (reused across steps)"""
    if not off:
        d[k] = v
        return
    _OFFLOAD.put(d, k, v)


def _ts_state_bytes(lr_state):
    """bytes held by the TTF state (accumulator pairs, short momentum, factored v), counted per stored tensor"""
    n = 0
    for k, v in lr_state.items():
        if k == "_ts":
            for kk, vv in v.items():
                if isinstance(vv, tuple):
                    n += sum(x.numel() * x.element_size() for x in vv if torch.is_tensor(x))
        elif isinstance(v, tuple):
            n += sum(x.numel() * x.element_size() for x in v if torch.is_tensor(x))
    return n


def _acc_turn(U, V, S_):
    """the basis turn at the end of a micro-batch step: one subspace iteration of the step's tally A' = A + sum u from
    U = lk U_0 + sum u V (the step's u entered on its basis; U_0 is not kept): X = A'^T U = V (U^T U) + (I - V V^T) S_
    with S_ = (sum u)^T U [K, r] (_sketch_cross), and A kept on the new basis, A <- U V^T Vn Vn^T. The step's mass outside
    V is not kept. A one-batch step instead turns exactly from U_0 (ts_layer / _ts_layer_factored).
    Measured against the alternatives (26 Oct, Qwen 0.8B P2h recipe, 1 x 16 micro-batches, 1000 steps): the mass added
    from k = 256 sketches of sum g (accfix 0412490) ran the moves away within 100 steps (noise ~sqrt(K / k) x the signal);
    the exact one-batch turn at every micro-batch trained worse (+0.08 train loss over steps 300-576, twice the moves)
    at 2.5-5x the update's cost at 27B shapes; this rule is ahead of the one before it (the turn from M u_M^T U, the last micro-batch's u x
    M: wiki 2.9478 vs 2.9536 at 999). Returns (U, V)"""
    S_ = S_ - V @ (V.T @ S_)
    Vn = _orth(V @ (U.T @ U) + S_); del S_
    return U @ (V.T @ Vn), Vn


def _sketch_cross(U, SPt, Sc, ia, ib, lr_ratio):
    """(sum u)^T U [K, r] without sum u: through the step's own k-row sketch of its gradient, SPt = (Sc sum g)^T [K, k],
    which the update keeps for v anyway (E[Sc^T Sc] = I; Sc regenerated from its seed), with the step's v (ia, ib):
    -lr diag(ib) SPt (Sc diag(ia) U)"""
    return (-lr_ratio) * ib[:, None] * (SPt.float() @ (Sc @ (ia[:, None] * U)))


def _step_scale(R, C, b2, n):
    """1 / a_i, 1 / b_j of the step's v (R, C after n steps), as the update's u uses them"""
    a = (R / R.mean().clamp_min(1e-30) / (1 - b2 ** n)).sqrt(); b = C.sqrt()
    return 1.0 / torch.maximum(a, a.max() * 1e-12 + 1e-30), 1.0 / torch.maximum(b, b.max() * 1e-12 + 1e-30)


def _micro_t(l, st, key, M_, c):
    """a layer's update count t (in micro-batches with --accum M > 1). From the trainer's micro-batch index (l._ts_micro)
    and step (c["step"]) when it gives them: a layer that sees no tokens in some micro-batches (an --moe expert) keeps
    its count aligned with the step (the leak on the step's first call, the step's sketch seed in all its calls), and a
    step it left without its last micro-batch drops that step's running v sums (no move, v not stepped for it). Without
    gaps this is the count + 1, as without the index"""
    t0 = st.get(("t", key), 0)
    i = getattr(l, "_ts_micro", None) if M_ > 1 else None
    if i is None or c.get("step") is None:
        t = t0 + 1
    else:
        s0 = st.get(("ls", key))
        ts = (t0 - 1) // M_ if (s0 == c["step"] and t0) else -(-t0 // M_)   # this step's index: as before, or the next
        if s0 != c["step"] and t0 % M_:
            st.pop(("mx", key), None)
        t = ts * M_ + i + 1; st[("ls", key)] = c["step"]
    st[("t", key)] = t
    return t


def _hspend_basis(st, key, x, V, c):
    """--ttf-hspend F: a move spends theta (D H~) instead of theta D, H~ = I + F (C - I), C the correlation of the layer's
    input factor x (an EMA, ~100 steps, of x^T x from the last micro-batch of each step; the second moment, not the
    covariance: a move's output change is dW x with x the input the trits multiply, already centred in a centred
    layer): a move at (i, j)
    also takes theta d C_jk from the row's entry (i, k) fed by a correlated input, whose evidence was for the same
    output change (P2h 500 -> 999: a row's moves changed the output 1.2-1.6 x what independent inputs would, chance
    1.0-1.1). Low rank exactly: U -= theta D (H~ V), the spend's basis Vs = H~ V [K, r]. State: H [K, K] fp32 and a count"""
    H, n = st.get(("hs", key), (None, 0))
    Hn = _dpm(torch.mm(x.T, x, out_dtype=torch.float32) if x.dtype == torch.bfloat16 else x.T.float() @ x.float())
    Hn = Hn / x.shape[0]
    w = max(0.01, 1.0 / (n + 1))
    H = Hn if H is None else H.mul_(1 - w).add_(Hn, alpha=w); del Hn
    st[("hs", key)] = (H, n + 1)
    d = H.diag().clamp_min(1e-30).rsqrt()
    CV = d[:, None] * (H @ (d[:, None] * V.float()))
    return V + float(c["hspend"]) * (CV - V)


def _band_fire(l, st, key, c, U, V, A_prev, u, lk, theta):
    """--ttf-band F: the exact tally on the entries near the threshold. A band of at most F N K entries (where |A| >=
    --ttf-band-lo x theta after the step's moves) keeps E, the exact tally there (lk A + u, the moves' spends) minus
    the low-rank accumulator's value (the part of each step's evidence the rank-r projection drops, and the part of a
    spend the low-rank spend misses); the moves are decided by A + E on the band, by A elsewhere. An entry entering the
    band starts at E = 0; one leaving it loses E. State: the band's flat indices (int64) and E (fp32). (Phase 1: the
    torch path, dense [N, K] temporaries and g from its factors; one batch per step.) Returns (U, moves)"""
    from .kernel import unpack_rows, pack_rows
    N, K = l.N, l.K
    A = U @ V.T
    bd = st.get(("bd", key))
    if bd is not None and A_prev is not None:
        idx, e = bd
        Ex = lk * (A_prev.view(-1)[idx] + e) + u.view(-1)[idx]       # the exact tally after this step's evidence
        A.view(-1)[idx] = Ex
    w0 = unpack_rows(l.wpacked, K).to(torch.int8)
    d = A.sign()
    fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
    D = torch.where(fire, d, torch.zeros_like(d)); nf = fire.sum(); del d, fire
    l.wpacked.copy_(pack_rows(w0 + D.to(torch.int8))); del w0
    sp = c["spend"] * theta
    A.sub_(sp * D)                                                   # the exact spend (on the band; elsewhere A is replaced)
    U = U - sp * (D @ V); del D
    Alr = U @ V.T
    E = torch.zeros(N * K, device=U.device)
    if bd is not None and A_prev is not None:
        E[idx] = A.view(-1)[idx] - Alr.view(-1)[idx]
    del A
    score = (Alr.view(-1) + E).abs_(); del Alr
    n_ = max(1, int(c["band"] * N * K))
    score[score < c.get("band_lo", 0.5) * theta] = 0
    top = score.topk(n_, sorted=False)
    keep = top.values > 0
    idx = top.indices[keep].sort().values; del score, top, keep
    st[("bd", key)] = (idx, E[idx])
    return U, nf


def _orth(X):
    tf = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False      # (--tf32: the Gram products and QR need full fp32)
    try:
        with _rf("ttf.orth"):
            return _orth_(X)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf


STEP_FP32 = [False]   # --ttf-step-fp32: the accumulator's U in fp32 between a step's micro-batches (see _ts_layer_factored)
ORTH = ["qr"]   # --ttf-orth: qr (Householder), cholqr2 (two Cholesky QR passes, QR where they would fail), or spanqr
_SPANQR = [None]  # (the spanqr package's orth: robust shifted Cholesky QR, no host sync; above r 512 block Gram-Schmidt)


def _orth_(X):
    """orthonormal basis of the columns of X (tall, K x r), Householder QR. Only the span matters to TTF
    (A = U V^T does not depend on the basis chosen). cholqr2: Q = X R^-1 with R^T R the Cholesky factorisation of
    X^T X, twice (the second pass restores orthogonality to fp32 rounding when the first is within 0.1 of it, i.e.
    cond(X) below ~1e3); otherwise (rank deficient, ill conditioned: Cholesky QR would lose directions) Householder QR.
    One host sync per call for that check."""
    if ORTH[0] == "qr" or X.dtype != torch.float32 or X.shape[0] < X.shape[1]:
        return torch.linalg.qr(X)[0]
    if ORTH[0] == "spanqr":
        # spanqr (github.com/oguzeldereli/spanqr; 4080 laptop, 5120 x 256: 0.90 ms against cholqr2's 1.11 and
        # Householder's 3.47). Above r 512 (r a multiple of 256) its block Gram-Schmidt over 256-column blocks (spanqr
        # 4986987: 22/22 robustness cases at 5120 / 6144 / 17408 x 1024; 5120 x 1024 10.5 ms against Householder's
        # 11.9). The shapes it does not take (r not a multiple of 32, r > 512 not a multiple of 256, CPU): Householder
        r = X.shape[1]
        if not (X.is_cuda and r % 32 == 0 and (r <= 512 or r % 256 == 0)):
            return torch.linalg.qr(X)[0]
        if _SPANQR[0] is None:
            from orth import orth as _o
            _SPANQR[0] = _o
        return _SPANQR[0](X.contiguous())
    L1, i1 = torch.linalg.cholesky_ex(X.T @ X)
    Q = torch.linalg.solve_triangular(L1.T, X, upper=True, left=False)
    G2 = Q.T @ Q
    L2, i2 = torch.linalg.cholesky_ex(G2)
    G2.diagonal().sub_(1.0)
    ok = (i1 == 0) & (i2 == 0) & (G2.abs().amax() < 0.1)
    if not bool(ok):
        return torch.linalg.qr(X)[0]
    return torch.linalg.solve_triangular(L2.T, Q, upper=True, left=False)


def ts_layer(l, g, state, c):
    """one layer of TTF (the no-momentum update): g is this layer's gradient (fp32), or its factors (gy, x) with
    --dw factored; c holds the settings and lr_ratio. Called from the layer's backward right after its input gradient
    is computed, so only one layer's fp32 gradient exists at a time.
    Per layer: v, the factored row x column EMA of g^2 (decay b2); u = -lr_ratio g / sqrt(v); A, the accumulator, rank
    r: A <- (1 - lr_ratio^p / tau) A + u projected on its basis (V turns with the evidence); a trit moves by sign(A_ij)
    when |A_ij| >= theta (and stays in {-1, 0, 1}), and the move spends theta: A -= theta D (kept low-rank: U -= theta
    D V). With --accum M > 1 every micro-batch's u enters the accumulator, normalised by the v of the steps before;
    v, the basis turn (_acc_turn), the firing and the spend once per step, on the last micro-batch."""
    from .kernel import unpack_rows, pack_rows
    if getattr(l, "ts_frozen", False):   # (a plug-in's frozen layer: its trits stay, no state)
        return
    if not torch.is_tensor(g):           # --dw factored: the backward passed (gy, x) instead of the dense gradient
        return _ts_layer_factored(l, g, state, c)
    r, lr_ratio, theta, tau, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    r = getattr(l, "ts_rank", None) or r   # a layer's own accumulator rank (set by a plug-in, e.g. thin adapter factors)
    if c.get("theta_sched"):   # --ttf-theta-sched T0:T1[:START]: the threshold moves linearly from T0 to T1
        theta = _theta_at(c["theta_sched"], c.get("progress", 0.0))
    M_ = int(c.get("micro") or 1)   # --accum M: the accumulator takes every micro-batch's u
    final_ = getattr(l, "_ts_final", True) if M_ > 1 else True   # firing and the spend only on the last one
    b2o = b2
    if M_ > 1:
        b2 = b2 ** (1.0 / M_)
    sa = _store_codec(c.get("store_a", ""))                                            # --ttf-store: real storage
    st = state.setdefault("_ts", {})
    off, dev = (c.get("offload", False) if getattr(l, "_ts_host", True) else False), g.device  # --ttf-offload async (--ttf-offload-keep)
    key = id(l); t = _micro_t(l, st, key, M_, c)
    if key in state and is_dense_state(state[key]):
        raise SystemExit("--dw dense: this layer's TTF state is a dense accumulator (a --dw factored run's, "
                         "update.dense_acc); resume it with --dw factored")
    mic_nm = M_ > 1
    if mic_nm:    # --accum M > 1: the accumulator takes every micro-batch's u, normalised by the v of the steps before
        # (this step's v needs all of its micro-batches); on the first step there is none, so each micro-batch uses its
        # own g^2 x M^2 (each g_i carries the loss / M). Firing and the spend once per step, on the last one. As Adam's
        # v includes the current gradient (which bounds a step at 1 / sqrt(1 - beta2) per weight), this micro-batch's
        # own square (x M^2, the batch's scale) joins with weight 1 - beta2; without it a gradient spike passes into the
        # accumulator whole (110M: a 6% flip burst at step ~55, 5.64 vs 4.42 at 600 steps)
        ts_ = (t - 1) // M_
        if g.is_cuda:   # one pass for both sums of g^2 (rounding-level differences only)
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
    if M_ > 1:    # --accum M > 1: v is the batch's, stepped once per step with the step's beta2 as without micro-batches:
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
            turn_ = (SP.T, PS) + _step_scale(R, C, b2o, t // M_)       # (Sc sum g)^T, (sum g) Sr: the basis turn's sketches
            # the full [N, K] v only where it is read: --ttf-vdiag (the micro-batch u uses the factored v of the steps before)
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
    if c.get("vdiag") and v is not None:   # --ttf-vdiag: this step's g^2 against the second moment it is divided by
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
    if c.get("start") and c.get("step", 0) < c["start"]:   # --ttf-start S: the trits stay frozen (no evidence kept,
        del u, v                                             # nothing fires) until step S; v keeps learning
        return
    del v
    ra = min(r, l.N, l.K)
    lk = 1 - lr_ratio / tau   # the leak, annealed with the learning rate: the memory is a distance, not a step count
    if mic_nm and key in state:   # --accum M > 1: the basis stays fixed within the step (every micro-batch's u is projected
        # on it: U += u_i V), and turns once, on the last micro-batch (_acc_turn); turning it per micro-batch lets each
        # micro-batch's noisier u rotate the evidence away (Qwen 0.8B, 1 x 16: +0.09 train loss by step 500, acc_q08a)
        U, V = _load_pair(_host_get(state, key, dev, off))
        if (t - 1) % M_ == 0:
            U = U * lk
        U = U + u @ V
        if final_:
            SPt, PS_, ia_, ib_ = turn_
            U, V = _acc_turn(U, V, _sketch_cross(U, SPt, Sc, ia_, ib_, lr_ratio))
    elif key not in state:
        gen = torch.Generator(device=g.device).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        V = _orth(u.T @ torch.randn(l.N, ra, device=g.device, generator=gen))
        U = u @ V
    else:
        U, V = _load_pair(_host_get(state, key, dev, off))
        U = U * lk                                    # the step is one batch: the turn from U_0 (_acc_turn), exact
        Vn = _orth(V @ (U.T @ U) + u.T @ U)
        U = U @ (V.T @ Vn) + u @ Vn; V = Vn
    if mic_nm and not final_:   # --accum M > 1: not the last micro-batch: keep the accumulator, no firing
        _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
        return
    del u
    A = U @ V.T
    codes = getattr(l, "codes", None)         # int-k codes (intk_emb.IntKMatrix: levels 0 .. l.levels) instead of trits
    w0 = codes if codes is not None else unpack_rows(l.wpacked, l.K).to(torch.int8)
    d = A.sign()
    if codes is not None:
        nw = w0.float() + d
        fire = (A.abs() >= theta) & (nw >= 0) & (nw <= l.levels); del nw
    else:
        fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
    del A
    before = l.wpacked.clone() if l.track else None
    D = torch.where(fire, d, torch.zeros_like(d))
    nf = fire.sum()                           # a device tensor: no host sync per layer
    if codes is not None:
        codes.add_(D.to(torch.int8))
    else:
        l.wpacked.copy_(pack_rows(w0 + D.to(torch.int8)))   # int8 add: exact, no [N, K] fp32 temporary
    U = U - c["spend"] * (theta * (D @ V))
    _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + fire.numel()
    if before is not None:
        l._record_flips(before, l.wpacked)


def _mmb(A, B):
    """A (bf16, the factor) @ B on the bf16 tensor cores with fp32 accumulation and output (B rounded to bf16, as the
    dense path rounds the weight gradient to bf16)"""
    if A.dtype != torch.bfloat16:
        return A @ B
    if not A.is_cuda:        # (CPU: no mm with out_dtype; the same bf16 rounding of B, fp32 products)
        return A.float() @ B.to(A.dtype).float()
    return torch.mm(A, B.to(A.dtype), out_dtype=torch.float32)


def _fg_mm_r(fg, a, W):
    """u @ W for u = -lr diag(a) g diag(b), g = gy^T x, without g: -lr a * (gy^T (x (b W))) (fp32; W is K x r)"""
    gy, x, b = fg
    return a[:, None] * _mmb(gy.T, _mmb(x, b[:, None] * W))


def _fg_add_r(U, fg, a, W, alpha, cs=1.0):
    """cs U + alpha (a * (gy^T (x (b W)))), the accumulator's term from one micro-batch (cs: the step's leak on its first
    one): with bf16 factors in fewer passes (b W written straight to bf16, x (b W) out of the GEMM in bf16, the second
    product, the scale and the add in one kernel: kernel.acc_gemm); the same products, rounded at the same points,
    except the GEMM's bf16 output and the fp32 order"""
    gy, x, b = fg
    if gy.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        return cs * U + alpha * _fg_mm_r(fg, a, W)
    bw = torch.empty(W.shape, dtype=torch.bfloat16, device=W.device)
    torch.mul(b[:, None], W, out=bw)
    if gy.is_cuda:
        from .kernel import acc_gemm
        return acc_gemm(gy, torch.mm(x, bw), None, a, U, cs, alpha)
    return torch.addcmul(cs * U, a[:, None], torch.mm(gy.T, torch.mm(x, bw), out_dtype=torch.float32), value=alpha)


def _fg_mm_l(fg, a, W):
    """u^T @ W (W is N x r) without g: -lr b * (x^T (gy (a W))) (the -lr is applied by the caller)"""
    gy, x, b = fg
    return b[:, None] * _mmb(x.T, _mmb(gy, a[:, None] * W))


DP_REDUCE = [None]     # data parallel (set by train.py): t -> the mean of t over the ranks, in place


def _dpm(t):
    """data parallel: every quantity the factored update reads from g is linear in g, so the ranks' products averaged
    are the products of the mean gradient: each rank then runs the same update and moves the same trits"""
    if DP_REDUCE[0] is not None:
        DP_REDUCE[0](t)
    return t


def _layer_gen(l, dev):
    """the layer's own random generator on dev, made once (a CUDA generator costs ~20 us to build; the callers re-seed it,
    which resets its whole state, so the numbers are those of a fresh one)"""
    g = getattr(l, "_ts_gen", None)
    if g is None or g.device != dev:
        g = l._ts_gen = torch.Generator(device=dev)
    return g


def _ts_layer_factored(l, fg, state, c):
    """--dw factored K: the TTF update (one batch per step, or --accum M > 1) from the factors of the weight gradient,
    g = gy^T x, which is never formed. Everything the update reads from g is a product with a thin matrix: the
    accumulator's projections u V and u^T U (u = -lr g / sqrt(v) with the factored v = a_i b_j, so u = -lr diag(1/a) g
    diag(1/b)), and v's row / column sums of g^2, estimated from a K-column random sketch g S (E[S S^T] = I; with
    --accum M > 1 the same sketch the cross terms use, so the batch's sums are ||(sum_i g_i) S||^2). Differences from the
    dense path: the sums of g^2 are sketch estimates (relative noise sqrt(2 / K) per row and step, before v's EMA), and
    the 1e-8 in sqrt(v) + 1e-8 is left out (a scale factor needs a product form). Cost per call 2 T (N + K)(K_s + 2 r)
    instead of the dense gradient's 2 T N K."""
    gy, x = fg                                # [T, N], [T, K] (bf16; products accumulate in fp32: _mmb)
    r, lr_ratio, theta, tau, b2 = c["r"], c["lr_ratio"], c["theta"], c["tau"], c["b2"]
    r = getattr(l, "ts_rank", None) or r
    if c.get("theta_sched"):
        theta = _theta_at(c["theta_sched"], c.get("progress", 0.0))
    M_ = int(c.get("micro") or 1)
    final_ = getattr(l, "_ts_final", True) if M_ > 1 else True
    st = state.setdefault("_ts", {})
    dev = gy.device
    off = c.get("offload", False) if getattr(l, "_ts_host", True) else False
    sa = _store_codec(c.get("store_a", ""))
    key = id(l); t = _micro_t(l, st, key, M_, c)
    lk = 1 - lr_ratio / tau
    k_ = int(c["factored"])
    ts_ = (t - 1) // M_
    gen = _layer_gen(l, dev).manual_seed(11_000_000 + 2 * getattr(l, "_ts_idx", 0) + 1_000_003 * ts_)
    Sr = torch.randn(l.K, k_, device=dev, generator=gen) / k_ ** 0.5
    Sc = torch.randn(k_, l.N, device=dev, generator=gen) / k_ ** 0.5
    if MICRO_FUSED[0] and M_ > 1 and dev.type == "cuda" and not c.get("vdiag"):
        # --accum M > 1 on the GPU: each sketch side's statistics (its squares, the cross terms with the step's
        # running sum, v's EMA, the scale and its bounded reciprocal) in two kernels (kernel.micro_sketch) instead of
        # ~50 small ops; the running sum of the column side is kept as (Sc g)^T [K, k], so no transpose copy
        from .kernel import micro_sketch
        gS = _dpm(_mmb(gy.T, _mmb(x, Sr)))                                # g Sr  [N, k]
        SgT = _dpm(_mmb(x.T, _mmb(gy, Sc.T)))                             # (Sc g)^T  [K, k]
        R0 = C0 = None
        if ("v", key) in st and ts_ > 0:
            R0, C0 = _host_get(st, ("v", key), dev, off)
        bc_ = 1 - b2 ** ts_ if ts_ > 0 else 1.0
        PS, SP, rs, cs, ds, es = st.pop(("mx", key), None) or (None,) * 6
        ia, rs, ds, PS = micro_sketch(gS, PS, rs, ds, R0, M_, l.K, b2, bc_, True)
        ib, cs, es, SP = micro_sketch(SgT, SP, cs, es, C0, M_, l.N, b2, bc_, False)
        del gS, SgT, R0, C0
        if final_:
            rs, cs = torch.maximum(rs, ds / M_), torch.maximum(cs, es / M_)
            if ("v", key) in st:
                R, C = _host_get(st, ("v", key), dev, off); R = b2 * R + (1 - b2) * rs; C = b2 * C + (1 - b2) * cs
            else:
                R, C = (1 - b2) * rs, (1 - b2) * cs
            _host_put(st, ("v", key), (R, C), off)
            turn_ = (SP, PS) + _step_scale(R, C, b2, ts_ + 1)       # (Sc sum g)^T [K, k], (sum g) Sr: the turn's sketches
        else:
            st[("mx", key)] = (PS, SP, rs, cs, ds, es)
        del PS, SP
    else:
        gS = _dpm(_mmb(gy.T, _mmb(x, Sr)))                                    # g Sr  [N, k]
        Sg = _dpm(_mmb(x.T, _mmb(gy, Sc.T)).T.contiguous())                   # Sc g  [k, K]
        rsq, csq = gS.square().sum(1), Sg.square().sum(0)                     # ~ row / column sums of g^2
        if M_ > 1:   # as ts_layer's micro-batch branch, with the sketched own squares
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
                turn_ = (SP.T, PS) + _step_scale(R, C, b2, ts_ + 1)
                if c.get("vdiag"):   # --ttf-vdiag: this micro-batch's g^2 against v (the u extremes need the dense u: not kept)
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
    if c.get("start") and c.get("step", 0) < c["start"]:                  # --ttf-start: frozen, v keeps learning
        return
    ra = min(r, l.N, l.K)
    if dense_acc(l, ra) and not c.get("band") and not c.get("hspend"):
        return _ts_dense_acc(l, fgb, ia, state, st, key, t, M_, final_, lk, lr_ratio, theta, c, sa, off, dev)
    A_prev = None
    if key not in state:
        g_ = torch.Generator(device=dev).manual_seed(4_000_001 + 2 * getattr(l, "_ts_idx", 0))
        V = _orth(-lr_ratio * _dpm(_fg_mm_l(fgb, ia, torch.randn(l.N, ra, device=dev, generator=g_))))
        U = -lr_ratio * _dpm(_fg_mm_r(fgb, ia, V))
    elif M_ > 1:   # basis fixed within the step (every micro-batch's u enters on it), turned on the last (_acc_turn)
        U, V = _load_pair(_host_get(state, key, dev, off))
        U = _fg_add_r(U, fgb, ia, V, -lr_ratio, lk if (t - 1) % M_ == 0 else 1.0)   # (data parallel: this rank's term,
        # averaged on the last one)
        if final_:
            # data parallel: the micro-batches' terms are linear in g and V is fixed within the step, so each rank
            # summed its own and one mean here replaces a mean per micro-batch (N x r per layer: 16 x less traffic)
            U = _dpm(U)
            SPt, PS_, ia_, ib_ = turn_
            U, V = _acc_turn(U, V, _sketch_cross(U, SPt, Sc, ia_, ib_, lr_ratio))
    else:
        U, V = _load_pair(_host_get(state, key, dev, off))
        A_prev = U @ V.T if c.get("band") else None
        U = U * lk                                    # the step is one batch: the turn from U_0 (_acc_turn), exact
        Vn = _orth(V @ (U.T @ U) - lr_ratio * _dpm(_fg_mm_l(fgb, ia, U)))
        U = U @ (V.T @ Vn) - lr_ratio * _dpm(_fg_mm_r(fgb, ia, Vn)); V = Vn
    if not final_:
        if STEP_FP32[0] and sa and not off:   # --ttf-step-fp32: within the step U stays fp32 (V is fixed until the last
            # micro-batch: kept as stored); encoded once, on the last micro-batch. Skips U's encode / decode per micro-batch
            enc_ = st.get(("venc", key))
            if enc_ is None or (t - 1) % M_ == 0:
                enc_ = st[("venc", key)] = sa(V)
            _host_put(state, key, (U,) + tuple(enc_), off)
        else:
            _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
        return
    st.pop(("venc", key), None)
    from .kernel import unpack_rows, pack_rows
    before = l.wpacked.clone() if l.track else None
    Vs = _hspend_basis(st, key, x, V, c) if c.get("hspend") else V
    if c.get("band"):
        if A_prev is not None:
            g_ = torch.mm(gy.T, x, out_dtype=torch.float32) if gy.dtype == torch.bfloat16 else gy.T.float() @ x.float()
            u_ = (-lr_ratio) * ia[:, None] * g_ * ib[None, :]; del g_
        else:
            u_ = None                                 # (the first step: the band starts empty)
        U, nf = _band_fire(l, st, key, c, U, V, A_prev, u_, lk, theta)
    elif FIRE_FUSED[0] and dev.type == "cuda":   # A, the moves, the repack and D V in one kernel
        from .kernel import ttf_fire
        DV, nf, dnnz = ttf_fire(U, V, l.wpacked, l.K, theta, Vs=Vs)
        DV = _dpm(DV)          # data parallel: the spend's sparse sums (index_add_) are not bit-reproducible: one mean
        l._after_fire(dnnz)    # the kernel writes the trits in place (no version bump): beta follows the moves
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
        U = U - c["spend"] * (theta * (D @ Vs))
    _host_put(state, key, _store_pair(sa, U, V) if sa else (U, V), off)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + l.N * l.K
    if before is not None:
        l._record_flips(before, l.wpacked)


DENSE_ACC = [True]     # the dense accumulator where the rank covers the rows (dense_acc); False: low-rank (tests, A/B)


def dense_acc(l, ra):
    """the layer's accumulator is held dense, A [N, K]: its rank is full, ra = min(N, K) (the adapters' factors at
    rank 1024, the 1024-row projections). A U V^T of that rank holds more numbers than A ((N + K) ra against N K); with
    ra = N < K its fixed basis V within a step drops the step's evidence outside span(V) (_acc_turn), with ra = K it is
    exact but pays the basis' products, orth and turn. Dense, A + u is exact and needs none of them"""
    return DENSE_ACC[0] and ra == min(l.N, l.K)


def _dense_mark(dev):
    return torch.zeros(0, dtype=torch.int8, device=dev)


def is_dense_state(t):
    """a stored accumulator tuple is a dense A (its codes and an empty int8 marker last), not a low-rank pair"""
    return len(t) >= 2 and t[-1].numel() == 0 and t[-1].dtype == torch.int8


def store_dense(sa, A):
    """A [N, K] -> the stored tuple: its codes (the --ttf-store codec, or fp32) and the marker"""
    return (sa(A) if sa else (A,)) + (_dense_mark(A.device),)


def load_dense(t, l):
    """the dense A from its stored tuple; from a low-rank pair (an earlier run's state): U V^T"""
    if not is_dense_state(t):
        U, V = _load_pair(t)
        return U @ V.T
    t = t[:-1]
    return t[0].float() if len(t) == 1 else _decode(t[0], t[1])


def _ts_dense_acc(l, fgb, ia, state, st, key, t, M_, final_, lk, lr_ratio, theta, c, sa, off, dev):
    """_ts_layer_factored with a dense accumulator (dense_acc): A <- lk A on the step's first micro-batch, A += u with
    u = -lr diag(ia) (gy^T (x diag(ib))) on every one (one GEMM, the factors in bf16 as the low-rank products), and on
    the last the firing on A itself (|A| >= theta, legal moves; A -= spend theta D). Stored as the low-rank pair is (the
    --ttf-store codec, fp32 within the step with --ttf-step-fp32)"""
    from .kernel import unpack_rows, pack_rows
    gy, x, ib = fgb
    A = load_dense(_host_get(state, key, dev, off), l) if key in state else None
    cs = lk if (t - 1) % M_ == 0 else 1.0
    if gy.is_cuda and gy.dtype == x.dtype == torch.bfloat16:   # (kernel.acc_gemm: x diag(ib) rounded to bf16 on load)
        from .kernel import acc_gemm
        A = acc_gemm(gy, x, ib, ia, A, cs, -lr_ratio)
    else:
        xb = torch.empty(x.shape, dtype=x.dtype, device=x.device)      # (bf16 factors: x diag(ib) rounded to bf16)
        torch.mul(x, ib[None, :], out=xb)
        term = _mmb(gy.T, xb) if gy.dtype == xb.dtype == torch.bfloat16 else gy.T.float() @ xb.float(); del xb
        A = (-lr_ratio) * ia[:, None] * term if A is None else torch.addcmul(A * cs, ia[:, None], term, value=-lr_ratio)
        del term
    if not final_:
        _host_put(state, key, store_dense(None if STEP_FP32[0] and not off else sa, A), off)
        return
    A = _dpm(A)            # data parallel: the micro-batches' terms are linear in g: one mean of the step's sum
    before = l.wpacked.clone() if l.track else None
    w0 = unpack_rows(l.wpacked, l.K).to(torch.int8)
    d = A.sign()
    fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
    D = torch.where(fire, d, torch.zeros_like(d)); del d
    nf = fire.sum(); del fire
    l.wpacked.copy_(pack_rows(w0 + D.to(torch.int8))); del w0
    A = A.sub_(D, alpha=c["spend"] * theta); del D
    _host_put(state, key, store_dense(sa, A), off)
    st["_nf"] = st.get("_nf", 0) + nf; st["_nt"] = st.get("_nt", 0) + l.N * l.K
    if before is not None:
        l._record_flips(before, l.wpacked)


MICRO_SKETCH = [int(os.environ.get("TERN_MICRO_K", "256"))]   # the dense path's sketch size: --dw-sketch's default
# (the v cross terms and the basis turn; TERN_MICRO_K: tests only)
FIRE_FUSED = [True]           # the factored path's firing through ttf.kernel.ttf_fire on CUDA (False: the torch reference; tests)
MICRO_FUSED = [os.environ.get("TERN_MICRO_FUSED", "1") == "1"]   # --accum M > 1: kernel.micro_sketch (TERN_MICRO_FUSED=0: the torch reference path)


def _micro_sketch(l, dev, step):
    """--accum M > 1: a layer's random maps (K x k, k x N; E[S S^T] = I) for the cross terms of the batch's v; fixed within
    a step, new every step, so their errors are independent across steps and v's EMA averages them out"""
    k = MICRO_SKETCH[0]
    gen = torch.Generator(device=dev).manual_seed(11_000_000 + 2 * getattr(l, "_ts_idx", 0) + 1_000_003 * step)
    return (torch.randn(l.K, k, device=dev, generator=gen) / k ** 0.5,
            torch.randn(k, l.N, device=dev, generator=gen) / k ** 0.5)


def _ts_rec(st, name, x):
    d = st.setdefault("_dg", {}); n = d.setdefault(name, [0.0, 0]); n[0] += x; n[1] += 1


def _ts_recmax(st, name, x):
    d = st.setdefault("_dgmax", {}); d[name] = max(d.get(name, -float("inf")), x)


def ts_step(model, state, c):
    """after the backward: the layers have already updated inside their backward, so this only collects the step's
    statistics (share of trits moved, --ttf-vdiag means / maxima)"""
    st = state.setdefault("_ts", {})
    nf, nt = st.pop("_nf", 0), st.pop("_nt", 0)
    info = {"ts_frac": float(nf) / max(nt, 1)}
    for k_, n_ in st.pop("_intk_nf", {}).items():          # --intk-emb: code moves this step, per matrix
        info[k_] = int(n_)
    for k_, (tot, n_) in st.pop("_dg", {}).items():      # --ttf-vdiag: means over layers this step
        info["dg_" + k_] = tot / max(n_, 1)
    for k_, x_ in st.pop("_dgmax", {}).items():           # --ttf-vdiag: maxima over layers this step
        info["dgmax_" + k_] = x_
    return info
