"""--attn-f16 (21 Nov): causal flash attention (forward + backward, Triton, flash-attention 2 order) with the
softmax-side GEMMs on fp16 MMA with fp16 accumulation per tile (GeForce Ada: 2x the fp32-accumulate rate, f16mm).

Which products accumulate where:
- the scores S = Q K^T and dP = dO V^T (contraction over the head dim, 64) accumulate in fp32: the softmax's exp and
  the dS = P (dP - delta) cancellation need them exact to fp32 sums of exact fp16 products;
- O += P V, dV += P^T dO, dK += dS^T Q, dQ += dS K (contraction over one key / query tile, 64-128 entries) are fp16
  MMAs with fp16 sums inside the tile, added into fp32 accumulators across tiles.
fp16 range: q, k, v, o are bf16 values converted exactly (|x| < 65504). P is in [0, 1]. dO is multiplied by a power
of two (on the device, from its max: no host sync) so its scale (the mean loss over ~32k tokens makes it ~1e-5) sits
inside fp16's normal range, and divided out at the end. dS and P^T rows get a per-row power of two before the fp16
MMA (from the row's max and the other operand's tile max) so a tile sum of products cannot overflow fp16; it is
divided out of that row's fp32 contribution exactly (powers of two).
Output: O bf16 and the log-sum-exp (base 2, of the scaled scores) fp32 [B, H, T] for the backward."""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
F16ACC = [True]
TILES = {"fwd": (128, 64, 4, 2), "dkdv": (64, 64, 4, 2), "dq": (64, 64, 4, 1), "bwd1": (64, 64, 4, 2)}  # (lab sweep 21 Nov)   # (BM, BN, warps, stages)


@triton.jit
def _pow2_below(x):
    """the largest power of two <= x (x > 0), elementwise, fp32"""
    return tl.exp2(tl.floor(tl.log2(x)))


@triton.jit
def _dot16(a, b, acc, F16: tl.constexpr):
    if F16:
        return acc + tl.dot(a, b, out_dtype=tl.float16).to(tl.float32)
    return tl.dot(a, b, acc)


@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, qk_scale, T, H, sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sgb, sgh, sgt, sob, soh, sot,
                D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F16: tl.constexpr):
    pm = tl.program_id(0); bh = tl.program_id(1); b = bh // H; h = bh % H
    rm = pm * BM + tl.arange(0, BM); rn = tl.arange(0, BN); rd = tl.arange(0, D)
    offq = b * sqb + h * sqh; offk = b * skb + h * skh; offv = b * svb + h * svh
    q = tl.load(q_ptr + offq + rm[:, None] * sqt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float16)
    m_i = tl.full((BM,), -float("inf"), tl.float32); l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    hi = tl.minimum((pm + 1) * BM, T)
    for n0 in range(0, hi, BN):
        cn = n0 + rn
        kt = tl.load(k_ptr + offk + cn[None, :] * skt + rd[:, None], mask=cn[None, :] < T, other=0.0).to(tl.float16)
        s = tl.dot(q, kt) * qk_scale
        if n0 + BN > pm * BM:   # (the diagonal tiles: causal mask)
            s = tl.where(rm[:, None] >= cn[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        al = tl.exp2(m_i - m_new)
        l_i = l_i * al + tl.sum(p, 1)
        v = tl.load(v_ptr + offv + cn[:, None] * svt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float16)
        acc = _dot16(p.to(tl.float16), v, acc * al[:, None], F16)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + b * sob + h * soh + rm[:, None] * sot + rd[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=rm[:, None] < T)
    tl.store(lse_ptr + bh * T + rm, m_i + tl.log2(l_i), mask=rm < T)


@triton.jit
def _pre_kernel(o_ptr, do_ptr, dl_ptr, T, H, sob, soh, sot, sdb, sdh, sdt, BM: tl.constexpr, D: tl.constexpr):
    """delta = rowsum(dO * O) (fp32, unscaled dO)"""
    pm = tl.program_id(0); bh = tl.program_id(1); b = bh // H; h = bh % H
    rm = pm * BM + tl.arange(0, BM); rd = tl.arange(0, D)
    o = tl.load(o_ptr + b * sob + h * soh + rm[:, None] * sot + rd[None, :], mask=rm[:, None] < T, other=0.0)
    do = tl.load(do_ptr + b * sdb + h * sdh + rm[:, None] * sdt + rd[None, :], mask=rm[:, None] < T, other=0.0)
    tl.store(dl_ptr + bh * T + rm, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1), mask=rm < T)


@triton.jit
def _dkdv_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, dl_ptr, dk_ptr, dv_ptr, sc_ptr, qk_scale, sm_scale, T, H,
                 sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sgb, sgh, sgt, sdb, sdh, sdt, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 F16: tl.constexpr):
    """one key tile [BN] against the query tiles at or after it: dV = P^T dO, dK = dS^T Q sm_scale"""
    pn = tl.program_id(0); bh = tl.program_id(1); b = bh // H; h = bh % H
    cn = pn * BN + tl.arange(0, BN); rd = tl.arange(0, D); rq = tl.arange(0, BM)
    offq = b * sqb + h * sqh; offk = b * skb + h * skh; offv = b * svb + h * svh; offg = b * sgb + h * sgh
    offd = b * sdb + h * sdh
    sdo = tl.load(sc_ptr)   # (dO's power of two)
    k = tl.load(k_ptr + offk + cn[:, None] * skt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float16)
    v = tl.load(v_ptr + offv + cn[:, None] * svt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float16)
    dk = tl.zeros((BN, D), tl.float32); dv = tl.zeros((BN, D), tl.float32)
    lo = (pn * BN // BM) * BM
    for m0 in range(lo, T, BM):
        rm = m0 + rq
        qt = tl.load(q_ptr + offq + rm[None, :] * sqt + rd[:, None], mask=rm[None, :] < T, other=0.0).to(tl.float16)
        st = tl.dot(k, qt) * qk_scale                                       # S^T [BN, BM], fp32
        lse = tl.load(lse_ptr + bh * T + rm, mask=rm < T, other=0.0)
        pt = tl.exp2(st - lse[None, :])
        pt = tl.where((rm[None, :] >= cn[:, None]) & (rm[None, :] < T), pt, 0.0)
        do = tl.load(do_ptr + offd + rm[:, None] * sdt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float32)
        do16 = (do * sdo).to(tl.float16)                                   # [BM, D]
        mdo = tl.maximum(tl.max(tl.max(tl.abs(do16.to(tl.float32)), 1), 0), 1e-30)
        # dV rows (keys): P^T rows <= 1; a tile sum of BM products <= BM * max|dO| (scaled): fp16-safe by construction
        dv = _dot16(pt.to(tl.float16), do16, dv, F16)
        dpt = tl.dot(v, tl.trans(do16)).to(tl.float32)                     # dP^T [BN, BM], fp32 (scaled by sdo)
        dl = tl.load(dl_ptr + bh * T + rm, mask=rm < T, other=0.0) * sdo
        dst = pt * (dpt - dl[None, :])
        # dK rows (keys): per-row power of two so BM * |dS^T row| * max|Q| <= 2^13
        mq = tl.maximum(tl.max(tl.max(tl.abs(qt.to(tl.float32)), 1), 0), 1e-30)
        rmax = tl.maximum(tl.max(tl.abs(dst), 1), 1e-30)
        rs = tl.minimum(_pow2_below(8192.0 / (BM * rmax * mq)), 1e30)
        if F16:
            dk += tl.dot((dst * rs[:, None]).to(tl.float16), tl.trans(qt), out_dtype=tl.float16).to(tl.float32) / rs[:, None]
        else:
            dk = tl.dot(dst.to(tl.float16), tl.trans(qt), dk)
    dk = dk * (sm_scale / sdo); dv = dv / sdo
    tl.store(dk_ptr + offk + cn[:, None] * skt + rd[None, :], dk.to(dk_ptr.dtype.element_ty), mask=cn[:, None] < T)
    tl.store(dv_ptr + offv + cn[:, None] * svt + rd[None, :], dv.to(dv_ptr.dtype.element_ty), mask=cn[:, None] < T)


@triton.jit
def _dq_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, dl_ptr, dq_ptr, sc_ptr, qk_scale, sm_scale, T, H,
               sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sgb, sgh, sgt, sdb, sdh, sdt, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, F16: tl.constexpr):
    """one query tile [BM] against the key tiles at or before it: dQ = dS K sm_scale"""
    pm = tl.program_id(0); bh = tl.program_id(1); b = bh // H; h = bh % H
    rm = pm * BM + tl.arange(0, BM); rd = tl.arange(0, D); rk = tl.arange(0, BN)
    offq = b * sqb + h * sqh; offk = b * skb + h * skh; offv = b * svb + h * svh; offg = b * sgb + h * sgh
    offd = b * sdb + h * sdh
    sdo = tl.load(sc_ptr)
    q = tl.load(q_ptr + offq + rm[:, None] * sqt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float16)
    do = tl.load(do_ptr + offd + rm[:, None] * sdt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float32)
    do16 = (do * sdo).to(tl.float16)
    lse = tl.load(lse_ptr + bh * T + rm, mask=rm < T, other=0.0)
    dl = tl.load(dl_ptr + bh * T + rm, mask=rm < T, other=0.0) * sdo
    dq = tl.zeros((BM, D), tl.float32)
    hi = tl.minimum((pm + 1) * BM, T)
    for n0 in range(0, hi, BN):
        cn = n0 + rk
        kt = tl.load(k_ptr + offk + cn[None, :] * skt + rd[:, None], mask=cn[None, :] < T, other=0.0).to(tl.float16)
        vt = tl.load(v_ptr + offv + cn[None, :] * svt + rd[:, None], mask=cn[None, :] < T, other=0.0).to(tl.float16)
        s = tl.dot(q, kt) * qk_scale
        p = tl.exp2(s - lse[:, None])
        p = tl.where(rm[:, None] >= cn[None, :], p, 0.0)
        dp = tl.dot(do16, vt)
        ds = p * (dp - dl[:, None])
        mk = tl.maximum(tl.max(tl.max(tl.abs(kt.to(tl.float32)), 1), 0), 1e-30)
        rmax = tl.maximum(tl.max(tl.abs(ds), 1), 1e-30)
        rs = tl.minimum(_pow2_below(8192.0 / (BN * rmax * mk)), 1e30)
        if F16:
            dq += tl.dot((ds * rs[:, None]).to(tl.float16), tl.trans(kt), out_dtype=tl.float16).to(tl.float32) / rs[:, None]
        else:
            dq = tl.dot(ds.to(tl.float16), tl.trans(kt), dq)
    dq = dq * (sm_scale / sdo)
    tl.store(dq_ptr + offq + rm[:, None] * sqt + rd[None, :], dq.to(dq_ptr.dtype.element_ty), mask=rm[:, None] < T)


@triton.jit
def _bwd1_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, dl_ptr, dk_ptr, dv_ptr, dq_ptr, sc_ptr, qk_scale, sm_scale, T,
                 H, sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sgb, sgh, sgt, sdb, sdh, sdt, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 F16: tl.constexpr):
    """_dkdv_kernel with dQ in the same pass (flash-attention 2's backward): S, P, dP formed once per (key tile, query
    tile); dQ's per-tile product dS K (fp16-accumulated, per-row power of two) added to an fp32 [B H T D] buffer with
    atomics (scaled by sm_scale / sdo afterwards)"""
    pn = tl.program_id(0); bh = tl.program_id(1); b = bh // H; h = bh % H
    cn = pn * BN + tl.arange(0, BN); rd = tl.arange(0, D); rq = tl.arange(0, BM)
    offq = b * sqb + h * sqh; offk = b * skb + h * skh; offv = b * svb + h * svh; offg = b * sgb + h * sgh
    offd = b * sdb + h * sdh
    sdo = tl.load(sc_ptr)
    k = tl.load(k_ptr + offk + cn[:, None] * skt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float16)
    v = tl.load(v_ptr + offv + cn[:, None] * svt + rd[None, :], mask=cn[:, None] < T, other=0.0).to(tl.float16)
    mk = tl.maximum(tl.max(tl.max(tl.abs(k.to(tl.float32)), 1), 0), 1e-30)
    dk = tl.zeros((BN, D), tl.float32); dv = tl.zeros((BN, D), tl.float32)
    lo = (pn * BN // BM) * BM
    for m0 in range(lo, T, BM):
        rm = m0 + rq
        qt = tl.load(q_ptr + offq + rm[None, :] * sqt + rd[:, None], mask=rm[None, :] < T, other=0.0).to(tl.float16)
        st = tl.dot(k, qt) * qk_scale
        lse = tl.load(lse_ptr + bh * T + rm, mask=rm < T, other=0.0)
        pt = tl.exp2(st - lse[None, :])
        pt = tl.where((rm[None, :] >= cn[:, None]) & (rm[None, :] < T), pt, 0.0)
        do = tl.load(do_ptr + offd + rm[:, None] * sdt + rd[None, :], mask=rm[:, None] < T, other=0.0).to(tl.float32)
        do16 = (do * sdo).to(tl.float16)
        dv = _dot16(pt.to(tl.float16), do16, dv, F16)
        dpt = tl.dot(v, tl.trans(do16))
        dl = tl.load(dl_ptr + bh * T + rm, mask=rm < T, other=0.0) * sdo
        dst = pt * (dpt - dl[None, :])                                      # dS^T [BN, BM]
        mq = tl.maximum(tl.max(tl.max(tl.abs(qt.to(tl.float32)), 1), 0), 1e-30)
        rmax = tl.maximum(tl.max(tl.abs(dst), 1), 1e-30)
        rs = tl.minimum(_pow2_below(8192.0 / (BM * rmax * mq)), 1e30)
        # dQ rows (queries): dS [BM, BN] = dst^T, per-column-of-dst power of two
        cmax = tl.maximum(tl.max(tl.abs(dst), 0), 1e-30)
        cs = tl.minimum(_pow2_below(8192.0 / (BN * cmax * mk)), 1e30)
        if F16:
            dk += tl.dot((dst * rs[:, None]).to(tl.float16), tl.trans(qt), out_dtype=tl.float16).to(tl.float32) / rs[:, None]
            dqp = tl.dot(tl.trans((dst * cs[None, :]).to(tl.float16)), k, out_dtype=tl.float16).to(tl.float32) / cs[:, None]
        else:
            dk = tl.dot(dst.to(tl.float16), tl.trans(qt), dk)
            dqp = tl.dot(tl.trans(dst.to(tl.float16)), k)
        tl.atomic_add(dq_ptr + bh * T * D + rm[:, None] * D + rd[None, :], dqp, mask=rm[:, None] < T, sem="relaxed")
    dk = dk * (sm_scale / sdo); dv = dv / sdo
    tl.store(dk_ptr + offk + cn[:, None] * skt + rd[None, :], dk.to(dk_ptr.dtype.element_ty), mask=cn[:, None] < T)
    tl.store(dv_ptr + offv + cn[:, None] * svt + rd[None, :], dv.to(dv_ptr.dtype.element_ty), mask=cn[:, None] < T)


ONE_PASS = [True]


def _gl(x):
    """an output [B, H, T, D] laid out as [B, T, H, D] (the model's layout; the gradients share it: strides G)"""
    B, H, T, D = x.shape
    return torch.empty(B, T, H, D, dtype=x.dtype, device=x.device).transpose(1, 2)


def _st(q, k, v, g):
    return tuple(t.stride(i) for t in (q, k, v, g) for i in (0, 1, 2))


def _bhtd(x):
    """[B, H, T, D] with unit stride along D and a common (b, h, t) stride layout for q, k, v, o"""
    return x if x.stride(-1) == 1 else x.contiguous()


class F16AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, T, D = q.shape
        q, k, v = _bhtd(q), _bhtd(k), _bhtd(v)
        sm = D ** -0.5
        o = _gl(q)
        ST = _st(q, k, v, o)
        lse = torch.empty(B * H, T, dtype=torch.float32, device=q.device)
        BM, BN, nw, ns = TILES["fwd"]
        _fwd_kernel[(triton.cdiv(T, BM), B * H)](q, k, v, o, lse, sm * LOG2E, T, H, *ST, o.stride(0), o.stride(1), o.stride(2), D=D, BM=BM,
                                                 BN=BN, F16=F16ACC[0], num_warps=nw, num_stages=ns)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        do = do if do.stride(-1) == 1 else do.contiguous()
        sm = D ** -0.5
        G = _gl(q); ST = _st(q, k, v, G)
        dl = torch.empty(B * H, T, dtype=torch.float32, device=q.device)
        _pre_kernel[(triton.cdiv(T, 128), B * H)](o, do, dl, T, H, o.stride(0), o.stride(1), o.stride(2),
                                                  do.stride(0), do.stride(1), do.stride(2), BM=128, D=D)
        # dO's power of two: max |dO| s <= 2^4 (dV's tile sums: BM * 16 <= 2^11), from the device (no sync)
        amax = do.abs().amax().float().clamp_min(1e-30)
        sdo = torch.exp2(torch.floor(torch.log2(16.0 / amax))).clamp_(2.0 ** -20, 2.0 ** 40).reshape(1)
        dk = _gl(k); dv = _gl(v)
        if ONE_PASS[0]:
            dq32 = torch.zeros(B, H, T, D, dtype=torch.float32, device=q.device)
            BM, BN, nw, ns = TILES["bwd1"]
            _bwd1_kernel[(triton.cdiv(T, BN), B * H)](q, k, v, do, lse, dl, dk, dv, dq32, sdo, sm * LOG2E, sm, T, H,
                                                      *ST, do.stride(0),
                                                      do.stride(1), do.stride(2), D=D, BM=BM, BN=BN, F16=F16ACC[0],
                                                      num_warps=nw, num_stages=ns)
            dq = _gl(q); dq.copy_(dq32 * (sm / sdo))
            return dq, dk, dv
        dq = _gl(q)
        BM, BN, nw, ns = TILES["dkdv"]
        _dkdv_kernel[(triton.cdiv(T, BN), B * H)](q, k, v, do, lse, dl, dk, dv, sdo, sm * LOG2E, sm, T, H,
                                                  *ST, do.stride(0), do.stride(1),
                                                  do.stride(2), D=D, BM=BM, BN=BN, F16=F16ACC[0], num_warps=nw,
                                                  num_stages=ns)
        BM, BN, nw, ns = TILES["dq"]
        _dq_kernel[(triton.cdiv(T, BM), B * H)](q, k, v, do, lse, dl, dq, sdo, sm * LOG2E, sm, T, H,
                                                *ST, do.stride(0), do.stride(1),
                                                do.stride(2), D=D, BM=BM, BN=BN, F16=F16ACC[0], num_warps=nw,
                                                num_stages=ns)
        return dq, dk, dv


def attention(q, k, v):
    """causal softmax(q k^T / sqrt(D)) v, q k v [B, H, T, D] bf16 (one shared layout)"""
    return F16AttnFn.apply(q, k, v)
