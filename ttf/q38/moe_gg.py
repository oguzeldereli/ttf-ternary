"""moe_gg.py: grouped GEMMs over expert segments in Triton (S, 5 Nov 2026). torch._grouped_mm on sm_120 (torch 2.13)
is a host loop over the groups (one .item() and one mm per expert: 8k syncs per layer and window on Flash-Next), so
the expert products are launch-bound. Here one launch per product, no host sync:

  gg(a, W, trans, sch) -> c       c[m] = a[m] @ B_e for the rows m of expert e, B_e = W[e] (trans False: W [E, R, O])
                                  or W[e]^T (trans True: W [E, O, R]); int8 a and W: exact int32 sums (the padded
                                  path's exact fp32 sums), returned as fp32; bf16 / fp32 a: fp32 accumulation
  gw(l, r, sch, E) -> P           P[e] = l_e^T r_e over the rows of expert e ([E, p, q], fp32): the weight gradient
  GG.apply(a, W, trans, sch)      autograd for a (bf16 / fp32) and W (dA = dC B_e^T, dW_e from gw)

sch = Sched(offs, M): the row tiles of the segments (built on the device from offs [E + 1], rows sorted by expert)."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

BM = 64


class Sched:
    """the BM-row tiles of each expert's segment: tile t -> (expert, first row, end row); tiles past the last are
    marked m0 = -1 (the grid is the static bound ceil(M / BM) + E, so no host sync)"""
    def __init__(self, offs, M, bm=BM):
        offs = offs.to(torch.int64)
        E = offs.numel() - 1
        cnt = offs[1:] - offs[:-1]
        nt = (cnt + bm - 1) // bm
        cum = torch.cumsum(nt, 0)
        self.n = (M + bm - 1) // bm + E
        t = torch.arange(self.n, device=offs.device)
        e = torch.searchsorted(cum, t, right=True)
        ec = e.clamp_max(E - 1)
        m0 = offs[ec] + (t - (cum[ec] - nt[ec])) * bm
        self.te = ec.to(torch.int32)
        self.m0 = torch.where(e < E, m0, torch.full_like(m0, -1)).to(torch.int32)
        self.m1 = offs[ec + 1].to(torch.int32)
        self.offs = offs.to(torch.int32)
        self.E, self.M, self.bm = E, M, bm


@triton.jit
def _gg_kernel(A, B, C, RS, ES, TE, TM0, TM1, R, O, sam, sar, sbe, sbr, sbo, scm, sco,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INT8: tl.constexpr, HAS_RS: tl.constexpr,
               HAS_ES: tl.constexpr):
    t = tl.program_id(0)
    pn = tl.program_id(1)
    m0 = tl.load(TM0 + t)
    if m0 < 0:
        return
    e = tl.load(TE + t).to(tl.int64)
    m1 = tl.load(TM1 + t)
    rm = m0 + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < m1
    mn = rn < O
    a_p = A + rm.to(tl.int64)[:, None] * sam + rk[None, :] * sar
    b_p = B + e * sbe + rk[:, None] * sbr + rn[None, :] * sbo
    if INT8:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, R, BK):
        mk = (k0 + rk) < R
        a = tl.load(a_p, mask=mm[:, None] & mk[None, :], other=0)
        b = tl.load(b_p, mask=mk[:, None] & mn[None, :], other=0)
        if INT8:
            acc += tl.dot(a, b)
        else:
            acc += tl.dot(a, b.to(a.dtype))
        a_p += BK * sar
        b_p += BK * sbr
    c = acc.to(tl.float32)
    if HAS_RS:      # (the epilogue in torch's order: (acc * rs[m]) * es[e], fp32, then one rounding to C's dtype)
        c = c * tl.load(RS + rm, mask=mm, other=0)[:, None]
    if HAS_ES:
        c = c * tl.load(ES + e)
    c_p = C + rm.to(tl.int64)[:, None] * scm + rn[None, :] * sco
    tl.store(c_p, c.to(C.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _gw_kernel(L, Rt, P, OFF, p, q, slm, slp, srm, srq, spe, spp, spq,
               BP: tl.constexpr, BQ: tl.constexpr, BMR: tl.constexpr):
    e = tl.program_id(0)
    pp = tl.program_id(1)
    pq = tl.program_id(2)
    r0 = tl.load(OFF + e)
    r1 = tl.load(OFF + e + 1)
    ip = pp * BP + tl.arange(0, BP)
    iq = pq * BQ + tl.arange(0, BQ)
    im = tl.arange(0, BMR)
    acc = tl.zeros((BP, BQ), dtype=tl.float32)
    for m0 in range(r0, r1, BMR):
        rm = m0 + im
        msk = rm < r1
        lt = tl.load(L + rm.to(tl.int64)[None, :] * slm + ip[:, None] * slp, mask=msk[None, :] & (ip[:, None] < p), other=0)
        rr = tl.load(Rt + rm.to(tl.int64)[:, None] * srm + iq[None, :] * srq, mask=msk[:, None] & (iq[None, :] < q), other=0)
        acc += tl.dot(lt, rr.to(lt.dtype))
    tl.store(P + e.to(tl.int64) * spe + ip[:, None] * spp + iq[None, :] * spq, acc,
             mask=(ip[:, None] < p) & (iq[None, :] < q))


def _bn(O):
    return 128 if O >= 128 else max(16, triton.next_power_of_2(O))


def gg(a, W, trans, sch, rs=None, es=None, out_dtype=torch.float32):
    """a [M, R] rows grouped by expert (sch), W [E, R, O] (trans False) or [E, O, R] (trans True) -> [M, O] fp32
    (or out_dtype); rs [M] / es [E] (fp32): the epilogue's row and expert scales"""
    M, R = a.shape
    if trans:
        O = W.shape[1]; sbr, sbo = W.stride(2), W.stride(1)
    else:
        O = W.shape[2]; sbr, sbo = W.stride(1), W.stride(2)
    assert W.shape[0] == sch.E and M == sch.M
    i8 = a.dtype == torch.int8
    assert i8 == (W.dtype == torch.int8) or not i8
    c = torch.empty(M, O, device=a.device, dtype=out_dtype)
    if M == 0:
        return c
    BN = _bn(O)
    BK = 64 if i8 else 32
    nw, ns = 4, 3
    cf = GG_CFG.get((i8, a.dtype == torch.bfloat16 and W.dtype == torch.int8, trans))
    if cf is not None and O >= cf[0]:       # (tuned tiles: BN, BK, warps, stages; the same sums, see gg_tune.py)
        BN, BK, nw, ns = cf
    grid = (sch.n, triton.cdiv(O, BN))
    rs_ = rs.float().contiguous() if rs is not None else c
    es_ = es.float().contiguous() if es is not None else c
    _gg_kernel[grid](a, W, c, rs_, es_, sch.te, sch.m0, sch.m1, R, O, a.stride(0), a.stride(1), W.stride(0), sbr, sbo,
                     c.stride(0), c.stride(1), BM=sch.bm, BN=BN, BK=BK, INT8=i8, HAS_RS=rs is not None,
                     HAS_ES=es is not None, num_warps=nw, num_stages=ns)
    return c


# (int8 a, bf16 a x int8 W, trans) -> (BN, BK, num_warps, num_stages), used where the output width >= BN. Tuned on
# the RTX PRO 6000 (sm_120) at the Q38 expert shapes (gg_tune.py, 6 Nov; every config bit-identical to the defaults):
# forward int8 gu 6.58 -> 2.92 ms, dn 2.74 -> 1.87; dx bf16 x int8 gu 10.29 -> 5.29, dn 5.12 -> 2.82
GG_CFG = {(True, False, True): (256, 128, 8, 3), (False, True, False): (256, 64, 8, 4)}


def gw(l, r, sch):
    """P[e] = l_e^T r_e, l [M, p], r [M, q] (rows grouped by expert) -> [E, p, q] fp32"""
    M, p = l.shape
    q = r.shape[1]
    P = torch.empty(sch.E, p, q, device=l.device, dtype=torch.float32)
    BP, BQ = _bn(p), _bn(q)
    grid = (sch.E, triton.cdiv(p, BP), triton.cdiv(q, BQ))
    _gw_kernel[grid](l, r, P, sch.offs, p, q, l.stride(0), l.stride(1), r.stride(0), r.stride(1),
                     P.stride(0), P.stride(1), P.stride(2), BP=BP, BQ=BQ, BMR=32, num_warps=4)
    return P


class GG(torch.autograd.Function):
    """c = a @ B_e per expert segment (fp32 out), with gradients for a and W"""
    @staticmethod
    def forward(ctx, a, W, trans, sch):
        ctx.save_for_backward(a, W)
        ctx.trans, ctx.sch = trans, sch
        return gg(a, W, trans, sch)

    @staticmethod
    def backward(ctx, gc):
        a, W = ctx.saved_tensors
        trans, sch = ctx.trans, ctx.sch
        g = gc.to(a.dtype).contiguous()
        ga = gW = None
        if ctx.needs_input_grad[0]:
            ga = gg(g, W, not trans, sch).to(a.dtype)
        if ctx.needs_input_grad[1]:
            gW = (gw(g, a, sch) if trans else gw(a, g, sch)).to(W.dtype)
        return ga, gW, None, None


@triton.jit
def _segsum_kernel(G, S, OFF, D, sgm, sgd, sse, ssd, BD: tl.constexpr, BMR: tl.constexpr):
    e = tl.program_id(0)
    pd = tl.program_id(1)
    r0 = tl.load(OFF + e)
    r1 = tl.load(OFF + e + 1)
    idd = pd * BD + tl.arange(0, BD)
    im = tl.arange(0, BMR)
    acc = tl.zeros((BMR, BD), dtype=tl.float32)
    for m0 in range(r0, r1, BMR):
        rm = m0 + im
        acc += tl.load(G + rm.to(tl.int64)[:, None] * sgm + idd[None, :] * sgd,
                       mask=(rm < r1)[:, None] & (idd < D)[None, :], other=0).to(tl.float32)
    tl.store(S + e.to(tl.int64) * sse + idd * ssd, tl.sum(acc, 0), mask=idd < D)


def segsum(g, sch):
    """S[e] = sum of g's rows of expert e ([M, D] -> [E, D] fp32), deterministic"""
    g2 = g.reshape(g.shape[0], -1)
    D = g2.shape[1]
    S = torch.empty(sch.E, D, device=g.device, dtype=torch.float32)
    BD = 128 if D >= 128 else max(16, triton.next_power_of_2(D))
    _segsum_kernel[(sch.E, triton.cdiv(D, BD))](g2, S, sch.offs, D, g2.stride(0), g2.stride(1), S.stride(0),
                                               S.stride(1), BD=BD, BMR=32, num_warps=4)
    return S.view(sch.E, *g.shape[1:])


class EVec(torch.autograd.Function):
    """v[e_of] (a per-expert [E, ...] tensor expanded to the expert-sorted rows); backward: the per-expert row sums
    (segsum: no atomics; v[e_of]'s own backward is a sorted index_put)"""
    @staticmethod
    def forward(ctx, v, e_of, sch):
        ctx.sch, ctx.dt = sch, v.dtype
        return v[e_of]

    @staticmethod
    def backward(ctx, g):
        return segsum(g.contiguous(), ctx.sch).to(ctx.dt), None, None


# ---- the per-expert row scales around the expert GEMM, fused (no [T, D] expansions of the per-expert vectors)
@triton.jit
def _pre_kernel(X, MU, Cs, Y, TE, TM0, TM1, D, sxm, sye, HAS_MU: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    pd = tl.program_id(1)
    m0 = tl.load(TM0 + t)
    if m0 < 0:
        return
    e = tl.load(TE + t).to(tl.int64)
    m1 = tl.load(TM1 + t)
    rm = m0 + tl.arange(0, BM)
    idd = pd * BD + tl.arange(0, BD)
    msk = (rm < m1)[:, None] & (idd < D)[None, :]
    x = tl.load(X + rm.to(tl.int64)[:, None] * sxm + idd[None, :], mask=msk, other=0).to(tl.float32)
    c = tl.load(Cs + e * D + idd, mask=idd < D, other=0)
    if HAS_MU:
        x = x - tl.load(MU + e * D + idd, mask=idd < D, other=0)[None, :]
    tl.store(Y + rm.to(tl.int64)[:, None] * sye + idd[None, :], x * c[None, :], mask=msk)


@triton.jit
def _pre_bwd_kernel(G, X, MU, Cs, GX, GC, OFF, D, sgm, sxm, HAS_MU: tl.constexpr, BD: tl.constexpr, BMR: tl.constexpr):
    """per (expert, column block): gx = g c_e for its rows, gc_e = sum g (x - mu_e)"""
    e = tl.program_id(0)
    pd = tl.program_id(1)
    r0 = tl.load(OFF + e)
    r1 = tl.load(OFF + e + 1)
    idd = pd * BD + tl.arange(0, BD)
    c = tl.load(Cs + e.to(tl.int64) * D + idd, mask=idd < D, other=0)
    if HAS_MU:
        mu = tl.load(MU + e.to(tl.int64) * D + idd, mask=idd < D, other=0)
    acc = tl.zeros((BMR, BD), dtype=tl.float32)
    for m0 in range(r0, r1, BMR):
        rm = m0 + tl.arange(0, BMR)
        msk = (rm < r1)[:, None] & (idd < D)[None, :]
        g = tl.load(G + rm.to(tl.int64)[:, None] * sgm + idd[None, :], mask=msk, other=0).to(tl.float32)
        x = tl.load(X + rm.to(tl.int64)[:, None] * sxm + idd[None, :], mask=msk, other=0).to(tl.float32)
        if HAS_MU:
            x = x - mu[None, :]
        acc += g * x
        tl.store(GX + rm.to(tl.int64)[:, None] * sgm + idd[None, :], g * c[None, :], mask=msk)
    tl.store(GC + e.to(tl.int64) * D + idd, tl.sum(acc, 0), mask=idd < D)


class Pre(torch.autograd.Function):
    """y = (x - mu[e]) * c[e] on the expert-sorted rows (fp32 arithmetic, one rounding to x's dtype); mu a buffer"""
    @staticmethod
    def forward(ctx, x, c, mu, sch):
        x = x.contiguous()
        M, D = x.shape
        y = torch.empty_like(x)
        cc = c.float().contiguous()
        mm = mu.float().contiguous() if mu is not None else cc
        if M:
            _pre_kernel[(sch.n, triton.cdiv(D, 128))](x, mm, cc, y, sch.te, sch.m0, sch.m1, D, x.stride(0), y.stride(0),
                                                       HAS_MU=mu is not None, BM=sch.bm, BD=128, num_warps=4)
        ctx.save_for_backward(x, cc, mm)
        ctx.sch, ctx.has_mu, ctx.cdt = sch, mu is not None, c.dtype
        return y

    @staticmethod
    def backward(ctx, g):
        x, cc, mm = ctx.saved_tensors
        sch = ctx.sch
        g = g.contiguous()
        M, D = x.shape
        gx = torch.empty_like(g)
        gc = torch.empty(sch.E, D, device=g.device, dtype=torch.float32)
        _pre_bwd_kernel[(sch.E, triton.cdiv(D, 128))](g, x, mm, cc, gx, gc, sch.offs, D, g.stride(0), x.stride(0),
                                                       HAS_MU=ctx.has_mu, BD=128, BMR=32, num_warps=4)
        return gx, gc.to(ctx.cdt), None, None


@triton.jit
def _post_kernel(YE, O, SA, Rs, B, Y, TE, TM0, TM1, N, sym, HAS_O: tl.constexpr, HAS_B: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    pn = tl.program_id(1)
    m0 = tl.load(TM0 + t)
    if m0 < 0:
        return
    e = tl.load(TE + t).to(tl.int64)
    m1 = tl.load(TM1 + t)
    rm = m0 + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    msk = (rm < m1)[:, None] & (rn < N)[None, :]
    off = rm.to(tl.int64)[:, None] * sym + rn[None, :]
    s = tl.load(YE + off, mask=msk, other=0).to(tl.float32)
    if HAS_O:
        s = s + tl.load(O + off, mask=msk, other=0) * tl.load(SA + e)
    y = s * tl.load(Rs + e * N + rn, mask=rn < N, other=0)[None, :]
    if HAS_B:
        y = y + tl.load(B + e * N + rn, mask=rn < N, other=0)[None, :]
    tl.store(Y + off, y, mask=msk)


@triton.jit
def _post_bwd_kernel(G, YE, O, SA, Rs, GYE, GO, GR, GB, GSA, OFF, N, sgm, nb, HAS_O: tl.constexpr, HAS_B: tl.constexpr,
                     BN: tl.constexpr, BMR: tl.constexpr):
    """per (expert, column block): g_ye = g r_e, g_o = g r_e sa_e; gr_e = sum g s, gb_e = sum g, gsa partial sum g r o"""
    e = tl.program_id(0)
    pn = tl.program_id(1)
    r0 = tl.load(OFF + e)
    r1 = tl.load(OFF + e + 1)
    rn = pn * BN + tl.arange(0, BN)
    r = tl.load(Rs + e.to(tl.int64) * N + rn, mask=rn < N, other=0)
    sa = tl.load(SA + e) if HAS_O else 0.0
    ar = tl.zeros((BMR, BN), dtype=tl.float32)
    ab = tl.zeros((BMR, BN), dtype=tl.float32)
    asa = tl.zeros((BMR, BN), dtype=tl.float32)
    for m0 in range(r0, r1, BMR):
        rm = m0 + tl.arange(0, BMR)
        msk = (rm < r1)[:, None] & (rn < N)[None, :]
        off = rm.to(tl.int64)[:, None] * sgm + rn[None, :]
        g = tl.load(G + off, mask=msk, other=0).to(tl.float32)
        s = tl.load(YE + off, mask=msk, other=0).to(tl.float32)
        gr_ = g * r[None, :]
        tl.store(GYE + off, gr_, mask=msk)
        if HAS_O:
            o = tl.load(O + off, mask=msk, other=0)
            s = s + o * sa
            asa += gr_ * o
            tl.store(GO + off, gr_ * sa, mask=msk)
        ar += g * s
        ab += g
    tl.store(GR + e.to(tl.int64) * N + rn, tl.sum(ar, 0), mask=rn < N)
    if HAS_B:
        tl.store(GB + e.to(tl.int64) * N + rn, tl.sum(ab, 0), mask=rn < N)
    if HAS_O:
        tl.store(GSA + e.to(tl.int64) * nb + pn, tl.sum(tl.sum(asa, 1), 0))


class Post(torch.autograd.Function):
    """y = (ye + o * sa[e]) * r[e] + b[e] on the expert-sorted rows (fp32, one rounding to ye's dtype); o (fp32) and
    sa, b optional"""
    @staticmethod
    def forward(ctx, ye, o, sa, r, b, sch):
        ye = ye.contiguous()
        M, N = ye.shape
        y = torch.empty_like(ye)
        rr = r.float().contiguous()
        oo = o.float().contiguous() if o is not None else ye
        ss = sa.float().contiguous() if sa is not None else rr
        bb = b.float().contiguous() if b is not None else rr
        if M:
            _post_kernel[(sch.n, triton.cdiv(N, 128))](ye, oo, ss, rr, bb, y, sch.te, sch.m0, sch.m1, N, ye.stride(0),
                                                        HAS_O=o is not None, HAS_B=b is not None, BM=sch.bm, BN=128,
                                                        num_warps=4)
        ctx.save_for_backward(ye, oo, ss, rr)
        ctx.sch, ctx.has_o, ctx.has_b = sch, o is not None, b is not None
        ctx.dts = (r.dtype, sa.dtype if sa is not None else None, b.dtype if b is not None else None)
        return y

    @staticmethod
    def backward(ctx, g):
        ye, oo, ss, rr = ctx.saved_tensors
        sch = ctx.sch
        g = g.contiguous()
        M, N = g.shape
        nb = triton.cdiv(N, 128)
        gye = torch.empty_like(ye)
        go = torch.empty(M, N, device=g.device, dtype=torch.float32) if ctx.has_o else gye
        gr = torch.empty(sch.E, N, device=g.device, dtype=torch.float32)
        gb = torch.empty(sch.E, N, device=g.device, dtype=torch.float32) if ctx.has_b else gr
        gsa = torch.empty(sch.E, nb, device=g.device, dtype=torch.float32) if ctx.has_o else gr
        _post_bwd_kernel[(sch.E, nb)](g, ye, oo, ss, rr, gye, go, gr, gb, gsa, sch.offs, N, g.stride(0), nb,
                                      HAS_O=ctx.has_o, HAS_B=ctx.has_b, BN=128, BMR=32, num_warps=4)
        rdt, sdt, bdt = ctx.dts
        return (gye, go if ctx.has_o else None, gsa.sum(1).to(sdt) if ctx.has_o else None, gr.to(rdt),
                gb.to(bdt) if ctx.has_b else None, None)


# ---- the core's TTF firing (ttf.kernel._fire / _fire_fix / _fire_apply / _spend kernels) batched over a chunk of
# experts: the same arithmetic per expert, program axis 2 = the expert in the chunk (one launch per chunk, not per expert)
@triton.jit
def _bfire_kernel(ub_ptr, vb_ptr, u_ptr, v_ptr, un_ptr, vn_ptr, p_ptr, idx_ptr, d_ptr, s_ptr, N, K, K5, R, spe, spn,
                  theta, eb, NPK, NT, SLOTS: tl.constexpr, RC: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1); j = tl.program_id(2).to(tl.int64)
    ub_ptr += j * N * R; vb_ptr += j * K * R; u_ptr += j * N * R; v_ptr += j * K * R
    un_ptr += j * N; vn_ptr += j * K; d_ptr += j * N * K; s_ptr += j * NT * SLOTS
    p_ptr += tl.load(idx_ptr + j) * spe
    rows = pn * BN + tl.arange(0, BN)
    k5 = pk * BK5 + tl.arange(0, BK5)
    rm = rows[:, None] < N
    pm = rm & (k5[None, :] < K5)
    b = tl.load(p_ptr + rows[:, None] * spn + k5[None, :], mask=pm, other=0).to(tl.int32)
    un = tl.load(un_ptr + rows, mask=rows < N, other=0.0)
    sbase = (pn * NPK + pk).to(tl.int64) * SLOTS
    nsl = 0
    pw = 1
    for i in tl.static_range(5):
        k = k5 * 5 + i
        km = k[None, :] < K
        dig = (b // pw) % 3
        a = tl.zeros((BN, BK5), dtype=tl.float32)
        for r0 in range(0, R, RC):
            rr = r0 + tl.arange(0, RC)
            u = tl.load(ub_ptr + rows[:, None] * R + rr[None, :], mask=rm & (rr[None, :] < R), other=0.0)
            v = tl.load(vb_ptr + k[:, None] * R + rr[None, :], mask=(k[:, None] < K) & (rr[None, :] < R), other=0.0)
            a += tl.dot(u, tl.trans(v))
        vn = tl.load(vn_ptr + k, mask=k < K, other=0.0)
        bd = eb * un[:, None] * vn[None, :]
        ok = pm & km
        amb = ok & (tl.abs(a) + bd >= theta) & (tl.abs(a) - bd < theta)
        flat = tl.reshape(amb.to(tl.int32), (BN * BK5,))
        na = tl.sum(flat, axis=0)
        if nsl + na > SLOTS:
            a = tl.zeros((BN, BK5), dtype=tl.float32)
            for r0 in range(0, R, RC):
                rr = r0 + tl.arange(0, RC)
                u = tl.load(u_ptr + rows[:, None] * R + rr[None, :], mask=rm & (rr[None, :] < R), other=0.0)
                v = tl.load(v_ptr + k[:, None] * R + rr[None, :], mask=(k[:, None] < K) & (rr[None, :] < R), other=0.0)
                a += tl.dot(u, tl.trans(v), input_precision="tf32x3")
            bd = tl.zeros((BN, BK5), dtype=tl.float32)
        else:
            pos = nsl + tl.cumsum(flat, axis=0) - 1
            ent = tl.reshape(rows[:, None].to(tl.int64) * K + k[None, :], (BN * BK5,))
            tl.store(s_ptr + sbase + pos, ent, mask=flat > 0)
            nsl += na
        d = tl.where(a > 0, 1, tl.where(a < 0, -1, 0))
        fire = ok & (tl.abs(a) - bd >= theta) & (tl.abs(dig - 1 + d) <= 1)
        tl.store(d_ptr + rows[:, None] * K + k[None, :], tl.where(fire, d, 0).to(tl.int8), mask=rm & km)
        pw = pw * 3


@triton.jit
def _bfire_fix_kernel(u_ptr, v_ptr, p_ptr, idx_ptr, d_ptr, s_ptr, n_slots, N, K, R, spe, spn, theta,
                      BE: tl.constexpr, RC: tl.constexpr):
    j = tl.program_id(1).to(tl.int64)
    u_ptr += j * N * R; v_ptr += j * K * R; d_ptr += j * N * K; s_ptr += j * n_slots
    p_ptr += tl.load(idx_ptr + j) * spe
    pe = tl.program_id(0) * BE + tl.arange(0, BE)
    ent = tl.load(s_ptr + pe, mask=pe < n_slots, other=-1)
    live = ent >= 0
    if tl.sum(live.to(tl.int32), axis=0) > 0:
        n = tl.where(live, ent // K, 0); k = tl.where(live, ent % K, 0)
        a = tl.zeros((BE,), dtype=tl.float32)
        for r0 in range(0, R, RC):
            rr = r0 + tl.arange(0, RC)
            m = live[:, None] & (rr[None, :] < R)
            u = tl.load(u_ptr + n[:, None] * R + rr[None, :], mask=m, other=0.0)
            v = tl.load(v_ptr + k[:, None] * R + rr[None, :], mask=m, other=0.0)
            a += tl.sum(u * v, axis=1)
        byte = tl.load(p_ptr + n * spn + k // 5, mask=live, other=0).to(tl.int32)
        pw = tl.where(k % 5 == 0, 1, tl.where(k % 5 == 1, 3, tl.where(k % 5 == 2, 9, tl.where(k % 5 == 3, 27, 81))))
        dig = (byte // pw) % 3
        d = tl.where(a > 0, 1, tl.where(a < 0, -1, 0))
        fire = live & (tl.abs(a) >= theta) & (tl.abs(dig - 1 + d) <= 1)
        tl.store(d_ptr + n * K + k, d.to(tl.int8), mask=fire)


@triton.jit
def _bfire_apply_kernel(p_ptr, idx_ptr, d_ptr, c_ptr, ek_ptr, er_ptr, n_ptr, N, K, K5, spe, spn, NPK, NT,
                        SL: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1); j = tl.program_id(2).to(tl.int64)
    p_ptr += tl.load(idx_ptr + j) * spe
    d_ptr += j * N * K; c_ptr += j * 2; ek_ptr += j * NT * SL; er_ptr += j * NT * SL; n_ptr += j * NT
    rows = pn * BN + tl.arange(0, BN)
    k5 = pk * BK5 + tl.arange(0, BK5)
    pm = (rows[:, None] < N) & (k5[None, :] < K5)
    b = tl.load(p_ptr + rows[:, None] * spn + k5[None, :], mask=pm, other=0).to(tl.int32)
    t = pn * NPK + pk
    nb = tl.zeros((BN, BK5), dtype=tl.int32)
    nmv = tl.zeros((BN, BK5), dtype=tl.int32); dnz = tl.zeros((BN, BK5), dtype=tl.int32)
    nl = 0
    pw = 1
    for i in tl.static_range(5):
        k = k5 * 5 + i
        dig = (b // pw) % 3
        dd = tl.load(d_ptr + rows[:, None] * K + k[None, :], mask=pm & (k[None, :] < K), other=0).to(tl.int32)
        mv = dd != 0
        nmv += tl.where(mv, 1, 0)
        dnz += tl.where(mv, tl.where(dig - 1 + dd != 0, 1, 0) - tl.where(dig - 1 != 0, 1, 0), 0)
        nb += (dig + dd) * pw
        flat = tl.reshape(mv.to(tl.int32), (BN * BK5,))
        pos = nl + tl.cumsum(flat, axis=0) - 1
        put = (flat > 0) & (pos < SL)
        tl.store(ek_ptr + t * SL + pos, tl.reshape(k[None, :] + rows[:, None] * 0, (BN * BK5,)), mask=put)
        er = tl.arange(0, BN)[:, None] + tl.where(dd < 0, 256, 0)
        tl.store(er_ptr + t * SL + pos, tl.reshape(er, (BN * BK5,)), mask=put)
        nl += tl.sum(flat, axis=0)
        pw = pw * 3
    tl.store(p_ptr + rows[:, None] * spn + k5[None, :], nb.to(tl.uint8), mask=pm)
    tl.store(n_ptr + t, nl)
    tl.atomic_add(c_ptr, tl.sum(tl.sum(nmv, axis=1), axis=0))
    tl.atomic_add(c_ptr + 1, tl.sum(tl.sum(dnz, axis=1), axis=0))


@triton.jit
def _bspend_kernel(d_ptr, v_ptr, o_ptr, ek_ptr, er_ptr, n_ptr, N, K, R, NPK, NT,
                   SL: tl.constexpr, BM: tl.constexpr, BK5: tl.constexpr, BR: tl.constexpr):
    pn = tl.program_id(0); j = tl.program_id(2).to(tl.int64)
    d_ptr += j * N * K; v_ptr += j * K * R; o_ptr += j * N * R
    ek_ptr += j * NT * SL; er_ptr += j * NT * SL; n_ptr += j * NT
    rows = pn * BM + tl.arange(0, BM)
    rr = tl.program_id(1) * BR + tl.arange(0, BR)
    acc = tl.zeros((BM, BR), dtype=tl.float32)
    sl = tl.arange(0, SL)
    for pk in range(0, NPK):
        t = pn * NPK + pk
        n = tl.load(n_ptr + t)
        if n > 0:
            if n <= SL:
                ek = tl.load(ek_ptr + t * SL + sl, mask=sl < n, other=0)
                er = tl.load(er_ptr + t * SL + sl, mask=sl < n, other=0)
                sg = tl.where(er >= 256, -1.0, 1.0)
                oh = tl.where((tl.arange(0, BM)[:, None] == (er % 256)[None, :]) & (sl < n)[None, :], sg[None, :], 0.0)
                v = tl.load(v_ptr + ek[:, None] * R + rr[None, :], mask=(sl < n)[:, None] & (rr[None, :] < R), other=0.0)
                db = oh.to(tl.bfloat16)
                vh = v.to(tl.bfloat16); r1 = v - vh.to(tl.float32); vm = r1.to(tl.bfloat16)
                vl = (r1 - vm.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(db, vh) + tl.dot(db, vm) + tl.dot(db, vl)
            else:
                for c0 in range(0, BK5 * 5, 32):
                    kk = pk * BK5 * 5 + c0 + tl.arange(0, 32)
                    d = tl.load(d_ptr + rows[:, None] * K + kk[None, :], mask=(rows[:, None] < N) & (kk[None, :] < K),
                                other=0)
                    v = tl.load(v_ptr + kk[:, None] * R + rr[None, :], mask=(kk[:, None] < K) & (rr[None, :] < R),
                                other=0.0)
                    db = d.to(tl.bfloat16)
                    vh = v.to(tl.bfloat16); r1 = v - vh.to(tl.float32); vm = r1.to(tl.bfloat16)
                    vl = (r1 - vm.to(tl.float32)).to(tl.bfloat16)
                    acc += tl.dot(db, vh) + tl.dot(db, vm) + tl.dot(db, vl)
    tl.store(o_ptr + rows[:, None] * R + rr[None, :], acc, mask=(rows[:, None] < N) & (rr[None, :] < R))


def bfire(U, V, packed, idx, K, theta):
    """ttf.kernel.ttf_fire for the experts idx of packed [E, N, K5] (U [C, N, R], V [C, K, R] fp32, contiguous):
    the legal moves applied in place; returns D V [C, N, R] fp32 and per expert (moves, change of nonzeros) [C, 2]"""
    from ttf.kernel import FIRE_SLOTS, FIRE_EB
    C, N, R = U.shape
    K5 = packed.shape[2]
    U = U.contiguous().float(); V = V.contiguous().float()
    Ub, Vb = U.to(torch.bfloat16), V.to(torch.bfloat16)
    un, vn = U.norm(dim=2), V.norm(dim=2)
    D = torch.empty(C, N, K, dtype=torch.int8, device=U.device)
    BN, BK5, RC = 64, 32, 64
    g = (triton.cdiv(N, BN), triton.cdiv(K5, BK5))
    NT = g[0] * g[1]
    S = torch.full((C, NT * FIRE_SLOTS), -1, dtype=torch.int64, device=U.device)
    idx = idx.to(device=U.device, dtype=torch.int64).contiguous()
    _bfire_kernel[(g[0], g[1], C)](Ub, Vb, U, V, un, vn, packed, idx, D, S, N, K, K5, R, packed.stride(0),
                                   packed.stride(1), float(theta), FIRE_EB, g[1], NT, SLOTS=FIRE_SLOTS, RC=RC, BN=BN,
                                   BK5=BK5, num_warps=4, num_stages=3)
    BE = 64
    ns = NT * FIRE_SLOTS
    _bfire_fix_kernel[(triton.cdiv(ns, BE), C)](U, V, packed, idx, D, S, ns, N, K, R, packed.stride(0), packed.stride(1),
                                               float(theta), BE=BE, RC=64)
    cnt = torch.zeros(C, 2, dtype=torch.int32, device=U.device)
    AN, AK5, SL = 16, 32, 16
    ga = (triton.cdiv(N, AN), triton.cdiv(K5, AK5))
    NTa = ga[0] * ga[1]
    ek = torch.empty(C, NTa * SL, dtype=torch.int32, device=U.device); er = torch.empty_like(ek)
    nmv = torch.empty(C, NTa, dtype=torch.int32, device=U.device)
    _bfire_apply_kernel[(ga[0], ga[1], C)](packed, idx, D, cnt, ek, er, nmv, N, K, K5, packed.stride(0),
                                           packed.stride(1), ga[1], NTa, SL=SL, BN=AN, BK5=AK5)
    dU = torch.empty(C, N, R, dtype=torch.float32, device=U.device)
    BR = 256
    _bspend_kernel[(ga[0], triton.cdiv(R, BR), C)](D, V, dU, ek, er, nmv, N, K, R, ga[1], NTa, SL=SL, BM=AN, BK5=AK5,
                                                   BR=BR)
    return dU, cnt


# ---- the MoE combine: y[t] = sum_j bf16(ys[inv[t k + j]] w[t, j]) in fp32, one rounding (the eager ys * w[order],
# gather, .float().sum(k) without the [T k, d] intermediates); backward per sorted slot s (t, j = divmod(order[s], k)):
# g_ys[s] = bf16(g[t] w[t, j]), g_w[t, j] = sum_d bf16(g[t] ys[s]) in fp32
@triton.jit
def _comb_kernel(YS, W, INV, Y, T, D, K: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    pt = tl.program_id(0)
    pd = tl.program_id(1)
    rt = pt * BT + tl.arange(0, BT)
    rd = pd * BD + tl.arange(0, BD)
    mt = rt < T
    msk = mt[:, None] & (rd < D)[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for j in tl.static_range(K):
        s = tl.load(INV + rt.to(tl.int64) * K + j, mask=mt, other=0).to(tl.int64)
        w = tl.load(W + rt.to(tl.int64) * K + j, mask=mt, other=0)
        v = tl.load(YS + s[:, None] * D + rd[None, :], mask=msk, other=0)
        acc += (v.to(tl.float32) * w.to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + rt.to(tl.int64)[:, None] * D + rd[None, :], acc.to(tl.bfloat16), mask=msk)


@triton.jit
def _comb_bwd_kernel(G, YS, W, ORD, GYS, GW, M, D, K: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    ps = tl.program_id(0)
    rs = ps * BS + tl.arange(0, BS)
    ms = rs < M
    f = tl.load(ORD + rs, mask=ms, other=0).to(tl.int64)
    t = f // K
    w = tl.load(W + f, mask=ms, other=0).to(tl.float32)
    acc = tl.zeros((BS, BD), dtype=tl.float32)
    for d0 in range(0, D, BD):
        rd = d0 + tl.arange(0, BD)
        msk = ms[:, None] & (rd < D)[None, :]
        g = tl.load(G + t[:, None] * D + rd[None, :], mask=msk, other=0).to(tl.float32)
        v = tl.load(YS + rs.to(tl.int64)[:, None] * D + rd[None, :], mask=msk, other=0).to(tl.float32)
        tl.store(GYS + rs.to(tl.int64)[:, None] * D + rd[None, :], (g * w[:, None]).to(tl.bfloat16), mask=msk)
        acc += (g * v).to(tl.bfloat16).to(tl.float32)
    tl.store(GW + f, tl.sum(acc, 1), mask=ms)


class Combine(torch.autograd.Function):
    """ys [T k, d] bf16 (the expert-sorted slots), w [T, k] bf16, order / inv the slot permutation -> y [T, d] bf16"""
    @staticmethod
    def forward(ctx, ys, w, order, inv, k):
        ys, w = ys.contiguous(), w.contiguous()
        M, D = ys.shape
        T = M // k
        y = torch.empty(T, D, device=ys.device, dtype=torch.bfloat16)
        if T:
            _comb_kernel[(triton.cdiv(T, 32), triton.cdiv(D, 128))](ys, w, inv, y, T, D, K=k, BT=32, BD=128, num_warps=4)
        ctx.save_for_backward(ys, w, order)
        ctx.k = k
        return y

    @staticmethod
    def backward(ctx, g):
        ys, w, order = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        M, D = ys.shape
        gys = torch.empty_like(ys)
        gw = torch.empty(w.numel(), device=w.device, dtype=torch.float32)
        if M:
            _comb_bwd_kernel[(triton.cdiv(M, 16),)](g, ys, w, order, gys, gw, M, D, K=ctx.k, BS=16, BD=256, num_warps=4)
        return gys, gw.view_as(w).to(w.dtype), None, None, None


@triton.jit
def _gdq_kernel(Q, S, IDX, Y, M, K, BM: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rk = pk * BK + tl.arange(0, BK)
    mm = rm < M
    msk = mm[:, None] & (rk < K)[None, :]
    src = tl.load(IDX + rm, mask=mm, other=0).to(tl.int64)
    s = tl.load(S + src, mask=mm, other=0).to(tl.float32)
    q = tl.load(Q + src[:, None] * K + rk[None, :], mask=msk, other=0).to(tl.float32)
    tl.store(Y + rm.to(tl.int64)[:, None] * K + rk[None, :], (q * s[:, None]).to(tl.bfloat16), mask=msk)


def gather_dq(q, s, idx):
    """bf16(q[idx] * s[idx, None]): q int8 [M, K], s bf16 [M] (torch's q.to(bf16) * s[:, None] after the gather, the same
    bits: q is exact in bf16, the product rounded once)"""
    q, s = q.contiguous(), s.contiguous()
    M, K = idx.numel(), q.shape[1]
    y = torch.empty(M, K, device=q.device, dtype=torch.bfloat16)
    if M:
        _gdq_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 128))](q, s, idx, y, M, K, BM=64, BK=128, num_warps=4)
    return y


# ---- DeltaNet's causal depthwise conv + silu on the [B, T, C] layout (the reference's torch fallback: F.conv1d on the
# transposed [B, C, T] copy, padding KC - 1, then silu): pre = bf16(sum_k w[c, k] x[t + k - KC + 1, c] + b[c]) in fp32,
# y = bf16(silu(pre)); backward g = bf16(gy silu'(pre)) (pre recomputed), gx[t] = sum_k w[k] g[t + KC - 1 - k],
# gw[c, k] = sum_t g[t] x[t + k - KC + 1], gb = sum_t g (fp32 sums)
@triton.jit
def _cc_pre(X, W, Bb, b, rt, rc, T, C, KC: tl.constexpr, HAS_B: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    mc = rc < C
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for k in tl.static_range(KC):
        ts = rt + (k - (KC - 1))
        m = ((ts >= 0) & (ts < T))[:, None] & mc[None, :]
        xv = tl.load(X + (b * T + ts.to(tl.int64))[:, None] * C + rc[None, :], mask=m, other=0).to(tl.float32)
        wv = tl.load(W + rc * KC + k, mask=mc, other=0).to(tl.float32)
        acc += xv * wv[None, :]
    if HAS_B:
        acc += tl.load(Bb + rc, mask=mc, other=0).to(tl.float32)[None, :]
    return acc.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _cc_fwd_kernel(X, W, Bb, Y, T, C, NTB, KC: tl.constexpr, HAS_B: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // NTB).to(tl.int64)
    rt = (pid % NTB) * BT + tl.arange(0, BT)
    rc = tl.program_id(1) * BC + tl.arange(0, BC)
    pre = _cc_pre(X, W, Bb, b, rt, rc, T, C, KC, HAS_B, BT, BC)
    y = pre * tl.sigmoid(pre)
    m = (rt < T)[:, None] & (rc < C)[None, :]
    tl.store(Y + (b * T + rt.to(tl.int64))[:, None] * C + rc[None, :], y.to(tl.bfloat16), mask=m)


@triton.jit
def _cc_bwd_g_kernel(X, W, Bb, GY, GP, GWp, GBp, T, C, NTB, KC: tl.constexpr, HAS_B: tl.constexpr, BT: tl.constexpr,
                     BC: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // NTB).to(tl.int64)
    rt = (pid % NTB) * BT + tl.arange(0, BT)
    rc = tl.program_id(1) * BC + tl.arange(0, BC)
    mc = rc < C
    m = (rt < T)[:, None] & mc[None, :]
    pre = _cc_pre(X, W, Bb, b, rt, rc, T, C, KC, HAS_B, BT, BC)
    s = tl.sigmoid(pre)
    gy = tl.load(GY + (b * T + rt.to(tl.int64))[:, None] * C + rc[None, :], mask=m, other=0).to(tl.float32)
    g = (gy * (s * (1 + pre * (1 - s)))).to(tl.bfloat16)
    tl.store(GP + (b * T + rt.to(tl.int64))[:, None] * C + rc[None, :], g, mask=m)
    g = tl.where(m, g.to(tl.float32), 0.0)
    if HAS_B:
        tl.store(GBp + pid.to(tl.int64) * C + rc, tl.sum(g, 0), mask=mc)
    for k in tl.static_range(KC):
        ts = rt + (k - (KC - 1))
        mx = ((ts >= 0) & (ts < T))[:, None] & mc[None, :]
        xv = tl.load(X + (b * T + ts.to(tl.int64))[:, None] * C + rc[None, :], mask=mx, other=0).to(tl.float32)
        tl.store(GWp + (pid.to(tl.int64) * C + rc) * KC + k, tl.sum(g * xv, 0), mask=mc)


@triton.jit
def _cc_bwd_x_kernel(GP, W, GX, T, C, NTB, KC: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // NTB).to(tl.int64)
    rt = (pid % NTB) * BT + tl.arange(0, BT)
    rc = tl.program_id(1) * BC + tl.arange(0, BC)
    mc = rc < C
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for k in tl.static_range(KC):
        ts = rt + ((KC - 1) - k)
        m = (ts < T)[:, None] & mc[None, :]
        gv = tl.load(GP + (b * T + ts.to(tl.int64))[:, None] * C + rc[None, :], mask=m, other=0).to(tl.float32)
        wv = tl.load(W + rc * KC + k, mask=mc, other=0).to(tl.float32)
        acc += gv * wv[None, :]
    m = (rt < T)[:, None] & mc[None, :]
    tl.store(GX + (b * T + rt.to(tl.int64))[:, None] * C + rc[None, :], acc.to(tl.bfloat16), mask=m)


class CConv(torch.autograd.Function):
    """x [B, T, C] bf16 (contiguous), w [C, KC], b [C] or None -> silu(causal depthwise conv) [B, T, C] bf16"""
    BT, BC = 64, 128

    @staticmethod
    def forward(ctx, x, w, b):
        x = x.contiguous()
        B, T, C = x.shape
        KC = w.shape[1]
        wc = w.contiguous(); bc = b.contiguous() if b is not None else wc
        y = torch.empty_like(x)
        ntb = triton.cdiv(T, CConv.BT)
        _cc_fwd_kernel[(B * ntb, triton.cdiv(C, CConv.BC))](x, wc, bc, y, T, C, ntb, KC=KC, HAS_B=b is not None,
                                                             BT=CConv.BT, BC=CConv.BC, num_warps=4)
        ctx.save_for_backward(x, wc, bc)
        ctx.has_b, ctx.wdt, ctx.bdt = b is not None, w.dtype, (b.dtype if b is not None else None)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, wc, bc = ctx.saved_tensors
        gy = gy.contiguous()
        B, T, C = x.shape
        KC = wc.shape[1]
        ntb = triton.cdiv(T, CConv.BT)
        nb = B * ntb
        gp = torch.empty_like(x)
        gwp = torch.empty(nb, C, KC, device=x.device, dtype=torch.float32)
        gbp = torch.empty(nb, C, device=x.device, dtype=torch.float32) if ctx.has_b else gwp
        grid = (nb, triton.cdiv(C, CConv.BC))
        _cc_bwd_g_kernel[grid](x, wc, bc, gy, gp, gwp, gbp, T, C, ntb, KC=KC, HAS_B=ctx.has_b, BT=CConv.BT,
                               BC=CConv.BC, num_warps=4)
        gx = torch.empty_like(x)
        _cc_bwd_x_kernel[grid](gp, wc, gx, T, C, ntb, KC=KC, BT=CConv.BT, BC=CConv.BC, num_warps=4)
        return gx, gwp.sum(0).to(ctx.wdt), (gbp.sum(0).to(ctx.bdt) if ctx.has_b else None)


def causal_conv1d_silu(x_bct, weight, bias=None, activation=None, **kw):
    """drop-in for the reference's causal_conv1d_fn (x [B, C, T], the transposed view of [B, T, C]; silu): the result
    as a [B, C, T] view of a contiguous [B, T, C] tensor (the caller transposes back)"""
    if activation not in ("silu", "swish") or not x_bct.is_cuda or x_bct.dtype != torch.bfloat16 or \
            not x_bct.transpose(1, 2).is_contiguous():
        return None
    return CConv.apply(x_bct.transpose(1, 2), weight, bias).transpose(1, 2)
