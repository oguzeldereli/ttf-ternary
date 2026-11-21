"""i8upd: the factored TTF update's products on the int8 tensor cores (--ttf-upd-i8).

The factored update (update._ts_layer_factored) reads the weight gradient g = gy^T x only through thin products, each a
pair of GEMMs: one contracting the layer's width (x M, gy M) and one contracting the tokens (gy^T Y, x^T Y). Both factors
already exist as int8 with one scale per token: x = xs * xq (the forward's input codes, act_bits 8) and gy = s * q1 (the
input-gradient pass's first int8 slice, layers.fused_bwd_rows). So

    x M        = xs * (xq @ Q(M))                              (Q: int8 per column of the thin M)
    gy^T Y     = q1^T @ (s * Y)        x^T Y = xq^T @ (xs * Y)

and the token-contracting product only needs the thin Y in int8, with one scale per block of 128 tokens and column
(block scaling along the reduction: Y's tokens differ in size by orders of magnitude). _rq_kernel forms Y = rs * (A8 @ B8)
* sB and writes it straight out block-quantized (no fp32 [T, J] round trip); _tn_kernel sums the blocks' int32 products
times their scales in fp32, split over the tokens. Rounding: 8-bit codes of the thin operands (relative error ~2^-8 per
entry, independent between entries), against bf16's 2^-9 on all four operands now; the int8 factors are those the
layer's own GEMMs use."""
import torch
import triton
import triton.language as tl

BT = 128          # the token block of Y's int8 scales (= _rq_kernel's row tile)


@triton.jit
def _rq_kernel(a_ptr, bt_ptr, sb_ptr, rs_ptr, yt_ptr, sy_ptr, T, D, J, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr):
    """Y = rs[t] * sB[j] * sum_d A8[t, d] B8[d, j] for a [BM, BN] tile (A8 [T, D], B8 given transposed [J, D]: both
    contiguous along d, the int8 MMA's layout); written transposed, Y8^T [J, T], with one scale per column and BM tokens"""
    pm = tl.program_id(0); pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM); cols = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, D, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rows[:, None] * D + kk[None, :], mask=(rows[:, None] < T) & (kk[None, :] < D), other=0)
        b = tl.load(bt_ptr + cols[None, :] * D + kk[:, None], mask=(kk[:, None] < D) & (cols[None, :] < J), other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
    rs = tl.load(rs_ptr + rows, mask=rows < T, other=0.0)
    sb = tl.load(sb_ptr + cols, mask=cols < J, other=0.0)
    y = acc.to(tl.float32) * rs[:, None] * sb[None, :]
    sy = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30) / 127.0
    q = tl.extra.cuda.libdevice.rint(y / sy[None, :])
    m = (rows[:, None] < T) & (cols[None, :] < J)
    tl.store(yt_ptr + cols[None, :] * T + rows[:, None], tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8), mask=m)
    tl.store(sy_ptr + pm * J + cols, sy, mask=cols < J)


@triton.jit
def _tn_kernel(at_ptr, yt_ptr, sy_ptr, o_ptr, T, D, J, NB_SPLIT, BM: tl.constexpr, BN: tl.constexpr,
               BTK: tl.constexpr, SA=None, SY=None, GROUP: tl.constexpr = 0):
    """partial[split, d, j] = sum over this split's token blocks of sY[blk, j] * sum_t A8[t, d] Y8[t, j], from the
    transposed int8 A8^T [D, T] and Y8^T [J, T] (both contiguous along the tokens, the reduction; row strides SA, SY,
    default T: a token-subset view of wider rows)"""
    if SA is None:
        SA = T
    if SY is None:
        SY = T
    if GROUP > 0:   # (grouped tile order on a 1-D grid: a wave's tiles share GROUP row blocks of A^T and their
        # column blocks of Y^T, so both stay in L2 when A^T is larger than it; 13824 x 5120: 82 -> ~200 TOPS)
        pid = tl.program_id(0); ps = tl.program_id(1)
        nm = tl.cdiv(D, BM); nn = tl.cdiv(J, BN)
        gid = pid // (GROUP * nn); fm = gid * GROUP; gs = tl.minimum(nm - fm, GROUP)
        pm = fm + (pid % (GROUP * nn)) % gs; pn = (pid % (GROUP * nn)) // gs
    else:
        pm = tl.program_id(0); pn = tl.program_id(1); ps = tl.program_id(2)
    rd = pm * BM + tl.arange(0, BM); cj = pn * BN + tl.arange(0, BN); rt = tl.arange(0, BTK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    b0 = ps * NB_SPLIT
    for bi in range(b0, b0 + NB_SPLIT):
        tt = bi * BTK + rt
        a = tl.load(at_ptr + rd[:, None] * SA + tt[None, :], mask=(tt[None, :] < T) & (rd[:, None] < D), other=0)
        y = tl.load(yt_ptr + cj[None, :] * SY + tt[:, None], mask=(tt[:, None] < T) & (cj[None, :] < J), other=0)
        sy = tl.load(sy_ptr + bi * J + cj, mask=(cj < J) & (bi * BTK < T), other=0.0)
        acc += tl.dot(a, y, out_dtype=tl.int32).to(tl.float32) * sy[None, :]
    tl.store(o_ptr + ps * D * J + rd[:, None] * J + cj[None, :], acc, mask=(rd[:, None] < D) & (cj[None, :] < J))


@triton.jit
def _tr_kernel(x_ptr, o_ptr, R, C, BR: tl.constexpr, BC: tl.constexpr):
    """int8 [R, C] -> [C, R]"""
    pr = tl.program_id(0); pc = tl.program_id(1)
    r = pr * BR + tl.arange(0, BR); c = pc * BC + tl.arange(0, BC)
    v = tl.load(x_ptr + r[:, None] * C + c[None, :], mask=(r[:, None] < R) & (c[None, :] < C), other=0)
    tl.store(o_ptr + c[None, :] * R + r[:, None], v, mask=(r[:, None] < R) & (c[None, :] < C))


def tr8(x):
    R, C = x.shape
    o = torch.empty(C, R, dtype=x.dtype, device=x.device)
    _tr_kernel[(triton.cdiv(R, 64), triton.cdiv(C, 64))](x, o, R, C, BR=64, BC=64, num_warps=4)
    return o


def q_cols_t(M):
    """thin M [D, J] (any float) -> (int8 M^T [J, D], fp32 scales [J]): one scale per column of M"""
    M = M.float()
    s = M.abs().amax(0).clamp_min(1e-30) / 127.0
    return torch.round(M.T / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous(), s.contiguous()


def rq(A8, B8t, sB, rs):
    """A8 int8 [T, D], B8^T int8 [J, D] (column scales sB), rs fp32 [T] -> Y = rs * (A8 @ B8) * sB as (int8 Y^T [J, T],
    fp32 [ceil(T / BT), J])"""
    T, D = A8.shape; J = B8t.shape[0]
    assert A8.is_contiguous() and B8t.is_contiguous() and A8.dtype == torch.int8 and B8t.dtype == torch.int8
    Yt = torch.empty(J, T, dtype=torch.int8, device=A8.device)
    sY = torch.empty(triton.cdiv(T, BT), J, dtype=torch.float32, device=A8.device)
    # tiles (t_i8tune.py, 4070 Ti Super, 20 Nov; T 32768): J <= 128 BN 64 w8 s3; wider J: BN 256 w8 s3 from D 1024
    # (D 2048 J 307: 410 -> 323 us), BN 128 w4 s2 below (D 768 J 307: 136 -> 124 us). Same codes: only the tiling changes
    if J <= 128:
        BN, nw, ns = 64, 8, 3
    elif D >= 1024:
        BN, nw, ns = 256, 8, 3
    else:
        BN, nw, ns = 128, 4, 2
    _rq_kernel[(triton.cdiv(T, BT), triton.cdiv(J, BN))](A8, B8t, sB, rs, Yt, sY, T, D, J, BM=BT, BN=BN, BK=128,
                                                       num_warps=nw, num_stages=ns)
    return Yt, sY


def _n_sm(dev):
    return torch.cuda.get_device_properties(dev).multi_processor_count


TN_GROUP = [8]     # tn without split: grouped tile order, GROUP row blocks per group (0: the 2-D grid; same sums)
TN_WIDE = [None]   # tn for wide J (the dense gradient, J = K >= 512): BM, BN, warps, stages, split x SMs


def tn(At, Yt, sY, T=None):
    """A8^T int8 [D, T], (Y8^T int8 [J, T], sY [ceil(T / BT), J]) -> A^T Y fp32 [D, J]. T: only the first T tokens
    (a token subset; the rows keep their full stride)"""
    D, Tf = At.shape; J = Yt.shape[0]
    T = Tf if T is None else T
    if J >= 512:      # (the dense gradient g = gy^T x; t_dg.py 21 Nov, 4070 Ti Super: 128 x 128 w8 up to ~1M outputs,
        # 128 x 64 w4 above: 768^2 222 us, 2048 x 768 497, 768 x 2048 542, 1024^2 303, 2816 x 1024 893)
        BM, BN, nw, ns, sm = TN_WIDE[0] if TN_WIDE[0] is not None else (
            (128, 128, 8, 3, 2) if D * J <= (1 << 20) + (1 << 18) else (128, 64, 4, 3, 2))
    else:
        BM, BN, nw, ns = 128, 64, 4, 3     # (t_i8tune.py: BN 64 w4 s3 best at every J; D 2048 J 307 359 -> 256 us with 4 x SMs of split)
        sm = 4 if D >= 2048 else 2
    tiles = triton.cdiv(D, BM) * triton.cdiv(J, BN)
    nb = triton.cdiv(T, BT)
    split = max(1, min(nb, (sm * _n_sm(At.device)) // max(tiles, 1)))
    per = triton.cdiv(nb, split); split = triton.cdiv(nb, per)
    part = torch.empty(split, D, J, dtype=torch.float32, device=At.device)
    if TN_GROUP[0] and split == 1:
        _tn_kernel[(tiles, 1)](At, Yt, sY, part, T, D, J, per, BM=BM, BN=BN, BTK=BT, SA=At.stride(0), SY=Yt.stride(0),
                               GROUP=TN_GROUP[0], num_warps=nw, num_stages=ns)
    else:
        _tn_kernel[(triton.cdiv(D, BM), triton.cdiv(J, BN), split)](At, Yt, sY, part, T, D, J, per, BM=BM, BN=BN,
                                                                    BTK=BT, SA=At.stride(0), SY=Yt.stride(0),
                                                                    num_warps=nw, num_stages=ns)
    return part[0] if split == 1 else part.sum(0)


@triton.jit
def _rqx_kernel(x_ptr, rs_ptr, yt_ptr, sy_ptr, T, D, BT_: tl.constexpr, BD: tl.constexpr):
    """Y = rs[t] * X8[t, d] for a [BT_, BD] tile, block-quantized as _rq_kernel's output (one scale per column and BT_
    tokens, rint, +-127), written transposed: Y8^T [D, T]"""
    pt = tl.program_id(0); pd = tl.program_id(1)
    rows = pt * BT_ + tl.arange(0, BT_); cols = pd * BD + tl.arange(0, BD)
    m = (rows[:, None] < T) & (cols[None, :] < D)
    x = tl.load(x_ptr + rows[:, None] * D + cols[None, :], mask=m, other=0).to(tl.float32)
    rs = tl.load(rs_ptr + rows, mask=rows < T, other=0.0)
    y = x * rs[:, None]
    sy = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30) / 127.0
    q = tl.extra.cuda.libdevice.rint(y / sy[None, :])
    tl.store(yt_ptr + cols[None, :] * T + rows[:, None], tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8), mask=m)
    tl.store(sy_ptr + pt * D + cols, sy, mask=cols < D)


def rqx(X8, rs):
    """X8 int8 [T, D], rs fp32 [T] -> rs * X8 as (int8 Y^T [D, T], fp32 [ceil(T / BT), D]): the requantized operand of
    the dense gradient (the row scales of both factors on one side)"""
    T, D = X8.shape
    Yt = torch.empty(D, T, dtype=torch.int8, device=X8.device)
    sY = torch.empty(triton.cdiv(T, BT), D, dtype=torch.float32, device=X8.device)
    _rqx_kernel[(triton.cdiv(T, BT), triton.cdiv(D, 64))](X8, rs, Yt, sY, T, D, BT_=BT, BD=64, num_warps=4)
    return Yt, sY


def dense_g(F, frac=1.0):
    """g = gy^T x fp32 [N, K] from the int8 factors: q1^T (block-quantized rs * xq), one int8 GEMM (2 T N K) instead of
    the factored update's four product chains (2 T (N + K)(2 k + 2 r)). frac < 1: the first frac of the tokens only,
    scaled by 1 / frac (--ttf-upd-tokens)"""
    T = F.q1.shape[0]
    Tu = T if frac >= 1 else max(BT, (int(T * frac) // BT) * BT)
    Yt, sY = F.yq if (F.yq is not None and Tu == T) else rqx(F.xq[:Tu], F.rs[:Tu])
    g = tn(F.q1t_(), Yt, sY, T=Tu)
    return g if Tu == T else g.mul_(T / Tu)


class I8Factors:
    """a ternary layer's gradient factors in int8: gy = s * q1 [T, N], x = xs * xq [T, K], with their transposes (the
    token-contracting products' layout); rs = s * xs (both row scales, which every product chain carries once)"""
    __slots__ = ("q1", "s", "xq", "xs", "rs", "_q1t", "_xqt", "gy_fn", "yq")

    def __init__(self, q1, s, xq, xs, gy_fn=None, q1t=None):
        self.q1, self.s, self.xq, self.xs = q1, s, xq, xs
        self.gy_fn = gy_fn          # () -> gy bf16, where the layer did not keep it (kernel.fused_bwd_rows save_f False)
        self.rs = (s * xs).contiguous()
        self._q1t, self._xqt = q1t, None    # the transposes, made on first use (the dense gradient needs q1^T only)
        self.yq = None      # rqx(xq, rs) when the dx kernel formed it (kernel.dx2_col rqx=True)

    def q1t_(self):
        if self._q1t is None:
            self._q1t = tr8(self.q1)
        return self._q1t

    def xqt_(self):
        if self._xqt is None:
            self._xqt = tr8(self.xq)
        return self._xqt

    @property
    def q1t(self):
        return self.q1t_()

    @property
    def xqt(self):
        return self.xqt_()

    def gy(self):
        """gy itself, bf16 (the paths that read it: band, dense accumulator, grad hooks)"""
        return self.gy_fn()

    def x(self):
        """x itself, bf16 (as layers passes it without --ttf-upd-i8: kernel.dequant_i8)"""
        from .kernel import dequant_i8
        return dequant_i8(self.xq, self.xs)


def mm_r(F, a, b, W):
    """u @ W / (-lr) = a * (gy^T (x (b W))): W [K, r] -> [N, r] fp32"""
    return a[:, None] * tn(F.q1t, *rq(F.xq, *q_cols_t(b[:, None] * W), F.rs))


def mm_l(F, a, b, W):
    """u^T @ W / (-lr) = b * (x^T (gy (a W))): W [N, r] -> [K, r] fp32"""
    return b[:, None] * tn(F.xqt, *rq(F.q1, *q_cols_t(a[:, None] * W), F.rs))


def g_right(F, S):
    """g S = gy^T (x S): S [K, k] -> [N, k] fp32"""
    return tn(F.q1t, *rq(F.xq, *q_cols_t(S), F.rs))


def g_left_t(F, S):
    """(S g)^T = x^T (gy S^T): S [k, N] -> [K, k] fp32"""
    return tn(F.xqt, *rq(F.q1, *q_cols_t(S.T), F.rs))
