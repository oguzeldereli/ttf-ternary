"""Decode-time kernels for TTF layers: y = beta * s * (q @ T^T) for one (or a few) tokens, reading the packed trits once
(5 per byte, base 3, digit j has weight 3^j) and decoding them in registers. q: int8 activation codes, s: their scales.
Memory-bound by design: a 27B model reads ~4.6 GB of trits per token instead of ~50 GB of bf16."""
import torch
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({"BN": bn, "BK5": bk, "SPLIT": sp}, num_warps=w)
                          for bn in (16, 32, 64, 128) for bk in (64, 128, 256) for w in (4, 8) for sp in (1, 2, 4, 8)],
                 key=["N", "K5"], reset_to_zero=["y_ptr"])
@triton.jit
def _tern_gemv_kernel(xp_ptr, w_ptr, y_ptr, N, K5, M, sxm, swn, sym, BN: tl.constexpr, BK5: tl.constexpr,
                      SPLIT: tl.constexpr):
    """xp: the activation codes as 5 planes, xp[m, j, k5] = q[m, 5 k5 + j] (int8), so each plane load is contiguous"""
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    ps = tl.program_id(2)                            # split-K: this program's share of the K5 range
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.int32)
    chunk = tl.cdiv(tl.cdiv(K5, SPLIT), BK5) * BK5
    for k0 in range(ps * chunk, tl.minimum((ps + 1) * chunk, K5), BK5):
        rk = k0 + tl.arange(0, BK5)
        mk = rk < K5
        b = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=(rn[:, None] < N) & mk[None, :],
                    other=121).to(tl.int32)          # 121 = all five digits 1 = trits 0 (padding)
        for j in tl.static_range(5):
            qd = (b * 171) >> 9                      # b // 3, exact for 0 <= b < 256
            t = b - qd * 3 - 1
            b = qd
            x = tl.load(xp_ptr + m * sxm + j * K5 + rk, mask=mk, other=0).to(tl.int32)
            acc += tl.sum(t * x[None, :], axis=1)
    tl.atomic_add(y_ptr + m * sym + rn, acc, mask=rn < N)


def planes(q, K5):
    """int8 [M, K] -> [M, 5, K5] with plane j holding columns j, j + 5, ... (zero padded)"""
    M, K = q.shape
    if K5 * 5 != K:
        q = torch.nn.functional.pad(q, (0, K5 * 5 - K))
    return q.view(M, K5, 5).transpose(1, 2).contiguous()


def tern_gemv(q, wpacked, K):
    """q int8 [M, K] (M small) @ trits^T -> int32 [M, N]"""
    M = q.shape[0]
    N, K5 = wpacked.shape
    xp = planes(q, K5)
    y = torch.zeros(M, N, dtype=torch.int32, device=q.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), M, meta["SPLIT"])
    _tern_gemv_kernel[grid](xp, wpacked, y, N, K5, M, xp.stride(0), wpacked.stride(0), y.stride(0))
    return y


@triton.jit
def _act_quant_planes_kernel(x_ptr, c_ptr, xp_ptr, s_ptr, K, K5, sxm, BK: tl.constexpr):
    """one program per row: x * c, per-row absmax int8 codes written as 5 planes, and the row's scale"""
    m = tl.program_id(0)
    amax = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + m * sxm + rk, mask=rk < K, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + rk, mask=rk < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x * c), axis=0))
    s = tl.maximum(amax, 1e-5) / 127.0
    tl.store(s_ptr + m, s)
    for k0 in range(0, K5 * 5, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + m * sxm + rk, mask=rk < K, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + rk, mask=rk < K, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.rint(x * c / s)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)
        j = rk % 5
        k5 = rk // 5
        tl.store(xp_ptr + m * 5 * K5 + j * K5 + k5, q, mask=rk < K5 * 5)


def act_quant_planes(x, col_scale, K5):
    """x [M, K] (any float) -> (codes as planes int8 [M, 5, K5], scales fp32 [M])"""
    M, K = x.shape
    xp = torch.empty(M, 5, K5, dtype=torch.int8, device=x.device)
    s = torch.empty(M, dtype=torch.float32, device=x.device)
    _act_quant_planes_kernel[(M,)](x, col_scale, xp, s, K, K5, x.stride(0), BK=1024)
    return xp, s


def tern_gemv_planes(xp, wpacked):
    """codes as planes [M, 5, K5] @ trits^T -> int32 [M, N]"""
    M = xp.shape[0]
    N, K5 = wpacked.shape
    y = torch.zeros(M, N, dtype=torch.int32, device=xp.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), M, meta["SPLIT"])
    _tern_gemv_kernel[grid](xp, wpacked, y, N, K5, M, xp.stride(0), wpacked.stride(0), y.stride(0))
    return y


# ---- fused decode path: two kernels per TTF layer ---------------------------------------------------------------
@triton.jit
def _pre_kernel(x_ptr, c_ptr, b_ptr, xp_ptr, s_ptr, r_ptr, K, K5, R, sxm, BK: tl.constexpr, BR: tl.constexpr):
    """one program per row: xc = x * c; per-row absmax int8 codes as 5 planes; scale s; adapter input r = xc @ B (R)"""
    m = tl.program_id(0)
    rr = tl.arange(0, BR)
    amax = tl.zeros((), dtype=tl.float32)
    racc = tl.zeros((BR,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        xc = tl.load(x_ptr + m * sxm + rk, mask=mk, other=0.0).to(tl.float32) * \
            tl.load(c_ptr + rk, mask=mk, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(xc), axis=0))
        bt = tl.load(b_ptr + rk[:, None] * R + rr[None, :], mask=mk[:, None] & (rr[None, :] < R), other=0.0)
        racc += tl.sum(xc.to(tl.bfloat16).to(tl.float32)[:, None] * bt.to(tl.float32), axis=0)
    s = tl.maximum(amax, 1e-5) / 127.0
    tl.store(s_ptr + m, s)
    tl.store(r_ptr + m * BR + rr, racc, mask=rr < R)
    for k0 in range(0, K5 * 5, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        xc = tl.load(x_ptr + m * sxm + rk, mask=mk, other=0.0).to(tl.float32) * \
            tl.load(c_ptr + rk, mask=mk, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(xc / s), -128.0), 127.0).to(tl.int8)
        tl.store(xp_ptr + m * 5 * K5 + (rk % 5) * K5 + rk // 5, q, mask=rk < K5 * 5)


@triton.autotune(configs=[triton.Config({"BN": bn, "BK5": bk, "SPLIT": sp}, num_warps=w)
                          for bn in (16, 32, 64) for bk in (64, 128, 256) for w in (4, 8) for sp in (1, 2, 4, 8)],
                 key=["N", "K5"], reset_to_zero=["y_ptr"])
@triton.jit
def _gemv_epi_kernel(xp_ptr, w_ptr, s_ptr, r_ptr, a_ptr, rs_ptr, y_ptr, beta, N, K5, R, M, swn, sym,
                     BN: tl.constexpr, BK5: tl.constexpr, BR: tl.constexpr, SPLIT: tl.constexpr):
    """y32[m, n] += (beta * s[m] * sum_k q t  [+ A[n] . r[m] on split 0]) * rowscale[n]; trits read once"""
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    ps = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros((BN,), dtype=tl.int32)
    chunk = tl.cdiv(tl.cdiv(K5, SPLIT), BK5) * BK5
    for k0 in range(ps * chunk, tl.minimum((ps + 1) * chunk, K5), BK5):
        rk = k0 + tl.arange(0, BK5)
        mk = rk < K5
        b = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=mn[:, None] & mk[None, :], other=121).to(tl.int32)
        for j in tl.static_range(5):
            qd = (b * 171) >> 9
            t = b - qd * 3 - 1
            b = qd
            x = tl.load(xp_ptr + m * 5 * K5 + j * K5 + rk, mask=mk, other=0).to(tl.int32)
            acc += tl.sum(t * x[None, :], axis=1)
    s = tl.load(s_ptr + m)
    rr = tl.arange(0, BR)
    rv = tl.load(r_ptr + m * BR + rr, mask=rr < R, other=0.0)
    av = tl.load(a_ptr + rn[:, None] * R + rr[None, :], mask=mn[:, None] & (rr[None, :] < R), other=0.0).to(tl.float32)
    ad = tl.sum(av * rv[None, :], axis=1) * (ps == 0).to(tl.float32)
    rs = tl.load(rs_ptr + rn, mask=mn, other=0.0).to(tl.float32)
    tl.atomic_add(y_ptr + m * sym + rn, (acc.to(tl.float32) * (s * beta) + ad) * rs, mask=mn)


def ttf_decode(x, col_scale, mag_B, mag_A, row_scale, wpacked, beta):
    """x [M, K] -> bf16 [M, N]: the whole TTF layer in two kernels (M small)"""
    M, K = x.shape
    N, K5 = wpacked.shape
    R = mag_A.shape[1]
    BR = triton.next_power_of_2(max(R, 2))
    xp = torch.empty(M, 5, K5, dtype=torch.int8, device=x.device)
    s = torch.empty(M, dtype=torch.float32, device=x.device)
    r = torch.empty(M, BR, dtype=torch.float32, device=x.device)
    _pre_kernel[(M,)](x, col_scale, mag_B, xp, s, r, K, K5, R, x.stride(0), BK=512, BR=BR)
    y = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), M, meta["SPLIT"])
    _gemv_epi_kernel[grid](xp, wpacked, s, r, mag_A, row_scale, y, beta, N, K5, R, M, wpacked.stride(0), y.stride(0),
                           BR=BR)
    return y.to(torch.bfloat16)


# ---- 2-bit layout + dp4a: ~1 integer op per trit ----------------------------------------------------------------
def repack_2bit(wpacked, K, chunk=2048):
    """base-3 packed [N, K5] -> 2-bit [N, K4p] uint8 (4 trits per byte, code t + 1 in bits 2i), K4p a multiple of 16;
    padding codes are 1 (trit 0)"""
    from bitnet.kernel import unpack_rows
    N = wpacked.shape[0]
    K4p = ((K + 3) // 4 + 15) // 16 * 16
    out = torch.full((N, K4p), 0x55, dtype=torch.uint8, device=wpacked.device)
    for r in range(0, N, chunk):
        c = (unpack_rows(wpacked[r:r + chunk], K).to(torch.int16) + 1)          # codes {0, 1, 2}
        pad = K4p * 4 - K
        if pad:
            c = torch.nn.functional.pad(c, (0, pad), value=1)
        c = c.view(c.shape[0], K4p, 4)
        out[r:r + chunk] = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).to(torch.uint8)
    return out


@triton.autotune(configs=[triton.Config({"BN": bn, "BW": bw, "SPLIT": sp}, num_warps=w)
                          for bn in (16, 32, 64) for bw in (32, 64, 128) for w in (4, 8) for sp in (1, 2, 4)],
                 key=["N", "KW"], reset_to_zero=["y_ptr"])
@triton.jit
def _gemv2_kernel(xq_ptr, w_ptr, y_ptr, N, KW, M, sxm, swn, sym, BN: tl.constexpr, BW: tl.constexpr,
                  SPLIT: tl.constexpr):
    """int32 y[m, n] += sum_k q[m, k] t[n, k]; w: 2-bit trits viewed as int32 words (16 trits per word), q viewed as
    int32 words of 4 int8 codes. Each byte of a weight word holds 4 codes c = t + 1 (2 bits each), spread to one per byte
    (unsigned; subtracting 1 per byte would borrow across bytes); dp4a (u8 x s8) multiplies them with 4 activation codes in
    one instruction, and sum (c - 1) x = sum c x - sum x: the caller subtracts the activations' sum."""
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    ps = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros((BN,), dtype=tl.int32)
    chunk = tl.cdiv(tl.cdiv(KW, SPLIT), BW) * BW
    for k0 in range(ps * chunk, tl.minimum((ps + 1) * chunk, KW), BW):
        rk = k0 + tl.arange(0, BW)
        mk = rk < KW
        wv = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=mn[:, None] & mk[None, :], other=0x55555555)
        for byte in tl.static_range(4):
            bb = (wv >> (8 * byte)) & 0xFF
            lanes = (bb & 0x3) | ((bb & 0xC) << 6) | ((bb & 0x30) << 12) | ((bb & 0xC0) << 18)   # codes t + 1
            xw = tl.load(xq_ptr + m * sxm + rk * 4 + byte, mask=mk, other=0)
            d = tl.inline_asm_elementwise("dp4a.u32.s32 $0, $1, $2, 0;", "=r,r,r",
                                          [lanes, xw[None, :].broadcast_to(BN, BW)],
                                          dtype=tl.int32, is_pure=True, pack=1)
            acc += tl.sum(d, axis=1)
    tl.atomic_add(y_ptr + m * sym + rn, acc, mask=mn)


def tern_gemv2(q, w2):
    """q int8 [M, K] (K padded to the layout's 4 * K4p) @ trits^T from the 2-bit layout -> int32 [M, N]"""
    M, K = q.shape
    N, K4p = w2.shape
    if K < K4p * 4:
        q = torch.nn.functional.pad(q, (0, K4p * 4 - K))
    q = q.contiguous()
    xw = q.view(torch.int32).view(M, K4p)                 # 4 int8 codes per int32 word, in k order
    ww = w2.view(torch.int32)                              # [N, K4p / 4] words, 16 trits each
    KW = K4p // 4
    y = torch.zeros(M, N, dtype=torch.int32, device=q.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), M, meta["SPLIT"])
    _gemv2_kernel[grid](xw, ww, y, N, KW, M, xw.stride(0), ww.stride(0), y.stride(0))
    return y - q.to(torch.int32).sum(1, keepdim=True)


# ---- embedding lookup straight from pinned host memory (unified addressing): no host sync, capturable in a graph --
@triton.jit
def _uva_embed_kernel(ids_ptr, tab_ptr, out_ptr, D, BD: tl.constexpr):
    i = tl.program_id(0)
    tok = tl.load(ids_ptr + i).to(tl.int64)
    for d0 in range(0, D, BD):
        rd = d0 + tl.arange(0, BD)
        v = tl.load(tab_ptr + tok * D + rd, mask=rd < D)
        tl.store(out_ptr + i * D + rd, v, mask=rd < D)


class _HostPtr:
    """a pinned host tensor's address as a Triton pointer argument (valid on the device under unified addressing)"""
    def __init__(self, t):
        self.t, self.dtype = t, t.dtype

    def data_ptr(self):
        return self.t.data_ptr()


_UVA_TABLES = {}


@torch.library.custom_op("ttf::uva_embed", mutates_args=())
def uva_embed(ids: torch.Tensor, key: int, dim: int) -> torch.Tensor:
    tab = _UVA_TABLES[key]
    flat = ids.reshape(-1).contiguous()
    out = torch.empty(flat.numel(), dim, dtype=tab.dtype, device=ids.device)
    _uva_embed_kernel[(flat.numel(),)](flat, _HostPtr(tab), out, dim, BD=1024)
    return out.view(*ids.shape, dim)


@uva_embed.register_fake
def _(ids, key, dim):
    return ids.new_empty(*ids.shape, dim, dtype=torch.bfloat16)



# ---- 2-bit GEMV with the TTF epilogue fused: y = (beta s (q . t) + A[n] . r) * rowscale, fp32 accumulate --------------
@triton.autotune(configs=[triton.Config({"BN": bn, "BW": bw, "SPLIT": sp}, num_warps=4)
                          for bn in (16, 32) for bw in (64, 128) for sp in (1, 2, 4)],
                 key=["N", "KW"], reset_to_zero=["y_ptr"])
@triton.jit
def _gemv2_epi_kernel(xq_ptr, w_ptr, y_ptr, s_ptr, sumx_ptr, r_ptr, a_ptr, rs_ptr, beta, N, KW, M, R,
                      sxm, swn, sym, BN: tl.constexpr, BW: tl.constexpr, SPLIT: tl.constexpr, BR: tl.constexpr):
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    ps = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros((BN,), dtype=tl.int32)
    chunk = tl.cdiv(tl.cdiv(KW, SPLIT), BW) * BW
    for k0 in range(ps * chunk, tl.minimum((ps + 1) * chunk, KW), BW):
        rk = k0 + tl.arange(0, BW)
        mk = rk < KW
        wv = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=mn[:, None] & mk[None, :], other=0x55555555)
        for byte in tl.static_range(4):
            bb = (wv >> (8 * byte)) & 0xFF
            lanes = (bb & 0x3) | ((bb & 0xC) << 6) | ((bb & 0x30) << 12) | ((bb & 0xC0) << 18)
            xw = tl.load(xq_ptr + m * sxm + rk * 4 + byte, mask=mk, other=0)
            d = tl.inline_asm_elementwise("dp4a.u32.s32 $0, $1, $2, 0;", "=r,r,r",
                                          [lanes, xw[None, :].broadcast_to(BN, BW)],
                                          dtype=tl.int32, is_pure=True, pack=1)
            acc += tl.sum(d, axis=1)
    first = (ps == 0).to(tl.int32)
    acc = acc - first * tl.load(sumx_ptr + m)            # sum (c - 1) x = sum c x - sum x, once
    s = tl.load(s_ptr + m)
    rr = tl.arange(0, BR)
    rv = tl.load(r_ptr + m * R + rr, mask=rr < R, other=0.0)
    av = tl.load(a_ptr + rn[:, None] * R + rr[None, :], mask=mn[:, None] & (rr[None, :] < R), other=0.0).to(tl.float32)
    ad = tl.sum(av * rv[None, :], axis=1) * first.to(tl.float32)
    rs = tl.load(rs_ptr + rn, mask=mn, other=0.0).to(tl.float32)
    tl.atomic_add(y_ptr + m * sym + rn, (acc.to(tl.float32) * (s * beta) + ad) * rs, mask=mn)


@torch.library.custom_op("ttf::layer2", mutates_args=())
def ttf_layer2(x: torch.Tensor, col_scale: torch.Tensor, mag_B: torch.Tensor, mag_A: torch.Tensor,
               row_scale: torch.Tensor, w2: torch.Tensor, beta: float) -> torch.Tensor:
    """a whole TTF layer for a few tokens on the 2-bit layout: x [M, K] bf16 -> bf16 [M, N]"""
    M, K = x.shape
    N, K4p = w2.shape
    xc = x * col_scale.to(x.dtype)
    s = xc.abs().amax(-1).clamp_min(1e-5).float() / 127.0
    q = (xc.float() / s[:, None]).round().clamp_(-128, 127).to(torch.int8)
    if K < K4p * 4:
        q = torch.nn.functional.pad(q, (0, K4p * 4 - K))
    sumx = q.to(torch.int32).sum(1)
    r = (xc.to(torch.bfloat16) @ mag_B).float().contiguous()
    R = mag_A.shape[1]
    xw = q.contiguous().view(torch.int32).view(M, K4p)
    ww = w2.view(torch.int32)
    y = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), M, meta["SPLIT"])
    _gemv2_epi_kernel[grid](xw, ww, y, s, sumx, r, mag_A, row_scale, beta, N, K4p // 4, M, R,
                            xw.stride(0), ww.stride(0), y.stride(0), BR=triton.next_power_of_2(max(R, 2)))
    return y.to(torch.bfloat16)


@ttf_layer2.register_fake
def _(x, col_scale, mag_B, mag_A, row_scale, w2, beta):
    return x.new_empty(x.shape[0], w2.shape[0], dtype=torch.bfloat16)
