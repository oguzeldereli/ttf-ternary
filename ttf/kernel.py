"""Triton packed-ternary GEMM.

Weight is stored packed 5 trits/byte along the K (input) dimension, one row per
output feature: wpacked[N, K5] uint8, K5 = ceil(K/5). The kernel loads a byte,
decodes 5 trits {-1,0,1} in registers, and accumulates x*trit — so the dense
bf16 weight is never materialized in global memory. y = x @ W^T.

Forward runs on the kernel (also the inference path). Backward re-derives the
weight transiently for one layer at a time (grad_x and the flip), which is
already memory-bounded.
"""
from __future__ import annotations
import os
import torch
import triton
import triton.language as tl

_P3 = [1, 3, 9, 27, 81]


_P3_DEV = {}


# --gemm cublas: the ternary GEMMs go through cuBLAS in bf16 on weights unpacked per call (transient [N, K] bf16).
# --gemm int8mm: the same, except the int8-activation forward, which runs int8 x int8 on the int8 tensor cores.
# int8 codes and trits are exact in bf16 and cuBLAS accumulates in fp32, so the result matches the Triton kernels up
# to the order of summation; on the A100 the Triton kernels were ~4x slower than cuBLAS (1.3B profile, 2 Oct).
GEMM_BACKEND = ["triton"]


def _mm_f32(a, b):
    """bf16 a @ b with an fp32 result (no bf16 rounding before the scales are applied)"""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


def _w_bf16(wpacked, K):
    return unpack_fast(wpacked, K, torch.bfloat16)


def pack_rows(w: torch.Tensor) -> torch.Tensor:
    """int8 {-1,0,1} [N,K] -> uint8 [N, ceil(K/5)] base-3, per row along K (digit j has weight 3^j). Horner in uint8
    (at most 242): one [N, K] uint8 temporary instead of int16 products and an int64 sum (0.76 GiB at 13824 x 5120)."""
    N, K = w.shape
    K5 = (K + 4) // 5
    pad = K5 * 5 - K
    x = (w.to(torch.int8) + 1).view(torch.uint8)      # {0,1,2}
    if pad:
        x = torch.cat([x, x.new_zeros(N, pad)], dim=1)
    x = x.view(N, K5, 5)
    out = x[:, :, 4].clone()
    for j in (3, 2, 1, 0):
        out.mul_(3).add_(x[:, :, j])
    return out                                        # [N, K5]


@triton.jit
def _unpack_kernel(p_ptr, o_ptr, N, K, K5, spn, son, BLOCK: tl.constexpr, OUT_BF16: tl.constexpr):
    """one row block of packed bytes -> 5 trits each, written as int8 (or bf16) {-1, 0, 1}; the padding past K is
    dropped"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = offs // K5
    k5 = offs % K5
    m = offs < N * K5
    b = tl.load(p_ptr + n * spn + k5, mask=m, other=0).to(tl.int32)
    for j in tl.static_range(5):
        t = (b % 3) - 1
        b = b // 3
        k = k5 * 5 + j
        mk = m & (k < K)
        if OUT_BF16:
            tl.store(o_ptr + n * son + k, t.to(tl.bfloat16), mask=mk)
        else:
            tl.store(o_ptr + n * son + k, t.to(tl.int8), mask=mk)


def unpack_fast(packed: torch.Tensor, K: int, dtype=torch.int8) -> torch.Tensor:
    """uint8 [N, K5] -> contiguous [N, K] of dtype (int8 or bf16) {-1, 0, 1}: unpack_rows in one kernel (same values)"""
    N, K5 = packed.shape
    out = torch.empty(N, K, dtype=dtype, device=packed.device)
    BLOCK = 1024
    _unpack_kernel[(triton.cdiv(N * K5, BLOCK),)](packed, out, N, K, K5, packed.stride(0), out.stride(0),
                                                 BLOCK=BLOCK, OUT_BF16=(dtype == torch.bfloat16))
    return out


def unpack_rows(packed: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [N,K5] -> int8 {-1,0,1} [N,K] (on the GPU: one kernel, contiguous; same values)."""
    if packed.is_cuda:
        return unpack_fast(packed, K)
    N, K5 = packed.shape
    x = packed.to(torch.int16)
    out = torch.empty(N, K5, 5, dtype=torch.int8, device=packed.device)
    for j in range(5):
        out[:, :, j] = (x % 3).to(torch.int8)
        x //= 3
    return out.view(N, K5 * 5)[:, :K].sub_(1)


_NZ_LUT = {}


def _nz_lut(device):
    """256-entry table: number of nonzero trits encoded in each packed byte."""
    t = _NZ_LUT.get(device)
    if t is None:
        v = torch.arange(256)
        c = torch.zeros(256, dtype=torch.int64)
        for j in range(5):
            c += ((v // 3 ** j) % 3 != 1).long()
        t = _NZ_LUT[device] = c.to(device)
    return t


def trit_beta(wpacked: torch.Tensor, K: int, chunk: int = 4096) -> torch.Tensor:
    """Per-tensor weight scale beta = 1/sqrt(K * rho), rho = nonzero-trit density.

    Plays the role of BitNet's absmean gamma (which comes from the latent weight,
    absent here): keeps rms(x @ W^T * beta) ~= rms(x). Padding trits (packed as
    -1 past K) are excluded. Returns a 0-dim float32 CUDA tensor (no host sync).
    """
    return beta_from_nnz(trit_nnz(wpacked, K, chunk), wpacked.shape[0])


def trit_nnz(wpacked: torch.Tensor, K: int, chunk: int = 4096) -> torch.Tensor:
    """the number of nonzero trits (padding past K excluded), a 0-dim int64 device tensor"""
    N, K5 = wpacked.shape
    lut = _nz_lut(wpacked.device)
    nnz = torch.zeros((), dtype=torch.int64, device=wpacked.device)
    for r in range(0, N, chunk):
        nnz += lut[wpacked[r:r + chunk].long()].sum()
    return nnz - N * (K5 * 5 - K)


def beta_from_nnz(nnz: torch.Tensor, N: int) -> torch.Tensor:
    return (nnz.float() / N).clamp_min(1.0).rsqrt()           # K*rho = nnz/N


_DIFF_LUT = {}
_POPC_LUT = {}


def _diff_lut(device):
    """[65536] uint8: bit j set where packed bytes old,new differ in trit slot j."""
    t = _DIFF_LUT.get(device)
    if t is None:
        a = torch.arange(256).view(256, 1)
        b = torch.arange(256).view(1, 256)
        m = torch.zeros(256, 256, dtype=torch.uint8)
        for j in range(5):
            d = ((a // 3 ** j) % 3) != ((b // 3 ** j) % 3)
            m |= (d.to(torch.uint8) << j)
        t = _DIFF_LUT[device] = m.reshape(-1).to(device)
    return t


def _popc_lut(device):
    t = _POPC_LUT.get(device)
    if t is None:
        v = torch.arange(256)
        c = torch.zeros(256, dtype=torch.int64)
        for j in range(5):
            c += (v >> j) & 1
        t = _POPC_LUT[device] = c.to(device)
    return t


def trit_diff(old: torch.Tensor, new: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """uint8 [N,K5] mask of trit slots that differ between two packed buffers."""
    lut = _diff_lut(old.device)
    out = torch.empty_like(old)
    for r in range(0, old.shape[0], chunk):
        o, n = old[r:r + chunk].long(), new[r:r + chunk].long()
        out[r:r + chunk] = lut[o * 256 + n]
    return out


def popcount(mask: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Number of set bits in a uint8 [N,K5] trit mask (0-dim int64, no host sync)."""
    lut = _popc_lut(mask.device)
    tot = torch.zeros((), dtype=torch.int64, device=mask.device)
    for r in range(0, mask.shape[0], chunk):
        tot += lut[mask[r:r + chunk].long()].sum()
    return tot


def _configs():
    cfgs = []
    for BM in (64, 128):
        for BN in (64, 128):
            for BK5 in (16, 32):
                for w in (4, 8):
                    for s in (2, 3):
                        cfgs.append(triton.Config(
                            {"BM": BM, "BN": BN, "BK5": BK5},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _tern_gemm_kernel(x_ptr, w_ptr, y_ptr, M, N, K, K5,
                      sxm, sxk, swn, swk, sym, syn,
                      BM: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k5 in range(0, K5, BK5):
        offk5 = k5 + tl.arange(0, BK5)
        w = tl.load(w_ptr + offn[:, None] * swn + offk5[None, :] * swk,
                    mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                    other=0).to(tl.int32)                       # [BN, BK5]
        for j in tl.static_range(5):
            trit = ((w // (3 ** j)) % 3 - 1).to(tl.bfloat16)      # [BN, BK5]
            offkx = offk5 * 5 + j                                # x column index
            xj = tl.load(x_ptr + offm[:, None] * sxm + offkx[None, :] * sxk,
                         mask=(offm[:, None] < M) & (offkx[None, :] < K),
                         other=0.0).to(tl.bfloat16)               # [BM, BK5]
            acc += tl.dot(xj, tl.trans(trit))
    y = acc.to(tl.bfloat16)
    tl.store(y_ptr + offm[:, None] * sym + offn[None, :] * syn, y,
             mask=(offm[:, None] < M) & (offn[None, :] < N))


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _dx_kernel(gy_ptr, w_ptr, gx_ptr, M, N, K, K5,
               sgm, sgn, swn, swk, sxm, sxk,
               BM: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    # grad_x[m, k] = sum_n gy[m, n] * W[n, k]   (contract over N)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offk5 = pid_k * BK5 + tl.arange(0, BK5)
    a0 = tl.zeros((BM, BK5), tl.float32)
    a1 = tl.zeros((BM, BK5), tl.float32)
    a2 = tl.zeros((BM, BK5), tl.float32)
    a3 = tl.zeros((BM, BK5), tl.float32)
    a4 = tl.zeros((BM, BK5), tl.float32)
    for n0 in range(0, N, BN):
        offn = n0 + tl.arange(0, BN)
        gy = tl.load(gy_ptr + offm[:, None] * sgm + offn[None, :] * sgn,
                     mask=(offm[:, None] < M) & (offn[None, :] < N),
                     other=0.0).to(tl.bfloat16)                  # [BM, BN]
        w = tl.load(w_ptr + offn[:, None] * swn + offk5[None, :] * swk,
                    mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                    other=0).to(tl.int32)                       # [BN, BK5]
        a0 += tl.dot(gy, (((w // 1) % 3 - 1)).to(tl.bfloat16))
        a1 += tl.dot(gy, (((w // 3) % 3 - 1)).to(tl.bfloat16))
        a2 += tl.dot(gy, (((w // 9) % 3 - 1)).to(tl.bfloat16))
        a3 += tl.dot(gy, (((w // 27) % 3 - 1)).to(tl.bfloat16))
        a4 += tl.dot(gy, (((w // 81) % 3 - 1)).to(tl.bfloat16))
    for j in tl.static_range(5):
        aj = a0 * (j == 0) + a1 * (j == 1) + a2 * (j == 2) + a3 * (j == 3) + a4 * (j == 4)
        offkx = offk5 * 5 + j
        tl.store(gx_ptr + offm[:, None] * sxm + offkx[None, :] * sxk,
                 aj.to(tl.bfloat16),
                 mask=(offm[:, None] < M) & (offkx[None, :] < K))


def tern_gemm_dx(gy: torch.Tensor, wpacked: torch.Tensor, K: int, w8=None) -> torch.Tensor:
    """grad_x = gy @ decode(wpacked), gy [M,N] -> [M,K]. Packed weight, no unpack (Triton); cuBLAS: unpacked to bf16
    per call, or cast from the cached int8 trits w8 (--w-cache)."""
    if GEMM_BACKEND[0] in ("cublas", "int8mm"):
        wb = w8.to(torch.bfloat16) if w8 is not None else _w_bf16(wpacked, K)
        return torch.mm(gy.contiguous().to(torch.bfloat16), wb)
    M, N = gy.shape
    Nw, K5 = wpacked.shape
    gy = gy.contiguous().to(torch.bfloat16)
    gx = torch.empty(M, K, dtype=torch.bfloat16, device=gy.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(K5, meta["BK5"]))
    _dx_kernel[grid](gy, wpacked, gx, M, N, K, K5,
                     gy.stride(0), gy.stride(1), wpacked.stride(0), wpacked.stride(1),
                     gx.stride(0), gx.stride(1))
    return gx


def tern_gemm(x: torch.Tensor, wpacked: torch.Tensor, K: int) -> torch.Tensor:
    """x [M,K] (fp16/bf16) @ decode(wpacked)^T -> y [M,N] fp16."""
    if GEMM_BACKEND[0] in ("cublas", "int8mm"):
        return torch.mm(x.contiguous().to(torch.bfloat16), _w_bf16(wpacked, K).t())
    M, Kx = x.shape
    assert Kx == K, (Kx, K)
    N, K5 = wpacked.shape
    x = x.contiguous().to(torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _tern_gemm_kernel[grid](
        x, wpacked, y, M, N, K, K5,
        x.stride(0), x.stride(1), wpacked.stride(0), wpacked.stride(1),
        y.stride(0), y.stride(1))
    return y


# =============================================================================
# int8 path (Phase 4a/4b): trits decode to int8 and feed the s8 tensor cores.
# Activations are already 8-bit (per-token absmax), so the FORWARD is lossless:
# |acc| <= K * 127 fits int32 exactly, where the bf16 path rounded partial sums.
# =============================================================================

def _i8_configs():
    cfgs = []
    for BM in (64, 128):
        for BN in (64, 128):
            for BK5 in (32, 64):          # s8 mma contracts k=32
                for w in (4, 8):
                    for st in (2, 3):
                        cfgs.append(triton.Config(
                            {"BM": BM, "BN": BN, "BK5": BK5}, num_warps=w, num_stages=st))
    return cfgs


def act_quant_i8(x: torch.Tensor, bits: int = 8):
    """Per-token symmetric absmax quantization -> (int8 codes, scale) with x ~ q*scale."""
    qp = 2 ** (bits - 1) - 1
    amax = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5).float()
    scale = amax / qp
    q = (x.float() / scale).round().clamp_(-qp - 1, qp).to(torch.int8)
    return q, scale.squeeze(-1)


@triton.autotune(configs=_i8_configs(), key=["N", "K"])
@triton.jit
def _tern_gemm_i8_kernel(x_ptr, w_ptr, y_ptr, xs_ptr, M, N, K, K5,
                         sxm, sxk, swn, swk, sym, syn, beta_ptr,
                         BM: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k5 in range(0, K5, BK5):
        offk5 = k5 + tl.arange(0, BK5)
        w = tl.load(w_ptr + offn[:, None] * swn + offk5[None, :] * swk,
                    mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                    other=0).to(tl.int32)                        # [BN, BK5]
        for j in tl.static_range(5):
            trit = ((w // (3 ** j)) % 3 - 1).to(tl.int8)          # [BN, BK5]
            offkx = offk5 * 5 + j
            xj = tl.load(x_ptr + offm[:, None] * sxm + offkx[None, :] * sxk,
                         mask=(offm[:, None] < M) & (offkx[None, :] < K),
                         other=0).to(tl.int8)                     # [BM, BK5]
            acc += tl.dot(xj, tl.trans(trit), out_dtype=tl.int32)
    xs = tl.load(xs_ptr + offm, mask=offm < M, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr).to(tl.float32)
    y = acc.to(tl.float32) * (xs[:, None] * beta)
    tl.store(y_ptr + offm[:, None] * sym + offn[None, :] * syn, y.to(tl.bfloat16),
             mask=(offm[:, None] < M) & (offn[None, :] < N))


def _dev_scalar(v, device):
    """A 0-dim fp32 device tensor: tensors pass through (no host sync), floats are filled on device."""
    if torch.is_tensor(v):
        return v.reshape(()).to(device=device, dtype=torch.float32)
    return torch.full((), float(v), dtype=torch.float32, device=device)


def tern_gemm_i8(xq: torch.Tensor, xs: torch.Tensor, wpacked: torch.Tensor, K: int,
                 beta=1.0, w8=None) -> torch.Tensor:
    """int8 x [M,K] (codes) @ decode(wpacked)^T * xs[:,None] * beta -> bf16 [M,N].
    beta: float or a 0-dim device tensor (read in the kernel, so no host sync). w8: the unpacked int8 trits [N, K] if
    the caller holds them (--w-cache), else they are unpacked here."""
    if GEMM_BACKEND[0] == "int8mm":
        # --gemm int8mm: int8 activation codes x int8 trits on the int8 tensor cores (cuBLASLt via torch._int_mm),
        # int32 accumulation: exact; the trits are unpacked per call to int8 (1 byte per weight, half of bf16)
        if w8 is None:
            w8 = unpack_fast(wpacked, K)
        M_, N_ = xq.shape[0], w8.shape[0]
        if M_ > 16 and K % 16 == 0 and N_ % 8 == 0:   # (K % 16: cuBLASLt int8 on Blackwell sm_120 refuses K 8 / 24
            y32 = torch._int_mm(xq.contiguous(), w8.t())  # with N 48, the rank-8 adapters of in_proj_a / b; Ada took them)
            if y32.is_contiguous():
                return i8_epilogue(y32, xs, beta)
            return (y32.float() * xs.float()[:, None] * beta).to(torch.bfloat16)
        return (_mm_f32(xq.to(torch.bfloat16), w8.to(torch.bfloat16).t()) * xs.float()[:, None] * beta).to(torch.bfloat16)
    if GEMM_BACKEND[0] in ("cublas", "int8mm"):
        y = _mm_f32(xq.to(torch.bfloat16), _w_bf16(wpacked, K).t()) * xs.float()[:, None] * beta
        return y.to(torch.bfloat16)
    M, Kx = xq.shape
    assert Kx == K, (Kx, K)
    N, K5 = wpacked.shape
    xq = xq.contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _tern_gemm_i8_kernel[grid](xq, wpacked, y, xs.contiguous().float(), M, N, K, K5,
                               xq.stride(0), xq.stride(1), wpacked.stride(0),
                               wpacked.stride(1), y.stride(0), y.stride(1),
                               _dev_scalar(beta, xq.device))
    return y


@triton.autotune(configs=_i8_configs(), key=["N", "K"])
@triton.jit
def _dx_i8_kernel(g_ptr, w_ptr, gx_ptr, gs_ptr, M, N, K, K5,
                  sgm, sgn, swn, swk, sxm, sxk,
                  BM: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    # grad_x[m, k] = sum_n gy[m, n] * W[n, k], gy quantized to int8 per row.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offk5 = pid_k * BK5 + tl.arange(0, BK5)
    a0 = tl.zeros((BM, BK5), tl.int32)
    a1 = tl.zeros((BM, BK5), tl.int32)
    a2 = tl.zeros((BM, BK5), tl.int32)
    a3 = tl.zeros((BM, BK5), tl.int32)
    a4 = tl.zeros((BM, BK5), tl.int32)
    for n0 in range(0, N, BN):
        offn = n0 + tl.arange(0, BN)
        gy = tl.load(g_ptr + offm[:, None] * sgm + offn[None, :] * sgn,
                     mask=(offm[:, None] < M) & (offn[None, :] < N),
                     other=0).to(tl.int8)                          # [BM, BN]
        w = tl.load(w_ptr + offn[:, None] * swn + offk5[None, :] * swk,
                    mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                    other=0).to(tl.int32)                          # [BN, BK5]
        a0 += tl.dot(gy, (((w // 1) % 3 - 1)).to(tl.int8), out_dtype=tl.int32)
        a1 += tl.dot(gy, (((w // 3) % 3 - 1)).to(tl.int8), out_dtype=tl.int32)
        a2 += tl.dot(gy, (((w // 9) % 3 - 1)).to(tl.int8), out_dtype=tl.int32)
        a3 += tl.dot(gy, (((w // 27) % 3 - 1)).to(tl.int8), out_dtype=tl.int32)
        a4 += tl.dot(gy, (((w // 81) % 3 - 1)).to(tl.int8), out_dtype=tl.int32)
    gs = tl.load(gs_ptr + offm, mask=offm < M, other=0.0).to(tl.float32)
    for j in tl.static_range(5):
        aj = a0 * (j == 0) + a1 * (j == 1) + a2 * (j == 2) + a3 * (j == 3) + a4 * (j == 4)
        offkx = offk5 * 5 + j
        tl.store(gx_ptr + offm[:, None] * sxm + offkx[None, :] * sxk,
                 (aj.to(tl.float32) * gs[:, None]).to(tl.bfloat16),
                 mask=(offm[:, None] < M) & (offkx[None, :] < K))


def tern_gemm_dx_i8(gy: torch.Tensor, wpacked: torch.Tensor, K: int) -> torch.Tensor:
    """grad_x = gy @ decode(wpacked) with gy quantized int8 per row. [M,N] -> [M,K]."""
    M, N = gy.shape
    gq, gs = act_quant_i8(gy)
    if GEMM_BACKEND[0] in ("cublas", "int8mm"):
        return (_mm_f32(gq.to(torch.bfloat16), _w_bf16(wpacked, K)) * gs.float()[:, None]).to(torch.bfloat16)
    gx = torch.empty(M, K, dtype=torch.bfloat16, device=gy.device)
    K5 = wpacked.shape[1]
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(K5, meta["BK5"]))
    _dx_i8_kernel[grid](gq, wpacked, gx, gs, M, N, K, K5,
                        gq.stride(0), gq.stride(1), wpacked.stride(0), wpacked.stride(1),
                        gx.stride(0), gx.stride(1))
    return gx


# --dx int8x2: the input gradient on the int8 tensor cores without losing precision. gy (bf16) is split per row into
# two int8 slices, q1 = round(gy / s) with s = max|gy_row| / 127 and q2 = round((gy - s q1) / (s / 254)), so that
# gy = s (q1 + q2 / 254) to about 2^-15 of the row's largest entry; two exact int8 x trit GEMMs (int32) and one epilogue
# give gx in fp32 before its bf16 rounding (relative error 4-7e-4 against 1.5e-3 for the bf16 GEMM, whose output is
# rounded to bf16). The trits are unpacked straight into the [K, N] layout the int8 GEMM wants.
@triton.jit
def _slice2_kernel(g_ptr, q1_ptr, q2_ptr, s_ptr, N, sgm, sqm, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    amax = tl.zeros((BLOCK,), tl.float32)
    for n0 in range(0, N, BLOCK):
        offs = n0 + tl.arange(0, BLOCK)
        g = tl.load(g_ptr + m * sgm + offs, mask=offs < N, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(g))
    s = tl.maximum(tl.max(amax, 0), 1e-30) / 127.0
    for n0 in range(0, N, BLOCK):
        offs = n0 + tl.arange(0, BLOCK)
        g = tl.load(g_ptr + m * sgm + offs, mask=offs < N, other=0.0).to(tl.float32)
        q1 = tl.extra.cuda.libdevice.rint(g / s)
        q2 = tl.extra.cuda.libdevice.rint((g - q1 * s) / (s / 254.0))
        tl.store(q1_ptr + m * sqm + offs, tl.minimum(tl.maximum(q1, -127.0), 127.0).to(tl.int8), mask=offs < N)
        tl.store(q2_ptr + m * sqm + offs, tl.minimum(tl.maximum(q2, -127.0), 127.0).to(tl.int8), mask=offs < N)
    tl.store(s_ptr + m, s)


@triton.jit
def _unpack_t_kernel(p_ptr, o_ptr, N, K, K5, spn, sok, BN: tl.constexpr, BK5: tl.constexpr):
    """packed rows [N, K5] -> int8 trits transposed, [K, N] (row k holds column k of the weight)"""
    offn = tl.program_id(0) * BN + tl.arange(0, BN)
    offk5 = tl.program_id(1) * BK5 + tl.arange(0, BK5)
    b = tl.load(p_ptr + offn[:, None] * spn + offk5[None, :], mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                other=0).to(tl.int32)
    for j in tl.static_range(5):
        t = ((b // (3 ** j)) % 3 - 1).to(tl.int8)
        k = offk5 * 5 + j
        tl.store(o_ptr + k[None, :] * sok + offn[:, None], t, mask=(offn[:, None] < N) & (k[None, :] < K))


@triton.jit
def _dx2_epi_kernel(y1_ptr, y2_ptr, s_ptr, o_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr):
    offm = tl.program_id(0) * BM + tl.arange(0, BM)
    offk = tl.program_id(1) * BK + tl.arange(0, BK)
    mk = (offm[:, None] < M) & (offk[None, :] < K)
    idx = offm[:, None] * K + offk[None, :]
    y1 = tl.load(y1_ptr + idx, mask=mk, other=0).to(tl.float32)
    y2 = tl.load(y2_ptr + idx, mask=mk, other=0).to(tl.float32)
    s = tl.load(s_ptr + offm, mask=offm < M, other=0.0)
    tl.store(o_ptr + idx, ((y1 + y2 * (1.0 / 254.0)) * s[:, None]).to(tl.bfloat16), mask=mk)


DX_I8X2_MIN_N = [int(os.environ.get("TTF_DX_I8X2_MIN_N", 4096))]   # below this summed width the overheads win: bf16


def unpack_t_i8(wpacked: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [N, K5] -> int8 {-1, 0, 1} [K, N], contiguous (the transposed trits)"""
    N, K5 = wpacked.shape
    out = torch.empty(K, N, dtype=torch.int8, device=wpacked.device)
    grid = (triton.cdiv(N, 64), triton.cdiv(K5, 32))
    _unpack_t_kernel[grid](wpacked, out, N, K, K5, wpacked.stride(0), out.stride(0), BN=64, BK5=32)
    return out


def tern_gemm_dx_i8x2(gy: torch.Tensor, wpacked: torch.Tensor, K: int) -> torch.Tensor:
    """grad_x = gy @ decode(wpacked), gy [M, N] -> bf16 [M, K], on the int8 tensor cores with gy in two int8 slices
    (--dx int8x2); shapes the int8 GEMM cannot take fall back to the bf16 path"""
    M, N = gy.shape
    if not (gy.is_cuda and M > 16 and N % 8 == 0 and K % 8 == 0 and N >= DX_I8X2_MIN_N[0]):
        return tern_gemm_dx(gy, wpacked, K)
    g = gy.contiguous()
    q1 = torch.empty(M, N, dtype=torch.int8, device=g.device); q2 = torch.empty_like(q1)
    sc = torch.empty(M, dtype=torch.float32, device=g.device)
    _slice2_kernel[(M,)](g, q1, q2, sc, N, g.stride(0), q1.stride(0), BLOCK=1024)
    wt = unpack_t_i8(wpacked, K).t()                     # [N, K], column-major: the layout torch._int_mm wants
    y1 = torch._int_mm(q1, wt); y2 = torch._int_mm(q2, wt)
    out = torch.empty(M, K, dtype=torch.bfloat16, device=g.device)
    _dx2_epi_kernel[(triton.cdiv(M, 32), triton.cdiv(K, 128))](y1, y2, sc, out, M, K, BM=32, BK=128)
    return out


@triton.jit
def _ttf_u_kernel(g_ptr, r_ptr, c_ptr, m_ptr, u_ptr, N, K, sgn, sun, lr, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    msk = offs < N * K
    i = offs // K
    j = offs % K
    g = tl.load(g_ptr + i * sgn + j, mask=msk, other=0.0)
    r = tl.load(r_ptr + i, mask=msk, other=1.0)
    c = tl.load(c_ptr + j, mask=msk, other=1.0)
    m = tl.load(m_ptr)
    s = tl.sqrt_rn(tl.div_rn(r * c, m))                    # sqrt(v_ij) from the factors, never stored
    u = tl.div_rn(-g, s + eps) * lr
    tl.store(u_ptr + i * sun + j, u, mask=msk)


def ttf_u(g: torch.Tensor, r: torch.Tensor, c: torch.Tensor, m: torch.Tensor, lr: float, eps: float = 1e-8) -> torch.Tensor:
    """u = -g / (sqrt(r_i c_j / m) + eps) * lr in one pass over g (fp32 [N, K]; r [N], c [K], m a 0-dim tensor), the
    same IEEE operations in the same order as the PyTorch expression (correctly rounded division and square root), so
    the result is bit-identical; the [N, K] sqrt(v) and the temporaries are never formed."""
    N, K = g.shape
    u = torch.empty_like(g)
    BLOCK = 1024
    _ttf_u_kernel[(triton.cdiv(N * K, BLOCK),)](g, r.contiguous(), c.contiguous(), m.reshape(1).float(), u, N, K,
                                                g.stride(0), u.stride(0), float(lr), float(eps), BLOCK=BLOCK)
    return u


@triton.jit
def _i8_epi_kernel(y_ptr, xs_ptr, b_ptr, o_ptr, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    msk = offs < M * N
    r = offs // N
    y = tl.load(y_ptr + offs, mask=msk, other=0).to(tl.float32)
    xs = tl.load(xs_ptr + r, mask=msk, other=0.0)
    b = tl.load(b_ptr)
    tl.store(o_ptr + offs, ((y * xs) * b).to(tl.bfloat16), mask=msk)


def i8_epilogue(y32: torch.Tensor, xs: torch.Tensor, beta) -> torch.Tensor:
    """(y32.float() * xs[:, None] * beta).to(bf16) in one pass over the int32 GEMM result (the same fp32 products in the
    same order, then the same round-to-nearest-even into bf16: bit-identical)"""
    M, N = y32.shape
    out = torch.empty(M, N, dtype=torch.bfloat16, device=y32.device)
    BLOCK = 2048
    _i8_epi_kernel[(triton.cdiv(M * N, BLOCK),)](y32, xs.float().contiguous(), _dev_scalar(beta, y32.device), out, M, N,
                                                BLOCK=BLOCK)
    return out


@triton.jit
def _actq_kernel(x_ptr, q_ptr, s_ptr, K, sxm, sqm, qp, rqp, lo, BK: tl.constexpr):
    row = tl.program_id(0)
    amax = tl.zeros((BK,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + row * sxm + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    a = tl.maximum(tl.max(amax, axis=0), lo)
    scale = a * rqp                       # torch's tensor / scalar is a * fp32(1 / scalar), not a true division
    tl.store(s_ptr + row, scale)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + row * sxm + offs, mask=offs < K, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(x, scale))        # round half to even, as torch.round
        q = tl.minimum(tl.maximum(q, -qp - 1.0), qp)
        tl.store(q_ptr + row * sqm + offs, q.to(tl.int8), mask=offs < K)


_ACTQ_C = {}


def _actq_consts(bits, dt):
    """qp, the fp32 reciprocal torch multiplies by, and 1e-5 in x's dtype (clamp_min(1e-5) happens in x's dtype in
    act_quant_i8); cached: two CPU tensors per call otherwise"""
    k = (bits, dt)
    if k not in _ACTQ_C:
        qp = float(2 ** (bits - 1) - 1)
        _ACTQ_C[k] = (qp, float(torch.tensor(1.0 / qp, dtype=torch.float32)), float(torch.tensor(1e-5, dtype=dt)))
    return _ACTQ_C[k]


def act_quant_i8_fast(x: torch.Tensor, bits: int = 8):
    """act_quant_i8 in one kernel per row (two passes over the row in registers / L2): the same amax, the same fp32
    scale (amax / qp), the same x / scale rounded half to even and clamped: bit-identical codes and scales"""
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M, K = x2.shape
    q = torch.empty(M, K, dtype=torch.int8, device=x.device)
    sc = torch.empty(M, dtype=torch.float32, device=x.device)
    qp, rqp, lo = _actq_consts(bits, x.dtype)
    _actq_kernel[(M,)](x2, q, sc, K, x2.stride(0), q.stride(0), qp, rqp, lo, BK=1024)
    return q.view(*x.shape[:-1], K), sc.view(x.shape[:-1])


@triton.jit
def _sq_sums_kernel(g_ptr, rp_ptr, cp_ptr, N, K, sgn, nkt, BR: tl.constexpr, BK: tl.constexpr):
    pr = tl.program_id(0); pk = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR); cols = pk * BK + tl.arange(0, BK)
    m = (rows[:, None] < N) & (cols[None, :] < K)
    g = tl.load(g_ptr + rows[:, None] * sgn + cols[None, :], mask=m, other=0.0).to(tl.float32)
    g2 = g * g
    tl.store(rp_ptr + rows * nkt + pk, tl.sum(g2, axis=1), mask=rows < N)          # per (row, column tile)
    tl.store(cp_ptr + pr * K + cols, tl.sum(g2, axis=0), mask=cols < K)            # per (row tile, column)


def sq_sums(g: torch.Tensor):
    """row sums and column sums of g^2 in one pass over g ([N, K]; deterministic two-stage reduction): what
    vector_norm(g, dim=1)^2, g2.mean(1) and their column versions read g four times (plus a g^2 copy) for"""
    N, K = g.shape
    BR, BK = 32, 256
    nrt, nkt = triton.cdiv(N, BR), triton.cdiv(K, BK)
    rp = torch.empty(N, nkt, dtype=torch.float32, device=g.device)
    cp = torch.empty(nrt, K, dtype=torch.float32, device=g.device)
    _sq_sums_kernel[(nrt, nkt)](g, rp, cp, N, K, g.stride(0), nkt, BR=BR, BK=BK)
    return rp.sum(1), cp.sum(0)


@triton.jit
def _msk_rows_kernel(g_ptr, p_ptr, sq_ptr, cr_ptr, n, k, sg, sp, HAS_P: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    """rows of one sketch side G [n, k] (fp32) and its running sum P [n, k] (bf16, updated in place): sq = sum G^2,
    cr = sum G P (P before this micro-batch), P <- bf16(P + G)"""
    pr = tl.program_id(0)
    rows = pr * BR + tl.arange(0, BR)
    sq = tl.zeros([BR], dtype=tl.float32); cr = tl.zeros([BR], dtype=tl.float32)
    for k0 in range(0, k, BK):
        cols = k0 + tl.arange(0, BK)
        m = (rows[:, None] < n) & (cols[None, :] < k)
        g = tl.load(g_ptr + rows[:, None] * sg + cols[None, :], mask=m, other=0.0)
        sq += tl.sum(g * g, axis=1)
        if HAS_P:
            p = tl.load(p_ptr + rows[:, None] * sp + cols[None, :], mask=m, other=0.0).to(tl.float32)
            cr += tl.sum(g * p, axis=1)
            g = p + g
        tl.store(p_ptr + rows[:, None] * sp + cols[None, :], g.to(tl.bfloat16), mask=m)
    tl.store(sq_ptr + rows, sq, mask=rows < n)
    tl.store(cr_ptr + rows, cr, mask=rows < n)


@triton.jit
def _msk_vec_kernel(sq_ptr, cr_ptr, rs_ptr, ds_ptr, v_ptr, o_ptr, n, c_sq, c_d, c_cr, b2, bc,
                    HAS_P: tl.constexpr, HAS_V: tl.constexpr, NORM: tl.constexpr, B: tl.constexpr):
    """one program over a side's vector (n entries): r0 = sq c_sq (then the v EMA: b2 v / bc + (1 - b2) r0), the
    scale a = sqrt(r0 / mean(r0)) (NORM) or sqrt(r0), o = 1 / max(a, max(a) 1e-12 + 1e-30); rs += sq / c_d + c_cr cr
    and ds += sq / c_d (HAS_P; else both = sq / c_d), in place"""
    offs = tl.arange(0, B)
    tot = tl.zeros([B], dtype=tl.float32)
    for i0 in range(0, n, B):                                   # pass 1: r0 (kept in o), its sum
        i = i0 + offs; m = i < n
        sq = tl.load(sq_ptr + i, mask=m, other=0.0)
        r0 = sq * c_sq
        if HAS_V:
            r0 = b2 * tl.load(v_ptr + i, mask=m, other=0.0) / bc + (1 - b2) * r0
        tl.store(o_ptr + i, r0, mask=m)
        tot += tl.where(m, r0, 0.0)
        d = sq / c_d
        if HAS_P:
            tl.store(rs_ptr + i, tl.load(rs_ptr + i, mask=m) + d + c_cr * tl.load(cr_ptr + i, mask=m), mask=m)
            tl.store(ds_ptr + i, tl.load(ds_ptr + i, mask=m) + d, mask=m)
        else:
            tl.store(rs_ptr + i, d, mask=m)
            tl.store(ds_ptr + i, d, mask=m)
    mean = tl.maximum(tl.sum(tot, axis=0) / n, 1e-30)
    amax = tl.zeros([B], dtype=tl.float32)
    for i0 in range(0, n, B):                                   # pass 2: a (kept in o), its max
        i = i0 + offs; m = i < n
        r0 = tl.load(o_ptr + i, mask=m, other=0.0)
        a = tl.sqrt(r0 / mean) if NORM else tl.sqrt(r0)
        tl.store(o_ptr + i, a, mask=m)
        amax = tl.maximum(amax, tl.where(m, a, 0.0))
    lo = tl.max(amax, axis=0) * 1e-12 + 1e-30
    for i0 in range(0, n, B):                                   # pass 3: the bounded reciprocal
        i = i0 + offs; m = i < n
        a = tl.load(o_ptr + i, mask=m, other=1.0)
        tl.store(o_ptr + i, 1.0 / tl.maximum(a, lo), mask=m)


def micro_sketch(G, P, rs, ds, v, M, n_other, b2, bc, norm):
    """--ttf-accum micro, one side of the factored update's statistics in two kernels (instead of ~25 torch ops):
    G [n, k] fp32, this micro-batch's sketch (g Sr, or (Sc g)^T); P [n, k] bf16, the running sum of the step's
    sketches (None on its first micro-batch); rs, ds [n] fp32, the running cross-term and own-square sums (None with
    P); v [n], the side's v (None before the first step). Returns (1 / a, rs, ds, P), P and rs / ds updated in place.
    The same quantities as update._ts_layer_factored's torch path; sums in another order (not bit-identical)."""
    n, k = G.shape
    G = G.contiguous()
    has_p = P is not None
    if not has_p:
        P = torch.empty(n, k, dtype=torch.bfloat16, device=G.device)
        rs = torch.empty(n, dtype=torch.float32, device=G.device); ds = torch.empty_like(rs)
    sq = torch.empty(n, dtype=torch.float32, device=G.device); cr = torch.empty_like(sq)
    BR = 16
    _msk_rows_kernel[(triton.cdiv(n, BR),)](G, P, sq, cr, n, k, G.stride(0), P.stride(0), HAS_P=has_p, BR=BR,
                                            BK=min(256, triton.next_power_of_2(k)))
    o = torch.empty_like(sq)
    _msk_vec_kernel[(1,)](sq, cr, rs, ds, v if v is not None else sq, o, n, float(M * M / n_other), float(n_other),
                          2.0 / n_other, float(b2), float(bc), HAS_P=has_p, HAS_V=v is not None, NORM=bool(norm), B=1024)
    return o, rs, ds, P


@triton.jit
def _fire_kernel(u_ptr, v_ptr, p_ptr, d_ptr, c_ptr, N, K, K5, R, sun, svk, spn, sdn, theta,
                 RC: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr, PREC: tl.constexpr):
    """one tile of packed bytes: for each of the 5 trit digits, A = U V^T on those columns (fp32 accumulation; PREC
    'tf32x3' keeps fp32 accuracy on the tensor cores), the legal moves applied to the digit, the moves written to D
    (int8), the bytes stored back; the tile's number of moves and change of the nonzero count added to c[0], c[1]"""
    pn = tl.program_id(0); pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    k5 = pk * BK5 + tl.arange(0, BK5)
    pm = (rows[:, None] < N) & (k5[None, :] < K5)
    b = tl.load(p_ptr + rows[:, None] * spn + k5[None, :], mask=pm, other=0).to(tl.int32)
    nb = tl.zeros((BN, BK5), dtype=tl.int32)
    nmv = tl.zeros((BN, BK5), dtype=tl.int32); dnz = tl.zeros((BN, BK5), dtype=tl.int32)
    pw = 1
    for i in tl.static_range(5):
        k = k5 * 5 + i
        dig = (b // pw) % 3
        a = tl.zeros((BN, BK5), dtype=tl.float32)
        for r0 in range(0, R, RC):
            rr = r0 + tl.arange(0, RC)
            u = tl.load(u_ptr + rows[:, None] * sun + rr[None, :], mask=(rows[:, None] < N) & (rr[None, :] < R), other=0.0)
            v = tl.load(v_ptr + k[:, None] * svk + rr[None, :], mask=(k[:, None] < K) & (rr[None, :] < R), other=0.0)
            a += tl.dot(u, tl.trans(v), input_precision=PREC)
        d = tl.where(a > 0, 1, tl.where(a < 0, -1, 0))
        fire = (tl.abs(a) >= theta) & (tl.abs(dig - 1 + d) <= 1) & (k[None, :] < K) & pm
        dd = tl.where(fire, d, 0)
        nmv += tl.where(fire, 1, 0)
        dnz += tl.where(fire, tl.where(dig - 1 + dd != 0, 1, 0) - tl.where(dig - 1 != 0, 1, 0), 0)
        nb += (dig + dd) * pw
        tl.store(d_ptr + rows[:, None] * sdn + k[None, :], dd.to(tl.int8), mask=(rows[:, None] < N) & (k[None, :] < K))
        pw = pw * 3
    tl.store(p_ptr + rows[:, None] * spn + k5[None, :], nb.to(tl.uint8), mask=pm)
    tl.atomic_add(c_ptr, tl.sum(tl.sum(nmv, axis=1), axis=0))
    tl.atomic_add(c_ptr + 1, tl.sum(tl.sum(dnz, axis=1), axis=0))


FIRE_TILE = None          # ttf_fire tiles (rows, packed bytes of 5 columns, rank chunk); None: by rank
FIRE_PREC = ["tf32x3"]     # ttf_fire's A: "tf32x3" (tensor cores, ~fp32 accuracy) or "ieee" (fp32 FMAs, slower)
FIRE_SPEND = ["dense"]     # ttf_fire's D V: "dense" (no host sync: three bf16 GEMMs of the int8 moves against V split into
                           # high / middle / low parts, ~fp32 accuracy) or "sparse" (the moves' indices: one host sync per
                           # layer, which stalls the CPU's queue of launches; before 23 Oct)


def ttf_fire(U: torch.Tensor, V: torch.Tensor, packed: torch.Tensor, K: int, theta: float):
    """TTF firing in one pass: A = U V^T (fp32 accumulation), every legal move (|A| >= theta, the trit stays in
    {-1, 0, 1}) applied to the packed trits in place; returns D V (D the moves, {-1, 0, 1}; from the moves alone, a
    sparse product) for the spend, and the number of moves (a device tensor). The fp32 [N, K] accumulator and the
    unpacked trits are never formed (D is int8). Differences from the unfused path: A's sums in another order and
    precision (rounding level), so an entry within rounding of theta may decide otherwise."""
    N, R = U.shape
    K5 = packed.shape[1]
    U = U.contiguous().float(); V = V.contiguous().float()
    D = torch.empty(N, K, dtype=torch.int8, device=U.device)
    cnt = torch.zeros(2, dtype=torch.int32, device=U.device)          # (the number of moves, the change of nonzeros)
    BN, BK5, RC = FIRE_TILE if FIRE_TILE else ((64, 32, 32) if R <= 512 else (64, 64, 32))   # best measured (4070 Ti S)
    _fire_kernel[(triton.cdiv(N, BN), triton.cdiv(K5, BK5))](U, V, packed, D, cnt, N, K, K5, R, U.stride(0), V.stride(0),
                                                              packed.stride(0), D.stride(0), float(theta),
                                                              RC=RC, BN=BN, BK5=BK5, PREC=FIRE_PREC[0])
    if FIRE_SPEND[0] == "dense":                                       # D V without the moves' indices (no host sync)
        Vh = V.to(torch.bfloat16); r_ = V - Vh.float(); Vm = r_.to(torch.bfloat16); Vl = (r_ - Vm.float()).to(torch.bfloat16)
        dU = torch.empty(N, R, dtype=torch.float32, device=U.device)
        for a0 in range(0, N, 8192):                                   # (row chunks: the bf16 copy of D stays small)
            Db = D[a0:a0 + 8192].to(torch.bfloat16)                    # (exact: D is -1 / 0 / 1)
            dU[a0:a0 + 8192] = (torch.mm(Db, Vh, out_dtype=torch.float32) + torch.mm(Db, Vm, out_dtype=torch.float32)
                                + torch.mm(Db, Vl, out_dtype=torch.float32))
        return dU, cnt[0], cnt[1]
    nz = D.nonzero()                                                    # the moves (one host sync)
    dU = torch.zeros(N, R, dtype=torch.float32, device=U.device)
    for c0 in range(0, nz.shape[0], 1 << 16):                          # in chunks: moves x r floats at a time
        z_ = nz[c0:c0 + (1 << 16)]
        dU.index_add_(0, z_[:, 0], D[z_[:, 0], z_[:, 1]].float()[:, None] * V[z_[:, 1]])
    # the change of the nonzero count, from the moved entries alone (a move is one level: it ends on a nonzero trit
    # (+1) or on zero (-1)), so the layer's beta follows exactly without a scan of all the trits
    byte = packed[nz[:, 0], torch.div(nz[:, 1], 5, rounding_mode="floor")].long()
    new_t = torch.div(byte, torch.pow(3, nz[:, 1] % 5), rounding_mode="floor") % 3 - 1
    dnnz = 2 * (new_t != 0).sum() - nz.shape[0]
    return dU, torch.tensor(nz.shape[0], device=U.device), dnnz


@triton.jit
def _pack12_kernel(x_ptr, s_ptr, o_ptr, N, n, np2, sxn, son, BN: tl.constexpr, BP: tl.constexpr):
    """X [N, n] fp32 / scale [n] -> fp16 -> 12 bits (sign, 5 exponent, 6 mantissa; nearest, ties to even) -> two values
    in 3 bytes: the integer operations of train._pack12, so the codes are identical"""
    pr = tl.program_id(0); pj = tl.program_id(1)
    rows = pr * BN + tl.arange(0, BN)
    j = pj * BP + tl.arange(0, BP)
    rm = rows[:, None] < N
    c0 = 2 * j; c1 = 2 * j + 1
    m0 = rm & (c0[None, :] < n); m1 = rm & (c1[None, :] < n)
    x0 = tl.load(x_ptr + rows[:, None] * sxn + c0[None, :], mask=m0, other=0.0)
    x1 = tl.load(x_ptr + rows[:, None] * sxn + c1[None, :], mask=m1, other=0.0)
    s0 = tl.load(s_ptr + c0, mask=c0 < n, other=1.0); s1 = tl.load(s_ptr + c1, mask=c1 < n, other=1.0)
    y0 = tl.where(m0, tl.math.div_rn(x0, s0[None, :]), 0.0).to(tl.float16)
    y1 = tl.where(m1, tl.math.div_rn(x1, s1[None, :]), 0.0).to(tl.float16)
    h0 = y0.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    h1 = y1.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    q0 = h0 & 0x7FFF; q0 = (q0 + 7 + ((q0 >> 4) & 1)) >> 4
    q1 = h1 & 0x7FFF; q1 = (q1 + 7 + ((q1 >> 4) & 1)) >> 4
    a = ((h0 >> 15) << 11) | q0
    b = ((h1 >> 15) << 11) | q1
    om = rm & (j[None, :] < np2)
    base = o_ptr + rows[:, None] * son + 3 * j[None, :]
    tl.store(base, (a & 0xFF).to(tl.uint8), mask=om)
    tl.store(base + 1, ((a >> 8) | ((b & 0xF) << 4)).to(tl.uint8), mask=om)
    tl.store(base + 2, (b >> 4).to(tl.uint8), mask=om)


def pack12_scaled(X: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """train._pack12((X / sc).to(fp16)) in one pass (X fp32 [N, n], sc [1, n] or [n]): identical codes"""
    N, n = X.shape
    np2 = (n + 1) // 2
    X = X.contiguous(); s = sc.reshape(-1).float().contiguous()
    out = torch.empty(N, 3 * np2, dtype=torch.uint8, device=X.device)
    BN, BP = 32, 64
    _pack12_kernel[(triton.cdiv(N, BN), triton.cdiv(np2, BP))](X, s, out, N, n, np2, X.stride(0), out.stride(0), BN=BN, BP=BP)
    return out


@triton.jit
def _unpack12_kernel(p_ptr, s_ptr, o_ptr, N, n, np2, spn, son, BN: tl.constexpr, BP: tl.constexpr):
    """inverse of _pack12_kernel times the per-column scale (train._decode of a 12-bit code): fp32 [N, n]"""
    pr = tl.program_id(0); pj = tl.program_id(1)
    rows = pr * BN + tl.arange(0, BN)
    j = pj * BP + tl.arange(0, BP)
    rm = rows[:, None] < N
    pm = rm & (j[None, :] < np2)
    base = p_ptr + rows[:, None] * spn + 3 * j[None, :]
    p0 = tl.load(base, mask=pm, other=0).to(tl.int32)
    p1 = tl.load(base + 1, mask=pm, other=0).to(tl.int32)
    p2 = tl.load(base + 2, mask=pm, other=0).to(tl.int32)
    a = p0 | ((p1 & 0xF) << 8)
    b = (p1 >> 4) | (p2 << 4)
    ha = ((a >> 11) << 15) | ((a & 0x7FF) << 4)
    hb = ((b >> 11) << 15) | ((b & 0x7FF) << 4)
    ya = ha.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    yb = hb.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    c0 = 2 * j; c1 = 2 * j + 1
    s0 = tl.load(s_ptr + c0, mask=c0 < n, other=0.0); s1 = tl.load(s_ptr + c1, mask=c1 < n, other=0.0)
    tl.store(o_ptr + rows[:, None] * son + c0[None, :], ya * s0[None, :], mask=rm & (c0[None, :] < n))
    tl.store(o_ptr + rows[:, None] * son + c1[None, :], yb * s1[None, :], mask=rm & (c1[None, :] < n))


def unpack12_scaled(P: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """train._decode(P, sc) of 12-bit codes in one pass: fp32 [N, n] (n = sc's length), identical values"""
    n = sc.shape[-1]
    N = P.shape[0]
    np2 = (n + 1) // 2
    s = sc.reshape(-1).float().contiguous(); P = P.contiguous()
    out = torch.empty(N, n, dtype=torch.float32, device=P.device)
    BN, BP = 32, 64
    _unpack12_kernel[(triton.cdiv(N, BN), triton.cdiv(np2, BP))](P, s, out, N, n, np2, P.stride(0), out.stride(0), BN=BN, BP=BP)
    return out


@triton.jit
def _addrow_fwd_kernel(y_ptr, a_ptr, r_ptr, t_ptr, o_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    """t = bf16(y + a), out = bf16(t * r) (r already bf16): the add and the row scale of a ternary layer in one pass"""
    pm = tl.program_id(0); pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM); cols = pn * BN + tl.arange(0, BN)
    m = (rows[:, None] < M) & (cols[None, :] < N)
    off = rows[:, None] * N + cols[None, :]
    y = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + off, mask=m, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    t = (y + a).to(tl.bfloat16)
    tl.store(t_ptr + off, t, mask=m)
    tl.store(o_ptr + off, (t.to(tl.float32) * r[None, :]).to(tl.bfloat16), mask=m)


@triton.jit
def _scale_bwd_kernel(g_ptr, s_ptr, x_ptr, gx_ptr, part_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    """the backward of out = x * s (s per column, bf16): gx = bf16(g * s), and per row block the column sums of
    bf16(g * x) in fp32 (the scale's gradient, summed over the blocks afterwards)"""
    pm = tl.program_id(0); pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM); cols = pn * BN + tl.arange(0, BN)
    m = (rows[:, None] < M) & (cols[None, :] < N)
    off = rows[:, None] * N + cols[None, :]
    g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    tl.store(gx_ptr + off, (g * s[None, :]).to(tl.bfloat16), mask=m)
    gs = (g * x).to(tl.bfloat16).to(tl.float32)
    tl.store(part_ptr + pm * N + cols, tl.sum(gs, axis=0), mask=cols < N)


def addrow_fwd(y, a, r_bf16):
    M, N = y.shape
    t = torch.empty_like(y); o = torch.empty_like(y)
    BM, BN = 32, 128
    _addrow_fwd_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](y, a, r_bf16, t, o, M, N, BM=BM, BN=BN)
    return t, o


def scale_bwd(g, s_bf16, x):
    """(bf16(g * s), column sums of bf16(g * x)) for out = x * s, s a per-column bf16 vector; deterministic"""
    M, N = g.shape
    gx = torch.empty_like(g)
    BM, BN = 64, 128
    nb = triton.cdiv(M, BM)
    part = torch.empty(nb, N, dtype=torch.float32, device=g.device)
    _scale_bwd_kernel[(nb, triton.cdiv(N, BN))](g, s_bf16, x, gx, part, M, N, BM=BM, BN=BN)
    return gx, part.sum(0)


@triton.jit
def _rms_fwd_kernel(x_ptr, w_ptr, y_ptr, r_ptr, K, sx, sy, eps, BK: tl.constexpr):
    """one row: y = (x * rsqrt(mean(x^2) + eps)) * (1 + w) in fp32, stored in y's dtype; rstd kept for the backward
    (Qwen3.5's RMSNorm: the gain is 1 + w and the product is taken before the cast)"""
    row = tl.program_id(0)
    k = tl.arange(0, BK); m = k < K
    x = tl.load(x_ptr + row * sx + k, mask=m, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)
    w = tl.load(w_ptr + k, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sy + k, (x * rstd * (1.0 + w)).to(y_ptr.dtype.element_ty), mask=m)
    tl.store(r_ptr + row, rstd)


@triton.jit
def _rms_bwd_kernel(x_ptr, w_ptr, g_ptr, r_ptr, gx_ptr, gw_ptr, M, K, sx, sg, sgx, RPB: tl.constexpr, BK: tl.constexpr):
    """RPB rows per program: gx = rstd (g' - xh mean(g' xh)), g' = g (1 + w), xh = x rstd; the gain's gradient
    sum(g xh) over the program's rows, written per program (summed afterwards: deterministic)"""
    pid = tl.program_id(0)
    k = tl.arange(0, BK); m = k < K
    w1 = 1.0 + tl.load(w_ptr + k, mask=m, other=0.0).to(tl.float32)
    gw = tl.zeros((BK,), dtype=tl.float32)
    for i in range(RPB):
        row = pid * RPB + i
        if row < M:
            x = tl.load(x_ptr + row * sx + k, mask=m, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + row * sg + k, mask=m, other=0.0).to(tl.float32)
            rstd = tl.load(r_ptr + row)
            xh = x * rstd
            gp = g * w1
            c = tl.sum(gp * xh, axis=0) / K
            tl.store(gx_ptr + row * sgx + k, (rstd * (gp - xh * c)).to(gx_ptr.dtype.element_ty), mask=m)
            gw += g * xh
    tl.store(gw_ptr + pid * K + k, gw, mask=m)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        sh = x.shape; x2 = x.reshape(-1, sh[-1]).contiguous(); M, K = x2.shape
        y = torch.empty_like(x2); r = torch.empty(M, dtype=torch.float32, device=x.device)
        BK = triton.next_power_of_2(K)
        _rms_fwd_kernel[(M,)](x2, w, y, r, K, x2.stride(0), y.stride(0), float(eps), BK=BK)
        ctx.save_for_backward(x2, w, r); ctx.sh = sh
        return y.view(sh)

    @staticmethod
    def backward(ctx, g):
        x2, w, r = ctx.saved_tensors; M, K = x2.shape
        g2 = g.reshape(M, K).contiguous()
        gx = torch.empty_like(x2)
        RPB = 16; nb = triton.cdiv(M, RPB); BK = triton.next_power_of_2(K)
        gw = torch.empty(nb, K, dtype=torch.float32, device=x2.device)
        _rms_bwd_kernel[(nb,)](x2, w, g2, r, gx, gw, M, K, x2.stride(0), g2.stride(0), gx.stride(0), RPB=RPB, BK=BK)
        return gx.view(ctx.sh), gw.sum(0).to(w.dtype), None


def rms_norm_qwen(x, w, eps):
    """Qwen3.5 RMSNorm, (x / rms(x)) (1 + w) in fp32 cast to x's dtype, forward and backward in one kernel each"""
    return RMSNormFn.apply(x, w, eps)


@triton.jit
def _swiglu_fwd_kernel(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """y = bf16(bf16(silu(g)) * u): torch's silu (fp32, rounded to bf16) then the product (rounded), in one pass"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + offs, (s * u).to(tl.bfloat16), mask=m)


@triton.jit
def _swiglu_bwd_kernel(g_ptr, u_ptr, gy_ptr, gg_ptr, gu_ptr, n, BLOCK: tl.constexpr):
    """the backward of y = silu(g) * u with the unfused path's roundings: gu = bf16(gy s), gs = bf16(gy u),
    gg = bf16(gs silu'(g)) (s = bf16(silu(g)))"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    s = (g * sig).to(tl.bfloat16).to(tl.float32)
    tl.store(gu_ptr + offs, (gy * s).to(tl.bfloat16), mask=m)
    gs = (gy * u).to(tl.bfloat16).to(tl.float32)
    tl.store(gg_ptr + offs, (gs * (sig * (1.0 + g * (1.0 - sig)))).to(tl.bfloat16), mask=m)


class SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        g_, u_ = g.contiguous(), u.contiguous()
        y = torch.empty_like(g_); n = g_.numel(); B = 2048
        _swiglu_fwd_kernel[(triton.cdiv(n, B),)](g_, u_, y, n, BLOCK=B)
        ctx.save_for_backward(g_, u_)
        return y

    @staticmethod
    def backward(ctx, gy):
        g_, u_ = ctx.saved_tensors
        gy = gy.contiguous(); gg = torch.empty_like(g_); gu = torch.empty_like(u_); n = g_.numel(); B = 2048
        _swiglu_bwd_kernel[(triton.cdiv(n, B),)](g_, u_, gy, gg, gu, n, BLOCK=B)
        return gg, gu


def swiglu(g, u):
    """silu(g) * u (bf16) in one kernel each way"""
    return SwiGLUFn.apply(g, u)


@triton.jit
def _gnorm_fwd_kernel(x_ptr, z_ptr, w_ptr, o_ptr, r_ptr, M, D, eps, RB: tl.constexpr, BD: tl.constexpr, WBF: tl.constexpr):
    """Qwen3.5's gated RMSNorm on RB rows of length D: hb = bf16(x rsqrt(mean(x^2) + eps)); t = w * hb (fp32, or
    rounded to bf16 when the gain is bf16: torch's type promotion); out = bf16(t * silu(z))"""
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB); d = tl.arange(0, BD)
    m = (rows[:, None] < M) & (d[None, :] < D)
    off = rows[:, None] * D + d[None, :]
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    hb = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + d, mask=d < D, other=0.0).to(tl.float32)
    t = w[None, :] * hb
    if WBF:
        t = t.to(tl.bfloat16).to(tl.float32)
    z = tl.load(z_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + off, (t * (z / (1.0 + tl.exp(-z)))).to(tl.bfloat16), mask=m)
    tl.store(r_ptr + rows, rstd, mask=rows < M)


@triton.jit
def _gnorm_bwd_kernel(x_ptr, z_ptr, w_ptr, r_ptr, g_ptr, gx_ptr, gz_ptr, gw_ptr, M, D,
                      RB: tl.constexpr, BD: tl.constexpr, WBF: tl.constexpr):
    """the backward with the unfused path's roundings: gt = g silu(z), gz = bf16(g t silu'(z)), ghb = bf16(gt w),
    gx = bf16(rstd (ghb - xh mean(ghb xh))) with xh = x rstd; the gain's gradient sum(gt hb) per program"""
    pid = tl.program_id(0)
    rows = pid * RB + tl.arange(0, RB); d = tl.arange(0, BD)
    m = (rows[:, None] < M) & (d[None, :] < D)
    off = rows[:, None] * D + d[None, :]
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + off, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
    rstd = tl.load(r_ptr + rows, mask=rows < M, other=0.0)
    xh = x * rstd[:, None]
    hb = xh.to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + d, mask=d < D, other=0.0).to(tl.float32)
    t = w[None, :] * hb
    if WBF:
        t = t.to(tl.bfloat16).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-z))
    gt = g * (z * sig)
    tl.store(gz_ptr + off, (g * t * (sig * (1.0 + z * (1.0 - sig)))).to(tl.bfloat16), mask=m)
    ghb = (gt * w[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.sum(ghb * xh, axis=1) / D
    tl.store(gx_ptr + off, (rstd[:, None] * (ghb - xh * c[:, None])).to(tl.bfloat16), mask=m)
    tl.store(gw_ptr + pid * D + d, tl.sum(gt * hb, axis=0), mask=d < D)


class GatedNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, w, eps):
        sh = x.shape; D = sh[-1]
        x2 = x.reshape(-1, D).contiguous(); z2 = z.reshape(-1, D).contiguous(); M = x2.shape[0]
        o = torch.empty(M, D, dtype=torch.bfloat16, device=x.device); r = torch.empty(M, dtype=torch.float32, device=x.device)
        RB = 32; BD = triton.next_power_of_2(D); wbf = w.dtype == torch.bfloat16
        _gnorm_fwd_kernel[(triton.cdiv(M, RB),)](x2, z2, w, o, r, M, D, float(eps), RB=RB, BD=BD, WBF=wbf)
        ctx.save_for_backward(x2, z2, w, r); ctx.sh = sh; ctx.zsh = z.shape; ctx.wbf = wbf
        return o.view(sh)

    @staticmethod
    def backward(ctx, g):
        x2, z2, w, r = ctx.saved_tensors; M, D = x2.shape
        g2 = g.reshape(M, D).contiguous()
        gx = torch.empty_like(x2); gz = torch.empty_like(z2)
        RB = 32; nb = triton.cdiv(M, RB); BD = triton.next_power_of_2(D)
        gw = torch.empty(nb, D, dtype=torch.float32, device=x2.device)
        _gnorm_bwd_kernel[(nb,)](x2, z2, w, r, g2, gx, gz, gw, M, D, RB=RB, BD=BD, WBF=ctx.wbf)
        return gx.view(ctx.sh), gz.view(ctx.zsh), gw.sum(0).to(w.dtype), None


def gated_norm_qwen(x, z, w, eps):
    """Qwen3_5RMSNormGated(x, gate z) for bf16 x / z in one kernel each way"""
    return GatedNormFn.apply(x, z, w, eps)


@triton.jit
def _rbf(x):
    """fp32 -> the nearest bf16 (ties to even), as fp32. x.to(tl.bfloat16).to(tl.float32) is not safe here: Triton
    folds that cast pair away when arithmetic follows (seen with an add after it), which skips the rounding"""
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


# --fused-layer: a ternary layer's elementwise work around its int8 GEMMs in a few passes. Each kernel repeats the
# unfused path's roundings in its order (every bf16 op rounds its fp32 result to bf16), so the layer's output and input
# gradient are bit-identical; the scale and bias gradients' sums run in fp32 without the final bf16 rounding.
@triton.jit
def _pro_kernel(x_ptr, mu_ptr, c_ptr, q_ptr, s_ptr, xc_ptr, K, sxm, qp, rqp, lo, HAS_MU: tl.constexpr,
                SAVE_XC: tl.constexpr, BK: tl.constexpr):
    """xc = bf16(x - mu), xs = bf16(xc * c), then act_quant_i8 of xs (_actq_kernel's arithmetic); xs is not stored"""
    row = tl.program_id(0)
    amax = tl.zeros((BK,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + row * sxm + offs, mask=offs < K, other=0.0).to(tl.float32)
        if HAS_MU:
            x = _rbf(x - tl.load(mu_ptr + offs, mask=offs < K, other=0.0).to(tl.float32))
        x = _rbf(x * tl.load(c_ptr + offs, mask=offs < K, other=0.0).to(tl.float32))
        amax = tl.maximum(amax, tl.abs(x))
    a = tl.maximum(tl.max(amax, axis=0), lo)
    scale = a * rqp
    tl.store(s_ptr + row, scale)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + row * sxm + offs, mask=offs < K, other=0.0).to(tl.float32)
        if HAS_MU:
            x = _rbf(x - tl.load(mu_ptr + offs, mask=offs < K, other=0.0).to(tl.float32))
            if SAVE_XC:
                tl.store(xc_ptr + row * K + offs, x.to(tl.bfloat16), mask=offs < K)
        x = _rbf(x * tl.load(c_ptr + offs, mask=offs < K, other=0.0).to(tl.float32))
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(x, scale))
        q = tl.minimum(tl.maximum(q, -qp - 1.0), qp)
        tl.store(q_ptr + row * K + offs, q.to(tl.int8), mask=offs < K)


def fused_prologue(x2, mu_b, c_b, bits, save_xc):
    """x2 [M, K] bf16 -> (int8 codes, fp32 row scales, xc = bf16(x - mu) if save_xc and mu_b is not None else None)"""
    M, K = x2.shape
    q = torch.empty(M, K, dtype=torch.int8, device=x2.device)
    sc = torch.empty(M, dtype=torch.float32, device=x2.device)
    xc = torch.empty(M, K, dtype=torch.bfloat16, device=x2.device) if (save_xc and mu_b is not None) else None
    qp, rqp, lo = _actq_consts(bits, x2.dtype)
    _pro_kernel[(M,)](x2, mu_b if mu_b is not None else c_b, c_b, q, sc, xc if xc is not None else q, K, x2.stride(0),
                      qp, rqp, lo, HAS_MU=mu_b is not None, SAVE_XC=xc is not None, BK=1024)
    return q, sc, xc


@triton.jit
def _epi_fwd_kernel(y_ptr, xs_ptr, b_ptr, r_ptr, bias_ptr, acc_ptr, t_ptr, o_ptr, M, N, HAS_BIAS: tl.constexpr,
                    HAS_ACC: tl.constexpr, SAVE_T: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """t = bf16(y32 * xs * beta) (_i8_epi_kernel's order), o = bf16(t * r), o = bf16(o + bias), o = bf16(acc + o)"""
    rows = tl.program_id(0) * BM + tl.arange(0, BM); cols = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rows[:, None] < M) & (cols[None, :] < N)
    off = rows[:, None] * N + cols[None, :]
    y = tl.load(y_ptr + off, mask=m, other=0).to(tl.float32)
    xs = tl.load(xs_ptr + rows, mask=rows < M, other=0.0)
    t = _rbf((y * xs[:, None]) * tl.load(b_ptr))
    if SAVE_T:
        tl.store(t_ptr + off, t.to(tl.bfloat16), mask=m)
    o = _rbf(t * tl.load(r_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)[None, :])
    if HAS_BIAS:
        o = _rbf(o + tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)[None, :])
    if HAS_ACC:
        o = tl.load(acc_ptr + off, mask=m, other=0.0).to(tl.float32) + o
    tl.store(o_ptr + off, o.to(tl.bfloat16), mask=m)


def fused_epilogue(y32, xs, beta, r_b, bias_b, acc, save_t):
    """the int32 GEMM result -> (t = the ternary product before the row scale if save_t, the layer's bf16 output)"""
    M, N = y32.shape
    o = torch.empty(M, N, dtype=torch.bfloat16, device=y32.device)
    t = torch.empty_like(o) if save_t else None
    BM, BN = 32, 128
    _epi_fwd_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        y32, xs, _dev_scalar(beta, y32.device), r_b, bias_b if bias_b is not None else r_b,
        acc if acc is not None else o, t if t is not None else o, o, M, N, HAS_BIAS=bias_b is not None,
        HAS_ACC=acc is not None, SAVE_T=save_t, BM=BM, BN=BN)
    return t, o


@triton.jit
def _bwd_rows_kernel(g_ptr, t_ptr, r_ptr, bb_ptr, f_ptr, q1_ptr, q2_ptr, s_ptr, pb_ptr, pr_ptr, M, N,
                     SLICES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """from the output gradient g [M, N]: gyf = bf16(bf16(g * r) * bf16(beta)) (the GEMM's gradient, stored), the
    column sums of g (bias) and of bf16(g * t) (row scale) per row block, and with SLICES gyf's two int8 slices
    (_slice2_kernel's arithmetic)"""
    pm = tl.program_id(0)
    rows = pm * BM + tl.arange(0, BM)
    bb = tl.load(bb_ptr).to(tl.float32)
    amax = tl.zeros((BM,), tl.float32)
    for n0 in range(0, N, BN):
        cols = n0 + tl.arange(0, BN)
        m = (rows[:, None] < M) & (cols[None, :] < N)
        off = rows[:, None] * N + cols[None, :]
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        t = tl.load(t_ptr + off, mask=m, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        f = _rbf(_rbf(g * r[None, :]) * bb)
        tl.store(f_ptr + off, f.to(tl.bfloat16), mask=m)
        amax = tl.maximum(amax, tl.max(tl.abs(f), axis=1))
        tl.store(pb_ptr + pm * N + cols, tl.sum(g, axis=0), mask=cols < N)
        tl.store(pr_ptr + pm * N + cols, tl.sum(_rbf(g * t), axis=0), mask=cols < N)
    if SLICES:
        s = tl.maximum(amax, 1e-30) / 127.0
        tl.store(s_ptr + rows, s, mask=rows < M)
        for n0 in range(0, N, BN):
            cols = n0 + tl.arange(0, BN)
            m = (rows[:, None] < M) & (cols[None, :] < N)
            off = rows[:, None] * N + cols[None, :]
            g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
            f = _rbf(_rbf(g * r[None, :]) * bb)
            q1 = tl.extra.cuda.libdevice.rint(f / s[:, None])
            q2 = tl.extra.cuda.libdevice.rint((f - q1 * s[:, None]) / (s[:, None] / 254.0))
            tl.store(q1_ptr + off, tl.minimum(tl.maximum(q1, -127.0), 127.0).to(tl.int8), mask=m)
            tl.store(q2_ptr + off, tl.minimum(tl.maximum(q2, -127.0), 127.0).to(tl.int8), mask=m)


def fused_bwd_rows(g, t, r_b, beta_b, slices):
    """-> (gyf bf16 [M, N], q1, q2, s (None without slices), bias gradient fp32 [N], row-scale gradient fp32 [N])"""
    M, N = g.shape
    f = torch.empty_like(g)
    BM, BN = 16, 256
    nb = triton.cdiv(M, BM)
    pb = torch.empty(nb, N, dtype=torch.float32, device=g.device); pr = torch.empty_like(pb)
    if slices:
        q1 = torch.empty(M, N, dtype=torch.int8, device=g.device); q2 = torch.empty_like(q1)
        s = torch.empty(M, dtype=torch.float32, device=g.device)
    else:
        q1 = q2 = s = None
    _bwd_rows_kernel[(nb,)](g, t, r_b, beta_b, f, q1 if slices else f, q2 if slices else f, s if slices else pb,
                            pb, pr, M, N, SLICES=slices, BM=BM, BN=BN)
    return f, q1, q2, s, pb.sum(0), pr.sum(0)


@triton.jit
def _dx_col_kernel(y1_ptr, y2_ptr, s_ptr, c_ptr, x_ptr, gx_ptr, pc_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr):
    """g = bf16((y1 + y2 / 254) * s) (_dx2_epi_kernel), then the column scale's backward (_scale_bwd_kernel):
    gx = bf16(g * c) and per row block the column sums of bf16(g * x)"""
    pm = tl.program_id(0)
    rows = pm * BM + tl.arange(0, BM); cols = tl.program_id(1) * BK + tl.arange(0, BK)
    m = (rows[:, None] < M) & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    y1 = tl.load(y1_ptr + off, mask=m, other=0).to(tl.float32)
    y2 = tl.load(y2_ptr + off, mask=m, other=0).to(tl.float32)
    s = tl.load(s_ptr + rows, mask=rows < M, other=0.0)
    g = _rbf((y1 + y2 * (1.0 / 254.0)) * s[:, None])
    c = tl.load(c_ptr + cols, mask=cols < K, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(gx_ptr + off, (g * c[None, :]).to(tl.bfloat16), mask=m)
    tl.store(pc_ptr + pm * K + cols, tl.sum(_rbf(g * x), axis=0), mask=cols < K)


def fused_dx_col(y1, y2, s, c_b, x):
    M, K = y1.shape
    gx = torch.empty(M, K, dtype=torch.bfloat16, device=y1.device)
    BM, BK = 64, 128
    nb = triton.cdiv(M, BM)
    pc = torch.empty(nb, K, dtype=torch.float32, device=y1.device)
    _dx_col_kernel[(nb, triton.cdiv(K, BK))](y1, y2, s, c_b, x, gx, pc, M, K, BM=BM, BK=BK)
    return gx, pc.sum(0)


@triton.jit
def _dq_kernel(q_ptr, s_ptr, o_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM); cols = tl.program_id(1) * BK + tl.arange(0, BK)
    m = (rows[:, None] < M) & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    s = _rbf(tl.load(s_ptr + rows, mask=rows < M, other=0.0))
    tl.store(o_ptr + off, (tl.load(q_ptr + off, mask=m, other=0).to(tl.float32) * s[:, None]).to(tl.bfloat16), mask=m)


def dequant_i8(q, s):
    """bf16(bf16(q) * bf16(s)[:, None]), as xq.to(bf16) * xs[:, None].to(bf16)"""
    M, K = q.shape
    o = torch.empty(M, K, dtype=torch.bfloat16, device=q.device)
    _dq_kernel[(triton.cdiv(M, 32), triton.cdiv(K, 128))](q, s, o, M, K, BM=32, BK=128)
    return o
