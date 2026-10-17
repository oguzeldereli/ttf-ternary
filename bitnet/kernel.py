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
    N, K5 = wpacked.shape
    lut = _nz_lut(wpacked.device)
    nnz = torch.zeros((), dtype=torch.int64, device=wpacked.device)
    for r in range(0, N, chunk):
        nnz += lut[wpacked[r:r + chunk].long()].sum()
    nnz -= N * (K5 * 5 - K)
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
    per call, or cast from the cached int8 trits w8."""
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


@triton.jit
def _flip_kernel(w_ptr, g_ptr, N, K, K5, sgn, sgk, gm_ptr, rate, g_ref, seed,
                 hump_g0, hump_alpha,
                 inv: tl.constexpr, hump: tl.constexpr, BLOCK: tl.constexpr):
    # in-place stochastic flip on packed bytes: decode 5 trits, flip each toward
    # -sign(grad) with prob ~ |grad|, re-encode, write. No dense weight round-trip.
    gmean = tl.load(gm_ptr).to(tl.float32)      # threshold scale, read on device (no host sync)
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b = tl.load(w_ptr + idx, mask=mask, other=0).to(tl.int32)
    newb = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t = (b // p3) % 3 - 1                                    # current trit {-1,0,1}
        k = k5 * 5 + j
        gmask = mask & (k < K)
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=gmask, other=0.0)
        gn = g / gmean
        ga = tl.abs(gn)
        ramp = tl.minimum(ga / g_ref, 1.0)
        # inv=1: flip the SMALLEST gradients first (reverse magnitude)
        prob = tl.where(inv == 1, 1.0 - ramp, ramp) * rate
        if hump == 1:
            # probability proportional to measured net gain per flip:
            # (g/g_ref) * (1 - (g/g0)^alpha): rises through the peak band, 0 at g0
            taper = 1.0 - tl.exp(hump_alpha * tl.log(tl.maximum(ga, 1e-12) / hump_g0))
            prob = tl.minimum((ga / g_ref) * tl.maximum(taper, 0.0) * rate, 1.0)
        r = tl.rand(seed, idx * 5 + j)
        fire = (r < prob) & gmask
        d = tl.where(gn > 0, -1, tl.where(gn < 0, 1, 0))         # -sign(grad)
        nt = tl.where(fire, tl.maximum(tl.minimum(t + d, 1), -1), t)
        newb += (nt + 1) * p3
    tl.store(w_ptr + idx, newb.to(tl.uint8), mask=mask)


@triton.jit
def _la_filter_kernel(p0_ptr, pc_ptr, g_ptr, g2_ptr, cnt_ptr, N, K, K5,
                      sgn, sgk, s2n, s2k, BLOCK: tl.constexpr):
    # look-ahead keep/revert on packed bytes. p0 = weights before the step, pc = current
    # candidate (p0 + surviving proposals). A changed trit (d = tc - t0 != 0) is kept iff
    # d * (g + g2) < 0 (its midpoint slope still descends), otherwise reverted to t0.
    # cnt[0] += trits changed on entry, cnt[1] += trits kept.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b0 = tl.load(p0_ptr + idx, mask=mask, other=0).to(tl.int32)
    bc = tl.load(pc_ptr + idx, mask=mask, other=0).to(tl.int32)
    newb = tl.zeros((BLOCK,), tl.int32)
    n_in = tl.zeros((BLOCK,), tl.int32)
    n_keep = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t0 = (b0 // p3) % 3 - 1
        tc = (bc // p3) % 3 - 1
        k = k5 * 5 + j
        gmask = mask & (k < K)
        d = tc - t0
        ch = gmask & (d != 0)
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=ch, other=0.0)
        g2 = tl.load(g2_ptr + n * s2n + k * s2k, mask=ch, other=0.0)
        keep = ch & (d.to(tl.float32) * (g + g2) < 0)
        nt = tl.where(keep, tc, t0)
        newb += (nt + 1) * p3
        n_in += ch.to(tl.int32)
        n_keep += keep.to(tl.int32)
    tl.store(pc_ptr + idx, newb.to(tl.uint8), mask=mask)
    tl.atomic_add(cnt_ptr, tl.sum(n_in, 0))
    tl.atomic_add(cnt_ptr + 1, tl.sum(n_keep, 0))


def lookahead_filter(p0: torch.Tensor, pc: torch.Tensor, g: torch.Tensor, g2: torch.Tensor,
                     cnt: torch.Tensor = None) -> torch.Tensor:
    """In place on `pc`: keep a changed trit iff (tc - t0) * (g + g2) < 0, else revert.
    Accumulates [changed on entry, kept] into the int32 `cnt` (2,) and returns it."""
    N, K5 = p0.shape
    K = g.shape[1]
    if cnt is None:
        cnt = torch.zeros(2, dtype=torch.int32, device=p0.device)
    BLOCK = 1024
    grid = (triton.cdiv(N * K5, BLOCK),)
    _la_filter_kernel[grid](p0, pc, g, g2, cnt, N, K, K5, g.stride(0), g.stride(1),
                            g2.stride(0), g2.stride(1), BLOCK=BLOCK)
    return cnt


def fused_flip(wpacked: torch.Tensor, grad_w: torch.Tensor, rate: float,
               g_ref: float, seed: int, gmean=None, inv: int = 0,
               hump: int = 0, hump_g0: float = 37.0, hump_alpha: float = 1.8):
    """Apply stochastic ternary flips directly on the packed buffer, in place.

    gmean: denominator of the flip threshold. Default (None) is this step's own
    mean|g| -- a RELATIVE scale, which cannot converge: if every gradient in the
    tensor shrinks, the ratio is unchanged and the same fraction keeps flipping.
    Pass a frozen per-layer scale to make the threshold absolute. A float or a 0-dim device
    tensor (the kernel reads it on device, so a tensor costs no host sync).
    """
    N, K5 = wpacked.shape
    K = grad_w.shape[1]
    if gmean is None:
        gmean = grad_w.abs().mean().clamp_min(1e-8)
    total = N * K5
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _flip_kernel[grid](wpacked, grad_w, N, K, K5,
                       grad_w.stride(0), grad_w.stride(1),
                       _dev_scalar(gmean, wpacked.device), rate, g_ref, seed, hump_g0, hump_alpha,
                       inv=inv, hump=hump, BLOCK=BLOCK)


@triton.jit
def _flip_ev_kernel(w_ptr, g_ptr, s_ptr, N, K, K5, sgn, sgk, ssn, gmean, rate,
                    g_ref, seed, smax, BLOCK: tl.constexpr):
    # integrate-and-fire with a signed saturating counter in {-smax..smax}.
    # smax=1 -> 3 levels (2-bit); smax=3 -> 7 levels (3-bit); etc. Each "active"
    # step nudges the counter toward -sign(grad); a flip fires only when the
    # counter is already saturated and pushed further. Deeper = more noise filtering.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b = tl.load(w_ptr + idx, mask=mask, other=0).to(tl.int32)
    newb = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t = (b // p3) % 3 - 1                          # current trit
        k = k5 * 5 + j
        gm = mask & (k < K)
        soff = n * ssn + k
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=gm, other=0.0)
        s = tl.load(s_ptr + soff, mask=gm, other=0).to(tl.int32)   # {-1,0,1}
        gn = g / gmean
        prob = tl.minimum(tl.abs(gn) / g_ref, 1.0) * rate
        r = tl.rand(seed, idx * 5 + j)
        active = (r < prob) & gm
        d = tl.where(gn > 0, -1, tl.where(gn < 0, 1, 0))           # -sign(grad)
        s2 = s + tl.where(active, d, 0)
        fire_up = active & (d > 0) & (s == smax)
        fire_dn = active & (d < 0) & (s == -smax)
        newt = tl.where(fire_up, tl.minimum(t + 1, 1),
                        tl.where(fire_dn, tl.maximum(t - 1, -1), t))
        news = tl.where(fire_up | fire_dn, 0,
                        tl.maximum(tl.minimum(s2, smax), -smax))
        tl.store(s_ptr + soff, news.to(tl.int8), mask=gm)
        newb += (newt + 1) * p3
    tl.store(w_ptr + idx, newb.to(tl.uint8), mask=mask)


def fused_flip_ev(wpacked, grad_w, evidence, rate, g_ref, seed, smax=1, gmean=None):
    """Flip with a signed saturating per-weight counter in {-smax..smax}, in place.
    smax=1 -> 2-bit (3 levels), smax=3 -> 3-bit (7 levels). `evidence` is int8 [N,K]."""
    N, K5 = wpacked.shape
    K = grad_w.shape[1]
    if gmean is None:
        gmean = grad_w.abs().mean().clamp_min(1e-8).item()
    total = N * K5
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _flip_ev_kernel[grid](wpacked, grad_w, evidence, N, K, K5,
                          grad_w.stride(0), grad_w.stride(1), evidence.stride(0),
                          gmean, rate, g_ref, seed, smax, BLOCK=BLOCK)


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
    the caller holds them (KernelTernaryLinear's weight cache), else they are unpacked here."""
    if GEMM_BACKEND[0] == "int8mm":
        # --gemm int8mm: int8 activation codes x int8 trits on the int8 tensor cores (cuBLASLt via torch._int_mm),
        # int32 accumulation: exact; the trits are unpacked per call to int8 (1 byte per weight, half of bf16)
        if w8 is None:
            w8 = unpack_fast(wpacked, K)
        M_, N_ = xq.shape[0], w8.shape[0]
        if M_ > 16 and K % 8 == 0 and N_ % 8 == 0:
            y32 = torch._int_mm(xq.contiguous(), w8.t())
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


def _dw_configs():
    cfgs = []
    for BN in (64, 128):
        for BK in (64, 128):
            for BM in (32, 64):
                for w in (4, 8):
                    cfgs.append(triton.Config({"BM": BM, "BN": BN, "BK": BK},
                                              num_warps=w, num_stages=2))
    return cfgs


@triton.autotune(configs=_dw_configs(), key=["N", "K"])
@triton.jit
def _dw_i8_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, san, sbm, sbk, scn, sck,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # C[n, k] = sum_m A[m, n] * B[m, k]  (both int8; contract over tokens)
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = pid_k * BK + tl.arange(0, BK)
    acc = tl.zeros((BN, BK), dtype=tl.int32)
    for m0 in range(0, M, BM):
        offm = m0 + tl.arange(0, BM)
        a = tl.load(a_ptr + offm[:, None] * sam + offn[None, :] * san,
                    mask=(offm[:, None] < M) & (offn[None, :] < N), other=0).to(tl.int8)
        b = tl.load(b_ptr + offm[:, None] * sbm + offk[None, :] * sbk,
                    mask=(offm[:, None] < M) & (offk[None, :] < K), other=0).to(tl.int8)
        acc += tl.dot(tl.trans(a), b, out_dtype=tl.int32)
    tl.store(c_ptr + offn[:, None] * scn + offk[None, :] * sck, acc,
             mask=(offn[:, None] < N) & (offk[None, :] < K))


def dw_i8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 [M,N]^T @ int8 [M,K] -> int32 [N,K]. With sign inputs this is the
    XNOR/popcount outer product (agreement count) on the s8 tensor cores."""
    M, N = a.shape
    Mb, K = b.shape
    assert M == Mb, (a.shape, b.shape)
    c = torch.empty(N, K, dtype=torch.int32, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), triton.cdiv(K, meta["BK"]))
    _dw_i8_kernel[grid](a.contiguous(), b.contiguous(), c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c


def sign_i8(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x).to(torch.int8)


def dw_int8(gy: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """dW[n,k] = sum_m gy[m,n] * x[m,k] on the s8 tensor cores, in float.

    x is already int8 codes (xq) with per-token scale xs. Folding xs into gy first
    and then quantizing gy with ONE global scale makes the remaining scale factor
    out of the sum exactly:
        dW = sum_m (gy[m,n]*xs[m]) * xq[m,k] ~= s * sum_m gq[m,n] * xq[m,k]
    """
    gyx = gy.float() * xs[:, None]
    s = gyx.abs().amax().clamp_min(1e-20) / 127.0
    gq = (gyx / s).round().clamp_(-128, 127).to(torch.int8)
    return dw_i8(gq, xq).float() * s


def dw_sign(gy: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """sign(dL/dy) (x) sign(x) as an XNOR/popcount outer product (agreement count)."""
    return dw_i8(sign_i8(gy), sign_i8(xq)).float()


def dw_cublas(gy: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Same factorization as dw_int8, but on cuBLASLt's int8 GEMM (torch._int_mm)."""
    gyx = gy.float() * xs[:, None]
    s = gyx.abs().amax().clamp_min(1e-20) / 127.0
    gq = (gyx / s).round().clamp_(-128, 127).to(torch.int8)
    return torch._int_mm(gq.transpose(0, 1).contiguous(), xq).float() * s


def expand_trit_mask(mask: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [N,K5] bit-per-trit-slot mask -> bool [N,K] (one entry per weight)."""
    N, K5 = mask.shape
    out = torch.empty(N, K5, 5, dtype=torch.bool, device=mask.device)
    for j in range(5):
        out[:, :, j] = ((mask >> j) & 1).bool()
    return out.view(N, K5 * 5)[:, :K]


@triton.jit
def _flip_lock_kernel(w_ptr, g_ptr, l_ptr, dr_ptr, N, K, K5, sgn, sgk, gmean, rate,
                      g_ref, seed, mode, BLOCK: tl.constexpr):
    """Stochastic flip with a per-weight lockout bit (1 bit/weight, bit j of byte).

    mode 0 ("once"):        a weight may flip at most once per epoch.
    mode 1 ("noreversal"):  after its first flip in an epoch it may only flip again
                            in the SAME direction, never back.
    The trainer clears the masks every T steps (the epoch).
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b = tl.load(w_ptr + idx, mask=mask, other=0).to(tl.int32)
    lk = tl.load(l_ptr + idx, mask=mask, other=0).to(tl.int32)
    dr = tl.load(dr_ptr + idx, mask=mask, other=0).to(tl.int32)
    newb = tl.zeros((BLOCK,), tl.int32)
    newlk = tl.zeros((BLOCK,), tl.int32)
    newdr = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t = (b // p3) % 3 - 1
        locked = (lk >> j) & 1
        updir = (dr >> j) & 1
        k = k5 * 5 + j
        gmask = mask & (k < K)
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=gmask, other=0.0)
        gn = g / gmean
        prob = tl.minimum(tl.abs(gn) / g_ref, 1.0) * rate
        r = tl.rand(seed, idx * 5 + j)
        fire = (r < prob) & gmask
        d = tl.where(gn > 0, -1, tl.where(gn < 0, 1, 0))         # -sign(grad)
        same_dir = (d > 0) == (updir == 1)
        allow = tl.where(mode == 0, locked == 0, (locked == 0) | same_dir)
        fire = fire & allow
        nt = tl.where(fire, tl.maximum(tl.minimum(t + d, 1), -1), t)
        moved = fire & (nt != t)
        newb += (nt + 1) * p3
        newlk += (tl.where(moved, 1, locked)) << j
        newdr += (tl.where(moved, tl.where(d > 0, 1, 0), updir)) << j
    tl.store(w_ptr + idx, newb.to(tl.uint8), mask=mask)
    tl.store(l_ptr + idx, newlk.to(tl.uint8), mask=mask)
    tl.store(dr_ptr + idx, newdr.to(tl.uint8), mask=mask)


def fused_flip_lock(wpacked, grad_w, lock, dirmask, rate: float, g_ref: float,
                    seed: int, gmean: float = None, mode: int = 0):
    """fused_flip + per-weight lockout. mode 0 = one flip per epoch, 1 = no reversal."""
    N, K5 = wpacked.shape
    K = grad_w.shape[1]
    if gmean is None:
        gmean = grad_w.abs().mean().clamp_min(1e-8).item()
    total = N * K5
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _flip_lock_kernel[grid](wpacked, grad_w, lock, dirmask, N, K, K5,
                            grad_w.stride(0), grad_w.stride(1), gmean, rate, g_ref,
                            seed, mode, BLOCK=BLOCK)


@triton.jit
def _flip_row_kernel(w_ptr, g_ptr, gm_ptr, N, K, K5, sgn, sgk, rate, g_ref, seed,
                     BLOCK: tl.constexpr):
    """fused_flip with PER-ROW normalisation: each output unit's threshold uses its
    own mean|g| instead of the whole tensor's. Changes WHICH weights clear the bar,
    not how many overall."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b = tl.load(w_ptr + idx, mask=mask, other=0).to(tl.int32)
    gmean = tl.load(gm_ptr + n, mask=mask, other=1.0)
    newb = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t = (b // p3) % 3 - 1
        k = k5 * 5 + j
        gmask = mask & (k < K)
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=gmask, other=0.0)
        gn = g / gmean
        prob = tl.minimum(tl.abs(gn) / g_ref, 1.0) * rate
        r = tl.rand(seed, idx * 5 + j)
        fire = (r < prob) & gmask
        d = tl.where(gn > 0, -1, tl.where(gn < 0, 1, 0))
        nt = tl.where(fire, tl.maximum(tl.minimum(t + d, 1), -1), t)
        newb += (nt + 1) * p3
    tl.store(w_ptr + idx, newb.to(tl.uint8), mask=mask)


def fused_flip_row(wpacked, grad_w, rate: float, g_ref: float, seed: int,
                   row_mean: torch.Tensor = None):
    """Per-row version of fused_flip. row_mean: [N] mean|g| per output unit."""
    N, K5 = wpacked.shape
    K = grad_w.shape[1]
    if row_mean is None:
        row_mean = grad_w.abs().mean(dim=1).clamp_min(1e-8)
    row_mean = row_mean.contiguous().float()
    BLOCK = 1024
    grid = (triton.cdiv(N * K5, BLOCK),)
    _flip_row_kernel[grid](wpacked, grad_w, row_mean, N, K, K5,
                           grad_w.stride(0), grad_w.stride(1), rate, g_ref, seed,
                           BLOCK=BLOCK)


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


def act_quant_i8_fast(x: torch.Tensor, bits: int = 8):
    """act_quant_i8 in one kernel per row (two passes over the row in registers / L2): the same amax, the same fp32
    scale (amax / qp), the same x / scale rounded half to even and clamped: bit-identical codes and scales"""
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M, K = x2.shape
    q = torch.empty(M, K, dtype=torch.int8, device=x.device)
    sc = torch.empty(M, dtype=torch.float32, device=x.device)
    qp = float(2 ** (bits - 1) - 1)
    rqp = float(torch.tensor(1.0 / qp, dtype=torch.float32))   # the fp32 reciprocal torch multiplies by
    lo = float(torch.tensor(1e-5, dtype=x.dtype))              # clamp_min(1e-5) happens in x's dtype in act_quant_i8
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
def _fire_kernel(u_ptr, v_ptr, p_ptr, d_ptr, N, K, K5, R, sun, svk, spn, sdn, theta,
                 RC: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr, PREC: tl.constexpr):
    """one tile of packed bytes: for each of the 5 trit digits, A = U V^T on those columns (fp32 accumulation; PREC
    'tf32x3' keeps fp32 accuracy on the tensor cores), the legal moves applied to the digit, the moves written to D
    (int8, the spend is applied from them sparsely), the bytes stored back"""
    pn = tl.program_id(0); pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    k5 = pk * BK5 + tl.arange(0, BK5)
    pm = (rows[:, None] < N) & (k5[None, :] < K5)
    b = tl.load(p_ptr + rows[:, None] * spn + k5[None, :], mask=pm, other=0).to(tl.int32)
    nb = tl.zeros((BN, BK5), dtype=tl.int32)
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
        nb += (dig + dd) * pw
        tl.store(d_ptr + rows[:, None] * sdn + k[None, :], dd.to(tl.int8), mask=(rows[:, None] < N) & (k[None, :] < K))
        pw = pw * 3
    tl.store(p_ptr + rows[:, None] * spn + k5[None, :], nb.to(tl.uint8), mask=pm)


FIRE_TILE = None          # ttf_fire tiles (rows, packed bytes of 5 columns, rank chunk); None: by rank
FIRE_PREC = ["tf32x3"]     # ttf_fire's A: "tf32x3" (tensor cores, ~fp32 accuracy) or "ieee" (fp32 FMAs, slower)


def ttf_fire(U: torch.Tensor, V: torch.Tensor, packed: torch.Tensor, K: int, theta: float):
    """--ts firing in one pass: A = U V^T (fp32 accumulation), every legal move (|A| >= theta, the trit stays in
    {-1, 0, 1}) applied to the packed trits in place; returns D V (D the moves, {-1, 0, 1}; from the moves alone, a
    sparse product) for the spend, and the number of moves (a device tensor). The fp32 [N, K] accumulator and the
    unpacked trits are never formed (D is int8). Differences from the unfused path: A's sums in another order and
    precision (rounding level), so an entry within rounding of theta may decide otherwise."""
    N, R = U.shape
    K5 = packed.shape[1]
    U = U.contiguous().float(); V = V.contiguous().float()
    D = torch.empty(N, K, dtype=torch.int8, device=U.device)
    BN, BK5, RC = FIRE_TILE if FIRE_TILE else ((64, 32, 32) if R <= 512 else (64, 64, 32))   # best measured (4070 Ti S)
    _fire_kernel[(triton.cdiv(N, BN), triton.cdiv(K5, BK5))](U, V, packed, D, N, K, K5, R, U.stride(0), V.stride(0),
                                                              packed.stride(0), D.stride(0), float(theta),
                                                              RC=RC, BN=BN, BK5=BK5, PREC=FIRE_PREC[0])
    nz = D.nonzero()                                                    # the moves (one host sync)
    dU = torch.zeros(N, R, dtype=torch.float32, device=U.device)
    for c0 in range(0, nz.shape[0], 1 << 16):                          # in chunks: moves x r floats at a time
        z_ = nz[c0:c0 + (1 << 16)]
        dU.index_add_(0, z_[:, 0], D[z_[:, 0], z_[:, 1]].float()[:, None] * V[z_[:, 1]])
    return dU, torch.tensor(nz.shape[0], device=U.device)
