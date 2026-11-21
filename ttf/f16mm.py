"""f16mm: GEMMs on the fp16 tensor cores with fp16 accumulation inside each K block, the blocks summed in fp32
(--f16-acc). GeForce Ada (the lab's 4070 Ti Super, the laptop's 4080) runs fp16/bf16 MMA with fp32 accumulation at
half the rate of fp16 MMA with fp16 accumulation; summing BK (32-64) products in fp16 and every block's partial into
fp32 keeps the error at the level of the output's own bf16 rounding (t_f16acc.py, 21 Nov: 8192 x 32768 x 768, rel err
1.7e-3 vs bf16 torch.mm 2.9e-3, 1.45x the throughput). Inputs are read in their own dtype (bf16 or fp16) and converted
to fp16 in registers (bf16 -> fp16 is exact for |x| in [6e-5, 65504]: bf16's 8-bit mantissa fits fp16's 11; smaller
values become fp16 subnormals). Callers keep operands well inside fp16's range (logits' inputs, weights, softmax
gradients, attention tiles)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mm16_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, alpha,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACC: tl.constexpr, F16ACC: tl.constexpr,
                 GM: tl.constexpr = 8, EVEN_K: tl.constexpr = False, sa_ptr=None, HAS_SA: tl.constexpr = False):
    """C[M, N] (+)= alpha * sum_k A[m, k] B[k, n]: A, B any strides and float dtype (fp16 in the MMA); per BK block an
    fp16-accumulated product (F16ACC) added to the fp32 accumulator; C fp32 (ACC: read and added) or bf16/fp16"""
    pid = tl.program_id(0); npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    g = pid // (GM * npn); fm = g * GM; gs = min(npm - fm, GM)
    pm = fm + (pid % gs); pn = (pid % (GM * npn)) // gs
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    am = rm < M; bn = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if HAS_SA:   # (a power of two, on the device: A's products kept above fp16's subnormals; undone at the end)
        sa = tl.load(sa_ptr)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        if EVEN_K:
            a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=am[:, None], other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=bn[None, :], other=0.0)
        else:
            a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=am[:, None] & (kk[None, :] < K), other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=(kk[:, None] < K) & bn[None, :], other=0.0)
        if HAS_SA:
            a = a.to(tl.float32) * sa
        a = a.to(tl.float16); b = b.to(tl.float16)
        if F16ACC:
            acc += tl.dot(a, b, out_dtype=tl.float16).to(tl.float32)
        else:
            acc = tl.dot(a, b, acc)
    acc = acc * alpha
    if HAS_SA:
        acc = acc / sa
    cp = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    cm = am[:, None] & bn[None, :]
    if ACC:
        acc += tl.load(cp, mask=cm, other=0.0)
    tl.store(cp, acc.to(c_ptr.dtype.element_ty), mask=cm)


TILE = {}   # (M, N, K) -> (BM, BN, BK, warps, stages): tuned per shape (t_f16mm_tune.py); default below
F16ACC = [True]


def _tile(M, N, K):
    t = TILE.get((M, N, K))
    if t is not None:
        return t
    # (t_f16mm_tune.py, lab 4070 Ti Super, 21 Nov, the 110M head n 2048 V 32768 d 768: z = h W^T 0.91 ms vs bf16 1.23;
    # dh = dz W (long K) 1.01 vs 1.25; dW += dz^T h 1.00 vs 1.22)
    if K <= 1024:
        return 128, 128, 32, 4, 3
    if K >= 8192:
        return 64, 128, 32, 4, 3
    return 64, 256, 32, 4, 3


def prescale(amax_a, amax_b, BK=64, cap=2.0 ** 14):
    """device fp32 scalar: the largest power of two s <= cap with BK * s * amax_a * amax_b <= 2^14 (an fp16 block sum
    of BK products stays 4x below fp16's max); no host sync"""
    lim = torch.as_tensor(2.0 ** 14 / BK, device=amax_b.device) / (amax_a * amax_b).clamp_min(1e-30)
    return torch.exp2(torch.floor(torch.log2(lim))).clamp_(1.0, cap).float().reshape(1)


def mm16(a, b, out=None, acc=False, out_dtype=torch.bfloat16, alpha=1.0, sa=None):
    """a [M, K] @ b [K, N] (any strides, bf16/fp16/fp32 read, fp16 in the MMA) -> out [M, N]; acc: out += (fp32 out).
    sa: a device fp32 [1] power of two multiplying A before the fp16 MMA, divided out at the end (prescale)"""
    M, K = a.shape; K2, N = b.shape
    assert K == K2
    if out is None:
        assert not acc
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    assert out.shape == (M, N) and (not acc or out.dtype == torch.float32)
    BM, BN, BK, nw, ns = _tile(M, N, K)
    _mm16_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
        a, b, out, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        float(alpha), BM=BM, BN=BN, BK=BK, ACC=acc, F16ACC=F16ACC[0], EVEN_K=(K % BK == 0), num_warps=nw,
        num_stages=ns, **({} if sa is None else dict(sa_ptr=sa, HAS_SA=True)))
    return out
