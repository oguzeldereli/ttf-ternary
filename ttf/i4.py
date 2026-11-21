"""i4: the int4 tensor-core GEMM (csrc/i4gemm.cu, CUTLASS) and 4-bit packing. Built on first use with
torch.utils.cpp_extension (needs nvcc 12.x: CUDA_HOME, and CUTLASS headers: TTF_CUTLASS)."""
import os
import torch
import triton
import triton.language as tl

_EXT = [None]


def ext():
    if _EXT[0] is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(__file__)
        cut = os.environ.get("TTF_CUTLASS", "/tmp/user/k4/cutlass")
        _EXT[0] = load("ttf_i4gemm", [os.path.join(here, "csrc", "i4gemm.cu")],
                       extra_include_paths=[os.path.join(cut, "include"), os.path.join(cut, "tools/util/include")],
                       extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89",
                                          "-gencode=arch=compute_86,code=sm_86", "--expt-relaxed-constexpr"],
                       verbose=False)
    return _EXT[0]


@triton.jit
def _pack4_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(x_ptr + 2 * i, mask=i < n, other=0).to(tl.int32)
    hi = tl.load(x_ptr + 2 * i + 1, mask=i < n, other=0).to(tl.int32)
    tl.store(o_ptr + i, ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8), mask=i < n)


def pack4(x8):
    """int8 codes in -8..7 [R, C] (C even) -> uint8 [R, C / 2], two's-complement nibbles, low nibble = even column"""
    R, C = x8.shape
    o = torch.empty(R, C // 2, dtype=torch.uint8, device=x8.device)
    n = R * C // 2
    _pack4_kernel[(triton.cdiv(n, 4096),)](x8.contiguous(), o, n, BLOCK=4096)
    return o


def mm(a4, b4, cfg=0):
    """a4 [M, K / 2], b4 [N, K / 2] packed -> int32 [M, N] = A B^T"""
    return ext().i4_mm(a4, b4, cfg)
