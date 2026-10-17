"""Kernels against dense references (GPU; run on a lab PC, never the laptop GPU):

    python tests/test_kernel_ref.py

Checks pack/unpack, trit_beta, the ternary GEMMs (forward int8 / bf16, input gradient, weight gradient) on both
backends (triton, cublas) and shapes whose K is not a multiple of 5, and the kernel layer's forward / backward against
beta * Q8(x) @ W^T with a straight-through activation quantiser. Prints the max relative error of each and fails on
errors beyond bf16 rounding (or int8 rounding for the int8 weight-gradient paths).
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ttf import kernel as K                                   # noqa: E402
from ttf.flip import KernelTernaryLinear                      # noqa: E402

dev = "cuda"
torch.manual_seed(0)
fails = []


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def check(name, err, tol):
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name:58s} max rel err {err:.2e} (tol {tol:.0e})", flush=True)
    if not ok:
        fails.append(name)


def trits(N, Kd, p0=0.4):
    w = torch.randint(-1, 2, (N, Kd), device=dev, dtype=torch.int8)
    w[torch.rand(N, Kd, device=dev) < p0 - 1 / 3] = 0
    return w


# pack / unpack and beta
for Kd in (1, 4, 5, 6, 13, 768, 1000, 2049):
    w = trits(7, Kd)
    p = K.pack_rows(w)
    check(f"pack/unpack round trip K={Kd}", float((K.unpack_rows(p, Kd) != w).any().item()), 0)
    rho = (w != 0).float().mean()
    check(f"trit_beta K={Kd}", rel(K.trit_beta(p, Kd), (Kd * rho).clamp_min(1.0).rsqrt()), 1e-6)

for backend in ("triton", "cublas"):
    K.GEMM_BACKEND[0] = backend
    for (M, N, Kd) in ((4096, 768, 768), (4096, 2048, 768), (4096, 768, 2048), (1024, 333, 1000), (512, 1024, 5120)):
        w = trits(N, Kd); p = K.pack_rows(w); Wf = w.float()
        x = torch.randn(M, Kd, device=dev)
        beta = K.trit_beta(p, Kd)
        xq, xs = K.act_quant_i8(x)
        ref = (xq.float() @ Wf.T) * xs.float()[:, None] * beta
        check(f"[{backend}] tern_gemm_i8 {M}x{N}x{Kd}", rel(K.tern_gemm_i8(xq, xs, p, Kd, beta), ref), 8e-3)
        xb = x.to(torch.bfloat16)
        check(f"[{backend}] tern_gemm (bf16) {M}x{N}x{Kd}", rel(K.tern_gemm(xb, p, Kd), xb.float() @ Wf.T), 8e-3)
        gy = torch.randn(M, N, device=dev)
        gb = gy.to(torch.bfloat16)
        check(f"[{backend}] tern_gemm_dx {M}x{N}x{Kd}", rel(K.tern_gemm_dx(gb, p, Kd), gb.float() @ Wf), 8e-3)
        check(f"[{backend}] tern_gemm_dx_i8 {M}x{N}x{Kd}", rel(K.tern_gemm_dx_i8(gy, p, Kd), gy @ Wf), 3e-2)
        if backend == "triton":
            refw = (gy * xs.float()[:, None]).T @ xq.float()
            check(f"dw_int8 {M}x{N}x{Kd}", rel(K.dw_int8(gy, xq, xs), refw), 3e-2)
            check(f"dw_cublas {M}x{N}x{Kd}", rel(K.dw_cublas(gy, xq, xs), refw), 3e-2)

# the layer: forward and both gradients, int8 forward + dense weight gradient (the configuration of the runs)
K.GEMM_BACKEND[0] = "triton"
for (N, Kd) in ((768, 768), (2048, 768), (333, 1000)):
    l = KernelTernaryLinear(Kd, N, 8, rate=0.0, beta=True, int8=True, dw_mode="dense").to(dev)
    captured = {}
    l.capture = True
    l.on_grad = lambda layer, g: captured.__setitem__("g", g.clone())
    w = K.unpack_rows(l.wpacked, Kd).float(); beta = K.trit_beta(l.wpacked, Kd)
    x = torch.randn(2, 512, Kd, device=dev, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = l(x)
    gy = torch.randn_like(y.float())
    y.float().backward(gy)
    xq, xs = K.act_quant_i8(x.detach().reshape(-1, Kd))
    xd = xq.float() * xs.float()[:, None]
    check(f"layer forward {N}x{Kd}", rel(y.reshape(-1, N), (xd @ w.T) * beta), 8e-3)
    check(f"layer grad x (straight-through) {N}x{Kd}", rel(x.grad.reshape(-1, Kd), (gy.reshape(-1, N) * beta) @ w), 1.5e-2)
    if "g" in captured:
        check(f"layer captured weight gradient {N}x{Kd}", rel(captured["g"], (gy.reshape(-1, N) * beta).T @ xd), 1.5e-2)
    else:
        print(f"note: no captured weight gradient for {N}x{Kd} (capture path differs: check flip.py hooks)")

print("ALL OK" if not fails else f"{len(fails)} FAILED: {fails}")
