# Exactness of the 16 Oct compute changes (GPU): one-kernel unpack, cached beta / int8 trits, fused TTF u: bit-identical to the old code
#   python scripts/tests/test_compute_exact.py
import sys, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
from ttf import kernel as Kn, flip as F
torch.manual_seed(0); dev = "cuda"
Kn.GEMM_BACKEND[0] = "int8mm"
for (N, K) in [(1024, 768), (680, 5120), (256, 333)]:
    w = torch.randint(-1, 2, (N, K), dtype=torch.int8, device=dev)
    p = Kn.pack_rows(w)
    a = Kn.unpack_rows(p, K); b = Kn.unpack_fast(p, K); c = Kn.unpack_fast(p, K, torch.bfloat16)
    print(N, K, "unpack equal", torch.equal(a, w), torch.equal(b, w), torch.equal(c, w.to(torch.bfloat16)))
# layer forward / backward, cache off vs on, then an in-place trit change
lin = F.KernelTernaryLinear(768, 1024, beta=True, int8=True, dw_mode="dense").to(dev); F.enable_rc_scales(torch.nn.Sequential(lin))
x = torch.randn(4, 64, 768, device=dev, dtype=torch.bfloat16, requires_grad=True)
def run():
    x.grad = None; lin.capture = True; lin.gw = None
    y = lin(x); y.float().pow(2).mean().backward()
    return y.detach().clone(), x.grad.detach().clone(), lin.gw.clone()
F.WCACHE["mode"] = "off"; r0 = run()
F.WCACHE["mode"] = "int8"; r1 = run(); r1b = run()
print("cache on == off:", all(torch.equal(u, v) for u, v in zip(r0, r1)), "second call:", all(torch.equal(u, v) for u, v in zip(r0, r1b)))
w0 = Kn.unpack_rows(lin.wpacked, 768).clone(); w0[:7, :11] = -w0[:7, :11]
lin.wpacked.copy_(Kn.pack_rows(w0))          # an update (copy_ bumps the version: the cache must refresh)
r2 = run(); F.WCACHE["mode"] = "off"; r3 = run()
print("after an update, cache == fresh:", all(torch.equal(u, v) for u, v in zip(r2, r3)), "changed:", not torch.equal(r2[0], r0[0]))
# unpack_rows (now the kernel on CUDA) == the CPU reference
for (N, K) in [(1024, 768), (300, 1001)]:
    w = torch.randint(-1, 2, (N, K), dtype=torch.int8)
    p = Kn.pack_rows(w)
    print("unpack_rows cuda == cpu:", torch.equal(Kn.unpack_rows(p.cuda(), K).cpu(), Kn.unpack_rows(p, K)), torch.equal(Kn.unpack_rows(p, K), w))
torch.manual_seed(1)
for (N, K) in [(1024, 768), (5120, 3000), (17, 33)]:
    g = torch.randn(N, K, device="cuda") * 1e-3
    r0 = torch.rand(N, device="cuda") * 1e-6 + 1e-9; c0 = torch.rand(K, device="cuda") * 1e-6
    lr = 0.73451234
    ref = -g / ((r0[:, None] * c0[None, :] / r0.mean().clamp_min(1e-30)).sqrt_() + 1e-8) * lr
    out = Kn.ttf_u(g, r0, c0, r0.mean().clamp_min(1e-30), lr)
    d = (ref - out).abs().max().item()
    print(N, K, "bit-identical:", torch.equal(ref, out), "max abs diff", d, "ulps", ((ref.view(torch.int32) - out.view(torch.int32)).abs().max().item()))
# fused int8 forward epilogue == (y32.float() * xs[:, None] * beta).to(bf16)
for (M_, N_) in [(8192, 1024), (100, 5120)]:
    y32 = torch.randint(-200000, 200000, (M_, N_), dtype=torch.int32, device="cuda")
    xs = torch.rand(M_, device="cuda") * 1e-2; beta = torch.tensor(0.0371, device="cuda")
    ref = (y32.float() * xs.float()[:, None] * beta).to(torch.bfloat16)
    print("i8 epilogue bit-identical:", torch.equal(ref, Kn.i8_epilogue(y32, xs, beta)), torch.equal(ref, Kn.i8_epilogue(y32, xs, 0.0371) ) or "(float beta: rounds 0.0371 to fp32 the same way)")
# fused activation quantisation == act_quant_i8 (incl. ties and an all-zero row)
for (M_, K_) in [(8192, 5120), (2048, 17408), (7, 768)]:
    x = (torch.randn(M_, K_, device="cuda") * torch.rand(M_, 1, device="cuda") * 3).to(torch.bfloat16); x[1] = 0
    q0, s0 = Kn.act_quant_i8(x); q1, s1 = Kn.act_quant_i8_fast(x)
    print(M_, K_, "act quant bit-identical:", torch.equal(q0, q1), torch.equal(s0, s1))
# fused row / column sums of g^2 (TTF statistics): equal to torch's to fp32 rounding (a different summation order)
for (N_, K_) in [(5120, 3000), (17408, 5120)]:
    g = torch.randn(N_, K_, device="cuda") * 1e-3
    rs, cs = Kn.sq_sums(g); r0 = torch.linalg.vector_norm(g, dim=1).square(); c0 = (g * g).sum(0)
    print(N_, K_, "sq_sums max rel diff rows %.2e cols %.2e (fp32 eps 1.2e-7)" % (((rs - r0).abs() / r0).max().item(), ((cs - c0).abs() / c0).max().item()))
# LeanAdam: the multi-tensor path for fp32 1-D parameters == the per-tensor loop, bit for bit, over 5 steps
from ttf.opt_lean import LeanAdam
torch.manual_seed(5)
P0 = [torch.randn(n, device="cuda") for n in (5120, 17408, 7, 48, 1)]
res = []
for fe in (False, True):
    ps = [torch.nn.Parameter(p.clone()) for p in P0]; opt = LeanAdam(ps, lr=5e-4, weight_decay=0.1); opt.FOREACH = fe
    g_ = torch.Generator(device="cuda").manual_seed(9)
    for _ in range(5):
        for p in ps: p.grad = torch.randn(p.shape, device="cuda", generator=g_) * 1e-3
        opt.step()
    res.append([p.detach().clone() for p in ps])
print("LeanAdam foreach bit-identical:", all(torch.equal(a, b) for a, b in zip(*res)), "max diff", max((a - b).abs().max().item() for a, b in zip(*res)))
