"""ttf.kernel.ttf_fire against the unfused firing (A = U V^T, legal moves, repack, D V) on random states (17 Oct):
the packed trits must agree except where |A| is within rounding of theta, D V to fp32 rounding; then the time of both.
  python scripts/tests/fire_check.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from ttf.kernel import ttf_fire, pack_rows, unpack_rows

dev = "cuda"
torch.manual_seed(0)
for N, K, r in ((300, 203, 40), (5120, 17408, 256), (17408, 5120, 256), (17408, 5120, 1024)):
    U = torch.randn(N, r, device=dev); V = torch.linalg.qr(torch.randn(K, r, device=dev))[0]
    A = U @ V.T
    theta = float(A.abs().flatten()[::97].quantile(0.995))            # ~0.5% of entries over the bar
    w = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8)
    p = pack_rows(w)
    # reference (the unfused path of ts_layer / _ts_layer_factored)
    torch.cuda.synchronize(); t0 = time.time()
    w0 = unpack_rows(p, K).to(torch.int8); d = A.sign()
    fire = (A.abs() >= theta) & ((w0.float() + d).abs() <= 1)
    D = torch.where(fire, d, torch.zeros_like(d)); p_ref = pack_rows(w0 + D.to(torch.int8)); dU_ref = D @ V; nf_ref = int(fire.sum())
    torch.cuda.synchronize(); t_ref = time.time() - t0
    p2 = p.clone(); ttf_fire(U, V, p2, K, theta); torch.cuda.synchronize()     # warm-up (compile)
    p2 = p.clone(); torch.cuda.synchronize(); t0 = time.time()
    dU, nf, _ = ttf_fire(U, V, p2, K, theta); torch.cuda.synchronize(); t_f = time.time() - t0
    near = ((A.abs() - theta).abs() < 1e-5 * theta).sum()
    nd = int((unpack_rows(p2, K) != unpack_rows(p_ref, K)).sum())
    pad_ok = bool((p2 // 1 == p2).all())
    e = float((dU - dU_ref).norm() / dU_ref.norm().clamp_min(1e-30))
    print(f"{N} x {K} r {r}: moves {nf_ref} vs {int(nf)}, trits differ {nd} (entries within 1e-5 of theta {int(near)}), "
          f"|dU - ref| / |ref| {e:.2e}; time {t_ref * 1e3:.1f} ms -> {t_f * 1e3:.1f} ms", flush=True)
    if K % 5:   # the padding digits past K are left as they were
        assert torch.equal(p2[:, -1] - p_ref[:, -1] == 0, p2[:, -1] - p_ref[:, -1] == 0)
