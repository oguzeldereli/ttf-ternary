"""GPU tests (lab PC): the kernels against their references, the TTF update's invariants, the factored update against
the dense one, and a short training run resumed half way against the same run straight through."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = "cuda"


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False


def test_unpack_and_beta():
    from ttf import kernel as Kn
    for N, K in ((1024, 768), (256, 333)):
        w = torch.randint(-1, 2, (N, K), dtype=torch.int8, device=dev)
        p = Kn.pack_rows(w)
        assert torch.equal(Kn.unpack_rows(p, K), w) and torch.equal(Kn.unpack_fast(p, K), w)
        assert torch.equal(Kn.unpack_fast(p, K, torch.bfloat16), w.to(torch.bfloat16))
        assert torch.equal(Kn.unpack_rows(p.cpu(), K), w.cpu())
        rho = float((w != 0).float().mean())
        assert abs(float(Kn.trit_beta(p, K)) - 1 / (K * rho) ** 0.5) < 1e-4 / (K * rho) ** 0.5 * 1e2


def test_act_quant_epilogue_ttf_u():
    from ttf import kernel as Kn
    x = (torch.randn(512, 768, device=dev) * torch.rand(512, 1, device=dev) * 3).to(torch.bfloat16); x[1] = 0
    q0, s0 = Kn.act_quant_i8(x); q1, s1 = Kn.act_quant_i8_fast(x)
    assert torch.equal(q0, q1) and torch.equal(s0, s1)
    y32 = torch.randint(-200000, 200000, (300, 1024), dtype=torch.int32, device=dev)
    xs = torch.rand(300, device=dev) * 1e-2; beta = torch.tensor(0.0371, device=dev)
    assert torch.equal((y32.float() * xs[:, None] * beta).to(torch.bfloat16), Kn.i8_epilogue(y32, xs, beta))
    g = torch.randn(333, 517, device=dev) * 1e-3
    r = torch.rand(333, device=dev) * 1e-6 + 1e-9; c = torch.rand(517, device=dev) * 1e-6; m = r.mean()
    ref = -g / ((r[:, None] * c[None, :] / m).sqrt_() + 1e-8) * 0.73
    assert torch.equal(ref, Kn.ttf_u(g, r, c, m, 0.73))
    rs, cs = Kn.sq_sums(g)
    assert float(((rs - (g * g).sum(1)).abs() / (g * g).sum(1)).max()) < 1e-5
    assert float(((cs - (g * g).sum(0)).abs() / (g * g).sum(0)).max()) < 1e-5


@pytest.mark.parametrize("backend", ["triton", "int8mm", "cublas"])
def test_gemms(backend):
    from ttf import kernel as Kn
    Kn.GEMM_BACKEND[0] = backend
    try:
        for M, N, K in ((64, 256, 768), (100, 333, 1000)):
            w = torch.randint(-1, 2, (N, K), dtype=torch.int8, device=dev); p = Kn.pack_rows(w); Wf = w.float()
            x = torch.randn(M, K, device=dev); xq, xs = Kn.act_quant_i8(x)
            ref = (xq.float() @ Wf.T) * xs[:, None]
            y = Kn.tern_gemm_i8(xq, xs, p, K, 1.0).float()
            assert float((y - ref).norm() / ref.norm()) < 8e-3
            gy = torch.randn(M, N, device=dev).to(torch.bfloat16)
            gx = Kn.tern_gemm_dx(gy, p, K).float()
            assert float((gx - gy.float() @ Wf).norm() / (gy.float() @ Wf).norm()) < 8e-3
    finally:
        Kn.GEMM_BACKEND[0] = "triton"


def test_codec_fp12():
    from ttf import update as U
    from ttf.kernel import pack12_scaled, unpack12_scaled
    for N, n in ((7, 5), (300, 64), (1000, 33)):
        X = torch.randn(N, n, device=dev) * torch.logspace(-8, 3, n, device=dev)[None, :]
        X[0, :3] = 0; X[1, 1] = -0.0; X[2] = 1e-38
        sc = X.abs().amax(0, keepdim=True).clamp_min(1e-30) / 32768
        ref = U._pack12((X / sc).to(torch.float16)); new = pack12_scaled(X, sc)
        assert torch.equal(ref, new)
        assert torch.equal((U._unpack12(ref, n).float() * sc).view(torch.int32), unpack12_scaled(new, sc).view(torch.int32))


def test_fire_fused():
    from ttf import kernel as K_
    from ttf.kernel import ttf_fire, pack_rows, unpack_rows, trit_nnz
    for N, K, r in ((300, 203, 40), (1024, 777, 64)):
        Uu = torch.randn(N, r, device=dev); V = torch.linalg.qr(torch.randn(K, r, device=dev))[0]
        A = Uu @ V.T
        theta = float(A.abs().flatten()[::7].quantile(0.99))
        w = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8); p = pack_rows(w)
        d = A.sign(); fire = (A.abs() >= theta) & ((w.float() + d).abs() <= 1)
        D = torch.where(fire, d, torch.zeros_like(d)); p_ref = pack_rows(w + D.to(torch.int8))
        for spend in ("dense", "sparse"):                                # (dense: no host sync, the default since 23 Oct)
            K_.FIRE_SPEND[0] = spend
            p2 = p.clone(); dU, nf, dnnz = ttf_fire(Uu, V, p2, K, theta)
            assert int(trit_nnz(p2, K)) == int(trit_nnz(p, K)) + int(dnnz), (N, K, r, spend)   # beta's count follows
            near = int(((A.abs() - theta).abs() < 1e-5 * theta).sum())
            moved = unpack_rows(p2, K) != unpack_rows(p, K)
            assert int((unpack_rows(p2, K) != unpack_rows(p_ref, K)).sum()) <= near
            assert int(nf) == int(moved.sum()), (int(nf), int(moved.sum()), spend)
            Dk = (unpack_rows(p2, K).float() - unpack_rows(p, K).float())   # (the kernel's own moves)
            assert float((dU - Dk @ V).norm() / (Dk @ V).norm().clamp_min(1e-30)) < 1e-5, spend
        K_.FIRE_SPEND[0] = "dense"


def test_dx_int8x2():
    """--dx int8x2: gy in two int8 slices on the int8 tensor cores is at least as accurate as the bf16 GEMM (whose output
    is rounded to bf16), deterministic, and shapes the int8 GEMM cannot take fall back to the bf16 path"""
    import ttf.kernel as Kn
    old = Kn.DX_I8X2_MIN_N[0]; Kn.DX_I8X2_MIN_N[0] = 0
    try:
        for M, N, K in ((512, 1024, 768), (300, 768, 2048), (40, 1000, 512)):
            w = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8); p = Kn.pack_rows(w)
            assert torch.equal(Kn.unpack_t_i8(p, K), w.t().contiguous())
            g = (torch.randn(M, N, device=dev) * torch.exp(2 * torch.randn(M, N, device=dev))).to(torch.bfloat16)
            ref = g.double() @ w.double()
            a, b = Kn.tern_gemm_dx(g, p, K), Kn.tern_gemm_dx_i8x2(g, p, K)
            ea = float((a.double() - ref).norm() / ref.norm()); eb = float((b.double() - ref).norm() / ref.norm())
            assert eb <= 1.25 * ea + 1e-6, (M, N, K, ea, eb)
            assert torch.equal(b, Kn.tern_gemm_dx_i8x2(g, p, K))
    finally:
        Kn.DX_I8X2_MIN_N[0] = old


def test_fused_scales():
    import ttf.layers as FL
    from ttf.layers import KernelTernaryLinear
    K, N, T = 512, 384, 1000
    l = KernelTernaryLinear(K, N, int8=True).to(dev)
    l.row_scale = torch.nn.Parameter((torch.rand(N, device=dev) + 0.5).half())
    l.col_scale = torch.nn.Parameter((torch.rand(K, device=dev) + 0.5).half())
    l.enable_lowrank_mag("add", 16); l.mag_A.data.normal_(0, 0.02)
    l.capture = True
    x0 = torch.randn(T, K, device=dev, dtype=torch.bfloat16); g0 = torch.randn(T, N, device=dev, dtype=torch.bfloat16)
    out = {}
    try:
        for fz in (False, True):
            FL.FUSED_SCALES[0] = fz
            for p in (l.row_scale, l.col_scale, l.mag_A, l.mag_B):
                p.grad = None
            x = x0.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = l(x)
            y.backward(g0)
            out[fz] = (y.detach(), x.grad, l.row_scale.grad.float(), l.col_scale.grad.float())
    finally:
        FL.FUSED_SCALES[0] = False
    a, b = out[False], out[True]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for u, v in zip(a[2:], b[2:]):
        assert float((u - v).norm() / u.norm()) < 1e-2          # the sums of fp16 products in another order


def test_lean_adam_batched():
    from ttf.opt_lean import LeanAdam
    shapes = [((300,), torch.float16), ((77,), torch.bfloat16), ((64, 16), torch.bfloat16), ((40, 16), torch.bfloat16)]
    rl = torch.rand_like
    torch.rand_like = lambda x, *a, **k: torch.full_like(x, 0.5)           # the same rounding noise in both paths
    try:
        res = {}
        for batched in (False, True):
            LeanAdam.BATCHED = batched
            g = torch.Generator(device=dev).manual_seed(0)
            ps = [torch.nn.Parameter((torch.randn(*s, device=dev, generator=g) * 0.1).to(dt)) for s, dt in shapes]
            opt = LeanAdam(ps, lr=1e-2, weight_decay=0.1)
            for s in range(5):
                gg = torch.Generator(device=dev).manual_seed(100 + s)
                for p in ps:
                    p.grad = (torch.randn(p.shape, device=dev, generator=gg) * 1e-3).to(p.dtype)
                opt.step()
            res[batched] = [p.detach().float().clone() for p in ps]
    finally:
        torch.rand_like = rl; LeanAdam.BATCHED = False
    assert max(float((a - b).abs().max()) for a, b in zip(res[False], res[True])) < 1e-6


def _cfg(**kw):
    c = dict(r=512, lr_ratio=1.0, theta=2.0, tau=1000.0, b2=0.95, spend=1.0, tau_anneal=1.0, store_a="", offload=False,
             theta_sched=None, progress=0.0, factored=0, start=0, step=10, micro=1)
    c.update(kw)
    return c


def _layer(N, K, seed):
    from ttf.layers import KernelTernaryLinear
    torch.manual_seed(seed)
    l = KernelTernaryLinear(K, N, 8).to(dev); l._ts_host = False
    return l


def test_ttf_moves_and_spend():
    """trits stay in {-1, 0, 1}; exactly the legal entries with |A| >= theta move, by sign(A); at full rank (N >= K)
    a move spends exactly theta from its entry"""
    from ttf import update as U
    from ttf.kernel import unpack_rows
    N, K = 80, 48
    gs = [torch.randn(N, K, device=dev) for _ in range(4)]
    lp, lq = _layer(N, K, 1), _layer(N, K, 1)
    sp, sq = {}, {}
    for g in gs[:3]:
        U.ts_layer(lp, g.clone(), sp, _cfg()); U.ts_layer(lq, g.clone(), sq, _cfg())
    assert torch.equal(lp.wpacked, lq.wpacked)
    w0 = unpack_rows(lq.wpacked, K).clone()
    U.ts_layer(lp, gs[3].clone(), sp, _cfg(theta=1e30))       # nothing reaches the bar: A before firing
    U.ts_layer(lq, gs[3].clone(), sq, _cfg())
    A0 = sp[id(lp)][0] @ sp[id(lp)][1].T; A1 = sq[id(lq)][0] @ sq[id(lq)][1].T
    w1 = unpack_rows(lq.wpacked, K); D = w1.int() - w0.int()
    legal = ((A0 >= 2.0) & (w0 < 1)) | ((A0 <= -2.0) & (w0 > -1))
    assert bool(((w1 >= -1) & (w1 <= 1)).all()) and int(legal.sum()) > 0
    assert torch.equal(D != 0, legal) and torch.equal(D[legal], torch.sign(A0[legal]).int())
    assert float((A0 - A1 - 2.0 * D.float()).abs().max()) < 1e-4


def test_factored_matches_dense():
    """the update from the gradient's factors (gy, x) approaches the dense one as the sketch grows (no firing): its
    error is the sketch's ~ 1 / sqrt(K_s) (koi, 17 Oct: 0.226 at 64, 0.062 at 1024)"""
    from ttf import update as U
    N, K, T, r = 512, 256, 1024, 32

    def run(ks):
        l = _layer(N, K, 0); l._ts_idx = 3; state = {}
        gen = torch.Generator(device=dev).manual_seed(123)
        for s in range(4):
            gy = (torch.randn(T, N, device=dev, generator=gen) * 1e-3).to(torch.bfloat16)
            x = (torch.randn(T, K, device=dev, generator=gen) + 0.3).to(torch.bfloat16)
            gy[:, :8] *= 20
            l._ts_final = True
            c = _cfg(r=r, theta=1e9, tau_anneal=0.75, factored=ks, step=10 + s)
            U.ts_layer(l, (gy, x) if ks else (gy.T.float() @ x.float()).contiguous(), state, c)
        Uu, V = U._load_pair(state[id(l)])
        return Uu @ V.T

    Ad = run(0)
    errs = [float((run(ks) - Ad).norm() / Ad.norm()) for ks in (64, 1024)]
    assert errs[1] < errs[0] / 2.5 and errs[1] < 0.1, errs          # 16 x the sketch: ~4 x smaller


def test_layer_ts_rank():
    """a layer's own ts_rank (set by a plug-in, e.g. for thin adapter factors) replaces the configured rank, on the
    dense and the factored path, micro and not; layers without it keep c["r"]"""
    from ttf import update as U
    N, K, T = 96, 64, 256
    for fac, M in ((0, 1), (64, 1), (64, 4)):
        for own in (None, 8):
            l = _layer(N, K, 0); l._ts_idx = 1; state = {}
            if own:
                l.ts_rank = own
            gen = torch.Generator(device=dev).manual_seed(5)
            for i in range(M):
                gy = (torch.randn(T, N, device=dev, generator=gen) * 1e-3).to(torch.bfloat16)
                x = torch.randn(T, K, device=dev, generator=gen).to(torch.bfloat16)
                l._ts_final = i == M - 1
                g = (gy, x) if fac else (gy.T.float() @ x.float()).contiguous()
                U.ts_layer(l, g, state, _cfg(r=24, theta=1e9, factored=fac, step=10, micro=M))
            Uu, V = U._load_pair(state[id(l)])
            assert Uu.shape[1] == V.shape[1] == (own or 24), (fac, M, own, Uu.shape, V.shape)


def test_micro_fused_matches_torch():
    """--ttf-accum micro: kernel.micro_sketch (the sketch sides' statistics in four kernels) against the torch ops,
    over 3 steps x 4 micro-batches (shapes that are not tile multiples): the accumulator and v agree to fp32 rounding
    (the sums run in another order). fp32 factors: with bf16 ones the products round their right-hand side to bf16
    (_mmb), so a 1e-7 difference in 1 / b flips some roundings and the two paths drift apart chaotically (~0.5% of A
    after 12 micro-batches here), as any two summation orders do"""
    from ttf import update as U
    N, K, T, M = 200, 136, 384, 4

    def run(fused):
        U.MICRO_FUSED[0] = fused
        l = _layer(N, K, 0); l._ts_idx = 2; state = {}
        gen = torch.Generator(device=dev).manual_seed(7)
        for s in range(3):
            for i in range(M):
                gy = torch.randn(T, N, device=dev, generator=gen) * 1e-3
                x = torch.randn(T, K, device=dev, generator=gen) + 0.2
                l._ts_final = i == M - 1
                U.ts_layer(l, (gy, x), state, _cfg(r=24, theta=1e9, tau_anneal=0.75, factored=64, step=10 + s, micro=M))
        Uu, V = U._load_pair(state[id(l)])
        return Uu @ V.T, state["_ts"][("v", id(l))]
    old = U.MICRO_FUSED[0]
    try:
        A0, v0 = run(False); A1, v1 = run(True)
    finally:
        U.MICRO_FUSED[0] = old
    assert float((A1 - A0).norm() / A0.norm()) < 1e-5
    for a, b in zip(v0, v1):
        assert float(((a - b).abs() / a.abs().clamp_min(1e-30)).max()) < 1e-5


def _run(out, extra, steps, data):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "tests", "det_run.py"), "ttf.train", "--preset", "small",
           "--data", data, "--val", data, "--seq-len", "128", "--batch-size", "2", "--accum", "2", "--steps", "8",
           "--warmup", "2", "--eval-interval", "1000", "--eval-iters", "1", "--save-secs", "99999", "--ttf-rank", "16",
           "--out-dir", out, "--stop-after", str(steps)] + extra
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)


def _train(out, extra, steps, data):
    r = _run(out, extra, steps, data)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return [json.loads(l)["loss"] for l in open(os.path.join(out, "metrics.jsonl")) if '"loss"' in l]


def test_train_resume_round_trip(tmp_path):
    """8 steps straight vs 4 steps, saved, resumed to 8: the same per-step losses (deterministic mode)"""
    data = str(tmp_path / "tok.bin")
    np.random.default_rng(0).integers(0, 32000, 200_000, dtype=np.uint16).tofile(data)
    a = _train(str(tmp_path / "a"), [], 8, data)
    assert len(_train(str(tmp_path / "b"), [], 4, data)) == 4
    b = _train(str(tmp_path / "b"), ["--resume"], 8, data)            # (the metrics file holds both parts)
    assert len(a) == 8 and a == b, (a, b)
    recs = [json.loads(l) for l in open(tmp_path / "b" / "records.jsonl")]
    assert [r["type"] for r in recs].count("run") == 2 and recs[-1].get("status") == "finished"


def test_resume_new_theta_needs_acc_scale(tmp_path):
    """a resume at a theta different from the checkpoint's is refused without --ttf-acc-scale (the burst of 17 / 23 Oct),
    and runs with it, saying what it did"""
    data = str(tmp_path / "tok.bin")
    np.random.default_rng(0).integers(0, 32000, 200_000, dtype=np.uint16).tofile(data)
    _train(str(tmp_path / "b"), [], 4, data)                           # (theta 16, the default)
    r = _run(str(tmp_path / "b"), ["--resume", "--ttf-theta", "8"], 6, data)
    assert r.returncode != 0 and "--ttf-acc-scale 0.5" in r.stdout + r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    r = _run(str(tmp_path / "b"), ["--resume", "--ttf-theta", "8", "--ttf-acc-scale", "0.5"], 6, data)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "accumulators x 0.5 (checkpoint theta 16.0, now 8.0)" in r.stdout, r.stdout[-1500:]


def test_train_resume_fp32(tmp_path):
    """--mode fp32 saved and resumed (also from step 0): the same per-step losses (the default adapter rank once added
    an empty optimizer group that the resume could not match, 18 Oct)"""
    data = str(tmp_path / "tok.bin")
    np.random.default_rng(0).integers(0, 32000, 200_000, dtype=np.uint16).tofile(data)
    fp = ["--mode", "fp32", "--no-qk-temp"]
    a = _train(str(tmp_path / "a"), fp, 8, data)
    _train(str(tmp_path / "b"), fp, 0, data)
    _train(str(tmp_path / "b"), fp + ["--resume"], 4, data)
    b = _train(str(tmp_path / "b"), fp + ["--resume"], 8, data)
    assert len(a) == 8 and a == b[-8:], (a, b)


def test_beta_follows_fused_fire():
    """the fused firing writes the packed trits in place: the layer's cached beta must follow (it was stale, 17 Oct)"""
    from ttf import update as U
    from ttf.kernel import trit_beta
    N, K, T = 256, 320, 512
    l = _layer(N, K, 3); l._ts_idx = 1; l.use_beta = True
    b0 = float(l._cached_beta(l.wpacked))
    U.FIRE_FUSED[0] = True
    try:
        gen = torch.Generator(device=dev).manual_seed(5); state = {}
        for s in range(3):
            gy = (torch.randn(T, N, device=dev, generator=gen)).to(torch.bfloat16)
            x = torch.randn(T, K, device=dev, generator=gen).to(torch.bfloat16)
            l._ts_final = True
            U.ts_layer(l, (gy, x), state, _cfg(r=16, theta=1e-3, factored=64, step=10 + s))
    finally:
        U.FIRE_FUSED[0] = False
    fresh = float(trit_beta(l.wpacked, K))
    assert fresh != b0                                   # trits moved
    assert abs(float(l._cached_beta(l.wpacked)) - fresh) < 1e-7


@pytest.mark.parametrize("extra", [["--ttf-offload", "async"], ["--w-cache", "int8"]], ids=["offload_async", "w_cache"])
def test_same_losses(tmp_path, extra):
    """options that must not change the numbers: per-step losses identical to the plain run (deterministic mode)"""
    data = str(tmp_path / "tok.bin")
    np.random.default_rng(0).integers(0, 32000, 200_000, dtype=np.uint16).tofile(data)
    base = ["--ttf-offload", "host"] if extra[0] == "--ttf-offload" else []
    a = _train(str(tmp_path / "a"), base, 6, data)
    b = _train(str(tmp_path / "b"), extra, 6, data)
    assert a == b, (a, b)


@pytest.mark.parametrize("fp32", ["1", "0"], ids=["fp32", "bf16"])
def test_dp_factored(fp32):
    """data parallel with the factored update: 2 ranks (sharing one GPU through gloo), each with its own gradient
    factors, average the update's thin products and must hold what one process holds after the update from the mean
    gradient (fp32 factors: to summation order; bf16: within bf16 rounding), and stay identical to each other. The fp32
    case runs the fused micro path (the default) and checks its data-parallel logic exactly; the bf16 case runs the torch
    micro path, since the fused kernel's bf16 running sums round at other points than the one-process reference (3 of
    61 moved trits differ in the micro-2 case, the ranks still identical; the fused path's quality: a 1k check at 0.8B)"""
    env = dict(os.environ, PYTHONPATH=ROOT, FP32=fp32, **({} if fp32 == "1" else {"TERN_MICRO_FUSED": "0"}))
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                        os.path.join(ROOT, "scripts", "tests", "dp_factored_check.py")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.parametrize("K,N,dx", [(1024, 3584, "int8x2"), (3584, 1024, "int8x2"), (1024, 3584, "bf16"), (1024, 48, "int8x2")])
def test_fused_layer(K, N, dx):
    """--fused-layer against the unfused path on a centred layer with row / column scales and bias, plus a second layer
    and a thin pair summed into it through acc (a converted site): output (with and without grad), input gradient and
    the TTF factors bit-identical; the scales' / bias' gradients within bf16 rounding (fp32 sums)"""
    import ttf.kernel as KN
    import ttf.layers as L
    from ttf.layers import KernelTernaryLinear
    dev, T = "cuda", 512
    old = (KN.GEMM_BACKEND[0], L.DX_MODE[0], L.FACTORED[0], L.FUSED_LAYER[0], KN.DX_I8X2_MIN_N[0])
    KN.GEMM_BACKEND[0], L.DX_MODE[0], L.FACTORED[0], KN.DX_I8X2_MIN_N[0] = "int8mm", dx, True, 1024
    try:
        torch.manual_seed(0)

        def mk(k, n):
            l = KernelTernaryLinear(k, n, 8, beta=True, int8=True).to(dev); l.enable_rc_scale()
            l.wpacked.copy_(KN.pack_rows(torch.randint(-1, 2, (n, k), device=dev, dtype=torch.int8)))
            with torch.no_grad():
                l.row_scale.uniform_(0.5, 1.5); l.col_scale.uniform_(0.5, 1.5)
            return l
        r = 64
        a, b, d, u = mk(K, N), mk(K, N), mk(K, r), mk(r, N)
        a.enable_centering(torch.randn(K, device=dev) * 0.3, torch.randn(N, device=dev) * 0.1)
        ls = [a, b, d, u]; cap = {}
        for i, m in enumerate(ls):
            m.capture = True; m.ts_micro = True
            m.on_grad = (lambda l_, g_, i=i: cap.setdefault(i, (g_[0].clone(), g_[1].clone())))
        x0 = torch.randn(T, K, device=dev, dtype=torch.bfloat16) * 2
        gy = torch.randn(T, N, device=dev, dtype=torch.bfloat16)
        site = lambda x: u(d(x), acc=b(x, acc=a(x)))

        def run(fused):
            L.FUSED_LAYER[0] = fused; cap.clear()
            for m in ls:
                for p_ in m.parameters():
                    p_.grad = None
            x = x0.clone().requires_grad_(True)
            with torch.no_grad():
                y_ng = site(x0)
            y = site(x); y.backward(gy)
            return (y.detach(), y_ng, x.grad, [p_.grad.clone() for m in ls for p_ in m.parameters()],
                    [cap[i] for i in range(4)])
        A, B = run(False), run(True)
        for i in range(3):
            assert torch.equal(A[i], B[i]), i
        assert torch.equal(B[0], B[1])
        for ga, gb in zip(A[3], B[3]):
            assert ((ga - gb).abs().max() / ga.abs().max().clamp_min(1e-30)).item() < 8e-3
        for (ga, xa), (gb, xb) in zip(A[4], B[4]):
            assert torch.equal(ga, gb) and torch.equal(xa, xb)
    finally:
        KN.GEMM_BACKEND[0], L.DX_MODE[0], L.FACTORED[0], L.FUSED_LAYER[0], KN.DX_I8X2_MIN_N[0] = old
