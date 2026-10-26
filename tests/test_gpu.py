"""GPU tests: the kernels against their references, the TTF update's invariants, the factored update against
the dense one, and a short training run resumed half way against the same run straight through."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import triton
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


def test_gemms():
    from ttf import kernel as Kn
    for M, N, K in ((64, 256, 768), (100, 333, 1000), (8, 256, 768)):     # (M <= 16: torch._int_mm's fallback)
        w = torch.randint(-1, 2, (N, K), dtype=torch.int8, device=dev); p = Kn.pack_rows(w); Wf = w.float()
        x = torch.randn(M, K, device=dev); xq, xs = Kn.act_quant_i8(x)
        ref = (xq.float() @ Wf.T) * xs[:, None]
        y = Kn.tern_gemm_i8(xq, xs, p, K, 1.0).float()
        assert float((y - ref).norm() / ref.norm()) < 8e-3
        gy = torch.randn(M, N, device=dev).to(torch.bfloat16)
        gx = Kn.tern_gemm_dx(gy, p, K).float()
        assert float((gx - gy.float() @ Wf).norm() / (gy.float() @ Wf).norm()) < 8e-3


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


@pytest.mark.parametrize("slots", [32, 1])
def test_fire_fused(slots):
    """ttf_fire against the fp64 decisions: the moves equal them outside a rounding band of theta, the trits, the
    move count, the nonzero count and D V follow the kernel's own moves; slots 1: the tiles overflow their list and
    decide in tf32x3"""
    from ttf import kernel as K_
    from ttf.kernel import ttf_fire, pack_rows, unpack_rows, trit_nnz
    K_.FIRE_SLOTS = slots
    try:
        for N, K, r in ((300, 203, 40), (1024, 777, 64), (700, 1500, 1024)):
            g = torch.Generator(device=dev).manual_seed(N)
            Uu = torch.randn(N, r, device=dev, generator=g)
            V = torch.linalg.qr(torch.randn(K, r, device=dev, generator=g))[0]
            A = Uu.double() @ V.double().T
            theta = float(A.abs().flatten()[::7].quantile(0.99))
            w = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8, generator=g); p = pack_rows(w)
            d = A.sign(); fire = (A.abs() >= theta) & ((w.double() + d).abs() <= 1)
            D = torch.where(fire, d, torch.zeros_like(d))
            p2 = p.clone(); dU, nf, dnnz = ttf_fire(Uu, V, p2, K, theta)
            assert int(trit_nnz(p2, K)) == int(trit_nnz(p, K)) + int(dnnz), (N, K, r)
            Dk = (unpack_rows(p2, K).double() - unpack_rows(p, K).double())   # (the kernel's own moves)
            near = (A.abs() - theta).abs() < 1e-5 * theta
            assert int(((Dk != D) & ~near).sum()) == 0, (N, K, r, int(((Dk != D) & ~near).sum()))
            assert int(nf) == int((Dk != 0).sum())
            assert float((dU.double() - Dk @ V.double()).norm() / (Dk @ V.double()).norm().clamp_min(1e-30)) < 1e-6
    finally:
        K_.FIRE_SLOTS = 32


def test_fire_near_theta():
    """entries placed within 1e-4 .. 1e-2 of theta (relative), far inside the bf16 error: the decisions are the fp64
    ones except within 1e-6 of theta"""
    from ttf.kernel import ttf_fire, pack_rows, unpack_rows
    N, K, r = 512, 640, 640                                # (square V: A0 is in its span)
    g = torch.Generator(device=dev).manual_seed(3)
    V = torch.linalg.qr(torch.randn(K, r, device=dev, generator=g, dtype=torch.float64))[0]
    A0 = torch.randn(N, K, device=dev, generator=g, dtype=torch.float64)
    theta = 1.0
    sel = torch.rand(N, K, device=dev, generator=g) < 0.05
    off = 10 ** (-2 - 2 * torch.rand(N, K, device=dev, generator=g, dtype=torch.float64))
    sg = torch.where(torch.rand(N, K, device=dev, generator=g) < 0.5, -1.0, 1.0).double()
    A0 = torch.where(sel, A0.sign() * theta * (1 + sg * off), A0 * 0.3)
    Uu = (A0 @ V).float()                                  # A = U V^T: the part of A0 in V's span
    A = Uu.double() @ V.double().T
    w = torch.zeros(N, K, device=dev, dtype=torch.int8); p = pack_rows(w)
    D = torch.where(A.abs() >= theta, A.sign(), torch.zeros_like(A))
    p2 = p.clone(); ttf_fire(Uu, V.float(), p2, K, theta)
    Dk = unpack_rows(p2, K).double()
    near = (A.abs() - theta).abs() < 1e-6
    assert int(((A.abs() - theta).abs() < 1e-2).sum()) > 1000             # (the test has uncertain entries)
    assert int(((Dk != D) & ~near).sum()) == 0


def test_dx_int8x2():
    """--dx int8x2: gy in two int8 slices on the int8 tensor cores is at least as accurate as the bf16 GEMM (whose output
    is rounded to bf16), deterministic, and shapes the int8 GEMM cannot take fall back to the bf16 path"""
    import ttf.kernel as Kn
    for M, N, K in ((512, 1024, 768), (300, 768, 2048), (40, 1000, 512)):
        w = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8); p = Kn.pack_rows(w)
        assert torch.equal(Kn.unpack_t_i8(p, K), w.t().contiguous())
        g = (torch.randn(M, N, device=dev) * torch.exp(2 * torch.randn(M, N, device=dev))).to(torch.bfloat16)
        ref = g.double() @ w.double()
        a, b = Kn.tern_gemm_dx(g, p, K), Kn.tern_gemm_dx_i8x2(g, p, K)
        ea = float((a.double() - ref).norm() / ref.norm()); eb = float((b.double() - ref).norm() / ref.norm())
        assert eb <= 1.25 * ea + 1e-6, (M, N, K, ea, eb)
        assert torch.equal(b, Kn.tern_gemm_dx_i8x2(g, p, K))


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
        torch.rand_like = rl; LeanAdam.BATCHED = True
    assert max(float((a - b).abs().max()) for a, b in zip(res[False], res[True])) < 1e-6


def _cfg(**kw):
    c = dict(r=512, lr_ratio=1.0, theta=2.0, tau=1000.0, b2=0.95, spend=1.0, store_a="", offload=False,
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
            c = _cfg(r=r, theta=1e9, factored=ks, step=10 + s)
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


def test_orth_cholqr2():
    """--ttf-orth cholqr2: an orthonormal basis of the same span as QR on well-conditioned X; on rank-deficient or
    ill-conditioned X (zero columns, nearly parallel ones) the QR fallback, still orthonormal"""
    from ttf import update as U
    g = torch.Generator(device=dev).manual_seed(3)
    X = torch.randn(4096, 256, device=dev, generator=g)
    Xz = X.clone(); Xz[:, 10:20] = 0
    Xp = X.clone(); Xp[:, 1] = Xp[:, 0] + 1e-5 * Xp[:, 1]
    old = U.ORTH[0]
    try:
        U.ORTH[0] = "cholqr2"
        for Y in (X, Xz, Xp, X * 1e-20):
            Q = U._orth_(Y)
            I = torch.eye(Q.shape[1], device=dev)
            assert float((Q.T @ Q - I).abs().max()) < 1e-4
        Q = U._orth_(X); Q0 = torch.linalg.qr(X)[0]
        assert float((Q @ (Q.T @ Q0) - Q0).abs().max()) < 1e-4          # same span
        try:
            import orth  # noqa: F401  (spanqr, when on PYTHONPATH)
        except ImportError:
            return
        U.ORTH[0] = "spanqr"
        Xw = torch.randn(5120, 1024, device=dev, generator=g)          # above r 512: spanqr's block Gram-Schmidt
        Xwz = Xw.clone(); Xwz[:, 300:340] = 0
        for Y in (X, Xz, Xp, Xw, Xwz, X[:, :200]):                      # (r 200: Householder)
            Q = U._orth_(Y)
            assert float((Q.T @ Q - torch.eye(Q.shape[1], device=dev)).abs().max()) < 1e-4
        Q = U._orth_(Xw); Q0 = torch.linalg.qr(Xw)[0]
        assert float((Q @ (Q.T @ Q0) - Q0).abs().max()) < 1e-4          # same span
    finally:
        U.ORTH[0] = old


def test_micro_fused_matches_torch():
    """--accum M > 1: kernel.micro_sketch (the sketch sides' statistics in four kernels) against the torch ops,
    over 3 steps x 4 micro-batches (shapes that are not tile multiples): v agrees to fp32 rounding (the sums run in
    another order), the accumulator to the running sketch's bf16 rounding (the basis turn reads it: update._sketch_cross;
    the two paths keep it in bf16 and round it at other points: 0.3% of A here). fp32 factors: with bf16 ones the products round their right-hand side to bf16
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
                U.ts_layer(l, (gy, x), state, _cfg(r=24, theta=1e9, factored=64, step=10 + s, micro=M))
        Uu, V = U._load_pair(state[id(l)])
        return Uu @ V.T, state["_ts"][("v", id(l))]
    old = U.MICRO_FUSED[0]
    try:
        A0, v0 = run(False); A1, v1 = run(True)
    finally:
        U.MICRO_FUSED[0] = old
    assert float((A1 - A0).norm() / A0.norm()) < 1e-2
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
        U.FIRE_FUSED[0] = True
    fresh = float(trit_beta(l.wpacked, K))
    assert fresh != b0                                   # trits moved
    assert abs(float(l._cached_beta(l.wpacked)) - fresh) < 1e-7


@pytest.mark.parametrize("extra", [["--ttf-offload", "async"]], ids=["offload_async"])
def test_same_losses(tmp_path, extra):
    """options that must not change the numbers: per-step losses identical to the plain run (deterministic mode)"""
    data = str(tmp_path / "tok.bin")
    np.random.default_rng(0).integers(0, 32000, 200_000, dtype=np.uint16).tofile(data)
    a = _train(str(tmp_path / "a"), [], 6, data)
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
    old = (L.DX_MODE[0], L.FACTORED[0], L.FUSED_LAYER[0])
    L.DX_MODE[0], L.FACTORED[0] = dx, True
    try:
        torch.manual_seed(0)

        def mk(k, n):
            l = KernelTernaryLinear(k, n, 8, beta=True).to(dev); l.enable_rc_scale()
            l.wpacked.copy_(KN.pack_rows(torch.randint(-1, 2, (n, k), device=dev, dtype=torch.int8)))
            with torch.no_grad():
                l.row_scale.uniform_(0.5, 1.5); l.col_scale.uniform_(0.5, 1.5)
            return l
        r = 64
        a, b, d, u = mk(K, N), mk(K, N), mk(K, r), mk(r, N)
        a.enable_centering(torch.randn(K, device=dev) * 0.3, torch.randn(N, device=dev) * 0.1)
        ls = [a, b, d, u]; cap = {}
        for i, m in enumerate(ls):
            m.capture = True
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
        L.DX_MODE[0], L.FACTORED[0], L.FUSED_LAYER[0] = old


def test_intk_codes_fire_in_range():
    """--intk-emb: the dense TTF update moves int-k codes by one level where the accumulator reaches theta, never out
    of 0 .. levels; the leaf's gradient reaches the group parameters"""
    from ttf import update as U
    from ttf.intk_emb import IntKMatrix
    torch.manual_seed(0)
    m = IntKMatrix(torch.randn(96, 128, device=dev), 2).to(dev); m._ts_idx = 7
    q0 = m.codes.clone(); st = {}
    for _ in range(3):
        W = m.leaf(); (W * torch.randn_like(W)).sum().backward()
        m.step(U.ts_layer, st, _cfg(r=16, theta=0.5))
    assert int(m.codes.min()) >= 0 and int(m.codes.max()) <= 3
    assert int((m.codes != q0).sum()) > 0 and int((m.codes.int() - q0.int()).abs().max()) <= 3
    assert m.lo.grad is not None and m.s.grad is not None


def test_intk_rows_matches_dense():
    """intk_emb.ttf_rows (the chunked update for 27B-size matrices) against update.ts_layer's dense path: from the same
    accumulator state the same v, the same turned basis span and the same moves (up to fp32 summation order)"""
    from ttf import update as U
    from ttf import intk_emb as IE
    torch.manual_seed(0)
    W = torch.randn(300, 128, device=dev)
    a, b = IE.IntKMatrix(W, 3).to(dev), IE.IntKMatrix(W, 3).to(dev)
    a._ts_idx = b._ts_idx = 3
    sa, sb = {}, {}
    c = _cfg(r=16, theta=0.3)
    gs = [torch.randn(300, 128, device=dev) for _ in range(4)]
    U.ts_layer(a, gs[0].clone(), sa, dict(c, micro=1))       # the first call: both start from ts_layer's state
    sb[id(b)] = tuple(x.clone() for x in sa[id(a)]); sb["_ts"] = {k_ if k_[1] != id(a) else (k_[0], id(b)): (tuple(x.clone() for x in v_) if isinstance(v_, tuple) else v_)
                                                              for k_, v_ in sa["_ts"].items() if isinstance(k_, tuple)}
    b.codes.copy_(a.codes)
    for g in gs[1:]:
        U.ts_layer(a, g.clone(), sa, dict(c, micro=1)); IE.ttf_rows(b, g.clone(), sb, c, rows=64)
    assert int((a.codes != b.codes).sum()) <= 3, int((a.codes != b.codes).sum())
    Ua, Va = sa[id(a)]; Ub, Vb = sb[id(b)]
    assert float((Ua @ Va.T - Ub @ Vb.T).norm() / (Ua @ Va.T).norm()) < 1e-3


@pytest.mark.parametrize("wdt", [torch.float32, torch.bfloat16])
def test_qwen_fused_kernels(wdt):
    """The fused Qwen kernels (always on): kernel.rms_norm_qwen, swiglu and gated_norm_qwen against the Hugging Face formulas (forward and
    the gradients of every input), bf16 activations; within bf16 rounding (fp32 internals in both)"""
    import torch.nn.functional as F
    import ttf.kernel as KN
    dev = "cuda"
    torch.manual_seed(0)
    M, D, eps = 300, 1000, 1e-6

    def close(a, b, tol):
        return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item() < tol

    def both(fk, fr, xs):
        a = [x.detach().clone().requires_grad_(True) for x in xs]
        b = [x.detach().clone().requires_grad_(True) for x in xs]
        ya, yb = fk(*a), fr(*b)
        g = torch.randn_like(yb)
        ya.backward(g); yb.backward(g)
        assert ya.dtype == yb.dtype and close(ya, yb, 1e-2)
        for p, q in zip(a, b):
            assert close(p.grad, q.grad, 2e-2), (p.shape, p.dtype)

    x = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 3
    w = (torch.randn(D, device=dev) * 0.1).to(wdt)
    both(lambda x, w: KN.rms_norm_qwen(x, w, eps),
         lambda x, w: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
                       * (1.0 + w.float())).type_as(x), [x, w])
    g, u = (torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 2 for _ in range(2))
    both(KN.swiglu, lambda g, u: F.silu(g) * u, [g, u])
    z = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 2
    wg = (1 + torch.randn(D, device=dev) * 0.1).float()
    both(lambda x, z, w: KN.gated_norm_qwen(x, z, w, eps),
         lambda x, z, w: ((w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype))
                          * F.silu(z.float())).to(x.dtype), [x, z, wg])


def test_w8_cache_follows_trits():
    """KernelTernaryLinear.w8: the cached int8 trits (both layouts) equal a fresh unpack after the fused firing (in place,
    no version bump) and after a copy_ into the packed buffer"""
    from ttf import update as U
    from ttf.kernel import unpack_fast, unpack_t_i8, pack_rows
    N, K, T = 256, 320, 512
    l = _layer(N, K, 3); l._ts_idx = 1
    from ttf import layers as L
    L.W8_CACHE[0] = True
    w0, t0 = l.w8().clone(), l.w8(t=True).clone()
    assert torch.equal(w0, unpack_fast(l.wpacked, K)) and torch.equal(t0, unpack_t_i8(l.wpacked, K))
    gen = torch.Generator(device=dev).manual_seed(5); state = {}
    for s in range(3):
        gy = (torch.randn(T, N, device=dev, generator=gen)).to(torch.bfloat16)
        x = torch.randn(T, K, device=dev, generator=gen).to(torch.bfloat16)
        l._ts_final = True
        U.ts_layer(l, (gy, x), state, _cfg(r=16, theta=1e-3, factored=64, step=10 + s))
    assert not torch.equal(unpack_fast(l.wpacked, K), w0)                    # trits moved
    assert torch.equal(l.w8(), unpack_fast(l.wpacked, K)) and torch.equal(l.w8(t=True), unpack_t_i8(l.wpacked, K))
    l.wpacked.copy_(pack_rows(torch.zeros(N, K, dtype=torch.int8, device=dev)))
    assert torch.equal(l.w8(), torch.zeros(N, K, dtype=torch.int8, device=dev)) and not l.w8(t=True).any()
    L.W8_CACHE[0] = False


@pytest.mark.parametrize("opt", ["lean", "adamw_fused"])
def test_deq_cache_cuda_optimizers(opt):
    """deq_cache follows the tail optimizers that write through .data on CUDA (LeanAdam's batched 2-D path, fused
    AdamW: no version bump)"""
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from test_intk_factored import _deq_cache_steps
    from ttf.opt_lean import LeanAdam
    _deq_cache_steps(dev, (lambda ps: LeanAdam(ps, lr=1e-2)) if opt == "lean" else
                     (lambda ps: torch.optim.AdamW(ps, lr=1e-2, fused=True)))


def test_group_sums():
    from ttf.kernel import group_sums
    g = torch.randn(300, 640, device=dev); q = torch.randint(0, 8, (300, 640), device=dev, dtype=torch.int8)
    s0, s1 = group_sums(g, q, 64)
    Gg = g.double().view(300, 10, 64)
    assert torch.allclose(s0.double(), Gg.sum(-1), atol=1e-5) and torch.allclose(s1.double(), (Gg * q.view(300, 10, 64).double()).sum(-1), atol=1e-4)


@pytest.mark.parametrize("N,K", [(48, 160), (160, 48)], ids=["rows", "cols"])
def test_dense_acc(N, K):
    """a layer whose rank covers its rows (N <= r, N < K) holds a dense accumulator: after one micro-batch it equals the
    low-rank one (whose first basis spans u's rows: exact there); over a step of 4 micro-batches it is lk A + the sum of
    the u (the low-rank one drops the evidence outside its fixed basis); a low-rank state from an earlier run becomes
    U V^T; the firing moves where |A| >= theta (legal) and spends theta"""
    from ttf import update as U
    from ttf.kernel import unpack_rows
    T = 256
    gen = torch.Generator(device=dev).manual_seed(5)
    data = [((torch.randn(T, N, device=dev, generator=gen) * 1e-3).to(torch.bfloat16),
             torch.randn(T, K, device=dev, generator=gen).to(torch.bfloat16)) for _ in range(8)]

    def run(dense, M, steps, theta=1e9, state=None):
        U.DENSE_ACC[0] = dense
        l = _layer(N, K, 0); l._ts_idx = 1; state = {} if state is None else state
        for s in range(steps):
            for i in range(M):
                l._ts_final = i == M - 1
                U.ts_layer(l, data[(s * M + i) % 8], state, _cfg(r=64, theta=theta, factored=32, step=10 + s, micro=M))
        return l, state
    try:
        l1, s1 = run(True, 1, 1); l0, s0 = run(False, 1, 1)
        A1 = U.load_dense(s1[id(l1)], l1); A0 = U.load_dense(s0[id(l0)], l0)
        assert U.is_dense_state(s1[id(l1)]) and not U.is_dense_state(s0[id(l0)])
        R, C = s1["_ts"][("v", id(l1))]                    # the step's v: u = -lr g / (a_i b_j), in fp64
        c_ = _cfg(); b2 = c_["b2"]
        a = (R.double() / R.double().mean() / (1 - b2)).sqrt(); b = C.double().sqrt()
        gy, x = data[0]
        u = -c_["lr_ratio"] * (gy.double().T @ x.double()) / a[:, None] / b[None, :]
        e1, e0 = float((A1 - u).norm() / u.norm()), float((A0 - u).norm() / u.norm())
        assert e1 < 5e-3 and e1 < e0, (e1, e0)             # (the low-rank's first basis: bf16 products, conditioning)
        # 4 micro-batches: the dense sum against the low-rank's: differs (its basis drops evidence), same scale
        l1, s1 = run(True, 4, 2); l0, s0 = run(False, 4, 2)
        A1 = U.load_dense(s1[id(l1)], l1); A0 = U.load_dense(s0[id(l0)], l0)
        rel = float((A1 - A0).norm() / A1.norm())
        assert (1e-3 < rel < 1.0) if N < K else rel < 2e-2, rel   # (ra = K: the low-rank basis spans all: exact)
        # mixed factors (bf16 gy, fp32 x): the fp32 product
        U.DENSE_ACC[0] = True
        l4 = _layer(N, K, 0); l4._ts_idx = 1; s4 = {}; l4._ts_final = True
        gy, x = data[0]
        U.ts_layer(l4, (gy, x.float()), s4, _cfg(r=64, theta=1e9, factored=32, step=10))
        assert U.is_dense_state(s4[id(l4)])
        with pytest.raises(SystemExit):              # a dense state under --dw dense: refused
            U.ts_layer(l4, (gy.T.float() @ x.float()).contiguous(), s4, _cfg(r=64, theta=1e9, step=11))
        # resume from a low-rank state: converted, then dense
        l2, s2 = run(True, 4, 1, state=s0)
        assert U.is_dense_state(s2[id(l2)])
        # firing: theta at A's 99th percentile
        A_ = U.load_dense(s1[id(l1)], l1); th = float(A_.abs().quantile(0.99))
        l3, s3 = run(True, 4, 2, theta=th)
        w = unpack_rows(l3.wpacked, K).float()
        moved = int((w != unpack_rows(_layer(N, K, 0).wpacked, K).float()).sum())
        assert moved > 0 and moved <= int(s3["_ts"]["_nf"])
    finally:
        U.DENSE_ACC[0] = True


@pytest.mark.parametrize("slots", [32, 1])
def test_intk_fire(slots):
    """kernel.intk_fire against intk_emb._fire's fp32 arithmetic: the same moves and codes (entries planted at
    theta (1 +- 2e-4) and rows with negative or tiny rho), (D / rho) V within fp32 rounding; slots 1: the tf32x3
    fallback for most tiles"""
    from ttf import kernel as KN
    n, K, r, G, levels = 200, 448, 96, 64, 3
    gen = torch.Generator(device=dev).manual_seed(3)
    U = torch.randn(n, r, device=dev, generator=gen) * 0.3
    V = torch.linalg.qr(torch.randn(K, r, device=dev, generator=gen)).Q
    rho = torch.randn(n, K // G, device=dev, generator=gen).abs() + 0.5
    rho[::7] *= -1; rho[5, 2] = 1e-8
    codes = torch.randint(0, levels + 1, (n, K), device=dev, generator=gen, dtype=torch.int8)
    rf = rho.repeat_interleave(G, 1)
    A = (U @ V.T) * rf
    theta = float(A.abs().quantile(0.97))
    # entries planted near theta: U row n += c v_k so that A_q(n, k) = theta (1 +- 2e-4)
    for i in range(40):
        a, b = (i * 37) % n, (i * 101) % K
        tgt = theta * (1 + (2e-4 if i % 2 else -2e-4)) * (1 if i % 3 else -1)
        U[a] += ((tgt / rf[a, b]) - (U[a] @ V[b])) * V[b]
    A = (U @ V.T) * rf
    d = A.sign(); nw = codes.float() + d
    fire = (A.abs() >= theta) & (nw >= 0) & (nw <= levels) & (rf.abs() > 1e-6)
    c_ref = codes + torch.where(fire, d, 0).to(torch.int8)
    dU_ref = torch.where(fire, d / rf, 0) @ V
    old = KN.IFIRE_SLOTS
    try:
        KN.IFIRE_SLOTS = slots
        c2 = codes.clone()
        dU, nf = KN.intk_fire(U, V, c2, rho, G, levels, theta)
    finally:
        KN.IFIRE_SLOTS = old
    assert torch.equal(c2, c_ref) and int(nf) == int(fire.sum()), (int((c2 != c_ref).sum()), int(nf), int(fire.sum()))
    assert float((dU - dU_ref).norm() / dU_ref.norm()) < 1e-5


def test_intk_fire_wired():
    """intk_emb._fire through kernel.intk_fire (FUSED_FIRE) against its torch ops, on a CUDA matrix in two row chunks
    with signed, varying s: the same codes, the same stored U within fp32 rounding, deq_cache's key moved"""
    from ttf import intk_emb as IE
    N, K, r = 300, 192, 32
    out = {}
    for fused in (True, False):
        torch.manual_seed(1)
        m = IE.IntKMatrix(torch.randn(N, K), 2).to(dev); m._ts_idx = 2
        m.s.data.mul_(torch.randn_like(m.s) + 0.3)
        st = {"_ts": {}}
        oldr, oldf = IE.ROWS[0], IE.FUSED_FIRE[0]
        try:
            IE.ROWS[0] = 160; IE.FUSED_FIRE[0] = fused
            S = IE._Store(m, st, id(m), None, False, dev); S.t = [None] * (S.ne * (1 + S.nch))
            g = torch.Generator(device=dev).manual_seed(4)
            V = torch.linalg.qr(torch.randn(K, r, device=dev, generator=g)).Q
            Us = [torch.randn(j - i, r, device=dev, generator=g) for i, j in map(S.rng, range(S.nch))]
            v0 = m.codes._version
            IE._fire(m, S, Us, V, 1.5, dict(spend=1.0), st["_ts"])
            assert m.codes._version != v0
            S2 = IE._Store(m, st, id(m), None, False, dev)
            out[fused] = (m.codes.clone(), torch.cat([S2.u(c) for c in range(S2.nch)]), int(st["_ts"]["_nf"]))
        finally:
            IE.ROWS[0], IE.FUSED_FIRE[0] = oldr, oldf
    (c1, U1, n1), (c0, U0, n0) = out[True], out[False]
    assert n0 > 0 and n1 == n0 and torch.equal(c1, c0)
    assert float((U1 - U0).norm() / U0.norm()) < 1e-5


@pytest.mark.parametrize("M,N,K", [(512, 1024, 768), (300, 208, 2048), (2048, 5120, 1024), (100, 40, 4096)])
def test_dx2_col(M, N, K):
    """kernel.dx2_col (one kernel: both slices' int8 GEMMs + the dx epilogue) against the two-GEMM path: the input
    gradient bit-identical; the column scale's gradient within fp32 summation order"""
    import ttf.kernel as Kn
    gen = torch.Generator(device=dev).manual_seed(M + N)
    gy = (torch.randn(M, N, device=dev, generator=gen) * 1e-3).to(torch.bfloat16)
    w = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8, generator=gen)
    w8t = w.t().contiguous()
    q1 = torch.empty(M, N, dtype=torch.int8, device=dev); q2 = torch.empty_like(q1); s = torch.empty(M, device=dev)
    Kn._slice2_kernel[(M,)](gy, q1, q2, s, N, gy.stride(0), q1.stride(0), BLOCK=1024)
    c = (torch.rand(K, device=dev, generator=gen) + 0.5).to(torch.bfloat16)
    x = torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16)
    mu = (torch.randn(K, device=dev, generator=gen) * 0.1).to(torch.bfloat16)
    y1 = (q1.double() @ w.double()).to(torch.int32).contiguous(); y2 = (q2.double() @ w.double()).to(torch.int32).contiguous()
    g = torch.empty(M, K, dtype=torch.bfloat16, device=dev)          # (the two-GEMM path's epilogue: the same fp32 ops)
    Kn._dx2_epi_kernel[(triton.cdiv(M, 32), triton.cdiv(K, 128))](y1, y2, s, g, M, K, BM=32, BK=128)
    g = g.float()
    gx_ref = (g * c.float()[None, :]).to(torch.bfloat16)
    xc = (x.float() - mu.float()[None, :]).to(torch.bfloat16).float()
    gc_ref = (g * xc).to(torch.bfloat16).double().sum(0)
    gx, gc = Kn.dx2_col(q1, q2, s, w8t, c, x, mu)
    assert torch.equal(gx, gx_ref)
    assert float((gc.double() - gc_ref).abs().max() / gc_ref.abs().max()) < 1e-5


@pytest.mark.parametrize("M,K,N,acc,bias", [(512, 1024, 768, False, True), (300, 208, 2048, True, False),
                                            (100, 2048, 20000, False, False), (64, 17000, 4096, True, True)])
def test_fused_fwd_gemm(M, K, N, acc, bias):
    """kernel.fused_fwd_gemm (the GEMM and its epilogue in one kernel; the cutlass path for long reductions) against
    torch._int_mm + fused_epilogue: t and the output bit-identical"""
    import ttf.kernel as Kn
    gen = torch.Generator(device=dev).manual_seed(M + K)
    xq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8, generator=gen)
    w8 = torch.randint(-1, 2, (N, K), device=dev, dtype=torch.int8, generator=gen)
    xs = torch.rand(M, device=dev, generator=gen) * 1e-2
    rb = (torch.rand(N, device=dev, generator=gen) + 0.5).to(torch.bfloat16)
    bb = torch.randn(N, device=dev, generator=gen).to(torch.bfloat16) if bias else None
    a = torch.randn(M, N, device=dev, generator=gen).to(torch.bfloat16) if acc else None
    beta = torch.full((), 0.7, device=dev)
    for save in (True, False):
        t0, o0 = Kn.fused_epilogue(torch._int_mm(xq, w8.t()), xs, beta, rb, bb, a, save)
        t1, o1 = Kn.fused_fwd_gemm(xq, w8, xs, beta, rb, bb, a, save)
        assert torch.equal(o0, o1) and (not save or torch.equal(t0, t1))


@pytest.mark.parametrize("T,N,K,ib,c", [(256, 300, 200, True, True), (100, 64, 520, False, True), (512, 1000, 96, True, False)])
def test_acc_gemm(T, N, K, ib, c):
    """kernel.acc_gemm = cs C + alpha diag(ia) (g^T bf16(x diag(ib))) within fp32 summation order"""
    import ttf.kernel as Kn
    gen = torch.Generator(device=dev).manual_seed(T + N)
    g = (torch.randn(T, N, device=dev, generator=gen) * 1e-2).to(torch.bfloat16)
    x = torch.randn(T, K, device=dev, generator=gen).to(torch.bfloat16)
    b = torch.rand(K, device=dev, generator=gen) + 0.5 if ib else None
    a = torch.rand(N, device=dev, generator=gen) + 0.5
    C = torch.randn(N, K, device=dev, generator=gen) if c else None
    xb = (x.float() * b[None, :]).to(torch.bfloat16) if ib else x
    term = g.double().T @ xb.double()
    ref = -0.3 * a.double()[:, None] * term + (0.9 * C.double() if c else 0)
    o = Kn.acc_gemm(g, x, b, a, C, 0.9, -0.3)
    assert float((o.double() - ref).abs().max() / (0.3 * a.double()[:, None] * term).abs().max()) < 1e-5


def test_intk_update_acc_gemm(monkeypatch):
    """the int-k head's micro-batch accumulation through kernel.acc_gemm (one dense part on the GPU) against the same
    update with acc_gemm replaced by its torch arithmetic (fp32 products of the same bf16 operands): U V^T within fp32
    summation order"""
    import ttf.kernel as Kn
    from ttf import intk_emb as IE
    N, K, T, M = 300, 128, 64, 3
    fused = Kn.acc_gemm

    def ref(g, x, ib, ia, C, cs, alpha):
        xb = (x.float() * ib[None, :]).to(torch.bfloat16) if ib is not None else x
        o = alpha * (ia.float()[:, None] * (g.float().T @ xb.float()))
        return o if C is None else cs * C + o
    res = {}
    for name, f in (("fused", fused), ("ref", ref)):
        monkeypatch.setattr(Kn, "acc_gemm", f)
        torch.manual_seed(1)
        m = IE.IntKMatrix(torch.randn(N, K), 2).to(dev); m._ts_idx = 2
        st = {}
        old = IE.ROWS[0]
        try:
            IE.ROWS[0] = 128
            g = torch.Generator(device=dev).manual_seed(7)
            for s in range(2):
                for i in range(M):
                    gy = (torch.randn(T, N, device=dev, generator=g) * 1e-2).to(torch.bfloat16)
                    x = torch.randn(T, K, device=dev, generator=g).to(torch.bfloat16)
                    m._ts_final = i == M - 1
                    c = dict(r=16, lr_ratio=1.0, theta=1e9, tau=1000.0, b2=0.95, spend=1.0, store_a="", offload=False,
                             theta_sched=None, progress=0.0, factored=16, start=0, step=10 + s, micro=M)
                    IE._intk_update(m, [("d", gy, x)], st, c)
            S = IE._Store(m, st, id(m), None, False, dev)
            res[name] = torch.cat([S.u(cc) for cc in range(S.nch)]) @ S.v().T
        finally:
            IE.ROWS[0] = old
    assert float((res["fused"] - res["ref"]).norm() / res["ref"].norm()) < 1e-3     # (2e-4: the turn and orth
    # carry the order differences)
