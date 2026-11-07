"""TernaryExperts / ExpertTTF against per-expert core layers (CPU, small shapes).
  python -m pytest -q tests/test_q38_moe.py"""
import torch
import pytest

from ttf.layers import KernelTernaryLinear
from ttf.kernel import pack_rows, unpack_rows
import ttf.update as TU
import ttf.layers as TL
from ttf.q38.moe import TernaryExperts, ExpertTTF, unpack3

E, K, N = 3, 40, 24
CNT = [5, 0, 7]


def _setup(seed=0, centring=True, dev="cpu"):
    torch.manual_seed(seed)
    mod = TernaryExperts(E, K, N, centring=centring, device=dev)
    cores = []
    for e in range(E):
        l = KernelTernaryLinear(K, N, beta=True).to(dev)
        l.enable_rc_scale()
        T = torch.randint(-1, 2, (N, K), dtype=torch.int8).to(dev)
        r, c = (torch.rand(N) + 0.5).to(dev), (torch.rand(K) + 0.5).to(dev)
        mu, bias = ((torch.randn(K) * 0.1).to(dev), (torch.randn(N) * 0.1).to(dev)) if centring else (None, None)
        with torch.no_grad():
            l.wpacked.copy_(pack_rows(T)); l.row_scale.copy_(r); l.col_scale.copy_(c)
        if centring:
            l.enable_centering(mu, bias)
        mod.load_expert(e, T, None, r, c, mu, bias)
        assert torch.allclose(mod.beta[e], l._cached_beta(l.wpacked).float())
        cores.append(l)
    offs = [0]
    for n in CNT:
        offs.append(offs[-1] + n)
    x = torch.randn(offs[-1], K, device=dev)
    return mod, cores, offs, x


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the core layer's forward needs CUDA")
@pytest.mark.parametrize("grouped", [False, True])
def test_forward_backward_factors(grouped):
    """padded bmm and moe_gg (int8, exact integer sums; bf16 dx as cuBLAS): both equal to the core layers"""
    from ttf.q38 import moe as moe_ttf
    moe_ttf.GROUPED[0] = grouped
    torch.backends.cuda.matmul.allow_tf32 = False
    tol = dict(atol=1e-5)          # (grouped: moe_gg int8, exact sums like the padded path)
    mod, cores, offs, x = _setup(dev="cuda")
    xm = x.clone().requires_grad_(True)
    mod.capture = True
    y = mod(xm, offs)
    gy = torch.randn_like(y)
    y.backward(gy)
    fac = mod.take_factors()
    for e in range(E):
        a, b = offs[e], offs[e + 1]
        if b == a:
            assert e not in fac
            continue
        l = cores[e]; got = {}
        l.capture = True; l.on_grad = lambda l_, g_: got.setdefault("fg", g_)
        TL.FACTORED[0] = True
        try:
            xe = x[a:b].clone().requires_grad_(True)
            ye = l(xe)
            ye.backward(gy[a:b])
        finally:
            TL.FACTORED[0] = False
        assert torch.allclose(y[a:b], ye, **tol), (e, (y[a:b] - ye).abs().max())
        assert torch.allclose(xm.grad[a:b], xe.grad, atol=1e-5)
        st = dict(atol=1e-4)
        assert torch.allclose(mod.row_scale.grad[e], l.row_scale.grad, **st)
        assert torch.allclose(mod.col_scale.grad[e], l.col_scale.grad, **st)
        gyc, xc = got["fg"]
        assert torch.allclose(fac[e][0], gyc, atol=1e-6) and torch.allclose(fac[e][1], xc, atol=1e-6)
    moe_ttf.GROUPED[0] = True


@pytest.mark.parametrize("steps", [3])
def test_update_matches_core(steps):
    mod, cores, offs, _ = _setup(1, centring=False)
    r = 8
    c = dict(r=r, lr_ratio=0.9, theta=0.3, tau=50.0, b2=0.95, spend=1.0, factored=16, micro=1, store_a="")
    st = ExpertTTF.new_state(E, N, K, r)
    upd = ExpertTTF(seeds="core")
    states = [{} for _ in range(E)]
    for e, l in enumerate(cores):
        l._ts_idx = 10 + e
    TU.FIRE_FUSED[0] = False
    moves = []
    try:
        for s in range(steps):
            fac = {}
            for e in range(E):
                n = CNT[e] if s != 1 else CNT[e] + 2        # (expert 1 gets tokens on step 1 only)
                if n == 0:
                    continue
                gy = torch.randn(n, N).to(torch.bfloat16); x = torch.randn(n, K).to(torch.bfloat16)
                fac[e] = (gy, x)
                TU._ts_layer_factored(cores[e], (gy, x), states[e], c)
            moves.append(upd.update(mod, fac, st, c, idx0=10))
            for e in range(E):
                Tm = unpack3(mod.wpacked, K)[e]
                Tc = unpack_rows(cores[e].wpacked, K)
                assert torch.equal(Tm, Tc), (s, e, int((Tm != Tc).sum()))
                assert torch.allclose(mod.beta[e], cores[e]._cached_beta(cores[e].wpacked).float())
                key = id(cores[e])
                if key in states[e]:
                    U, V = TU._load_pair(states[e][key])
                    assert torch.allclose(st["U"][e] @ st["V"][e].T, U @ V.T, atol=1e-5), (s, e)
        assert sum(moves) > 0 and moves[-1] > 0, moves      # (the firing and the spend are exercised)
    finally:
        TU.FIRE_FUSED[0] = True


@pytest.mark.parametrize("dev,orth,fire", [("cpu", "qr", "torch")] + ([("cuda", "qr", "torch"), ("cuda", "rchol", "torch"),
                                                                   ("cuda", "rchol", "kernel")] if torch.cuda.is_available() else []))
def test_update_b_matches_loop(dev, orth, fire):
    """the batched update against the per-expert loop (both seeds "layer"), an expert joining late, one skipping a
    step. Each step starts both from the loop's state (so a rounding-level difference cannot compound); a move may
    differ only where |A| is within rounding of theta."""
    import copy
    torch.manual_seed(3)
    E_, K_, N_, r = 6, 40, 24, 8
    c = dict(r=r, lr_ratio=0.9, theta=0.3, tau=50.0, b2=0.95, spend=1.0, factored=16)
    m0 = TernaryExperts(E_, K_, N_, device=dev)
    T0 = torch.randint(-1, 2, (E_, N_, K_), dtype=torch.int8)
    for e in range(E_):
        m0.load_expert(e, T0[e], None, torch.ones(N_), torch.ones(K_))
    s0 = ExpertTTF.new_state(E_, N_, K_, r, device=dev)
    upd = ExpertTTF(seeds="layer", chunk=4)
    from ttf.q38 import moe as moe_ttf
    moe_ttf.ORTH_B[0] = orth            # (the loop keeps Householder: the batched bases must give the same accumulator)
    moe_ttf.FIRE_B[0] = fire            # (kernel: the core's ttf_fire batched over a chunk, moe_gg.bfire)
    tot = 0
    for s in range(4):
        fac = {}
        for e in range(E_):
            # (counts >= r: with fewer tokens than r the basis has arbitrary extra directions, QR's choice, which the
            # spend then projects onto, so loop and batch would differ there as two QR implementations do)
            n = r + (e * 3 + s * 5) % 10 if not (e == 5 and s < 2) and not (e == 0 and s == 1) else 0
            if n:
                fac[e] = (torch.randn(n, N_, device=dev).to(torch.bfloat16), torch.randn(n, K_, device=dev).to(torch.bfloat16))
        m1, s1 = copy.deepcopy(m0), copy.deepcopy(s0)
        n0 = upd.update(m0, fac, s0, c, idx0=7, step_seed=s)
        n1 = upd.update_b(m1, fac, s1, c, idx0=7, step_seed=s)
        tot += n0
        d = int((unpack3(m0.wpacked, K_) != unpack3(m1.wpacked, K_)).sum())
        assert abs(n0 - n1) <= 2 and d <= 2, (s, n0, n1, d)
        A0 = s0["U"] @ s0["V"].transpose(1, 2); A1 = s1["U"] @ s1["V"].transpose(1, 2)
        if d == 0:
            # (bf16 factors: the inner product is rounded to bf16 before the outer one, as update._mmb, so mm vs bmm
            # accumulation order flips some of those roundings: up to ~6e-4 of max|A| on CUDA, 7e-6 with fp32 factors)
            assert float((A0 - A1).abs().max()) <= 2e-3 * float(A0.abs().max()) + 1e-5, (s, (A0 - A1).abs().max())
            assert torch.allclose(m0.beta, m1.beta)
        assert torch.equal(s0["t"].cpu(), s1["t"].cpu())
    moe_ttf.ORTH_B[0] = "qr"; moe_ttf.FIRE_B[0] = "kernel"
    assert tot > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm is CUDA only")
def test_adapter_grouped_matches_padded():
    """the grouped_mm adapter (bf16) against the padded one (fp32), values and gradients, an empty expert included"""
    from ttf.q38.moe import pack3, _Pad, _Grp
    torch.manual_seed(6)
    E_, K_, N_, ra = 4, 64, 32, 16
    m = TernaryExperts(E_, K_, N_, adapter_rank=ra, device="cuda")
    with torch.no_grad():
        m.ta_packed.copy_(pack3(torch.randint(-1, 2, (E_, N_, ra), dtype=torch.int8)).cuda())
        m.tb_packed.copy_(pack3(torch.randint(-1, 2, (E_, ra, K_), dtype=torch.int8)).cuda())
        m.ad_cb.copy_(torch.rand(E_, ra)); m.ad_sa.copy_(torch.rand(E_))
    offs = [0, 9, 9, 25, 40]
    x = torch.randn(40, K_, device="cuda")
    g = torch.randn(40, N_, device="cuda")
    outs = []
    for pd in (_Pad(offs, E_, "cuda"), _Grp(offs, E_, "cuda", 40)):
        m.capture = True
        m._ad_leaf = None
        o = m._adapter(x, pd)
        TA, TB = m._ad_leaf
        outs.append((o.float(), torch.autograd.grad(o, (TA, TB, m.ad_cb, m.ad_sa), g.to(o.dtype))))
    (o0, g0), (o1, g1) = outs
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
    assert rel(o1, o0) < 1e-2, rel(o1, o0)
    for u, v in zip(g1, g0):
        assert rel(u, v) < 2e-2, rel(u, v)


def test_adapter_padded_matches_loop():
    """the padded-bmm adapter (and its gradients into TA / TB / cb / sa) against the explicit per-expert formula"""
    from ttf.q38.moe import pack3
    torch.manual_seed(5)
    E_, K_, N_, ra = 4, 20, 12, 3
    m = TernaryExperts(E_, K_, N_, adapter_rank=ra)
    with torch.no_grad():
        m.ta_packed.copy_(pack3(torch.randint(-1, 2, (E_, N_, ra), dtype=torch.int8)))
        m.ad_cb.copy_(torch.rand(E_, ra)); m.ad_sa.copy_(torch.rand(E_))
    offs = [0, 3, 3, 8, 10]
    x = torch.randn(10, K_)
    from ttf.q38.moe import _Pad
    m.capture = True
    out = m._adapter(x, _Pad(offs, E_, "cpu"))
    TA, TB = m._ad_leaf
    ref = torch.zeros_like(out)
    for e in range(E_):
        a, b = offs[e], offs[e + 1]
        ref[a:b] = ((x[a:b] @ TB[e].T) * m.ad_cb[e]) @ TA[e].T * m.ad_sa[e]
    assert torch.allclose(out, ref, atol=1e-5)
    g = torch.randn_like(out)
    ga = torch.autograd.grad(out, (TA, TB, m.ad_cb, m.ad_sa), g, retain_graph=True)
    gr = torch.autograd.grad(ref, (TA, TB, m.ad_cb, m.ad_sa), g)
    for u, v in zip(ga, gr):
        assert torch.allclose(u, v, atol=1e-5)
