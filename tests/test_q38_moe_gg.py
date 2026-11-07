"""moe_gg vs a per-expert loop (CUDA): exact for int8, fp32-accumulation tolerance for bf16, and the gradients"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _case(E=7, M=300, seed=0, skew=True):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(E, generator=g) ** (4 if skew else 1)
    p[2] = 0                                            # an empty expert
    e = torch.multinomial(p / p.sum(), M, replacement=True, generator=g).sort().values
    offs = torch.zeros(E + 1, dtype=torch.long)
    offs[1:] = torch.cumsum(torch.bincount(e, minlength=E), 0)
    return offs.cuda()


def _loop(a, W, trans, offs):
    out = torch.zeros(a.shape[0], W.shape[1] if trans else W.shape[2], device=a.device, dtype=torch.float64)
    o = offs.tolist()
    for e in range(len(o) - 1):
        B = W[e].double().T if trans else W[e].double()
        out[o[e]:o[e + 1]] = a[o[e]:o[e + 1]].double() @ B
    return out


@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("shape", [(96, 80), (256, 1280), (160, 16), (16, 200)])
def test_gg_int8_exact(trans, shape):
    from ttf.q38.moe_gg import Sched, gg
    R, O = shape
    offs = _case(M=517)
    E = offs.numel() - 1
    a = torch.randint(-127, 128, (517, R), dtype=torch.int8, device="cuda")
    W = torch.randint(-1, 2, (E, O, R) if trans else (E, R, O), dtype=torch.int8, device="cuda")
    c = gg(a, W, trans, Sched(offs, 517))
    assert torch.equal(c.double(), _loop(a, W, trans, offs))


@pytest.mark.parametrize("trans", [False, True])
def test_gg_bf16(trans):
    from ttf.q38.moe_gg import Sched, gg
    offs = _case(M=411)
    E = offs.numel() - 1
    a = torch.randn(411, 144, device="cuda").bfloat16()
    W = torch.randint(-1, 2, (E, 72, 144) if trans else (E, 144, 72), device="cuda").bfloat16()
    c = gg(a, W, trans, Sched(offs, 411))
    ref = _loop(a, W, trans, offs)
    assert (c.double() - ref).abs().max() <= 1e-4 * ref.abs().max()


@pytest.mark.parametrize("trans", [False, True])
def test_GG_grads(trans):
    from ttf.q38.moe_gg import Sched, GG
    offs = _case(M=333, seed=3)
    E = offs.numel() - 1
    a = torch.randn(333, 48, device="cuda", requires_grad=True)
    W = torch.randn((E, 32, 48) if trans else (E, 48, 32), device="cuda", requires_grad=True)
    c = GG.apply(a, W, trans, Sched(offs, 333))
    gc = torch.randn_like(c)
    c.backward(gc)
    a2, W2 = a.detach().double().requires_grad_(True), W.detach().double().requires_grad_(True)
    o = offs.tolist(); outs = []
    for e in range(E):
        B = W2[e].T if trans else W2[e]
        outs.append(a2[o[e]:o[e + 1]] @ B)
    torch.cat(outs).backward(gc.double())
    tol = lambda x, y: (x.double() - y).abs().max() / y.abs().max().clamp_min(1e-30)
    assert tol(c.detach(), torch.cat(outs).detach()) < 1e-2        # (tf32 dot for fp32 operands)
    assert tol(a.grad, a2.grad) < 1e-2 and tol(W.grad, W2.grad) < 1e-2


def test_evec_segsum():
    from ttf.q38.moe_gg import Sched, EVec
    offs = _case(M=401, seed=5)
    E = offs.numel() - 1
    e_of = torch.repeat_interleave(torch.arange(E, device="cuda"), offs[1:] - offs[:-1])
    v = torch.randn(E, 37, device="cuda", requires_grad=True)
    y = EVec.apply(v, e_of, Sched(offs, 401))
    g = torch.randn_like(y)
    y.backward(g)
    v2 = v.detach().clone().requires_grad_(True)
    v2[e_of].backward(g)
    assert torch.equal(y, v.detach()[e_of]) and torch.allclose(v.grad, v2.grad, atol=1e-5)


@pytest.mark.parametrize("has", [(True, True), (False, False)])
def test_pre_post(has):
    from ttf.q38.moe_gg import Sched, Pre, Post
    offs = _case(M=389, seed=7)
    E = offs.numel() - 1
    e_of = torch.repeat_interleave(torch.arange(E, device="cuda"), offs[1:] - offs[:-1])
    sch = Sched(offs, 389)
    x = torch.randn(389, 150, device="cuda", requires_grad=True)
    c = torch.randn(E, 150, device="cuda", requires_grad=True)
    mu = torch.randn(E, 150, device="cuda") if has[0] else None
    y = Pre.apply(x, c, mu, sch)
    x2, c2 = x.detach().clone().requires_grad_(True), c.detach().clone().requires_grad_(True)
    y2 = ((x2 - mu[e_of]) if mu is not None else x2) * c2[e_of]
    g = torch.randn_like(y); y.backward(g); y2.backward(g)
    assert torch.allclose(y, y2, atol=1e-5) and torch.allclose(x.grad, x2.grad, atol=1e-5) and torch.allclose(c.grad, c2.grad, atol=1e-4)
    ye = torch.randn(389, 130, device="cuda", requires_grad=True)
    o = torch.randn(389, 130, device="cuda", requires_grad=True) if has[1] else None
    sa = torch.randn(E, device="cuda", requires_grad=True) if has[1] else None
    r = torch.randn(E, 130, device="cuda", requires_grad=True)
    b = torch.randn(E, 130, device="cuda", requires_grad=True) if has[1] else None
    z = Post.apply(ye, o, sa, r, b, sch)
    L = [t for t in (ye, o, sa, r, b) if t is not None]
    L2 = [t.detach().clone().requires_grad_(True) for t in L]
    if has[1]:
        ye2, o2, sa2, r2, b2 = L2
        z2 = (ye2 + o2 * sa2[e_of][:, None]) * r2[e_of] + b2[e_of]
    else:
        ye2, r2 = L2
        z2 = ye2 * r2[e_of]
    g = torch.randn_like(z); z.backward(g); z2.backward(g)
    assert torch.allclose(z, z2, atol=1e-5)
    for t, t2 in zip(L, L2):
        assert torch.allclose(t.grad, t2.grad, atol=1e-3, rtol=1e-4), (t.shape, (t.grad - t2.grad).abs().max())


@pytest.mark.parametrize("shape", [(160, 96, 48), (1280, 2560, 384), (2560, 640, 192)])
def test_bfire_matches_ttf_fire(shape):
    """bfire over a chunk of experts == the core's ttf_fire one expert at a time: same moves, trits, D V, counts"""
    from ttf.q38.moe_gg import bfire
    from ttf.kernel import ttf_fire, pack_rows
    N, K, R = shape
    E, C = 9, 4
    g = torch.Generator(device="cuda").manual_seed(11)
    T = torch.randint(-1, 2, (E, N, K), device="cuda", dtype=torch.int8, generator=g)
    packed = torch.stack([pack_rows(T[e]) for e in range(E)])
    U = torch.randn(C, N, R, device="cuda", generator=g) * 0.05
    V = torch.randn(C, K, R, device="cuda", generator=g) * 0.05
    th = float((U[0] @ V[0].T).abs().quantile(0.999)) if N * K < 2 ** 24 else 0.07
    idx = torch.tensor([7, 2, 5, 0], device="cuda")
    p1 = packed.clone()
    dU, cnt = bfire(U, V, p1, idx, K, th)
    p2 = packed.clone()
    for j, e in enumerate(idx.tolist()):
        d2, n2, z2 = ttf_fire(U[j], V[j], p2[e], K, th)
        assert torch.equal(dU[j], d2) and int(cnt[j, 0]) == int(n2) and int(cnt[j, 1]) == int(z2)
    assert torch.equal(p1, p2) and int(cnt[:, 0].sum()) > 0
