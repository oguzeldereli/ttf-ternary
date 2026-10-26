"""--intk-update factored (ttf/intk_emb.py): the int-k embedding / head trained by TTF without a dense matrix or
gradient. CPU tests: the chunked update against the ternary layers' factored update, chunking and the sparse
(embedding) form against the dense one, the head loss and the lookup against autograd, the tied matrix, the dense-path
state taken over."""
import math

import pytest
import torch
import torch.nn.functional as F

from ttf import intk_emb as IE
from ttf import update as U


def _cfg(**kw):
    c = dict(r=16, lr_ratio=1.0, theta=1e9, tau=1000.0, b2=0.95, spend=1.0, store_a="", offload=False,
             theta_sched=None, progress=0.0, factored=16, start=0, step=10, micro=1)
    c.update(kw)
    return c


class _Stub:
    """the attributes _ts_layer_factored reads of a ternary layer (no firing at theta 1e9)"""
    def __init__(self, N, K, idx):
        from ttf.kernel import pack_rows
        self.N, self.K, self._ts_idx, self.track = N, K, idx, False
        self.wpacked = pack_rows(torch.zeros(N, K, dtype=torch.int8))

    def _after_fire(self, n):
        pass


def _mat(N, K, bits=2, idx=5, s_const=True):
    torch.manual_seed(1)
    m = IE.IntKMatrix(torch.randn(N, K), bits)
    if s_const:
        m.s.data.fill_(0.1)
    m._ts_idx = idx
    return m


def _factors(T, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, N, generator=g), torch.randn(T, K, generator=g)


def _AV(state, key, m=None):
    t = state[key]
    if m is None:
        return t[0] @ t[1].T
    S = IE._Store(m, state, key, None, False, "cpu")
    return torch.cat([S.u(c) for c in range(S.nch)]) @ S.v().T


@pytest.mark.parametrize("M", [1, 4])
def test_matches_ternary_factored(M):
    """one row chunk: the int-k update's accumulator equals _ts_layer_factored's (torch branch) step for step"""
    N, K, T = 200, 64, 32
    m, l = _mat(N, K), _Stub(N, K, 5)
    sa, sb = {}, {}
    IE.ROWS[0] = 4096
    for step in range(3):
        for i in range(M):
            gy, x = _factors(T, N, K, 100 * step + i)
            c = _cfg(micro=M)
            m._ts_final = l._ts_final = i == M - 1
            U._ts_layer_factored(l, (gy, x), sa, c)
            IE._intk_update(m, [("d", gy, x)], sb, c)
    A1, A2 = _AV(sa, id(l)), _AV(sb, id(m), m)
    assert float((A1 - A2).norm() / A1.norm()) < 1e-5


@pytest.mark.parametrize("M", [1, 3])
def test_chunked_equals_one_chunk(M):
    N, K, T = 300, 64, 24
    res = []
    for rows in (4096, 64):
        IE.ROWS[0] = rows
        m = _mat(N, K, s_const=False); st = {}
        for step in range(3):
            for i in range(M):
                gy, x = _factors(T, N, K, 10 * step + i)
                m._ts_final = i == M - 1
                IE._intk_update(m, [("d", gy, x)], st, _cfg(micro=M, theta=0.05))
        res.append((m.codes.clone(), _AV(st, id(m), m)))
    IE.ROWS[0] = 8192
    assert int((res[0][0] != res[1][0]).sum()) <= 2
    assert float((res[0][1] - res[1][1]).norm() / res[0][1].norm()) < 1e-4
    assert int((res[0][0] != _mat(N, K).codes).sum()) > 0       # (theta 0.05: codes moved)


def test_sparse_equals_dense_one_hot():
    """the embedding's form (distinct rows u, their summed dL/dh G) against the same gradient as one-hot dense factors"""
    N, K = 150, 64
    IE.ROWS[0] = 64
    res = []
    for sparse in (False, True):
        m = _mat(N, K, s_const=False); st = {}
        for step in range(3):
            g = torch.Generator().manual_seed(step)
            idx = torch.randint(0, N, (40,), generator=g); G = torch.randn(40, K, generator=g)
            u, inv = torch.unique(idx, return_inverse=True)
            Gu = torch.zeros(u.numel(), K).index_add_(0, inv, G)
            part = ("s", u, Gu) if sparse else ("d", F.one_hot(idx, N).float(), G)
            IE._intk_update(m, [part], st, _cfg(theta=0.05))
        res.append((m.codes.clone(), _AV(st, id(m), m)))
    IE.ROWS[0] = 8192
    assert int((res[0][0] != res[1][0]).sum()) <= 2
    assert float((res[0][1] - res[1][1]).norm() / res[0][1].norm()) < 1e-4


@pytest.mark.parametrize("kd", [None, "lp", "topk"])
def test_head_loss_grads(kd):
    """the int-k head loss: loss, dL/dh and the group parameters' gradients against autograd through W = lo + s q, and
    the factors handed to the update give dL/dW"""
    N, K, T = 300, 128, 40
    torch.manual_seed(0)
    m = IE.IntKMatrix(torch.randn(N, K) * 0.1, 3)
    IE.ROWS[0] = 64
    h = torch.randn(T, K, requires_grad=True); tgt = torch.randint(0, N, (T,)); tgt[3] = -1
    kda = None
    if kd == "lp":
        kda = ("lp", F.log_softmax(torch.randn(T, N), -1), 0.7)
    elif kd == "topk":
        ids = torch.randint(0, N, (T, 8)); kda = ("topk", ids, F.log_softmax(torch.randn(T, 8), -1) - 0.5, 0.7, "rest")
    got = []
    m.add_part = lambda p: got.append(p)
    loss = IE.head_loss(m, h, tgt, kda, chunk=64) / 7.0
    loss.backward()
    dh, glo, gs = h.grad.clone(), m.lo.grad.clone(), m.s.grad.clone()
    # reference
    lo, s = m.lo.detach().clone().requires_grad_(True), m.s.detach().clone().requires_grad_(True)
    h2 = h.detach().clone().requires_grad_(True)
    W = (lo[..., None] + s[..., None] * m.codes.view(N, K // 64, 64).float()).reshape(N, K)
    W.retain_grad()
    if kd is None:
        ref = F.cross_entropy(F.linear(h2, W), tgt, ignore_index=-1, reduction="sum")
    elif kd == "lp":
        from ttf.model import BitTransformer
        ref = BitTransformer._loss_chunk_kd(h2, W, tgt, kda[1], 0.7)
    else:
        from ttf.qwen import _kd_topk_chunk
        ref = _kd_topk_chunk(h2, W, tgt, kda[1], kda[2], 0.7, "rest")
    (ref / 7.0).backward()
    IE.ROWS[0] = 8192
    assert abs(float(loss) - float(ref / 7.0)) < 1e-4 * abs(float(ref / 7.0))
    for a, b in ((dh, h2.grad), (glo, lo.grad), (gs, s.grad)):
        assert float((a - b).norm() / b.norm()) < 2e-2, float((a - b).norm() / b.norm())   # (dL/dlogits kept in bf16)
    k_, gy, x = got[0]
    assert k_ == "d"
    gW = gy.float().T @ x.float()
    assert float((gW - W.grad).norm() / W.grad.norm()) < 2e-2


def test_embedding_lookup_grads():
    N, K = 120, 64
    torch.manual_seed(0)
    m = IE.IntKMatrix(torch.randn(N, K), 2)
    got = []
    m.add_part = lambda p: got.append(p)
    emb = IE._Emb(m).train()
    idx = torch.randint(0, N, (3, 17))
    y = emb(idx); go = torch.randn_like(y.float())
    (y.float() * go).sum().backward()
    W = (m.lo.detach()[..., None] + m.s.detach()[..., None] * m.codes.view(N, 1, 64).float()).reshape(N, K)
    assert torch.allclose(y.float(), W[idx], atol=1e-2)
    gob = go.to(y.dtype).float()                       # (the lookup's output is bf16: its gradient arrives rounded)
    gW = torch.zeros(N, K).index_add_(0, idx.reshape(-1), gob.reshape(-1, K))
    _, u, Gu = got[0]
    gW2 = torch.zeros(N, K); gW2[u] = Gu
    assert torch.allclose(gW, gW2, atol=1e-5)
    assert torch.allclose(m.lo.grad, gW.view(N, 1, 64).sum(-1), atol=1e-4)
    assert torch.allclose(m.s.grad, (gW.view(N, 1, 64) * m.codes.view(N, 1, 64).float()).sum(-1), atol=1e-4)


def test_tied_update_takes_both_parts():
    """a tied matrix (n_parts 2): one update per micro-batch from the head's and the embedding's factors together, the
    same as one call with both parts"""
    N, K = 100, 64
    IE.ROWS[0] = 8192
    a, b = _mat(N, K), _mat(N, K)
    sa, sb = {}, {}
    a.n_parts = 2; a._ts_state, a._ts_cfg = sa, _cfg(theta=0.05)
    for step in range(3):
        gy, x = _factors(16, N, K, step)
        u = torch.tensor([3, 7, 50]); G = torch.randn(3, K)
        a.add_part(("d", gy, x)); assert ("t", id(a)) not in sa.get("_ts", {}) or sa["_ts"][("t", id(a))] == step
        a.add_part(("s", u, G))
        IE._intk_update(b, [("d", gy, x), ("s", u, G)], sb, _cfg(theta=0.05))
    assert torch.equal(a.codes, b.codes)
    assert torch.allclose(_AV(sa, id(a), a), _AV(sb, id(b), b))


def test_dense_state_taken_over():
    """a --intk-update dense checkpoint's (U, V) continues under the factored update: the accumulator is kept and the
    update count becomes micro-batches"""
    N, K = 130, 64
    m = _mat(N, K); st = {}
    U.ts_layer(m, torch.randn(N, K), st, _cfg(micro=1))
    A0 = st[id(m)][0] @ st[id(m)][1].T
    IE.ROWS[0] = 64
    S = IE._Store(m, st, id(m), None, False, "cpu")
    st["_ts"][("t", id(m))] *= 4
    A1 = torch.cat([S.u(c) for c in range(S.nch)]) @ S.v().T
    assert float((A0 - A1).norm() / A0.norm()) < 1e-6
    m._ts_final = False
    IE._intk_update(m, [("d", *_factors(8, N, K, 0))], st, _cfg(micro=4))
    IE.ROWS[0] = 8192
    assert st["_ts"][("t", id(m))] == 4 * 1 * 4 + 1 or st["_ts"][("t", id(m))] >= 5


def test_fire_scale_ratio():
    """the within-row scale ratio at firing: with s constant along every row it is 1 (moves as without it); a row whose
    second group has twice the step fires there first"""
    N, K = 4, 128
    m = _mat(N, K, s_const=True)
    m.codes.fill_(1)
    m.s.data[:, 1] = 0.2                     # group 2 (columns 64..127): twice the step of group 1
    st = {"_ts": {}}
    S = IE._Store(m, st, id(m), None, False, "cpu"); S.t = [None] * 2
    V = torch.zeros(K, 1); V[:, 0] = 1.0 / math.sqrt(K)
    Uc = torch.full((N, 1), 0.9 * math.sqrt(K))        # A = 0.9 everywhere
    IE._fire(m, S, [Uc], V, 1.0, _cfg(), st["_ts"])
    rho2 = 0.2 / math.sqrt((0.1 ** 2 + 0.2 ** 2) / 2)   # 1.26: 0.9 x 1.26 >= 1 in group 2 only
    assert rho2 * 0.9 >= 1.0
    assert int((m.codes[:, 64:] == 2).sum()) == N * 64 and int((m.codes[:, :64] == 1).sum()) == N * 64


@pytest.mark.parametrize("codec", ["fp12s", "bf16"])
def test_store_codecs(codec):
    """the per-chunk encoded store (--ttf-store): the accumulator after a few micro-batch steps close to the fp32 one"""
    N, K, T, M = 300, 64, 24, 2
    IE.ROWS[0] = 64
    res = []
    for sa in ("", codec):
        m = _mat(N, K); st = {}
        for step in range(3):
            for i in range(M):
                gy, x = _factors(T, N, K, 10 * step + i)
                m._ts_final = i == M - 1
                IE._intk_update(m, [("d", gy, x)], st, _cfg(micro=M, store_a=sa))
        S = IE._Store(m, st, id(m), U._store_codec(sa), False, "cpu")
        res.append(torch.cat([S.u(c) for c in range(S.nch)]) @ S.v().T)
    IE.ROWS[0] = 8192
    assert float((res[0] - res[1]).norm() / res[0].norm()) < 0.03


def test_head_dense_g_equals_factors():
    """the int-k head's two gradient forms (factors / dense [V, d]) give the same group gradients and accumulator (up to
    the factors' bf16 rounding: 0.5%), and with firing (a quarter of the codes move in 3 steps) nearly the same moves
    (at theta 0.05 nearly every code moves every step: the residual left is small and turned by the sketch's noise, and
    0.5% in the gradient became 12% of the moves)"""
    N, K, T = 200, 64, 48
    IE.ROWS[0] = 64
    for theta in (1e9, 1.0):
        res = []
        for d in (False, True):
            IE.DENSE_G[0] = d
            torch.manual_seed(0)
            m = IE.IntKMatrix(torch.randn(N, K) * 0.1, 3); m._ts_idx = 2
            st = {}; m._ts_state, m._ts_cfg = st, _cfg(theta=theta)
            for step in range(3):
                g = torch.Generator().manual_seed(step)
                h = torch.randn(T, K, generator=g, requires_grad=True); tgt = torch.randint(0, N, (T,), generator=g)
                (IE.head_loss(m, h, tgt, None, chunk=64) / T).backward()
            res.append((m.codes.clone(), m.lo.grad.clone(), m.s.grad.clone(), _AV(st, id(m), m)))
        if theta > 1:
            for a, b in zip(res[0][1:], res[1][1:]):
                assert float((a - b).norm() / b.norm()) < 1e-2
        else:
            assert int((res[0][0] != res[1][0]).sum()) < 0.03 * N * K
    IE.DENSE_G[0] = None; IE.ROWS[0] = 8192


def test_fire_negative_step():
    """a group whose step s is negative: evidence to raise W moves its code down (W = lo + s q)"""
    N, K = 2, 64
    m = _mat(N, K, s_const=True)
    m.codes.fill_(1); m.s.data[1, 0] = -0.1
    st = {"_ts": {}}
    S = IE._Store(m, st, id(m), None, False, "cpu"); S.t = [None] * 2
    V = torch.full((K, 1), 1.0 / math.sqrt(K))
    Uc = torch.full((N, 1), 1.5 * math.sqrt(K))       # A = +1.5: raise W everywhere
    IE._fire(m, S, [Uc], V, 1.0, _cfg(), st["_ts"])
    W = m.dequant()
    assert int((m.codes[0] == 2).sum()) == K and int((m.codes[1] == 0).sum()) == K
    assert bool((W - (m.lo.detach()[:, :1] + m.s.detach()[:, :1] * 1.0)).gt(0).all())


def test_bf16_group_parameters():
    """the 27B model keeps lo / s in bf16 (box, 25 Oct: index_add_ bf16 self / fp32 source in the lookup's backward):
    the lookup and the head loss return their gradients in the parameters' dtype, close to the fp32 ones"""
    N, K, T = 120, 64, 30
    grads = {}
    for dt in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        m = IE.IntKMatrix(torch.randn(N, K) * 0.1, 2)
        m.lo.data, m.s.data = m.lo.data.to(dt), m.s.data.to(dt)
        m.add_part = lambda p: None
        idx = torch.randint(0, N, (2, 15))
        y = IE._Emb(m).train()(idx)
        (y.float() * torch.randn(y.shape, generator=torch.Generator().manual_seed(3))).sum().backward()
        h = torch.randn(T, K, generator=torch.Generator().manual_seed(4))
        IE.head_loss(m, h, torch.randint(0, N, (T,), generator=torch.Generator().manual_seed(5)), None, chunk=16).backward()
        assert m.lo.grad.dtype == dt and m.s.grad.dtype == dt
        grads[dt] = (m.lo.grad.float(), m.s.grad.float())
    for a, b in zip(grads[torch.bfloat16], grads[torch.float32]):
        assert float((a - b).norm() / b.norm()) < 3e-2


def test_bf16_update():
    """the update with bf16 lo / s and bf16 factors (27B) runs and fires the same codes as with fp32 ones, up to ties"""
    N, K = 100, 64
    IE.ROWS[0] = 8192
    out = {}
    for dt in (torch.float32, torch.bfloat16):
        m = _mat(N, K, s_const=False); m.lo.data, m.s.data = m.lo.data.to(dt), m.s.data.to(dt)
        st = {}
        for step in range(3):
            gy, x = _factors(16, N, K, step)
            u = torch.tensor([3, 7, 50]); G = torch.randn(3, K, generator=torch.Generator().manual_seed(step))
            IE._intk_update(m, [("d", gy.to(dt), x.to(dt)), ("s", u, G)], st, _cfg(theta=0.05))
        out[dt] = m.codes.clone()
    ch = (out[torch.float32] != _mat(N, K).codes).float().mean()
    agree = (out[torch.float32] == out[torch.bfloat16]).float().mean()
    assert float(ch) > 0.01 and float(agree) > 0.97, (float(ch), float(agree))


@pytest.mark.parametrize("path", ["factored", "dense"])
def test_theta_mult(path):
    """--intk-theta-mult: inf freezes the codes (the group parameters still get their gradients); a larger theta fires
    fewer codes"""
    N, K = 100, 64
    IE.ROWS[0] = 8192
    moved = {}
    for tm in (1.0, 4.0, math.inf):
        m = _mat(N, K, s_const=False); m.theta_mult = tm; c0 = m.codes.clone(); st = {}
        for step in range(3):
            gy, x = _factors(16, N, K, step)
            if path == "factored":
                m._ts_state, m._ts_cfg = st, _cfg(theta=0.05)
                m.add_part(("d", gy, x))
            else:
                m._G = gy.T @ x
                m.step(U.ts_layer, st, _cfg(theta=0.05))
                assert m.lo.grad is not None
        moved[tm] = int((m.codes != c0).sum())
    assert moved[1.0] > moved[4.0] > 0 and moved[math.inf] == 0, moved


@pytest.mark.parametrize("path", ["factored", "dense"])
def test_moves_per_matrix(path):
    """update.ts_step logs the step's code moves per int-k matrix (intk_moves_emb / intk_moves_head); one step: each
    moved code moved once"""
    N, K = 100, 64
    IE.ROWS[0] = 8192
    st, want = {}, {}
    for idx, name in ((20_000, "emb"), (20_001, "head")):
        m = _mat(N, K, idx=idx, s_const=False); c0 = m.codes.clone()
        gy, x = _factors(16, N, K, idx)
        if path == "factored":
            m._ts_state, m._ts_cfg = st, _cfg(theta=0.05)
            m.add_part(("d", gy, x))
        else:
            m._G = gy.T @ x
            m.step(U.ts_layer, st, _cfg(theta=0.05))
        want["intk_moves_" + name] = int((m.codes != c0).sum())
    info = U.ts_step(None, st, {})
    assert all(info[k] == v for k, v in want.items()) and min(want.values()) > 0, (info, want)
    assert info["ts_frac"] == sum(want.values()) / (2 * N * K)


def test_dense_step_bf16_group_parameters():
    """--intk-update dense with bf16 lo / s (a bf16 / lean tail): step() hands their gradients in their dtype (25 Oct:
    an fp32 gradient assigned to a bf16 parameter raised)"""
    N, K = 64, 64
    m = _mat(N, K); m.lo.data, m.s.data = m.lo.data.bfloat16(), m.s.data.bfloat16()
    gy, x = _factors(8, N, K, 0)
    m._G = gy.T @ x
    m.step(U.ts_layer, {}, _cfg(theta=0.05))
    assert m.lo.grad.dtype == torch.bfloat16 and m.s.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_dequant_bf16_chunks(monkeypatch, dt):
    # the row-chunked bf16 dequantization equals the whole fp32 matrix rounded once (fp32 and bf16 group parameters)
    torch.manual_seed(0)
    m = IE.IntKMatrix(torch.randn(37, 128), 3, 64)
    m.lo.data, m.s.data = m.lo.data.to(dt), m.s.data.to(dt)
    monkeypatch.setattr(IE.IntKMatrix, "DEQ_ROWS", 8)
    ref = m.dequant().detach().to(torch.bfloat16)
    assert torch.equal(m.dequant_bf16(), ref)
    with torch.no_grad():
        assert torch.equal(m.leaf(), ref)
    W = m.leaf()
    assert W.requires_grad and W.is_leaf and torch.equal(W.detach(), ref)
