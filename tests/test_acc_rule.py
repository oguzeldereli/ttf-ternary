"""update._acc_turn, the accumulator's one rule (CPU): every micro-batch's u enters on the step's basis, the basis turns
once from the step's whole tally (its cross term through sums), so a step's accumulator does not depend on the order of
its micro-batches (the rule before turned the basis from the last micro-batch's u x M: order-dependent); dense and
factored paths."""
import pytest
import torch

from ttf import update as UP
from ttf.kernel import pack_rows


def _cfg(**kw):
    c = dict(r=12, lr_ratio=1.0, theta=1e9, tau=1000.0, b2=0.95, spend=1.0, store_a="", offload=False,
             theta_sched=None, progress=0.0, factored=16, start=0, step=10, micro=4)
    c.update(kw)
    return c


class _Stub:
    def __init__(self, N, K):
        self.N, self.K, self._ts_idx, self.track = N, K, 3, False
        self.wpacked = pack_rows(torch.zeros(N, K, dtype=torch.int8))

    def _after_fire(self, n):
        pass


def _run(order, dense, N=40, K=24, T=16, M=4, steps=3, **kw):
    l, st = _Stub(N, K), {}
    for s in range(steps):
        g = torch.Generator().manual_seed(s)
        fs = [(torch.randn(T, N, generator=g), torch.randn(T, K, generator=g) + 0.3) for _ in range(M)]
        if s == steps - 1:
            fs = [fs[i] for i in order]
        for i, (gy, x) in enumerate(fs):
            l._ts_final = i == M - 1
            UP.ts_layer(l, (gy.T @ x) if dense else (gy, x), st, _cfg(micro=M, **kw))
    U, V = UP._load_pair(st[id(l)])
    return U @ V.T


@pytest.mark.parametrize("dense", [False, True])
def test_step_independent_of_micro_batch_order(dense):
    A0 = _run([0, 1, 2, 3], dense)
    A1 = _run([3, 1, 0, 2], dense)
    tol = 1e-5 if dense else 3e-3      # factored: its running sketches kept in bf16 between micro-batches
    assert float((A1 - A0).norm() / A0.norm()) < tol


@pytest.mark.parametrize("store", ["", "bf16"])
def test_running_sums_codec(store):
    """the running sums in the --ttf-store codec: the same accumulator as fp32 within the codec's rounding"""
    A0 = _run([0, 1, 2, 3], False)
    A1 = _run([0, 1, 2, 3], False, store_a=store)
    tol = 1e-6 if not store else 2e-2
    assert float((A1 - A0).norm() / A0.norm()) < tol


def test_acc_turn_rule():
    """_acc_turn: one subspace iteration of A' = A0 + u from U (= U0 + u V): span(Vn) = span(A'^T U), A = U V^T V_n V_n^T"""
    g = torch.Generator().manual_seed(0)
    U0, V = torch.randn(30, 6, generator=g), torch.linalg.qr(torch.randn(20, 6, generator=g)).Q
    u = torch.randn(30, 20, generator=g)
    U1 = U0 + u @ V
    U, Vn = UP._acc_turn(U1, V, u.T @ U1)
    Ap = U0 @ V.T + u
    Vr = torch.linalg.qr(Ap.T @ U1).Q
    assert torch.allclose(Vn @ Vn.T, Vr @ Vr.T, atol=1e-5)
    assert torch.allclose(U @ Vn.T, U1 @ V.T @ Vr @ Vr.T, atol=1e-4)


def test_sketch_cross():
    """_sketch_cross: (sum u)^T U from the k-row sketch Sc g, its error falls as 1 / sqrt(k) (E[Sc^T Sc] = I)"""
    g = torch.Generator().manual_seed(0)
    N, K, r = 200, 50, 4
    G, U = torch.randn(N, K, generator=g), torch.randn(N, r, generator=g)
    ia, ib = torch.rand(N, generator=g) + 0.5, torch.rand(K, generator=g) + 0.5
    ex = (-0.1 * ia[:, None] * G * ib[None, :]).T @ U
    err = []
    for k in (64, 1024):
        Sc = torch.randn(k, N, generator=g) / k ** 0.5
        err.append(float((UP._sketch_cross(U, (Sc @ G).T, Sc, ia, ib, 0.1) - ex).norm() / ex.norm()))
    assert 0.15 < err[1] / err[0] < 0.35, err      # sqrt(64 / 1024) = 0.25


def test_acc_turn_keeps_mass():
    """_acc_turn with mass: A' itself kept on the new basis, A = A' Vn Vn^T (A' = A0 + u, the turn from U = U0 + u V)"""
    g = torch.Generator().manual_seed(1)
    U0, V = torch.randn(30, 6, generator=g), torch.linalg.qr(torch.randn(20, 6, generator=g)).Q
    u = torch.randn(30, 20, generator=g)
    U1 = U0 + u @ V
    U, Vn = UP._acc_turn(U1, V, u.T @ U1, lambda W: u @ W)
    Ap = U0 @ V.T + u
    assert torch.allclose(U @ Vn.T, Ap @ Vn @ Vn.T, atol=1e-4)


def test_sketch_mass():
    """_sketch_mass: (sum u) W from the k-column sketch (sum g) Sr; its error falls as 1 / sqrt(k)"""
    g = torch.Generator().manual_seed(0)
    N, K, r = 50, 200, 4
    G, W = torch.randn(N, K, generator=g), torch.randn(K, r, generator=g)
    ia, ib = torch.rand(N, generator=g) + 0.5, torch.rand(K, generator=g) + 0.5
    ex = (-0.1 * ia[:, None] * G * ib[None, :]) @ W
    err = []
    for k in (64, 1024):
        Sr = torch.randn(K, k, generator=g) / k ** 0.5
        err.append(float((UP._sketch_mass(G @ Sr, Sr, ia, ib, 0.1)(W) - ex).norm() / ex.norm()))
    assert 0.15 < err[1] / err[0] < 0.35, err
