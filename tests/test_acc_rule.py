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


def test_one_batch_step_is_the_turn():
    """--accum 1 (dense): a step is exactly one subspace iteration of lk A + u from the stored (U, V), the whole tally kept
    on the new basis (v stepped with this g first)"""
    N, K, r = 40, 24, 6
    l, st = _Stub(N, K), {}
    c = _cfg(micro=1, r=r)
    g = torch.Generator().manual_seed(5)
    G = [torch.randn(N, K, generator=g) for _ in range(3)]
    for i in range(2):
        UP.ts_layer(l, G[i], st, c)
    U0, V0 = (x.clone() for x in UP._load_pair(st[id(l)]))
    R0, C0 = (x.clone() for x in st["_ts"][("v", id(l))])
    UP.ts_layer(l, G[2], st, c)
    b2, t = c["b2"], 3
    R, C = b2 * R0 + (1 - b2) * G[2].square().mean(1), b2 * C0 + (1 - b2) * G[2].square().mean(0)
    v = R[:, None] * C[None, :] / R.mean() / (1 - b2 ** t)
    u = -G[2] / (v.sqrt() + 1e-8) * c["lr_ratio"]
    Ul = U0 * (1 - c["lr_ratio"] / c["tau"])
    Vn = torch.linalg.qr(V0 @ (Ul.T @ Ul) + u.T @ Ul).Q
    A_ref = (Ul @ V0.T + u) @ Vn @ Vn.T
    U1, V1 = UP._load_pair(st[id(l)])
    assert float((U1 @ V1.T - A_ref).norm() / A_ref.norm()) < 1e-5


def _run_idx(dense, skip=(), M=4, steps=3, idx=True):
    """_run with the trainer's micro-batch index and step (update._micro_t); skip: (step, micro) calls left out"""
    N, K, T = 40, 24, 16
    l, st = _Stub(N, K), {}
    ts = []
    for s in range(steps):
        g = torch.Generator().manual_seed(s)
        fs = [(torch.randn(T, N, generator=g), torch.randn(T, K, generator=g) + 0.3) for _ in range(M)]
        for i, (gy, x) in enumerate(fs):
            if (s, i) in skip:
                continue
            l._ts_final = i == M - 1
            if idx:
                l._ts_micro = i
            UP.ts_layer(l, (gy.T @ x) if dense else (gy, x), st, _cfg(micro=M, step=s))
            ts.append(st["_ts"][("t", id(l))])
    U, V = UP._load_pair(st[id(l)])
    return U @ V.T, ts


@pytest.mark.parametrize("dense", [False, True])
def test_micro_index_without_gaps_changes_nothing(dense):
    A0, t0 = _run_idx(dense, idx=False)
    A1, t1 = _run_idx(dense)
    assert t0 == t1 == list(range(1, 13)) and torch.equal(A0, A1)


@pytest.mark.parametrize("dense", [False, True])
def test_micro_index_keeps_the_count_aligned(dense):
    """an --moe expert without tokens in some micro-batches (here step 1's micro-batches 1 and 3, its last): every call
    keeps its place in the step (the leak on the step's first call, the step's sketch seed), the next step starts
    clean (step 1's running v sums dropped)"""
    A, ts = _run_idx(dense, skip={(1, 1), (1, 3)})
    assert ts == [1, 2, 3, 4, 5, 7, 9, 10, 11, 12]
    assert torch.isfinite(A).all()
