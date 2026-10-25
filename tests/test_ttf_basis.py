"""--ttf-basis small (ttf/update.py _ts_layer_factored, ttf/train.py _reorient): the accumulator basis on the output side
of an N < K layer. CPU tests: at full rank (r = N) both sides hold the whole tally, so they give the same A and move the
same trits; the re-split of a stored pair keeps A."""
import torch

from ttf import update as UP
from ttf.kernel import pack_rows, unpack_rows
from ttf.train import _reorient


def _cfg(**kw):
    c = dict(r=24, lr_ratio=1.0, theta=0.02, tau=1000.0, b2=0.95, spend=1.0, tau_anneal=1.0, store_a="", offload=False,
             theta_sched=None, progress=0.0, factored=16, start=0, step=10, micro=1)
    c.update(kw)
    return c


class _Stub:
    """the attributes _ts_layer_factored reads of a ternary layer"""
    def __init__(self, N, K, idx, tb):
        self.N, self.K, self._ts_idx, self.track, self._ts_tb = N, K, idx, False, tb
        self.wpacked = pack_rows(torch.zeros(N, K, dtype=torch.int8))

    def _after_fire(self, n):
        pass


def _run(tb, steps, theta, N=24, K=64, T=32):
    l, st = _Stub(N, K, 3, tb), {}
    for step in range(steps):
        g = torch.Generator().manual_seed(step)
        UP._ts_layer_factored(l, (torch.randn(T, N, generator=g), torch.randn(T, K, generator=g)), st, _cfg(theta=theta))
    U, V = st[id(l)]
    return (V @ U.T if tb else U @ V.T), unpack_rows(l.wpacked, K)


def test_full_rank_sides_agree():
    """no firing: the same A on either side (r = N holds the whole tally either way)"""
    A1, _ = _run(False, 6, 1e9)
    A2, _ = _run(True, 6, 1e9)
    assert float((A1 - A2).norm() / A1.norm()) < 1e-5


def test_out_basis_spend_exact():
    """firing: the same trits move; on the output side (r = N) the spend takes theta D off A exactly, where the input
    side's basis (N of K directions) holds only D's projection D V V^T"""
    A0, _ = _run(True, 1, 1e9)
    A1, w1 = _run(False, 1, 0.02)
    A2, w2 = _run(True, 1, 0.02)
    assert int((w2 != 0).sum()) > 50 and torch.equal(w1, w2)
    assert float((A2 - (A0 - 0.02 * w2.float())).norm() / A0.norm()) < 1e-5
    assert float((A1 - (A0 - 0.02 * w1.float())).norm() / A0.norm()) > 1e-3


def test_reorient_keeps_A():
    g = torch.Generator().manual_seed(0)
    U, V = torch.randn(40, 8, generator=g), torch.linalg.qr(torch.randn(90, 8, generator=g)).Q
    U2, V2 = _reorient(U, V)
    assert U2.shape == (90, 8) and V2.shape == (40, 8)
    assert torch.allclose(V2.T @ V2, torch.eye(8), atol=1e-5)
    assert torch.allclose(V2 @ U2.T, U @ V.T, atol=1e-4)
    U3, V3 = _reorient(U2, V2)
    assert torch.allclose(U3 @ V3.T, U @ V.T, atol=1e-4)


def test_micro_out_basis_holds_every_micro_batch():
    """--ttf-accum micro on the output side: the shapes hold through the micro-batches and the basis turn, the basis
    stays orthonormal, the tally grows"""
    N, K, T, M = 24, 64, 16, 4
    l, st = _Stub(N, K, 3, True), {}
    norms = []
    for step in range(3):
        for i in range(M):
            g = torch.Generator().manual_seed(10 * step + i)
            l._ts_final = i == M - 1
            UP._ts_layer_factored(l, (torch.randn(T, N, generator=g), torch.randn(T, K, generator=g)), st,
                                  _cfg(micro=M, theta=1e9))
            U, V = st[id(l)]
            assert U.shape == (K, N) and V.shape == (N, N)
            norms.append(float(U.norm()))
    assert torch.allclose(V.T @ V, torch.eye(N), atol=1e-5)
    assert norms[-1] > norms[0]
