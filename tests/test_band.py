"""update._band_fire (--ttf-band, CPU): on the band the tally is exact. With the whole matrix in the band and no
firing, a rank-4 accumulator plus its band holds what a full-rank one holds (after the first step, which both keep as
their own low-rank / full projection); with firing, the band stays within its cap and the moves happen."""
import torch

from ttf import update as UP
from ttf.kernel import pack_rows, unpack_rows


def _cfg(**kw):
    c = dict(r=4, lr_ratio=1.0, theta=1e9, tau=1000.0, b2=0.95, spend=1.0, store_a="", offload=False,
             theta_sched=None, progress=0.0, factored=16, start=0, step=10, micro=1, band=0.0, band_lo=0.5)
    c.update(kw)
    return c


class _Stub:
    def __init__(self, N, K):
        self.N, self.K, self._ts_idx, self.track = N, K, 3, False
        self.wpacked = pack_rows(torch.zeros(N, K, dtype=torch.int8))

    def _after_fire(self, n):
        pass


def _tally(l, st):
    A = UP.load_dense(st[id(l)], l)                     # (a pair: U V^T; the full-rank reference: dense)
    bd = st["_ts"].get(("bd", id(l)))
    if bd is not None:
        A.view(-1)[bd[0]] += bd[1]
    return A


def _run(steps, N=24, K=16, T=32, **kw):
    l, st, As = _Stub(N, K), {}, []
    for s in range(steps):
        g = torch.Generator().manual_seed(s)
        gy, x = torch.randn(T, N, generator=g), torch.randn(T, K, generator=g) + 0.3
        l._ts_final = True
        UP.ts_layer(l, (gy, x), st, _cfg(**kw))
        As.append(_tally(l, st))
    return l, st, As


def test_band_whole_matrix_is_full_rank():
    lk = 1 - 1.0 / 1000.0
    _, _, Ab = _run(4, band=1.0, band_lo=0.0)
    _, _, Af = _run(4, r=16)
    db = Ab[-1] - lk ** 3 * Ab[0]
    df = Af[-1] - lk ** 3 * Af[0]
    assert float((db - df).norm() / df.norm()) < 1e-4


def test_band_fires_within_cap():
    l, st, As = _run(6, band=0.05, band_lo=0.5, theta=0.5)
    idx, e = st["_ts"][("bd", id(l))]
    assert 0 < idx.numel() <= int(0.05 * 24 * 16) and torch.isfinite(e).all()
    assert int((unpack_rows(l.wpacked, 16) != 0).sum()) > 0


def test_hspend_identity_and_coupling():
    """--ttf-hspend: F = 0 or white inputs spend theta D (the plain spend); with correlated inputs a move also takes
    evidence from the row's correlated entries: U -= theta D (H~ V)"""
    torch.manual_seed(0)
    K, r = 16, 4
    V = torch.linalg.qr(torch.randn(K, r)).Q
    st = {}
    x = torch.randn(4096, K)
    Vs = UP._hspend_basis(st, 1, x, V, {"hspend": 1.0})
    assert float((Vs - V).norm() / V.norm()) < 0.1                     # white: C ~ I
    st = {}
    z = torch.randn(4096, 1); xc = torch.cat([z + 0.1 * torch.randn(4096, 1), z + 0.1 * torch.randn(4096, 1), torch.randn(4096, K - 2)], 1)
    Vs = UP._hspend_basis(st, 1, xc, V, {"hspend": 1.0})
    C = torch.corrcoef(xc.T)
    assert torch.allclose(Vs, C @ V, atol=0.05)
    assert torch.allclose(UP._hspend_basis({}, 1, xc, V, {"hspend": 0.0}), V)
