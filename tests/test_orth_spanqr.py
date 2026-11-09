"""--ttf-orth spanqr (9 Nov 2026): the spanqr package's basis, with ranks that are not a multiple of 32 (<= 512) or of
256 (above) padded with zero columns, spans the same space as Householder QR's, orthonormal to fp32 rounding, on
well-conditioned, ill-conditioned and rank-deficient inputs at the trainer's shapes."""
import pytest
import torch

pytestmark = pytest.mark.gpu
orth = pytest.importorskip("orth")
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)

from ttf import update as U

SHAPES = [(768, 512), (2048, 256), (1024, 307), (1024, 62), (1024, 31), (1024, 32), (1024, 154), (1024, 160),
          (3584, 96), (5120, 600), (6144, 1000), (17408, 256)]   # (32, 96, 160: spanqr's own NaN widths, padded to 64)


def _x(K, r, kind, g):
    X = torch.randn(K, r, device="cuda", generator=g)
    if kind == "ill":      # condition number 1e6, as U^T U squares U's conditioning
        Qa = torch.linalg.qr(torch.randn(K, r, device="cuda", generator=g))[0]
        Qb = torch.linalg.qr(torch.randn(r, r, device="cuda", generator=g))[0]
        X = Qa @ torch.diag(torch.logspace(0, -6, r, device="cuda")) @ Qb
    elif kind == "deficient":   # a quarter of the columns zero, another quarter copies
        q = max(1, r // 4)
        X[:, :q] = 0
        X[:, q:2 * q] = X[:, 2 * q:3 * q]
    return X


@pytest.fixture
def spanqr():
    old = U.ORTH[0]; U.ORTH[0] = "spanqr"
    yield
    U.ORTH[0] = old


@pytest.mark.parametrize("K,r", SHAPES)
@pytest.mark.parametrize("kind", ["rand", "ill", "deficient"])
def test_span_and_orthonormal(spanqr, K, r, kind):
    g = torch.Generator(device="cuda").manual_seed(K + r)
    X = _x(K, r, kind, g)
    Q = U._orth_(X)
    assert Q.shape == (K, r) and torch.isfinite(Q).all()
    I = torch.eye(r, device="cuda")
    assert (Q.T @ Q - I).abs().max().item() < 1e-4
    # X lies in span(Q): the residual of the projection, relative to X
    res = (X - Q @ (Q.T @ X)).norm() / X.norm().clamp_min(1e-30)
    assert res.item() < (1e-4 if kind != "ill" else 1e-3)
    if kind == "rand":   # full rank: the same space as Householder's
        Qh = torch.linalg.qr(X)[0]
        assert (Q @ (Q.T @ Qh) - Qh).norm().item() / r ** 0.5 < 1e-4


def test_cpu_falls_back_to_householder(spanqr):
    X = torch.randn(256, 31)
    assert torch.equal(U._orth_(X), torch.linalg.qr(X)[0])
