"""hf_init.mag_svd_fit (--adapter-init svd, CPU): the float32 randomized top-R fit matches the full float64 SVD it
replaced (27 Oct: that one ran a 24 GB card out of memory in the 27B streaming conversion), for a positive-definite
covariance (Cholesky) and a singular one (eigh fallback), and leaves the global RNG as it was."""
import torch

from ttf.hf_init import mag_svd_fit, ternarize


class _Lin:
    pass


def _exact(W, T, a, c, beta, Hc, r):   # the pre-27 Oct body: full float64 SVD of G L
    cd = c.double().clamp_min(1e-12)
    Hp = cd[:, None] * Hc.double() * cd[None, :]
    Hp.diagonal().add_(1e-4 * Hp.diagonal().mean() + 1e-30)
    G = (beta / a.double().clamp_min(1e-12))[:, None] * (W.double() - a.double()[:, None] * c.double()[None, :] * T.double()) / cd[None, :]
    L, info = torch.linalg.cholesky_ex(Hp)
    if int(info) == 0:
        U, S, Vh = torch.linalg.svd(G @ L, full_matrices=False)
        B = torch.linalg.solve_triangular(L.T, Vh[:r].T, upper=True)
    else:
        lam, Q = torch.linalg.eigh(Hp)
        lam = lam.clamp_min(1e-6 * float(lam.max()))
        U, S, Vh = torch.linalg.svd(G @ ((Q * lam.sqrt()) @ Q.T), full_matrices=False)
        B = ((Q / lam.sqrt()) @ Q.T) @ Vh[:r].T
    return (U[:, :r] * S[:r]) @ B.T, float((S[:r] ** 2).sum() / (S ** 2).sum())


def _case(N, K, n_x, seed):
    g = torch.Generator().manual_seed(seed)
    # a weight with a decaying spectrum (as trained weights have), so the rounding error has a clear top part
    W = torch.randn(N, 8, generator=g) @ torch.randn(8, K, generator=g) + 0.1 * torch.randn(N, K, generator=g)
    X = torch.randn(n_x, K, generator=g) * torch.linspace(0.2, 2.0, K)
    T, a, c = ternarize(W, "als")
    return W, T, a, c, X.T @ X / n_x


def test_matches_full_svd():
    for N, K, n_x in ((96, 64, 256), (64, 96, 32)):    # (n_x < K: a singular covariance, the eigh branch)
        W, T, a, c, H = _case(N, K, n_x, seed=N)
        ref, kept_ref = _exact(W, T, a, c, 1.0, H, 4)
        lin = _Lin()
        kept = mag_svd_fit(lin, W, T, a, c, 1.0, H, 4)
        A, B = lin._mag_init
        assert A.dtype == B.dtype == torch.float32 and A.shape == (N, 4) and B.shape == (K, 4)
        err = float((A.double() @ B.double().T - ref).norm() / ref.norm())
        assert err < 1e-4, (N, K, err)
        assert abs(kept - kept_ref) < 1e-4, (kept, kept_ref)


def test_global_rng_untouched():
    W, T, a, c, H = _case(48, 40, 128, seed=3)
    torch.manual_seed(123); x0 = torch.rand(4)
    torch.manual_seed(123); mag_svd_fit(_Lin(), W, T, a, c, 1.0, H, 4); x1 = torch.rand(4)
    assert torch.equal(x0, x1)
