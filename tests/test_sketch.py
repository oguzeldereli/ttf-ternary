"""--dw-sketch-kind / --dw-sketch-warm: the row sketch keeps E[Sc^T Sc] = I, is exact on the warm basis, and the default
draws the same numbers as before the options."""
import types

import torch

from ttf.update import _row_sketch, _orth


def _mean_sts(N, k, c, Q=None, n=400):
    l = types.SimpleNamespace(N=N)
    acc = torch.zeros(N, N)
    for s in range(n):
        g = torch.Generator().manual_seed(s)
        S = _row_sketch(l, torch.device("cpu"), g, k, c, Q)
        assert S.shape == (k, N)
        acc += S.T @ S
    return acc / n


def test_gauss_default_draw_unchanged():
    l = types.SimpleNamespace(N=48)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(_row_sketch(l, torch.device("cpu"), g1, 16, {}), torch.randn(16, 48, generator=g2) / 16 ** 0.5)


def test_unbiased_all_kinds():
    N, k = 32, 8
    Q = _orth(torch.randn(N, 3, generator=torch.Generator().manual_seed(9)))
    for c, q in (({}, None), ({"sketch_kind": "count"}, None), ({}, Q), ({"sketch_kind": "count"}, Q)):
        E = _mean_sts(N, k, c, q)
        assert (E - torch.eye(N)).abs().max() < 0.25, (c, q is not None)
        assert (torch.diagonal(E) - 1).abs().mean() < 0.05


def test_warm_exact_on_basis():
    N, k = 40, 10
    Q = _orth(torch.randn(N, 4, generator=torch.Generator().manual_seed(1)))
    S = _row_sketch(types.SimpleNamespace(N=N), torch.device("cpu"), torch.Generator().manual_seed(2), k, {}, Q)
    g = torch.randn(N, 7); W = Q @ torch.randn(4, 5)            # W in span(Q): g^T Sc^T Sc W = g^T W exactly
    assert torch.allclose(g.T @ S.T @ (S @ W), g.T @ W, atol=1e-4)
