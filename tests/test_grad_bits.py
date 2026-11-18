"""--grad-bits (18 Nov 2026): the b-bit stochastic quantizer of the gradient entering a ternary product"""
import pytest
import torch

from ttf.layers import grad_quant


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_grid_and_range(bits):
    torch.manual_seed(0)
    g = torch.randn(64, 256)
    qi, s, gq = grad_quant(g, bits)
    qmax = (1 << (bits - 1)) - 1
    assert qi.dtype == torch.int8 and int(qi.abs().max()) <= qmax
    assert torch.allclose(gq.float(), (qi.float() * s[:, None]).bfloat16().float())
    assert torch.allclose(s, g.abs().amax(-1) / qmax)


@pytest.mark.parametrize("bits", [2, 4])
def test_unbiased(bits):
    torch.manual_seed(0)
    g = torch.randn(8, 128)
    acc = torch.zeros_like(g)
    for _ in range(4000):
        qi, s, _ = grad_quant(g, bits)
        acc += qi.float() * s[:, None]
    m = acc / 4000
    assert (m - g).abs().max() < 0.06 * g.abs().max()


def test_group_scales():
    torch.manual_seed(0)
    g = torch.randn(4, 128); g[:, 0] = 100.0     # one outlier per row: only its group's scale is large
    qi, s, gq = grad_quant(g, 2, group=32)
    assert s.shape == (4, 4) and torch.allclose(s[:, 1:], g[:, 32:].view(4, 3, 32).abs().amax(-1))
