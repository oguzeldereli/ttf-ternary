"""--tern-adapter R (21 Nov): ternary low-rank adapters from scratch, the all-ternary replacement for the float adapter
(--adapter-rank). Every ternary linear of the decoder blocks becomes y = T x + U (D x), D [r, K] and U [N, r] ternary
layers (KernelTernaryLinear, their own row / column scales and accumulators, trained by TTF like the main layer),
r = R min(N, K) rounded to a multiple of 16 (at least 16). The up layer's row scale starts at UP_INIT (the adapter
starts as a smaller random term, not zero: an all-zero ternary layer has no nonzero density for its beta and its trits'
normalized update is scale-free anyway). Same structure as ttfconv.adapters' QTad (state-dict keys .lin, .ad.0, .ad.1)."""
import torch.nn as nn

UP_INIT = [0.5]


class TernAd(nn.Module):
    def __init__(self, lin, down, up):
        super().__init__()
        self.lin, self.ad = lin, nn.ModuleList([down, up])
        self.in_features, self.out_features = lin.K, lin.N

    def forward(self, x):
        return self.ad[1](self.ad[0](x), acc=self.lin(x))


def rank_of(N, K, R):
    return max(16, int(round(R * min(N, K) / 16)) * 16)


def enable(model, R, make_linear):
    """wrap every ternary linear inside model.layers; returns (wrapped, adapter trits)"""
    from .layers import KernelTernaryLinear
    n = t = 0
    for blk in model.layers:
        for name, m in list(blk.named_modules()):
            if not isinstance(m, KernelTernaryLinear) or ".ad." in f".{name}." or name.endswith(".lin"):
                continue
            par, _, cn = name.rpartition(".")
            r = rank_of(m.N, m.K, R)
            dev = m.wpacked.device
            q = TernAd(m, make_linear(m.K, r).to(dev), make_linear(r, m.N).to(dev))
            setattr(blk.get_submodule(par) if par else blk, cn, q)
            n += 1; t += r * (m.N + m.K)
    return n, t


def init_up_scales(model):
    for m in model.modules():
        if isinstance(m, TernAd) and m.ad[1].row_scale is not None:
            m.ad[1].row_scale.data.fill_(UP_INIT[0])
