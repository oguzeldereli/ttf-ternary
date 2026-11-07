"""update.GRAD_HOOKS: a hook sees every ts_layer call, can replace the gradient, or skip the update (CPU)."""
import torch

from ttf import update as UP
from ttf.kernel import pack_rows


class _Stub:
    def __init__(self, N, K):
        self.N, self.K, self._ts_idx, self.track = N, K, 3, False
        self.wpacked = pack_rows(torch.zeros(N, K, dtype=torch.int8))

    def _after_fire(self, n):
        pass


def _cfg():
    return dict(r=8, lr_ratio=1.0, theta=1e9, tau=1000.0, b2=0.95, spend=1.0, store_a="", offload=False,
                factored=16, start=0, step=10, micro=1)


def _acc(st, l):
    U, V = UP._load_pair(st[id(l)])
    return U @ V.T


def _grad(N=24, K=16, seed=0):
    return torch.randn(N, K, generator=torch.Generator().manual_seed(seed))


def test_observe_replace_skip():
    seen = []
    l = _Stub(24, 16)
    try:
        UP.add_grad_hook(lambda l_, g, st, c: seen.append(g.clone()))
        st = {}
        UP.ts_layer(l, _grad(), st, _cfg())
        assert len(seen) == 1 and torch.equal(seen[0], _grad())
        ref = _acc(st, l)

        UP.GRAD_HOOKS.clear()                     # replace: the update sees another gradient (u = -g / sqrt(v) is scale-free)
        h = UP.add_grad_hook(lambda l_, g, st, c: _grad(seed=1))
        st2 = {}
        UP.ts_layer(l, _grad(), st2, _cfg())
        assert not torch.allclose(_acc(st2, l), ref)
        UP.remove_grad_hook(h)
        st3 = {}
        UP.ts_layer(l, _grad(), st3, _cfg())      # removed: back to the reference
        assert torch.allclose(_acc(st3, l), ref)

        UP.add_grad_hook(lambda l_, g, st, c: UP.SKIP)
        st4 = {}
        UP.ts_layer(l, _grad(), st4, _cfg())
        assert st4 == {}                          # skipped: no state made, nothing moved
    finally:
        UP.GRAD_HOOKS.clear()


def test_factored_gradient_reaches_hook_as_factors():
    got = []
    l = _Stub(24, 16)
    gy, x = torch.randn(8, 24), torch.randn(8, 16)
    try:
        UP.add_grad_hook(lambda l_, g, st, c: got.append(g) or UP.SKIP)
        UP.ts_layer(l, (gy, x), {}, _cfg())
        assert isinstance(got[0], tuple) and got[0][0] is gy and got[0][1] is x
    finally:
        UP.GRAD_HOOKS.clear()
