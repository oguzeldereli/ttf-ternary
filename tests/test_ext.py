"""--ext (ttf/ext.py) on the CPU: plug-ins load by module name in order, `having` lists their hooks, `base` unwraps
wrappers of wrappers, `structural` keeps only the plug-ins that change the model (define build)."""
import sys
import textwrap

import pytest

from ttf import ext as X


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    (tmp_path / "pl_build.py").write_text(textwrap.dedent('''
        class P:
            def __init__(self, spec): self.spec = spec
            def build(self, model, swapped, make_linear, log): return swapped
            def fitted(self, *a): pass
        def ttf_ext(spec): return P(spec)
    '''))
    (tmp_path / "pl_end.py").write_text(textwrap.dedent('''
        class P:
            def __init__(self, spec): self.spec = spec
            def end(self, model, log): pass
        def ttf_ext(spec): return P(spec)
    '''))
    (tmp_path / "pl_bad.py").write_text("x = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    X.load([])
    for m in ("pl_build", "pl_end", "pl_bad"):
        sys.modules.pop(m, None)


def test_load_order_and_hooks(plugins):
    es = X.load(["pl_build:rank=2", "pl_end"])
    assert [e.spec for e in es] == ["rank=2", ""]
    assert len(X.having("build")) == 1 and len(X.having("end")) == 1 and X.having("weight") == []


def test_bad_plugin(plugins):
    with pytest.raises(SystemExit):
        X.load(["pl_bad"])


def test_structural(plugins):
    assert X.structural(["pl_build:rank=2", "pl_end:out=x"]) == ["pl_build:rank=2"]
    assert X.structural([]) == []


def test_base_unwraps():
    class T:   # a ternary layer
        wpacked = 1

    class W:
        def __init__(self, lin): self.lin = lin
    t = T()
    assert X.base(t) is t and X.base(W(t)) is t and X.base(W(W(t))) is t


def test_kd_sampled_weights():
    """--kd-cache-mode sample: top k exact plus S tail samples at p_rest / S each; the weights sum to 1 per position and
    their expectation over the sampling is the teacher's distribution"""
    import numpy as np
    from ttf.data import _kd_sampled
    rng = np.random.default_rng(0)
    V, k, S = 50, 5, 8
    p = rng.dirichlet(np.ones(V) * 0.3)
    top = np.argsort(-p)[:k]; tail = np.setdiff1d(np.arange(V), top); pt = p[tail] / p[tail].sum()
    est = np.zeros(V); n = 4000
    for _ in range(n):
        s = rng.choice(tail, S, p=pt)
        ids, lw = _kd_sampled(top[None].astype(np.int32), np.log(p[top])[None].astype(np.float16), s[None].astype(np.int32))
        w = np.exp(lw[0]); assert abs(w.sum() - 1) < 2e-3
        np.add.at(est, ids[0], w / n)
    assert np.abs(est - p).max() < 0.01


def test_ts_frozen_layer_skipped():
    """a plug-in's frozen layer (ts_frozen, ttfconv.adapter_ctl): ts_layer returns before reading the gradient or
    making any state"""
    import types
    import torch
    from ttf.update import ts_layer
    st = {}
    ts_layer(types.SimpleNamespace(ts_frozen=True), torch.zeros(4, 4), st, {})
    ts_layer(types.SimpleNamespace(ts_frozen=True), (torch.zeros(2, 4), torch.zeros(2, 4)), st, {})
    assert st == {}
