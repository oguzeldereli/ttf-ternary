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
