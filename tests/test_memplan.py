"""--ckpt-skip auto (ttf/memplan.py): the controller's choices from simulated peaks (CPU; torch.cuda's memory
calls replaced by a model of the step's peak: base + d per kept block + the trit cache's bytes)"""
import torch

from ttf import layers as L
from ttf import memplan as MP


class _Lin(L.KernelTernaryLinear):
    def __init__(self, N, K):
        torch.nn.Module.__init__(self)
        self.N, self.K = N, K


class _Blk(torch.nn.Module):
    """a block whose training forward adds st["dblk"] to the simulated allocated bytes (its kept activations)"""
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, x):
        self.st["alloc"] += self.st.get("dblk", 0)
        return x


class _Model(torch.nn.Module):
    def __init__(self, n, st=None):
        super().__init__()
        st = st if st is not None else {}
        self.layers = torch.nn.ModuleList([_Blk(st) for _ in range(n)])
        self.lin = _Lin(1 << 14, 1 << 14)                 # trit cache: 512 MiB
        self.ckpt_skip = 0


def _run(monkeypatch, n, avail, base, d, steps=12, pp=None, d_first=None, fwd=False):
    st = {"peak": 0, "alloc": 0, "dblk": d}
    m = _Model(n, st)
    m.pp = pp
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (avail, avail))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: st["peak"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: st["alloc"])
    ck = MP.CkptAuto(m, log=lambda s: None)
    try:
        for _ in range(steps):
            dk = d if d_first is None or m.ckpt_skip != 1 else d_first     # (d_first: the step-1 measure, low)
            st["peak"] = int(base + dk * m.ckpt_skip + (2 * m.lin.N * m.lin.K if L.W8_CACHE[0] else 0))
            assert st["peak"] <= ck.budget or m.ckpt_skip <= 1     # never planned past the budget (after measuring)
            if fwd:     # the step's forward through the blocks (the hooks measure the kept ones)
                x = torch.zeros(1, requires_grad=True)
                for l_ in m.layers:
                    l_(x)
            ck.after_step()
        return m.ckpt_skip, L.W8_CACHE[0]
    finally:
        L.W8_CACHE[0] = False


def test_all_blocks_then_trit_cache(monkeypatch):
    G = 2**30
    assert _run(monkeypatch, 8, 20 * G, 4 * G, 1 * G) == (8, True)       # 4 + 8 + 0.5 fits 17.6


def test_partial_when_short(monkeypatch):
    G = 2**30
    k, w8 = _run(monkeypatch, 32, 20 * G, 4 * G, 1 * G)
    assert not w8 and 9 <= k <= 13                                      # (17.6 - 4) / 1.25 .. / 1


def test_backs_off_when_over(monkeypatch):
    G = 2**30
    k, w8 = _run(monkeypatch, 32, 12 * G, 11 * G, 1 * G)                # base above the budget: no block kept
    assert k == 0 and not w8


def test_pipe_grows_two_a_step(monkeypatch):
    G = 2**30           # --pipe: stage 0's first measure 0.6 of the real d (27B on 2 GPUs: 1.38 vs 2.33 GiB): without the cap
    k, w8 = _run(monkeypatch, 32, 40 * G, 10 * G, 2 * G, steps=20, pp={"lo": 0, "hi": 32}, d_first=1.2 * G)
    assert not w8 and 9 <= k <= 12                                      # (35.2 - 10) / 2.5 .. / 2, never past the budget


def test_kept_bytes_measured_directly(monkeypatch):
    G = 2**30           # the step-1 peak difference reads 0.3 of a block (0.8B 2 x 8, 9 Nov: 0.14 vs ~0.8 GiB): the
    k, w8 = _run(monkeypatch, 24, 16 * G, 9 * G, 0.8 * G, d_first=0.24 * G, fwd=True)   # hooks' direct measure keeps
    assert not w8 and 3 <= k <= 6                                       # the plan within the budget: (14.1 - 9) / 1 .. / 0.8
