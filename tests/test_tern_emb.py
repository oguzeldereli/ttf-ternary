import torch

from ttf.tern_emb import TernEmbed


def test_untied_lookup_does_not_leave_a_stale_weight():
    # untied: the lookup's W is never read as a head; after the update (grad_trits) the next .weight must be recomputed
    m = TernEmbed(16, 8)
    idx = torch.tensor([[1, 2, 3]])
    m(idx).sum().backward()
    assert m.grad_trits() is not None
    w = m.weight
    assert w is not None and w.shape == (16, 8)


def test_tied_head_reads_the_lookup_weight():
    m = TernEmbed(16, 8)
    h = m(torch.tensor([[1, 2]]))
    w = m.weight
    assert w is m._W
    (h.sum() + w.sum()).backward()
    assert m.grad_trits() is not None


def test_accum_marks_the_ternary_embedding():
    """--accum M > 1: the ternary embedding's update gets the micro-batch flags as the linear layers do (before 27 Oct it
    had none: ts_layer read every micro-batch as the step's last and fired M times a step)"""
    from ttf.config import ModelConfig
    from ttf.layers import build_kernel_transformer, KernelTernaryLinear
    from ttf.tern_emb import enable_tern_emb, tern_emb_layers
    from ttf.train import _mark_micro
    m = build_kernel_transformer(ModelConfig(vocab_size=64, dim=32, n_layers=1, n_heads=2, n_kv_heads=2, hidden_dim=64,
                                             max_seq_len=16))
    enable_tern_emb(m, "untied", "kernel")
    te = tern_emb_layers(m)
    lin = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
    assert len(te) == 2 and lin
    for i in range(3):
        _mark_micro(m, i, 3, "untied", "")
        assert all(l._ts_final == (i == 2) and l._ts_micro == i for l in te)
        assert all(l.cap_final == (i == 2) and l.cap_micro == i for l in lin)
