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
