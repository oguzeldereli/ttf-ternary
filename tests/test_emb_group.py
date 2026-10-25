import torch
import torch.nn as nn

from ttf.train import _split_emb_group


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(4))
        self.other = nn.Linear(4, 4, bias=False)


def _groups(m, wd):
    return [{"params": [m.embed_tokens.weight, m.lm_head.weight, m.other.weight], "weight_decay": wd},
            {"params": [m.norm.weight], "weight_decay": 0.0}]


def test_embedding_and_head_get_their_own_group_whatever_the_weight_decay():
    for wd in (0.1, 0.0):      # (before 26 Oct, --weight-decay 0 left them in place: --emb-lr-mult had no effect)
        m = _M()
        g = _split_emb_group(m, _groups(m, wd), 0.0, 0.05)
        assert len(g) == 3
        assert g[0]["params"] == [m.other.weight] and g[1]["params"] == [m.norm.weight]
        assert g[2]["params"] == [m.embed_tokens.weight, m.lm_head.weight]
        assert g[2]["weight_decay"] == 0.0 and g[2]["emb_mult"] == 0.05


def test_scales_group_is_left_alone():
    m = _M()
    gs = [{"params": [m.embed_tokens.weight], "weight_decay": 0.0, "rc_lr": 1e-3}]
    assert _split_emb_group(m, gs, 0.0, 0.05) == gs and len(gs) == 1
