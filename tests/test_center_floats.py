"""--center-floats (qwen.CenteredLinear, center_float_readers): the Gated DeltaNet's float in_proj_a / in_proj_b on the
centred input their in_proj_qkv reads. Exact at a fresh start, restored on a resume of a run that had it, left alone on a
resume of one without, meta placeholders for other pipeline stages; and the point of it: the weight gradient loses the
input's common direction (Qwen3.8-27B's massive dimension after ttfconv.fold, 31 Oct 2026)."""
import torch
import torch.nn as nn

from ttf.qwen import CenteredLinear, center_float_readers, set_center_bias

K, H = 64, 4


def _model(device="cpu"):
    m = nn.Module()
    m.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        la = nn.Module()
        la.in_proj_qkv = nn.Identity()
        la.in_proj_a = nn.Linear(K, H, bias=False, device=device)
        la.in_proj_b = nn.Linear(K, H, bias=False, device=device)
        layer.linear_attn = la
        m.layers.append(layer)
    return m


def _ck(m, mu):
    ck = {k: v.clone() for k, v in m.state_dict().items()}
    for i in range(2):
        ck[f"layers.{i}.linear_attn.in_proj_qkv.in_mean"] = mu.clone()
    return ck


def test_fresh_start_is_exact():
    torch.manual_seed(0)
    m = _model(); mu = torch.randn(K) * 3
    x = mu + torch.randn(10, K) * 0.1
    want = [m.layers[i].linear_attn.in_proj_b(x) for i in range(2)]
    todo = center_float_readers(m, _ck(m, mu), fresh=True, log=lambda *_: None)
    assert len(todo) == 4 and all(isinstance(l, CenteredLinear) for l in todo)
    set_center_bias(todo)
    for i in range(2):
        torch.testing.assert_close(m.layers[i].linear_attn.in_proj_b(x), want[i], rtol=1e-5, atol=1e-4)
    sd = m.state_dict()
    for i in range(2):
        for cn in ("in_proj_a", "in_proj_b"):
            p = f"layers.{i}.linear_attn.{cn}"
            assert {f"{p}.weight", f"{p}.in_mean", f"{p}.out_bias"} <= set(sd)


def test_resume_restores_and_old_runs_unchanged():
    torch.manual_seed(1)
    m = _model(); mu = torch.randn(K)
    todo = center_float_readers(m, _ck(m, mu), fresh=True, log=lambda *_: None)
    set_center_bias(todo)
    with torch.no_grad():
        m.layers[0].linear_attn.in_proj_a.out_bias.add_(1.5)
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    m2 = _model()                     # a resume of that run: restored from its own keys, then loaded
    assert center_float_readers(m2, saved, fresh=False, log=lambda *_: None) == []
    m2.load_state_dict(saved)
    x = torch.randn(5, K)
    torch.testing.assert_close(m2.layers[0].linear_attn.in_proj_a(x), m.layers[0].linear_attn.in_proj_a(x))
    m3 = _model()                     # a resume of a run without them: left as it was
    assert center_float_readers(m3, _ck(m3, mu), fresh=False, log=lambda *_: None) == []
    assert type(m3.layers[0].linear_attn.in_proj_b) is nn.Linear


def test_meta_stages_get_placeholders():
    m = _model(device="meta"); cn = {f"layers.{i}.linear_attn.in_proj_qkv" for i in range(2)}
    assert center_float_readers(m, {}, fresh=True, cn=cn, log=lambda *_: None) == []
    lin = m.layers[1].linear_attn.in_proj_b
    assert isinstance(lin, CenteredLinear) and lin.in_mean.is_meta and lin.out_bias.is_meta
    m2 = _model(device="meta")        # not fresh and the other stage did not have them: nothing
    center_float_readers(m2, {}, fresh=False, cn=cn, log=lambda *_: None)
    assert type(m2.layers[0].linear_attn.in_proj_a) is nn.Linear


def test_gradient_loses_the_common_direction():
    """input = a large token-independent vector + small token-varying part (the folded massive dimension): the plain
    linear's weight gradient lies almost entirely along the common direction, the centred one's hardly at all"""
    torch.manual_seed(2)
    u = torch.randn(K); u = u / u.norm()
    x = 8.0 * u + torch.randn(256, K) * 0.15        # ~98% of |x|^2 in one direction, as layer 1 of the 27B
    dy = torch.randn(256, H)
    m = _model(); lin = m.layers[0].linear_attn.in_proj_b
    lin(x).backward(dy); g_plain = lin.weight.grad.clone(); lin.weight.grad = None
    todo = center_float_readers(m, _ck(m, x.mean(0)), fresh=True, log=lambda *_: None); set_center_bias(todo)
    lin(x).backward(dy); g_cent = lin.weight.grad.clone()
    along = lambda g: float((g @ u).norm() / g.norm())
    assert along(g_plain) > 0.95 and along(g_cent) < 0.2, (along(g_plain), along(g_cent))
