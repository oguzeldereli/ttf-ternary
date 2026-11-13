"""ttf.export: the .ttf container round trip, and on a GPU the spec's Arithmetic against the
training layers (bit-exact) and the exported IR against PyTorch through the reference runner (float path)."""
import json
import os

import pytest
import torch

gguf = pytest.importorskip("gguf")
from ttf.export.io import write_ttf, read_ttf   # noqa: E402


def test_io_roundtrip(tmp_path):
    sd = {"a.wpacked": torch.randint(0, 243, (7, 13), dtype=torch.uint8),
          "b.weight": torch.randn(5, 3).to(torch.bfloat16), "c.row_scale": torch.randn(11)}
    ir = {"ir_version": 1, "graphs": {}}
    p = str(tmp_path / "t.ttf")
    write_ttf(p, sd, {"a.beta": torch.tensor([0.25])}, ir, "test", {"x": 1}, "bf16")
    kv, W = read_ttf(p)
    assert kv["ttf.family"] == "test" and json.loads(kv["ttf.ir"]) == ir and kv["ttf.layout.trits"] == "rows5"
    for k, v in sd.items():
        assert W[k].dtype == v.dtype and torch.equal(W[k], v), k
    assert float(W["a.beta"][0]) == 0.25


def _tiny(beta=True):
    from ttf.config import ModelConfig
    from ttf import layers as TL
    c = ModelConfig(vocab_size=512, dim=128, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=256, max_seq_len=64)
    m = TL.build_kernel_transformer(c, grad_checkpoint=False, beta=beta).cuda()
    TL.enable_rc_scales(m)
    g = torch.Generator(device="cuda").manual_seed(0)
    for n, l in m.named_modules():
        if isinstance(l, TL.KernelTernaryLinear):
            l.row_scale.data = 1 + 0.1 * torch.randn(l.N, device="cuda", generator=g)
            l.col_scale.data = 1 + 0.1 * torch.randn(l.K, device="cuda", generator=g)
            l.enable_lowrank_mag("add", 4)
            l.mag_A.data = 0.01 * torch.randn(l.N, 4, device="cuda", generator=g)
        if type(l).__name__ == "Attention":
            l.qk_logscale = torch.nn.Parameter(0.1 * torch.randn(l.n_heads, device="cuda", generator=g))
    return m.eval()


@pytest.mark.gpu
def test_arithmetic_matches_training():
    """refdump.linear_ref (the spec's steps 1-8) equals the training layer bit for bit: bf16 / f32, fused / unfused,
    M > 16 (int_mm / fused GEMM) and M = 1 (the small-M branch)"""
    from ttf import layers as TL
    from ttf.export.refdump import linear_ref
    g = torch.Generator(device="cuda").manual_seed(1)
    l = TL.KernelTernaryLinear(256, 192, beta=True).cuda()
    l.enable_rc_scale()
    l.row_scale.data = 1 + 0.1 * torch.randn(192, device="cuda", generator=g)
    l.col_scale.data = 1 + 0.1 * torch.randn(256, device="cuda", generator=g)
    l.enable_centering(0.05 * torch.randn(256, device="cuda", generator=g), 0.02 * torch.randn(192, device="cuda", generator=g))
    old = TL.FUSED_LAYER[0]
    try:
        for D in (torch.bfloat16, torch.float32):
            ld = l.to(D)
            for fused in (False, True):
                TL.FUSED_LAYER[0] = fused
                for M in (40, 1):
                    x = torch.randn(M, 256, device="cuda", generator=g).to(D)
                    with torch.no_grad():
                        y = ld(x)
                    yr, _ = linear_ref(x, ld, D)
                    assert torch.equal(yr, y), (D, fused, M)
    finally:
        TL.FUSED_LAYER[0] = old


@pytest.mark.gpu
def test_export_float_path(tmp_path):
    """core model -> .ttf -> the reference runner: float path vs PyTorch to fp32 rounding; incremental decode = prefill"""
    from ttf.export.emit import build_ir, float_check
    m = _tiny()
    ir, derived = build_ir(m, "bit_transformer", {"name": "lm", "graphs": {"step": "step"}, "attr": {}})
    ops = {n["op"] for n in ir["graphs"]["step"]["nodes"]}
    assert ops == {"ttf.embed", "rms_norm", "ttf.linear", "reshape", "mul", "rope", "attn", "add", "glu", "ttf.head"}
    p = str(tmp_path / "tiny.ttf")
    write_ttf(p, m.state_dict(), derived, ir, "bit_transformer", {}, "bf16")
    r, i, t1 = float_check(m, p, n=10, n_pre=6)
    assert r < 1e-5 and i < 1e-5 and t1 == 1.0, (r, i, t1)
