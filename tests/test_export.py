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
    ir, tensors = build_ir(m, "bit_transformer", {"name": "lm", "graphs": {"step": "step"}, "attr": {}}, m.state_dict())
    ops = {n["op"] for n in ir["graphs"]["step"]["nodes"]}
    assert ops == {"ttf.embed", "rms_norm", "ttf.linear", "reshape", "mul", "rope", "attn", "add", "glu", "ttf.head"}
    p = str(tmp_path / "tiny.ttf")
    write_ttf(p, tensors, {}, ir, "bit_transformer", {}, "bf16")
    r, i, t1 = float_check(m, p, n=10, n_pre=6)
    assert r < 1e-5 and i < 1e-5 and t1 == 1.0, (r, i, t1)


def test_ir_state_ops_stream():
    """delta_rule against HF's recurrent rule (plain torch), and both stateful ops: chunked calls == one call"""
    from ttf.export import ir as I
    from ttf.export.qwen35 import _ref_delta
    g = torch.Generator().manual_seed(0)
    T, Hk, Hv, dk, dv = 9, 2, 4, 8, 6
    q, k = torch.randn(T, Hk, dk, generator=g), torch.randn(T, Hk, dk, generator=g)
    v = torch.randn(T, Hv, dv, generator=g)
    gl, b = -torch.rand(T, Hv, generator=g), torch.rand(T, Hv, generator=g)
    node = {"op": "delta_rule", "attr": {"state": "rec.0", "qk_l2norm": True, "scale": dk ** -0.5}}
    full = I.op_delta_rule([q, k, v, gl, b], node, {}, {}, torch.float32)[0]
    r = Hv // Hk
    ref, _ = _ref_delta(q.repeat_interleave(r, 1)[None], k.repeat_interleave(r, 1)[None], v[None], gl[None], b[None],
                        use_qk_l2norm_in_kernel=True)
    assert torch.allclose(full, ref[0], atol=1e-6)
    st = {}
    parts = [I.op_delta_rule([t[i:j] for t in (q, k, v, gl, b)], node, {}, st, torch.float32)[0]
             for i, j in ((0, 5), (5, 6), (6, 9))]
    assert torch.allclose(torch.cat(parts), full, atol=1e-6)
    C, K = 5, 4
    W = {"c.weight": torch.randn(C, 1, K, generator=g)}
    x = torch.randn(C, 11, generator=g)
    cn = {"op": "conv1d_causal", "p": "c", "attr": {"ctx": K - 1, "groups": C, "act": "silu", "state": "conv.0"}}
    full = I.op_conv1d_causal([x], cn, W, {}, torch.float32)[0]
    st = {}
    parts = [I.op_conv1d_causal([x[:, i:j]], cn, W, st, torch.float32)[0] for i, j in ((0, 2), (2, 3), (3, 11))]
    assert torch.allclose(torch.cat(parts, -1), full, atol=1e-6)
    ref = torch.nn.functional.silu(torch.nn.functional.conv1d(x[None], W["c.weight"], padding=K - 1, groups=C)[0, :, :11])
    assert torch.allclose(full, ref, atol=1e-6)


@pytest.mark.gpu
def test_dumpcheck_on_reference_dump(tmp_path):
    """a dump in the runtime's format made from the reference's own values checks at 0 on every op"""
    import numpy as np
    from ttf.export.emit import build_ir
    from ttf.export import ir as I
    from ttf.export.dumpcheck import check_dump
    m = _tiny()
    ir, tensors = build_ir(m, "bit_transformer", {"name": "lm", "graphs": {"step": "step"}, "attr": {}}, m.state_dict())
    p = str(tmp_path / "tiny.ttf")
    write_ttf(p, tensors, {}, ir, "bit_transformer", {}, "bf16")
    kv, W = read_ttf(p)
    R = I.Runner(json.loads(kv["ttf.ir"]), W, torch.bfloat16)
    R("step", ids=torch.arange(7) * 13, pos=torch.arange(7))
    d = tmp_path / "dump"
    d.mkdir()
    idx = {}
    for k, v in R.last_values.items():
        f32 = v.is_floating_point()
        np.asarray(v.cpu().float().numpy() if f32 else v.cpu().to(torch.int32).numpy()).tofile(d / f"{k}.bin")
        idx[k] = {"type": "f32" if f32 else "i32", "ne": list(reversed(v.shape)) + [1] * (4 - v.dim())}
    json.dump(idx, open(d / "index.json", "w"))
    res = check_dump(p, str(d), log=lambda *a: None)
    assert res and all(r[0] == 0 for r in res.values()), res
