"""The fused head loss's Triton kernel (headloss.ce_kd_dz_fused, 9 Nov 2026) against the torch path (ce_kd_dz on the
same bf16 logits made fp32): loss to fp32 summation order, dz to one bf16 rounding, CE and CE + KD (a < 1 and a = 1),
padded targets, the Qwen vocabulary and a width that is not a multiple of the block."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)

from ttf import headloss as HL


@pytest.mark.parametrize("V", [248320, 5000])
@pytest.mark.parametrize("a", [None, 0.5, 1.0])
def test_fused_matches_torch(V, a):
    g = torch.Generator(device="cuda").manual_seed(V)
    n = 512
    zb = (torch.randn(n, V, device="cuda", generator=g) * 4).to(torch.bfloat16)
    tgt = torch.randint(0, V, (n,), device="cuda", generator=g)
    tgt[::7] = -1
    tlp = None
    if a is not None:
        tz = torch.randn(n, V, device="cuda", generator=g) * 4
        tlp = F.log_softmax(tz, -1).to(torch.bfloat16)
    l_ref, dz_ref = HL.ce_kd_dz(zb.float(), tgt, tlp, a or 0.0)
    l_k, dz_k = HL.ce_kd_dz_fused(zb, tgt, tlp, a or 0.0)
    assert dz_k.dtype == torch.bfloat16 and torch.isfinite(dz_k.float()).all()
    assert abs(l_k.item() - l_ref.item()) <= 1e-5 * abs(l_ref.item()) + 1e-3
    torch.testing.assert_close(dz_k.float(), dz_ref.to(torch.bfloat16).float(), rtol=1.6e-2, atol=1e-9)


def test_head_loss_chunk_fused_vs_torch():
    g = torch.Generator(device="cuda").manual_seed(0)
    n, d, V = 512, 1024, 248320
    h = (torch.randn(n, d, device="cuda", generator=g)).to(torch.bfloat16).requires_grad_()
    w = (torch.randn(V, d, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    tgt = torch.randint(0, V, (n,), device="cuda", generator=g)
    tlp = F.log_softmax(torch.randn(n, V, device="cuda", generator=g) * 4, -1).to(torch.bfloat16)
    out = {}
    for fused in (False, True):
        HL.FUSED_CEKD[0] = fused
        h.grad = None
        loss = HL.head_loss_chunk(h, w, tgt, tlp, 0.75)
        loss.backward()
        out[fused] = (loss.item(), h.grad.float().clone())
    HL.FUSED_CEKD[0] = True
    assert abs(out[True][0] - out[False][0]) <= 1e-5 * abs(out[False][0])
    torch.testing.assert_close(out[True][1], out[False][1], rtol=2e-2, atol=1e-4)
