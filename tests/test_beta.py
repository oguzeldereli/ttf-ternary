"""beta = 1/sqrt(K*rho) on the kernel path: matches a dense reference (fwd, grad_x),
restores unit output scale, and un-collapses attention on a fresh model."""
import math, torch
from ttf.kernel import pack_rows, unpack_rows, trit_beta
from ttf.flip import KernelTernaryLinear, build_kernel_transformer, _act_quant_plain
from ttf.config import ModelConfig
from ttf import model as M

torch.manual_seed(0)
dev = "cuda"
for K, N in [(512, 1408), (1408, 512), (768, 777)]:        # 777/768: exercise padding
    lay = KernelTernaryLinear(K, N, beta=True, rate=0.0).to(dev)
    w = unpack_rows(lay.wpacked, K).float()
    rho = (w != 0).float().mean()
    b = trit_beta(lay.wpacked, K)
    assert abs(b.item() - 1 / math.sqrt(K * rho.item())) < 1e-6, (b, rho)
    x = torch.randn(3, 50, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    y = lay(x)
    xd = x.detach().float().requires_grad_(True)
    # dense reference: the layer's own bf16 act-quant (STE), then beta * x@W^T in fp32
    xq = xd + (_act_quant_plain(x.detach(), 8).float() - xd).detach()
    bb = b.to(torch.bfloat16).float()          # the layer multiplies by beta in bf16
    yd = bb * (xq @ w.T)
    # tolerance: the kernel itself matches bf16 F.linear exactly; beta adds a second
    # bf16 rounding of the product, so allow ~2 bf16 ulp worst-case, tight in norm.
    assert (y.float() - yd).abs().max() / yd.abs().max() < 1e-2
    assert (y.float() - yd).norm() / yd.norm() < 3e-3
    gy = torch.randn_like(yd)
    y.backward(gy.to(y.dtype)); yd.backward(gy)
    rel = (x.grad.float() - xd.grad).abs().max() / xd.grad.abs().max()
    assert rel < 1e-2, rel
    ratio = (y.float().pow(2).mean().sqrt() / x.float().pow(2).mean().sqrt()).item()
    print(f"K={K} N={N}: rho {rho:.3f} beta {b.item():.4f} | fwd/grad match | rms out/in {ratio:.2f}")
    assert 0.5 < ratio < 2.0

c = ModelConfig(vocab_size=32000, dim=768, n_layers=4, n_heads=12, n_kv_heads=12,
                hidden_dim=2048, max_seq_len=256)
for beta in (False, True):
    m = build_kernel_transformer(c, grad_checkpoint=False, beta=beta).to(dev).eval()
    ents = []
    orig = M.Attention.forward
    def fwd(self, x, fc):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim)
        q, k = M.apply_rope(q, k, fc)
        lg = (q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        lg = lg.masked_fill(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), -float("inf"))
        pr = lg.softmax(-1)
        ents.append(-(pr * pr.clamp_min(1e-30).log()).sum(-1)[..., 64:].mean().item())
        return orig(self, x, fc)
    M.Attention.forward = fwd
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(torch.randint(0, 32000, (2, 256), device=dev))
    M.Attention.forward = orig
    print(f"fresh 4-layer model beta={beta}: attn entropy per layer {[round(e, 2) for e in ents]} (uniform over ~160 ctx ~ 5.1)")
    if beta:
        assert min(ents) > 1.0
print("OK: beta kernel path")
