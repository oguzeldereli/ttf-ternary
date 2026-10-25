import torch

import ttf.opt8 as opt8


def _reference(ps, grads, lr, wd, b1=0.9, b2=0.95, eps=1e-8):
    # the unchunked step (the original implementation)
    st = [(torch.zeros_like(p, dtype=torch.bfloat16), torch.zeros_like(p, dtype=torch.bfloat16)) for p in ps]
    for k, gs in enumerate(grads, 1):
        for p, g, (ms, vs) in zip(ps, gs, st):
            g = g.float()
            m = ms.float().mul_(b1).add_(g, alpha=1 - b1)
            v = vs.float().mul_(b2).addcmul_(g, g, value=1 - b2)
            ms.copy_(m); vs.copy_(v)
            denom = (v / (1 - b2 ** k)).sqrt_().add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(m / (1 - b1 ** k), denom, value=-lr)


def test_chunked_step_matches_the_whole_tensor_step(monkeypatch):
    monkeypatch.setattr(opt8, "CHUNK", 777)
    for dt in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        init = [torch.randn(1000, 37).to(dt), torch.randn(50, 20).to(dt).T]   # (the second: not contiguous)
        grads = [[torch.randn_like(p) for p in init] for _ in range(5)]
        ref = [p.clone() for p in init]
        _reference(ref, grads, 1e-2, 0.1)
        ps = [torch.nn.Parameter(p.clone()) for p in init]
        opt = opt8.Adam8bit(ps, lr=1e-2, weight_decay=0.1)
        for gs in grads:
            for p, g in zip(ps, gs):
                p.grad = g
            opt.step()
        assert all(torch.equal(p.detach(), r) for p, r in zip(ps, ref))
