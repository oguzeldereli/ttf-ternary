"""Diagnostic: does per-layer training transient scale with depth?
Same 27B layer geometry, vary n_layers, measure peak VRAM for 1 fwd+bwd."""
import torch
from ttf.config import ModelConfig
from ttf.layers import build_stateless_transformer
from ttf.opt8 import Adam8bit

dev = "cuda"; SEQ = 1024
for L in [4, 8, 16, 24]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    c = ModelConfig(vocab_size=32000, dim=5120, n_layers=L, n_heads=40,
                    n_kv_heads=8, hidden_dim=13824, max_seq_len=SEQ)
    m = build_stateless_transformer(c, grad_checkpoint=True, rate=2e-2)
    for p in m.float_tail_parameters():
        p.data = p.data.to(torch.bfloat16)
    m = m.to(dev).train()
    opt = Adam8bit(m.float_tail_parameters(), lr=3e-4)
    resident = torch.cuda.memory_allocated() / 2**30
    x = torch.randint(0, 32000, (1, SEQ), device=dev)
    y = torch.randint(0, 32000, (1, SEQ), device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"L={L:2d} ({c.n_params()/1e9:4.1f}B): resident {resident:.2f} GiB | "
          f"step peak {peak:.2f} GiB | transient {peak-resident:.2f} GiB", flush=True)
    del m, opt
