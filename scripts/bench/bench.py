"""Throughput sweep: tok/s for sizes that fit on the local GPU, to ground a
27B extrapolation. Ternary matmuls run in bf16 here (unpack then F.linear), so
throughput is compute-bound ~ 1/params, like any bf16 model."""
import time, sys, torch
from ttf.config import ModelConfig
from ttf.flip import build_stateless_transformer
from ttf.opt8 import Adam8bit

SEQ = 1024; BS = 1; dev = "cuda"
# (name, dim, layers, heads, kv, hidden)
SIZES = [
    ("340M", 1024, 24, 16, 16, 2816),
    ("730M", 1536, 24, 16, 8, 4096),
    ("1.5B", 2048, 24, 16, 8, 5632),
    ("3B",   2560, 32, 20, 4, 6912),
]

def bench(name, dim, L, h, kv, hid):
    c = ModelConfig(vocab_size=32000, dim=dim, n_layers=L, n_heads=h,
                    n_kv_heads=kv, hidden_dim=hid, max_seq_len=SEQ)
    m = build_stateless_transformer(c, grad_checkpoint=True, rate=2e-2)
    for p in m.float_tail_parameters():
        p.data = p.data.to(torch.bfloat16)
    m = m.to(dev).train()
    opt = Adam8bit(m.float_tail_parameters(), lr=3e-4)
    def step():
        opt.zero_grad(set_to_none=True)
        x = torch.randint(0, c.vocab_size, (BS, SEQ), device=dev)
        y = torch.randint(0, c.vocab_size, (BS, SEQ), device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); opt.step()
    for _ in range(2): step()
    torch.cuda.synchronize()
    N = 5; t0 = time.time()
    for _ in range(N): step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    toks = BS * SEQ * N
    print(f"{name:>5} ({c.n_params()/1e9:5.2f}B): {toks/dt:8.0f} tok/s | "
          f"{dt/N*1000:6.0f} ms/step | peak {peak:.2f} GiB", flush=True)
    del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return c.n_params(), toks / dt

pts = []
for s in SIZES:
    try:
        pts.append(bench(*s))
    except torch.OutOfMemoryError:
        print(f"{s[0]:>5}: OOM", flush=True)
        torch.cuda.empty_cache()

if len(pts) >= 2:
    # compute-bound: tok/s * params ~ const. Extrapolate to 27B.
    k = sum(n * t for n, t in pts) / len(pts)
    est = k / 27e9
    print(f"\nextrapolated 27B: ~{est:.1f} tok/s  ({27e9/ (k):.2e} s/token-ish)")
    print(f"  -> {est*3600:,.0f} tok/hour, {est*86400/1e6:.2f} M tok/day")
