"""Build 27B once, sweep seq lengths to find what fits on the local GPU."""
import time, sys, torch
from ttf.config import ModelConfig
from ttf.flip import build_kernel_transformer as build_stateless_transformer
from ttf.opt8 import Adam8bit

VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
dev = "cuda"; MAXSEQ = 1024

c = ModelConfig(vocab_size=VOCAB, dim=5120, n_layers=83, n_heads=40,
                n_kv_heads=8, hidden_dim=13824, max_seq_len=MAXSEQ)
print(f"b27 {c.n_params()/1e9:.1f}B | vocab {VOCAB}", flush=True)
t0 = time.time()
m = build_stateless_transformer(c, grad_checkpoint=True, rate=2e-2)
for p in m.float_tail_parameters():
    p.data = p.data.to(torch.bfloat16)
m = m.to(dev).train()
opt = Adam8bit(m.float_tail_parameters(), lr=3e-4)
torch.cuda.synchronize()
print(f"built + on GPU {time.time()-t0:.1f}s | resident "
      f"{torch.cuda.memory_allocated()/2**30:.2f} GiB\n", flush=True)

def run(SEQ, BS=1):
    def step():
        opt.zero_grad(set_to_none=True)
        x = torch.randint(0, VOCAB, (BS, SEQ), device=dev)
        y = torch.randint(0, VOCAB, (BS, SEQ), device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); opt.step()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2): step()
    torch.cuda.synchronize()
    N = 5; t0 = time.time()
    for _ in range(N): step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"seq {SEQ:4d} bs {BS}: {dt/N*1000:6.0f} ms/step | "
          f"{BS*SEQ*N/dt:7.0f} tok/s | peak {peak:.2f} GiB", flush=True)

for SEQ in [256, 512, 768, 1024]:
    try:
        run(SEQ)
    except torch.OutOfMemoryError:
        print(f"seq {SEQ:4d}: OOM", flush=True)
        torch.cuda.empty_cache()
