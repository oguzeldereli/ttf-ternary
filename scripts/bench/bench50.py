import time, torch
from ttf.config import ModelConfig
from ttf.layers import build_kernel_transformer
from ttf.opt8 import Adam8bit
dev='cuda'
# ~50M: dim512 L12 hidden1408, vocab32k
c=ModelConfig(vocab_size=32000,dim=512,n_layers=12,n_heads=8,n_kv_heads=8,hidden_dim=1408,max_seq_len=512)
print(f'model {c.n_params()/1e6:.0f}M params ({c.n_ternary_params()/1e6:.0f}M ternary)',flush=True)
m=build_kernel_transformer(c,grad_checkpoint=True)
for p in m.float_tail_parameters(): p.data=p.data.to(torch.bfloat16)
m=m.to(dev).train()
opt=Adam8bit(m.float_tail_parameters(),lr=3e-4)
def step(bs,seq):
    opt.zero_grad(set_to_none=True)
    x=torch.randint(0,32000,(bs,seq),device=dev); y=torch.randint(0,32000,(bs,seq),device=dev)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        _,loss=m(x,y)
    loss.backward(); opt.step()
for bs in [16,32,48]:
    seq=512
    try:
        for _ in range(3): step(bs,seq)   # warmup+autotune
        torch.cuda.synchronize(); t=time.time(); N=10
        for _ in range(N): step(bs,seq)
        torch.cuda.synchronize(); dt=time.time()-t
        tps=bs*seq*N/dt; peak=torch.cuda.max_memory_allocated()/2**30
        eta=1e9/tps/86400
        print(f'bs{bs} seq{seq}: {tps:,.0f} tok/s | {dt/N*1000:.0f} ms/step | peak {peak:.2f} GiB | 1B tok ETA {eta:.2f} days',flush=True)
        torch.cuda.reset_peak_memory_stats()
    except torch.OutOfMemoryError:
        print(f'bs{bs}: OOM',flush=True); torch.cuda.empty_cache()
