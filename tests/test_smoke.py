"""Smoke test: proves master-free ternary training actually learns, in-place,
with no full-precision weight copy of the BitLinear matrices."""
import torch
from ttf import BitTransformer, ModelConfig, STATE
from ttf.bitlinear import BitLinear

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# tiny config
c = ModelConfig(vocab_size=256, dim=128, n_layers=2, n_heads=4, n_kv_heads=4,
                hidden_dim=256, max_seq_len=64)
m = BitTransformer(c, grad_checkpoint=True).to(dev).train()

# confirm BitLinear weight is int8, not float, and has no nn.Parameter
bl = next(mod for mod in m.modules() if isinstance(mod, BitLinear))
assert bl.latent.dtype == torch.int8, bl.latent.dtype
assert len(list(bl.parameters())) == 0, "BitLinear must own no nn.Parameter"
before = bl.latent.clone()

tail = m.float_tail_parameters()
opt = torch.optim.AdamW(tail, lr=1e-3)
STATE.lr = 1e-2
STATE.updates_enabled = True
STATE.grad_scale = 1.0

# overfit one fixed batch
x = torch.randint(0, 256, (4, 64), device=dev)
y = torch.randint(0, 256, (4, 64), device=dev)
first = None
for step in range(60):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward()
    opt.step()
    if first is None:
        first = loss.item()
print(f"loss {first:.3f} -> {loss.item():.3f}")

changed = (bl.latent != before).float().mean().item()
print(f"ternary latent cells changed: {changed*100:.1f}%")
if dev == "cuda":
    print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")

assert loss.item() < first, "loss did not decrease -> not learning"
assert changed > 0, "int8 latent never updated -> hooks not firing"
print("OK: learns in place, master-free")
