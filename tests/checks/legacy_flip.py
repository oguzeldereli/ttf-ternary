"""Smoke test: ternary weights trained ONLY by predicted flips.

Verifies: weights stay in {-1,0,1}, the flip predictor receives gradient,
flips actually happen, and the loss goes down."""
import torch
from ttf.legacy.config import ModelConfig
from ttf.legacy.flip import build_flip_transformer, apply_flips, TernaryFlipLinear
from ttf.legacy.bitlinear import STATE  # unused knobs; imported for parity

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

c = ModelConfig(vocab_size=256, dim=128, n_layers=2, n_heads=4, n_kv_heads=4,
                hidden_dim=256, max_seq_len=64)
model, predictor = build_flip_transformer(c, grad_checkpoint=False,
                                          theta=8.0, e_gain=4.0, step=0.05)
model = model.to(dev).train()

bl = next(m for m in model.modules() if isinstance(m, TernaryFlipLinear))
assert set(bl.wstore.unpack().unique().tolist()) <= {-1, 0, 1}
assert len(list(bl.parameters())) == 0, "flip layer must own no nn.Parameter"

# tail = embeddings + norms + the shared flip predictor
tail = model.float_tail_parameters()
assert any(p is predictor.net[0].weight for p in tail), "predictor must be trained"
opt = torch.optim.AdamW(tail, lr=1e-3)

x = torch.randint(0, 256, (4, 64), device=dev)
y = torch.randint(0, 256, (4, 64), device=dev)

first = None
total_flips = 0
pred_grad_seen = False
for step in range(120):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    if predictor.net[0].weight.grad is not None and \
       predictor.net[0].weight.grad.abs().sum() > 0:
        pred_grad_seen = True
    opt.step()
    total_flips += apply_flips(model)   # predicted flips applied here
    if first is None:
        first = loss.item()

print(f"loss {first:.3f} -> {loss.item():.3f}")
print(f"total predicted flips: {total_flips}")
print(f"weight values still ternary: {sorted(bl.wstore.unpack().unique().tolist())}")
print(f"predictor received gradient: {pred_grad_seen}")

assert set(bl.wstore.unpack().unique().tolist()) <= {-1, 0, 1}, "weights left ternary set!"
assert pred_grad_seen, "flip predictor got no gradient"
assert total_flips > 0, "no flips happened"
assert loss.item() < first, "loss did not decrease"
print("OK: pure-ternary weights, trained by predicted flips")
