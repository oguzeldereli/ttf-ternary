"""1.6-bit ternary packing: 5 trits per byte (3^5 = 243 <= 256).

Used for RESIDENT weight storage during training, not just deploy. A 36B-trit
model is 7.2 GB packed instead of 36 GB as int8. Each layer unpacks to {-1,0,1}
transiently in its forward (one layer at a time under gradient checkpointing);
flips are written back into the packed buffer.
"""
from __future__ import annotations
import torch

_POW3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)


def pack5(t: torch.Tensor) -> torch.Tensor:
    """int8 {-1,0,1} [..] -> uint8 packed [ceil(numel/5)]."""
    flat = (t.reshape(-1).to(torch.int16) + 1)          # {0,1,2}
    pad = (-flat.numel()) % 5
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, 5)
    w = _POW3.to(g.device)
    return (g * w).sum(1).to(torch.uint8)               # 0..242


def unpack5(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """uint8 packed -> int8 {-1,0,1} flat tensor of length numel."""
    x = packed.to(torch.int16)
    out = torch.empty(packed.numel(), 5, dtype=torch.int8, device=packed.device)
    for i in range(5):
        out[:, i] = (x % 3).to(torch.int8)
        x //= 3
    out.sub_(1)                              # {0,1,2} -> {-1,0,1}, in place
    return out.view(-1)[:numel]


class TritStore:
    """Packed ternary buffer bound to an nn.Module (registers one uint8 buffer).

    Not an nn.Module itself; owns no parameters. Holds the packed weight plus the
    logical shape, and unpacks/repacks on demand.
    """
    def __init__(self, module, name: str, trits: torch.Tensor):
        self.module = module
        self.name = name
        self.shape = tuple(trits.shape)
        self.numel = trits.numel()
        module.register_buffer(name, pack5(trits))

    @property
    def packed(self) -> torch.Tensor:
        return getattr(self.module, self.name)

    def unpack(self) -> torch.Tensor:
        return unpack5(self.packed, self.numel).view(self.shape)

    def write(self, trits: torch.Tensor):
        self.packed.copy_(pack5(trits))
