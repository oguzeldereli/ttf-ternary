"""ttf.export.refdump: per-part teacher-forced reference dumps for the runtime's exactness tests.

The ternary linear's arithmetic: every TTF part's forward as steps 1-9. This module has
  linear_ref(x, layer, D)    the spec's arithmetic in plain torch, step by step, returning the intermediates
  dump_parts(model, run, out, D)
                             hooks every KernelTernaryLinear / QTad / IntKMatrix of any model built from ttf modules,
                             runs `run()` once (one teacher-forced forward), and per part saves the input as the
                             layer saw it and the spec's q, s, acc, t, out. It also checks the spec against the
                             training code: the spec's out must equal the module's own out bit for bit
                             (`exact` per part in the index).
Output: OUT/parts.pt = {"D", "parts": {name: {...}}, "index": [(name, kind, K, N, exact, max_abs_diff)]}.
  python -m ttf.export.refdump core [--dtype bf16]
Other models (conversions): build the model with its own code and call dump_parts(model, run, out, D).       a random-init core BitTransformer on CUDA (spec-vs-training self test)
"""
import os, sys, math
import torch

from ..layers import KernelTernaryLinear
from ..kernel import unpack_rows, trit_nnz, beta_from_nnz

DT = {"bf16": torch.bfloat16, "f32": torch.float32}


def _beta(l):
    return beta_from_nnz(trit_nnz(l.wpacked, l.K), l.N).float().reshape(()) if l.use_beta else torch.ones((), device=l.wpacked.device)


@torch.no_grad()
def linear_ref(x, l, D):
    """Arithmetic steps 1-8 for one KernelTernaryLinear; x [M, K] in D. Returns (out [M, N] in D, intermediates)."""
    x = x.to(D)
    mu = getattr(l, "in_mean", None)
    if mu is not None:                                   # 1. x = rD(x - rD(mu))
        x = x - mu.to(D)
    if l.col_scale is not None:                          # 2. x = rD(x * rD(c))
        x = x * l.col_scale.to(D)
    qp = 2 ** (l.act_bits - 1) - 1                       # 3. quantiser
    amax = x.abs().amax(-1, keepdim=True)
    a = torch.maximum(amax, torch.tensor(1e-5, dtype=D, device=x.device)).float()
    s = a * torch.tensor(1.0 / qp, dtype=torch.float32)
    q = torch.round(x.float() / s).clamp_(-qp - 1, qp)
    T = unpack_rows(l.wpacked, l.K)                      # 4. exact integer product (fp64 holds |acc| <= 127 K exactly)
    acc = (q.double() @ T.double().t()).to(torch.int64)
    beta = _beta(l)
    t = ((acc.float() * s) * beta).to(torch.bfloat16)    # 5. bf16 point, whatever D is
    y = t.to(D)
    if getattr(l, "mag_A", None) is not None:            # 6. legacy float adapter (tolerance class)
        y = y + (x @ l.mag_B.to(D)) @ l.mag_A.to(D).T
    if l.row_scale is not None:                          # 7. y = rD(y * rD(r))
        y = y * l.row_scale.to(D)
    ob = getattr(l, "out_bias", None)
    if ob is not None:                                   # 8. y = rD(y + rD(b))
        y = y + ob.to(D)
    return y, {"q": q.to(torch.int8).cpu(), "s": s.squeeze(-1).cpu(), "acc": acc.to(torch.int32).cpu(),
               "t": t.cpu(), "beta": beta.cpu()}


def _is_qtad(m):
    return type(m).__name__ == "QTad"


def _is_intk(m):
    return type(m).__name__ == "IntKMatrix"


@torch.no_grad()
def dump_parts(model, run, out, D, max_tokens=64):
    """hook every TTF part, call run() once, save per part (first call only, at most max_tokens rows)"""
    parts, index, hooks = {}, [], []
    names = {m: n for n, m in model.named_modules()}

    def lin_hook(mod, args, kwargs, y):
        n = names[mod]
        if n in parts:
            return
        x = args[0]
        acc_in = kwargs.get("acc", args[1] if len(args) > 1 else None)
        x2 = x.reshape(-1, mod.K)[:max_tokens]
        yr, mid = linear_ref(x2, mod, x.dtype)
        if acc_in is not None:                           # (QTad's up layer: out = rD(acc + up))
            yr = acc_in.reshape(-1, mod.N)[:max_tokens].to(x.dtype) + yr
        ym = y.reshape(-1, mod.N)[:max_tokens]
        d = float((yr.float() - ym.float()).abs().max())
        parts[n] = {"kind": "linear", "D": str(x.dtype), "K": mod.K, "N": mod.N, "act_bits": mod.act_bits, "x": x2.cpu(), "out": ym.cpu(),
                    "acc_in": acc_in.reshape(-1, mod.N)[:max_tokens].cpu() if acc_in is not None else None, **mid}
        index.append((n, "linear", mod.K, mod.N, bool(torch.equal(yr, ym)), d))

    def qtad_hook(mod, args, y):
        n = names[mod]
        if n not in parts:
            parts[n] = {"kind": "qtad", "x": args[0].reshape(-1, mod.in_features)[:max_tokens].cpu(),
                        "out": y.reshape(-1, mod.out_features)[:max_tokens].cpu()}
            index.append((n, "qtad", mod.in_features, mod.out_features, None, None))

    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            hooks.append(m.register_forward_hook(lin_hook, with_kwargs=True))
        elif _is_qtad(m):
            hooks.append(m.register_forward_hook(qtad_hook))
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    os.makedirs(out, exist_ok=True)
    torch.save({"D": str(D), "parts": parts, "index": index}, f"{out}/parts.pt")
    ex = [i for i in index if i[4] is not None]
    bad = [i for i in ex if not i[4]]
    print(f"{len(index)} parts dumped to {out}/parts.pt; spec == training on {len(ex) - len(bad)} / {len(ex)} TTF linears",
          flush=True)
    for i in bad[:10]:
        print("   NOT EXACT", i, flush=True)
    return index


def _core_selftest(D):
    from ..config import ModelConfig
    from .. import layers as TL
    c = ModelConfig(vocab_size=1024, dim=256, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=512, max_seq_len=128)
    m = TL.build_kernel_transformer(c, grad_checkpoint=False, beta=True).cuda()
    TL.enable_rc_scales(m)
    g = torch.Generator(device="cuda").manual_seed(0)
    for l in m.modules():                                # non-trivial scales, centring and bias on every part
        if isinstance(l, KernelTernaryLinear):
            l.row_scale.data = 1 + 0.1 * torch.randn(l.N, device="cuda", generator=g)
            l.col_scale.data = 1 + 0.1 * torch.randn(l.K, device="cuda", generator=g)
            l.enable_centering(0.05 * torch.randn(l.K, device="cuda", generator=g), 0.02 * torch.randn(l.N, device="cuda", generator=g))
    m = m.to(D).eval()
    idx = torch.randint(0, 1024, (1, 40), device="cuda", generator=g)
    for fused in (False, True):     # --fused-layer off / on (training's two forward paths)
        TL.FUSED_LAYER[0] = fused
        for M in (40, 1):           # M > 16: int_mm / fused path; M = 1: the small-M branch (decode)
            print(f"fused {fused} M {M}:", end=" ")
            dump_parts(m, lambda: m(idx[:, :M]), f"/tmp/refdump_core_{str(D)[6:]}_f{int(fused)}_M{M}", D)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "core"
    D = DT[sys.argv[sys.argv.index("--dtype") + 1]] if "--dtype" in sys.argv else torch.bfloat16
    if which == "core":
        _core_selftest(D)
