"""prof_q38.py: where a Flash-Next training step's time goes, on the first NL layers only (on an RTX PRO 6000 the
step was ~600 s, forward ~9 s per 2048-token window). Loads layers 0 .. NL-1 (real weights + the conversion), runs one
real window through them, and times with CUDA syncs:
  forward per layer and per sub-module (hyper-connection mixes, PLE, attention / DeltaNet, router, routed gu / dn and
  their adapters, shared expert), the routing skew (largest expert count / mean: the padding factor), the backward
  recompute per layer, ExpertTTF.update_b per module split into its parts, AdapterTTF.update, AdamW, and the layer /
  state moves; then a torch.profiler table of the top ops of one layer's forward + backward.
  python scripts/prof/prof_q38.py --model DIR --conv DIR --data TOKENS.bin --data-u32 [--layers 4] [--micro 2] [--rank-ratio 0.3]
Small: ~15 GB GPU, ~20 GB host; runs beside a training run."""
import argparse, os, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38 import train as QT
from ttf.q38 import moe as MT

T_ = defaultdict(float)


def sync():
    torch.cuda.synchronize()


class Tm:
    def __init__(self, k):
        self.k = k

    def __enter__(self):
        sync(); self.t = time.time()

    def __exit__(self, *a):
        sync(); T_[self.k] += time.time() - self.t


def hook_times(L, tag):
    """CUDA-synced forward times of L's sub-modules (pre / post hooks)"""
    hs = []
    names = {"attn_hyper_connection": "hc_attn", "mlp_hyper_connection": "hc_mlp", "ple": "ple",
             "linear_attn": "deltanet", "self_attn": "attention", "mlp.gate": "router", "mlp.gu": "gu",
             "mlp.dn": "dn", "mlp.shared_expert": "shared", "mlp.shared_expert_gate": "shared_gate"}
    for path, short in names.items():
        mod = QT._get(L, path)
        if mod is None:
            continue
        st = {}
        def pre(m, a, st=st):
            sync(); st["t"] = time.time()
        def post(m, a, o, st=st, k=f"{tag}.{short}"):
            sync(); T_[k] += time.time() - st["t"]
        hs += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--conv"); ap.add_argument("--data"); ap.add_argument("--data-u32", action="store_true")
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--rank-ratio", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--sketch", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda"
    cfg = QM.text_config(a.model)
    m = QM.build(cfg, layers=a.layers, device="meta")
    rd = QM.Reader(a.model)
    QM.load_globals(m, rd, "cpu")
    c0 = torch.load(os.path.join(a.conv, "L0.pt"), weights_only=False, map_location="cpu")
    ra = int(next(iter(c0.values()))["ad_cb"].shape[1]); del c0
    for i, L in enumerate(m.layers):
        QM.load_layer(L, i, rd, "cpu", experts=False)
        QT.swap_ternary(L, adapter_rank=ra)
        for n, sd in torch.load(os.path.join(a.conv, f"L{i}.pt"), weights_only=False).items():
            QT._get(L, n).load_state_dict(sd)
        if L.ple is not None:
            L.ple.ple_embedding.table, L.ple.ple_embedding.scale = QM.ngram_table_memmap(a.model, i)
    m.embed_tokens.to(dev); m.rotary.to(dev)
    data = np.memmap(a.data, dtype=np.uint32 if a.data_u32 else np.uint16, mode="r")
    ids = [torch.from_numpy(data[j * 100_003: j * 100_003 + a.seq].astype(np.int64))[None].to(dev) for j in range(a.micro)]
    pe = m.rope(a.seq, dev)
    # routing skew per layer (the padding factor of the batched expert GEMMs)
    skew = {}
    def cap_hook(L, i):
        def f(xs, offsets, order, w):
            c = (offsets[1:] - offsets[:-1]).float()
            skew[i] = (c.max().item(), c.mean().item(), (c == 0).sum().item())
        L.mlp.capture = f; L.mlp.capture_dn = lambda *x: None
    # ---- forward (no grad), per layer and per sub-module
    hs = [m.embed(x) for x in ids]
    H = []
    for i, L in enumerate(m.layers):
        with Tm("move.layer_on"):
            L.to(dev)
        cap_hook(L, i)
        hk = hook_times(L, f"fwd.L{i}")
        H.append([h.clone() for h in hs])
        with torch.no_grad(), Tm(f"fwd.L{i}.total"):
            hs = [L(h, pe, x) for h, x in zip(hs, ids)]
        for h in hk:
            h.remove()
        L.mlp.capture = None
    # ---- backward recompute + updates, per layer
    upd, aupd, opt = MT.ExpertTTF(seeds="layer", chunk=a.chunk), MT.AdapterTTF(), QT.AdamW()
    c = dict(lr_ratio=1.0, theta=16.0, tau=1000.0, b2=0.95, spend=1.0, factored=a.sketch)
    gs = [torch.randn_like(h) * 1e-3 for h in hs]
    for i in reversed(range(len(m.layers))):
        L = m.layers[i]
        tms = QT.tern_modules(L)
        for _, t in tms:
            t.capture = True
        hk = hook_times(L, f"bwd_recompute.L{i}")
        for mb in range(a.micro):
            h = H[i][mb].requires_grad_(True)
            with Tm(f"bwd.L{i}.recompute"):
                y = L(h, pe, ids[mb])
            with Tm(f"bwd.L{i}.backward"):
                y.backward(gs[mb])
            gs[mb] = h.grad
        for h_ in hk:
            h_.remove()
        for j, (n, t) in enumerate(tms):
            t.capture = False
            with Tm(f"upd.L{i}.take_grouped"):
                fac = t.take_grouped()
            r = max(1, int(a.rank_ratio * min(t.N, t.K)))
            with Tm(f"upd.L{i}.state_alloc"):
                st = MT.ExpertTTF.new_state(t.E, t.N, t.K, r, device=dev)
            with Tm(f"upd.L{i}.update_b.{'experts' if t.E > 1 else 'dense'}"):
                upd.update_b(t, fac, st, c, idx0=1000 * i + j, step_seed=0)
            with Tm(f"upd.L{i}.update_b2.{'experts' if t.E > 1 else 'dense'}"):      # (t > 1: the turn, not the init)
                fac2 = (fac[0], fac[1], fac[2]) if fac is not None else None
                upd.update_b(t, fac2, st, c, idx0=1000 * i + j, step_seed=1)
            if t.ra:
                with Tm(f"upd.L{i}.adapter"):
                    ast = MT.AdapterTTF.new_state(t.E, t.N, t.K, t.ra, device=dev)
                    aupd.update(t, ast, c)
            del st
        with Tm(f"upd.L{i}.adamw"):
            opt.step(i, QT.layer_floats(L), 1e-5)
    # ---- the update's parts on the largest module (gu of the last layer): QR vs the rest
    L = m.layers[-1]; t = L.mlp.gu
    r = max(1, int(a.rank_ratio * min(t.N, t.K)))
    X = torch.randn(a.chunk, t.K, r, device=dev)
    with Tm("parts.qr_batched_chunk"):
        torch.linalg.qr(X)
    A = torch.randn(a.chunk, t.N, r, device=dev) @ torch.randn(a.chunk, r, t.K, device=dev)
    with Tm("parts.UVt_chunk"):
        A = torch.randn(a.chunk, t.N, r, device=dev) @ torch.randn(a.chunk, r, t.K, device=dev)
    with Tm("parts.unpack3_chunk"):
        MT.unpack3(t.wpacked[:a.chunk], t.K)
    with Tm("parts.pack3_chunk"):
        MT.pack3(torch.zeros(a.chunk, t.N, t.K, dtype=torch.int8, device=dev))
    with Tm("parts.w_float_layer"):
        t.invalidate(); t.w_float(torch.bfloat16)
    # ---- report
    print("routing (max count, mean count, experts with no token) per layer:", {k: tuple(round(x, 1) for x in v) for k, v in skew.items()})
    tot = defaultdict(float)
    for k, v in sorted(T_.items()):
        print(f"{k:45s} {v:8.3f} s")
    print(f"(per window: the forward times above are for {a.micro} windows)")
    # ---- torch.profiler: one layer forward + backward (the last layer)
    from torch.profiler import profile, ProfilerActivity
    L = m.layers[-1]
    for _, t in QT.tern_modules(L):
        t.capture = True
    h = H[-1][0].detach().requires_grad_(True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        y = L(h, pe, ids[0]); y.backward(torch.randn_like(y) * 1e-3); sync()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
