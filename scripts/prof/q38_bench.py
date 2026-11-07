"""q38_bench.py:
a Qwen3.8-Flash-Next training step on the first NL layers at the real shapes, with random weights, through ttf.q38.train's
Trainer (the run's own step: layer-major forward, recompute + backward, ExpertTTF / AdapterTTF updates, AdamW, the TTF
state on the host), so a lab GPU (3090 Ti) can be timed and profiled without the checkpoint or the KD cache.

Random parts (timing does not depend on the values, except routing skew; reported): trits uniform over {-1, 0, 1}
(the packed bytes uniform on 0..242), scales 0.02, float weights randn / sqrt(fan in), the PLE n-gram table random
with --ple-rows rows (the real one is ~320 M rows on disk, read on demand), KD targets random top-64 + 64 tail.

  python scripts/prof/q38_bench.py --config CONFIG.json --layers 4 [--steps 4] [--profile] [run flags as ttf.q38.train:
      --prefetch --expert-orth rchol --resident --wbatch 8 --compile-hc --micro 16 --seq 2048 ...]
  (from the ttf-core dir)

Prints per step: wall time and --timing's phases; per layer type (DeltaNet / full attention): the share; with
--profile, a torch.profiler table of one step (CUDA and CPU self time, top ops) to OUT/prof_*.txt and a chrome trace."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import model as QM
from ttf.q38 import train as QT
from ttf.q38 import moe as MT
from ttf.data import _kd_sampled


def materialize(mod, g):
    """meta parameters / buffers of mod -> CPU tensors (random: 2-D+ randn / sqrt(fan in), A_log randn x 0.1, else 0)"""
    for name, m in mod.named_modules():
        for k, p in list(m._parameters.items()):
            if p is not None and p.is_meta:
                if p.dim() >= 2:
                    v = torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5
                elif k == "A_log":
                    v = torch.randn(p.shape, generator=g) * 0.1
                else:
                    v = torch.zeros(p.shape)
                m._parameters[k] = nn.Parameter(v.to(p.dtype), requires_grad=p.requires_grad)
        for k, b in list(m._buffers.items()):
            if b is not None and b.is_meta:
                m._buffers[k] = torch.zeros(b.shape, dtype=b.dtype)


def fill_ternary(L, g):
    for n, t in QT.tern_modules(L):
        with torch.no_grad():
            t.wpacked.copy_(torch.randint(0, 243, t.wpacked.shape, generator=g, dtype=torch.int16).to(torch.uint8))
            t.row_scale.fill_(0.02); t.col_scale.fill_(1.0)
            if t.ra:
                t.ta_packed.copy_(torch.randint(0, 243, t.ta_packed.shape, generator=g, dtype=torch.int16).to(torch.uint8))
        t._nnz = None; t._wc = None


class ModTable:
    """a small random n-gram table standing in for the ~320 M-row host table: rows taken modulo its size, on the CPU"""

    def __init__(self, t):
        self.t = t

    def __getitem__(self, flat):
        return self.t[flat.cpu() % self.t.shape[0]]


def kd_batch(M, T, V, seed, k=64, s=64):
    g = np.random.default_rng(seed)
    out = []
    for _ in range(M):
        w = torch.from_numpy(g.integers(0, V, T + 1).astype(np.int64))
        top = np.stack([g.choice(V, k, replace=False) for _ in range(T)]) if T <= 64 else g.integers(0, V, (T, k))
        lp = np.log(np.sort(g.dirichlet(np.ones(k) * 0.3, T), -1)[:, ::-1] * 0.95 + 1e-9).astype(np.float32)
        ids, lw = _kd_sampled(top, lp, g.integers(0, V, (T, s)))
        out.append({"ids": w[None, :-1], "tgt": w[1:], "kd_ids": torch.from_numpy(ids.astype(np.int64)),
                    "kd_lw": torch.from_numpy(lw)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--layer-ids", default="", help="which real layer indices to build (default 0..NL-1): e.g. 0,1,2,3")
    ap.add_argument("--conv", default="", help="dir with real converted layers L{i}.pt (used where present)")
    ap.add_argument("--ple-rows", type=int, default=1 << 20); ap.add_argument("--out", default="q38_bench")
    ap.add_argument("--profile", action="store_true"); ap.add_argument("--bench-steps", type=int, default=4)
    ap.add_argument("--micro", type=int, default=16); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--theta", type=float, default=16.0); ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--sketch", type=int, default=256); ap.add_argument("--rank-ratio", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--head-chunk", type=int, default=512); ap.add_argument("--expert-chunk", type=int, default=32)
    ap.add_argument("--adapter-rank", type=int, default=16); ap.add_argument("--no-pin", dest="pin", action="store_false")
    ap.add_argument("--prefetch", action="store_true"); ap.add_argument("--expert-orth", default="qr", choices=["qr", "rchol"])
    ap.add_argument("--dense-orth", default=None, choices=["qr", "rchol"]); ap.add_argument("--fire", default="kernel")
    ap.add_argument("--no-grouped", action="store_true"); ap.add_argument("--resident", action="store_true")
    ap.add_argument("--compile-hc", action="store_true"); ap.add_argument("--wbatch", type=int, default=1)
    a = ap.parse_args()
    a.steps, a.save_every, a.eval_every, a.timing, a.diag = 1000, 10 ** 9, 10 ** 9, True, False
    a.kd_cache_mode = "sample"
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    raw = json.load(open(a.config))
    t = dict(raw.get("text_config", raw))
    ids = [int(x) for x in a.layer_ids.split(",")] if a.layer_ids else list(range(a.layers))
    # the chosen real layers, renumbered 0..n-1 (their types and PLE placement kept)
    t["layer_types"] = [t["layer_types"][i] for i in ids]
    if "ple_layer_ids" in t:
        t["ple_layer_ids"] = [ids.index(i) for i in t["ple_layer_ids"] if i in ids]
    t["num_hidden_layers"] = len(ids)
    cfg = QM.text_config({"text_config": t})
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    m = QM.build(cfg, device="meta")
    for i, L in enumerate(m.layers):
        QT.swap_ternary(L, adapter_rank=a.adapter_rank)
        materialize(L, g)
        fill_ternary(L, g)
        cp = os.path.join(a.conv, f"L{ids[i]}.pt") if a.conv else ""
        if cp and os.path.exists(cp):                # the real converted ternary state (trits, scales, adapters)
            for n_, sd in torch.load(cp, weights_only=False, map_location="cpu").items():
                QT._get(L, n_).load_state_dict(sd)
            print(f"layer {i}: real conversion {cp}", flush=True)
        if L.ple is not None:                        # (a host table gathered on the CPU, as the run's MmapTable)
            L.ple.ple_embedding.table = ModTable(torch.randn(a.ple_rows, L.ple.ple_embedding.dim, generator=g).to(torch.bfloat16))
        L.to(torch.bfloat16)                        # (bf16 floats, fp32 ternary scales: as the run)
        for n_, tm in QT.tern_modules(L):
            tm.float()
        print(f"layer {i} ({cfg.layer_types[i]}{', PLE' if L.ple is not None else ''}) built, {time.time() - t0:.0f} s", flush=True)
    for mod in (m.embed_tokens, m.lm_head, m.hyper_connection_mixer):
        materialize(mod, g); mod.to(torch.bfloat16)
    materialize(m, g)
    for mod in (m.embed_tokens, m.lm_head, m.hyper_connection_mixer):
        mod.to(dev)
    m.rotary.to(dev)
    tr = QT.Trainer(m, a, dev)
    V = cfg.vocab_size
    hostGB = sum(v.numel() * v.element_size() for s_ in tr.ts.values() for v in s_.values() if torch.is_tensor(v)) / 2 ** 30
    print(f"built {len(ids)} layers in {time.time() - t0:.0f} s; TTF host state {hostGB:.1f} GiB "
          f"({hostGB / len(ids):.2f} per layer)", flush=True)
    recs = []
    for s in range(a.bench_steps):
        b = kd_batch(a.micro, a.seq, V, 100 + s)
        torch.cuda.reset_peak_memory_stats()
        r = tr.train_step(b)
        r["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        recs.append(r)
        mv = {k: v for k, v in r["moves"].items() if k != "by"}
        print(f"step {r['step']} {r['t']} s peak {r['peak_gib']} GiB loss {r['loss']:.3f} moves {mv} ph {r.get('ph')}", flush=True)
    json.dump(recs, open(os.path.join(a.out, "steps.json"), "w"))
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        b = kd_batch(a.micro, a.seq, V, 999)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            tr.train_step(b); torch.cuda.synchronize()
        ka = prof.key_averages()
        open(os.path.join(a.out, "prof_cuda.txt"), "w").write(ka.table(sort_by="self_cuda_time_total", row_limit=60))
        open(os.path.join(a.out, "prof_cpu.txt"), "w").write(ka.table(sort_by="self_cpu_time_total", row_limit=40))
        prof.export_chrome_trace(os.path.join(a.out, "trace.json"))
        print(ka.table(sort_by="self_cuda_time_total", row_limit=30))
    print("done", flush=True)


if __name__ == "__main__":
    main()
