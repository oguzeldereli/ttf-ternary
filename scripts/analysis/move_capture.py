"""move_capture.py (6 Nov 2026; the move-quality measurement of a conversion run at scale): a setup-only --ext plug-in
(not structural, so a resume accepts it). The trainer resumes a snapshot as usual (trits, floats, adapters, teacher, KD
loss); the plug-in then
  - turns off every update: a grad hook (ttf.update.add_grad_hook, SKIP) captures each layer's gradient instead of
    updating it (no trit moves, no accumulator change), ts_step and the float optimizers' steps are no-ops, so the
    weights stay the snapshot's;
  - replaces the batch stream by a fixed one (its own seed, the same windows from every snapshot);
  - per ternary layer of the chosen blocks sums G = gy^T x (what the accumulator sees) per micro-batch: s1, s2 = sum G^2,
    sA = the even micro-batches (split halves);
  - [trits=CKPT: the trits of another checkpoint into these layers first (counterfactual: the moves alone)];
  - after nb micro-batches writes OUT (mu, var of mu, half means, trits) and exits.
scripts/analysis/move_stats.py turns the files of several snapshots into move agreement, rho along the moves and the
net loss change by evidence quintile.

  PYTHONPATH=scripts/analysis python -m ttf.train <the run's own args> --resume \
      --ext move_capture:out=F.pt,blocks=2:8:14:20,nb=1024[,trits=CKPT][,seed=777]
(run name = a directory holding the snapshot as ckpt.pt)"""
import os, re
import numpy as np
import torch


class _Ana:
    def __init__(self, spec):
        kv = dict(p.split("=", 1) for p in spec.split(",") if p)
        self.out = kv["out"]
        self.blocks = {int(b) for b in kv.get("blocks", "2:8:14:20").split(":")}
        self.nb = int(kv.get("nb", 1024))
        self.trits = kv.get("trits")
        self.gen = torch.Generator().manual_seed(int(kv.get("seed", 777)))

    def setup(self, model, log):
        import sys
        _mm = sys.modules.get("__main__")      # python -m ttf.train runs the trainer as __main__: patch that module (an
        if getattr(getattr(_mm, "__spec__", None), "name", None) == "ttf.train":   # import ttf.train is a second copy
            TT = _mm
        else:
            import ttf.train as TT
        from ttf.layers import KernelTernaryLinear
        from ttf.kernel import unpack_rows, pack_rows
        mods = {}
        for n, m in model.named_modules():
            if isinstance(m, KernelTernaryLinear):
                b = re.search(r"layers\.(\d+)\.", n)
                if b and int(b.group(1)) in self.blocks:
                    mods[n] = m
        if self.trits:
            blob = torch.load(self.trits, map_location="cpu", mmap=True, weights_only=False)
            sd = blob["model"] if "model" in blob else blob
            with torch.no_grad():
                for n, m in mods.items():
                    m.wpacked.copy_(sd[n + ".wpacked"].to(m.wpacked.device))
                    if hasattr(m, "_invalidate"):
                        m._invalidate()
            del blob, sd
            log(f"move_capture: trits of {self.trits} in {len(mods)} layers (counterfactual)")
        ids = {id(m): n for n, m in mods.items()}
        dev = next(model.parameters()).device
        s1 = {n: torch.zeros(m.N, m.K, device=dev) for n, m in mods.items()}
        s2 = {n: torch.zeros(m.N, m.K, device=dev) for n, m in mods.items()}
        sA = {n: torch.zeros(m.N, m.K, device=dev) for n, m in mods.items()}
        W = {n: unpack_rows(m.wpacked, m.K).to(torch.int8).cpu() for n, m in mods.items()}
        k = [0]
        log(f"move_capture: patching {TT.__name__}; {len(mods)} ternary layers in blocks {sorted(self.blocks)}, "
            f"{sum(m.N * m.K for m in mods.values()) / 1e6:.1f} M entries, nb {self.nb}")

        import ttf.update as U

        def tl(l, g, state, c):
            n = ids.get(id(l))
            if n is not None:
                G = g.float() if torch.is_tensor(g) else (g[0].float().T @ g[1].float())
                s1[n] += G; s2[n] += G * G
                if k[0] % 2 == 0:
                    sA[n] += G
            return U.SKIP                       # no layer updates at all (the snapshot's weights stay)

        U.add_grad_hook(tl)
        TT.ts_step = lambda model_, state, c: {"ts_frac": 0.0}
        from ttf.opt_lean import LeanAdam
        LeanAdam.step = lambda self_, *a, **kw: None
        torch.optim.AdamW.step = lambda self_, *a, **kw: None
        me = self

        def gb(data, bs, seq_len, device, gen=None):
            k[0] += 1
            if k[0] > me.nb:          # the last micro-batch's backward is done: write and leave
                nb = me.nb; h = (nb + 1) // 2
                out = {"nb": nb, "W": W,
                       "mu": {n: (s1[n] / nb).cpu() for n in mods},
                       "var": {n: ((s2[n] / nb - (s1[n] / nb).square()).clamp_min(0) / nb).cpu() for n in mods},
                       "muA": {n: (sA[n] / h).cpu() for n in mods},
                       "muB": {n: ((s1[n] - sA[n]) / (nb - h)).cpu() for n in mods}}
                torch.save(out, me.out)
                log(f"move_capture: wrote {me.out}")
                os._exit(0)
            ix = torch.randint(len(data) - seq_len - 1, (bs,), generator=me.gen)
            x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix.tolist()])
            y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix.tolist()])
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        TT.get_batch = gb
        k[0] = 0
        # (k counts get_batch calls; the even / odd split uses the count at the layer's backward: k is the current batch)


def ttf_ext(spec):
    return _Ana(spec)
