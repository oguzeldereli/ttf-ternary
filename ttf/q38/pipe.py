"""q38_pipe.py: q38_train's step across PCs, layer stages in a pipeline (P, 5 Nov 2026; user: "can we run the
conversion + training on lab for qwen next flash", "get it as fast as possible").

Why recompute and not 1F1B: the update needs each layer's captured factors (beta gy, x) for the whole step, ~7 GB per
layer; holding them from the backward to the update for a stage's ~10 layers does not fit a lab PC's host RAM. So every
stage recomputes its layers once more, with capture, after the pipelined passes:

  (A) forward, pipelined over groups of --pipe-group micro-batches: stage r takes a group's hidden states from r - 1
      (stage 0 embeds), runs them through its layers (each layer's input kept on the host, as q38_train's stash) and
      sends the outputs on; the last stage takes the head loss and dL/dh per micro-batch as its groups arrive.
  (B) input gradients, pipelined backward over the same groups: stage r takes the group's gradient from r + 1 (the last
      stage: the head's), keeps each of its layers' output gradients on the host, propagates them down its layers
      with torch.autograd.grad w.r.t. the layer input only (no capture, no parameter gradients) and sends the result
      to r - 1.
  (C) each stage on its own (no communication): q38_train's backward + update, top layer down: recompute the layer from
      its stashed input with capture, backward with the stored output gradient (the same gradient q38_train chains),
      then the layer's TTF / adapter updates and AdamW (Trainer._update_layer, the same seeds: global layer index).

The arithmetic is q38_train's: every layer's backward and update sees the same inputs, output gradients and weights
(test_q38_pipe.py: 2 stages == one Trainer, CPU, fp32). The cost over q38_train: (B) adds one layer forward + input
backward per layer; the passes (A) and (B) are pipelined, (C) runs on all stages at once.

  torchrun --nnodes N --node_rank R --nproc_per_node 1 --master_addr HOST --master_port P q38_pipe.py \
      --model DIR --conv DIR --kd-cache DIR --data TOKENS.bin --out DIR [--pp-split 10,10,10,9,9] [q38_train flags]
  (NCCL over sockets on the lab: NCCL_SOCKET_IFNAME=^lo,docker NCCL_IB_DISABLE=1, as lab/pp27_lab.sh)
Every rank reads its own layers (model, conv) and the PLE table if it holds a PLE layer; the last rank reads the data
and KD cache and broadcasts the token ids; checkpoints: one file per global layer in OUT/ckpt (each rank its own).
"""
from __future__ import annotations
import argparse, json, math, os, signal, sys, time
import torch
import torch.distributed as dist
import torch.nn as nn

from ttf.q38 import model as QM
from ttf.q38 import train as QT
from ttf.q38 import moe as MT


def pp_split(n, k, spec=""):
    c = [int(v) for v in spec.split(",")] if spec else [n // k + (i < n % k) for i in range(k)]
    assert len(c) == k and sum(c) == n, f"--pp-split {spec}: {k} counts summing to {n}"
    lo = [sum(c[:i]) for i in range(k)]
    return [(lo[i], lo[i] + c[i]) for i in range(k)]


class PipeTrainer(QT.Trainer):
    """QT.Trainer over this stage's layers (model.layers = the stage's layers, global [i0, i1))"""

    def __init__(self, model, args, dev, rank, world, i0):
        super().__init__(model, args, dev)
        self.i0, self.rank, self.world = i0, rank, world
        self.first, self.last = rank == 0, rank == world - 1
        self.D = model.cfg.hc_count * model.cfg.hidden_size
        self.dt = model.embed_tokens.weight.dtype
        self.tm = {}

    # ---- communication (NCCL on the device, gloo on the CPU: the tensors live on self.dev either way)
    def _recv(self, n, T, src):
        t = torch.empty(n, T, self.D, dtype=self.dt, device=self.dev)
        return t, dist.irecv(t, src)

    def _ids(self, mbs, M, T):
        """the step's token ids [M, T] on every rank (the last rank reads the data)"""
        ids = torch.empty(M, T, dtype=torch.int64, device=self.dev)
        if self.last:
            ids.copy_(torch.cat([b["ids"] for b in mbs]))
        dist.broadcast(ids, self.world - 1)
        return [ids[j:j + 1] for j in range(M)]

    def _pgroups(self, M):
        G = max(1, getattr(self.a, "pipe_group", 4))
        return [list(range(j, min(M, j + G))) for j in range(0, M, G)]

    def _sub(self, g):
        """a pipeline group's micro-batches in --wbatch windows per layer call"""
        w = max(1, getattr(self.a, "wbatch", 1))
        return [g[j:j + w] for j in range(0, len(g), w)]

    # ---- (A) the forward, pipelined; returns the last stage's per-micro-batch head losses / gradients
    @torch.no_grad()
    def _fwd(self, ids, mbs, stash=True, grad=True):
        a, dev = self.a, self.dev
        M, T = len(ids), ids[0].shape[1]
        pe = self.m.rope(T, dev)
        if stash:
            shp = (self.nL, M, 1, T, self.D)
            if getattr(self, "_stash", None) is None or self._stash.shape != shp:
                self._stash = torch.empty(shp, dtype=self.dt, pin_memory=a.pin and dev != "cpu")
        groups = self._pgroups(M)
        pend, sends = None, []
        loss, gs = 0.0, [None] * M
        if not self.first:
            pend = self._recv(len(groups[0]), T, self.rank - 1)
        for k, g in enumerate(groups):
            if self.first:
                h = torch.cat([self.m.embed(ids[j]) for j in g])
            else:
                h, w = pend; w.wait()
                pend = self._recv(len(groups[k + 1]), T, self.rank - 1) if k + 1 < len(groups) else None
            hs = list(h.split(1))
            for i in range(self.nL):
                L = self._on(i)
                if stash:
                    for j, x in zip(g, hs):
                        self._stash[i, j].copy_(x, non_blocking=True)
                hs = self._layer_fwd(L, hs, pe, [{"ids": ids[j]} for j in g])
                self._off(i, writeback=False)
            if not self.last:
                y = torch.cat(hs).contiguous()
                sends.append((dist.isend(y, self.rank + 1), y))
            else:
                for j, x in zip(g, hs):
                    b = mbs[j]
                    l, gj = self.head_loss(x, (b["tgt"].to(dev), b["kd_ids"].to(dev), b["kd_lw"].to(dev)), grad=grad)
                    loss += l; gs[j] = gj
        for w, _ in sends:
            w.wait()
        return loss, gs, pe

    # ---- (B) input gradients, pipelined; the stage's output gradients per layer on the host
    def _bwd_input(self, ids, gs, pe):
        a, dev = self.a, self.dev
        M, T = len(ids), ids[0].shape[1]
        shp = (self.nL, M, 1, T, self.D)
        if getattr(self, "_gst", None) is None or self._gst.shape != shp:
            self._gst = torch.empty(shp, dtype=self.dt, pin_memory=a.pin and dev != "cpu")
        groups = self._pgroups(M)
        pend, sends = None, []
        if not self.last:
            pend = self._recv(len(groups[0]), T, self.rank + 1)
        for k, g in enumerate(groups):
            if self.last:
                gh = [gs[j] for j in g]
            else:
                t, w = pend; w.wait()
                pend = self._recv(len(groups[k + 1]), T, self.rank + 1) if k + 1 < len(groups) else None
                gh = list(t.split(1))
            for i in reversed(range(self.nL)):
                for j, x in zip(g, gh):
                    self._gst[i, j].copy_(x, non_blocking=True)
                if self.first and i == 0:
                    break                          # (the embedding's gradient is not needed)
                L = self._on(i)
                out = []
                for sg in self._sub(list(range(len(g)))):
                    h = torch.cat([self._stash[i, g[j]].to(dev, non_blocking=True) for j in sg]).requires_grad_(True)
                    with torch.enable_grad():
                        y = L(h, pe, torch.cat([ids[g[j]] for j in sg]))
                        gx, = torch.autograd.grad(y, h, torch.cat([gh[j] for j in sg]))
                    out += list(gx.split(1))
                    del y, h
                gh = out
                self._off(i, writeback=False)
            if not self.first:
                y = torch.cat(gh).contiguous()
                sends.append((dist.isend(y, self.rank - 1), y))
        for w, _ in sends:
            w.wait()

    # ---- (C) the stage's own backward with capture + updates (q38_train's loop, output gradients from the host)
    def _bwd_update(self, ids, pe, lr, ph, moves):
        a, dev = self.a, self.dev
        M = len(ids)
        pf = getattr(a, "prefetch", False) and dev != "cpu"
        if pf and getattr(self, "cs", None) is None:
            self.cs = torch.cuda.Stream()
        cur_st = None
        for i in reversed(range(self.nL)):
            L = self._on(i)
            if pf:
                cur_st = self._pf_start(i)
            tms = QT.tern_modules(L)
            for _, t in tms:
                t.capture = True
            with ph("bwd"):
                for sg in self._sub(list(range(M))):
                    h = torch.cat([self._stash[i, j].to(dev, non_blocking=True) for j in sg])
                    y = L(h, pe, torch.cat([ids[j] for j in sg]))
                    y.backward(torch.cat([self._gst[i, j].to(dev, non_blocking=True) for j in sg]))
                    del y, h
            for _, t in tms:
                t._wc = None
            self._update_layer(i, L, tms, ph, moves, cur_st, pf)
            with ph("adamw"):
                self.opt.step(i, QT.layer_floats(L), lr)
            self._off(i)
            cur_st = None

    def train_step(self, mbs, M=None, T=None):
        a, dev = self.a, self.dev
        t0 = time.time()
        lr = QT.lr_at(self.step, a.steps, a.lr, a.lr_min, a.warmup)
        th = QT.theta_at(self.step, a.steps, a.theta, getattr(a, "theta_end", None), getattr(a, "theta_rise_from", 0.0),
                         getattr(a, "theta_warm", None), getattr(a, "theta_warm_steps", 0))
        self.c["lr_ratio"], self.c["tau"] = a.theta / th, a.tau * a.theta / th
        M, T = M or a.micro, T or a.seq
        ids = self._ids(mbs, M, T)
        if self.last:
            a.norm = sum(int((b["tgt"] != -1).sum()) for b in mbs)
        ph = QT._Ph(getattr(a, "timing", False))
        loss, gs, pe = self._fwd(ids, mbs)
        if dev != "cpu":
            torch.cuda.synchronize()
        tA = time.time()
        self._bwd_input(ids, gs, pe)
        del gs
        if dev != "cpu":
            torch.cuda.synchronize()
        tB = time.time()
        moves = {"exp": 0, "dense": 0, "adapter": 0}
        self._dg, self._dgl = {}, {}
        self._bwd_update(ids, pe, lr, ph, moves)
        if dev != "cpu":
            torch.cuda.synchronize()
        tC = time.time()
        # (the step's loss from the last rank, the moves summed over the stages, the stages' phase times)
        v = torch.tensor([loss, moves["exp"], moves["dense"], moves["adapter"]], dtype=torch.float64, device=dev)
        dist.all_reduce(v)
        tt = torch.zeros(self.world, 3, dtype=torch.float64, device=dev)
        tt[self.rank] = torch.tensor([tA - t0, tB - tA, tC - tB], dtype=torch.float64)
        dist.all_reduce(tt)
        nrm = torch.tensor([float(getattr(a, "norm", 0)) if self.last else 0.0], dtype=torch.float64, device=dev)
        dist.all_reduce(nrm)
        self.step += 1
        by = moves.get("by", {})
        return {"step": self.step, "loss": float(v[0] / nrm[0]), "lr": lr, "theta": th,
                "moves": {"exp": int(v[1]), "dense": int(v[2]), "adapter": int(v[3]), "by_stage": by},
                "t": round(time.time() - t0, 2),
                "stages": [[round(x, 2) for x in r] for r in tt.tolist()],      # per rank: (A) fwd, (B) input bwd, (C)
                **({"ph": {k: round(x, 2) for k, x in ph.t.items()}} if ph.on else {})}

    @torch.no_grad()
    def eval_loss(self, mbs, M=None, T=None):
        a = self.a
        M, T = M or a.val_micro, T or a.seq
        ids = self._ids(mbs, M, T)
        if self.last:
            a.norm = sum(int((b["tgt"] != -1).sum()) for b in mbs)
        loss, _, _ = self._fwd(ids, mbs, stash=False, grad=False)
        v = torch.tensor([loss, float(getattr(a, "norm", 0)) if self.last else 0.0], dtype=torch.float64, device=self.dev)
        dist.all_reduce(v)
        return float(v[0] / v[1])

    # ---- checkpoints: the global layer index in the file names (each rank writes its own layers)
    def save(self, d, slim=False):
        os.makedirs(d, exist_ok=True)
        if self.dev != "cpu":
            torch.cuda.synchronize()
        for i, L in enumerate(self.m.layers):
            gi = self.i0 + i
            blob = {"layer": {k: v.cpu() for k, v in L.state_dict().items()}}
            if not slim:
                blob.update(ts={n: s for (k, n), s in self.ts.items() if k == i},
                            ad={n: s for (k, n), s in self.ad.items() if k == i}, opt=self.opt.state_dict(i))
            torch.save(blob, os.path.join(d, f"L{gi}.pt.tmp")); os.replace(os.path.join(d, f"L{gi}.pt.tmp"), os.path.join(d, f"L{gi}.pt"))
        dist.barrier()
        if self.rank == 0:
            json.dump({"step": self.step, "slim": slim}, open(os.path.join(d, "meta.json.tmp"), "w"))
            os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
        dist.barrier()

    def load(self, d):
        meta = json.load(open(os.path.join(d, "meta.json")))
        assert not meta.get("slim"), f"{d}: a slim (model only) save, not resumable"
        self.step = meta["step"]
        for i, L in enumerate(self.m.layers):
            blob = torch.load(os.path.join(d, f"L{self.i0 + i}.pt"), weights_only=False)
            L.load_state_dict(blob["layer"])
            for n, s in blob["ts"].items():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        self.ts[(i, n)][k].copy_(v)
            for n, s in blob["ad"].items():
                self.ad[(i, n)] = s
            self.opt.load(i, blob["opt"])


def stage_model(m, lo, hi):
    """m with only layers [lo, hi) (the others dropped: meta, never loaded)"""
    m.layers = nn.ModuleList([m.layers[i] for i in range(lo, hi)])
    return m


def run(tr, args, batches, val_batches):
    """q38_train.run for the pipeline: every rank runs the loop; the data / logging on the last rank; a stop signal on
    any rank stops all after the same step"""
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "metrics.jsonl"), "a") if tr.last else None
    stop = []

    def _sig(num, frame):
        stop.append(num)
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    def _stop_all():
        f = torch.tensor([1.0 if stop else 0.0], device=tr.dev)
        dist.all_reduce(f, op=dist.ReduceOp.MAX)
        return f.item() > 0

    def _say(rec, txt):
        if tr.last:
            print(json.dumps(rec), file=log, flush=True); print(txt, flush=True)

    while tr.step < args.steps:
        if _stop_all():
            break
        if tr.step % args.eval_every == 0:
            v = tr.eval_loss(val_batches() if tr.last else None)
            _say({"step": tr.step, "val": v}, f"step {tr.step} val {v:.4f}")
        r = tr.train_step(batches(tr.step) if tr.last else None)
        _say(r, f"step {r['step']} loss {r['loss']:.4f} moves {[r['moves'][k] for k in ('exp', 'dense', 'adapter')]} "
                f"{r['t']}s stages {r['stages']}")
        assert math.isfinite(r["loss"]), "loss not finite"
        if _stop_all():
            break
        if tr.step % args.save_every == 0 or tr.step == args.steps:
            tr.save(os.path.join(args.out, "ckpt"))
    if stop or tr.step < args.steps:
        mode = getattr(args, "stop_save", "slim")
        if mode != "none":
            tr.save(os.path.join(args.out, "stop" if mode == "slim" else "ckpt"), slim=mode == "slim")
        _say({"step": tr.step, "stopped": True, "save": mode}, f"stopped at step {tr.step}, {mode} save")
        return
    v = tr.eval_loss(val_batches() if tr.last else None)
    _say({"step": tr.step, "val": v, "final": True}, f"final val {v:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp-split", default="", help="layers per stage, e.g. 10,10,10,9,9 (default even)")
    ap.add_argument("--pipe-group", type=int, default=4, help="micro-batches per pipeline group (passes A and B)")
    # (q38_train's flags, the same names and defaults)
    ap.add_argument("--model"); ap.add_argument("--conv"); ap.add_argument("--kd-cache"); ap.add_argument("--data")
    ap.add_argument("--data-u32", action="store_true"); ap.add_argument("--out", default="q38_pipe_run")
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--theta", type=float, default=16.0); ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--theta-end", type=float, default=None); ap.add_argument("--theta-rise-from", type=float, default=0.0)
    ap.add_argument("--theta-warm", type=float, default=None); ap.add_argument("--theta-warm-steps", type=int, default=300)
    ap.add_argument("--dense-theta-mult", type=float, default=1.0)
    ap.add_argument("--fire", choices=["kernel", "torch"], default="kernel")
    ap.add_argument("--dense-orth", choices=["qr", "rchol"], default=None); ap.add_argument("--diag", action="store_true")
    ap.add_argument("--sketch", type=int, default=256); ap.add_argument("--rank-ratio", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--head-chunk", type=int, default=512); ap.add_argument("--expert-chunk", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=250); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-offset", type=int, default=16128); ap.add_argument("--val-micro", type=int, default=8)
    ap.add_argument("--adapter-rank", type=int, default=-1); ap.add_argument("--no-pin", dest="pin", action="store_false")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--expert-orth", default="qr", choices=["qr", "rchol"]); ap.add_argument("--no-grouped", action="store_true")
    ap.add_argument("--resident", action="store_true"); ap.add_argument("--compile-hc", action="store_true")
    ap.add_argument("--wbatch", type=int, default=1); ap.add_argument("--timing", action="store_true")
    ap.add_argument("--update-tf32", action="store_true")
    ap.add_argument("--stop-save", choices=("slim", "full", "none"), default="slim")
    args = ap.parse_args()
    args.kd_cache_mode = "sample"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dist.init_process_group("nccl" if dev == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if dev == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    cfg = QM.text_config(args.model)
    lo, hi = pp_split(cfg.num_hidden_layers, world, args.pp_split)[rank]
    print(f"rank {rank}/{world}: layers [{lo}, {hi})", flush=True)
    m = QM.build(cfg, device="meta")
    rd = QM.Reader(args.model)
    QM.load_globals(m, rd, "cpu")
    if args.adapter_rank < 0:
        c0 = torch.load(os.path.join(args.conv, f"L{lo}.pt"), weights_only=False, map_location="cpu")
        args.adapter_rank = int(next(iter(c0.values()))["ad_cb"].shape[1]) if any("ad_cb" in v for v in c0.values()) else 0
        del c0
    for i in range(lo, hi):
        L = m.layers[i]
        QM.load_layer(L, i, rd, "cpu", experts=False)
        QT.swap_ternary(L, adapter_rank=args.adapter_rank)
        L.to("cpu")
        conv = torch.load(os.path.join(args.conv, f"L{i}.pt"), weights_only=False)
        for n, sd in conv.items():
            QT._get(L, n).load_state_dict(sd)
        if L.ple is not None:
            L.ple.ple_embedding.table, L.ple.ple_embedding.scale = QM.ngram_table_memmap(args.model, i)
        print(f"rank {rank}: layer {i} loaded", flush=True)
    stage_model(m, lo, hi)
    if rank == 0:
        m.embed_tokens.to(dev)
    if rank == world - 1:
        m.lm_head.to(dev); m.hyper_connection_mixer.to(dev)
    for keep, mod in ((rank == 0, m.embed_tokens), (rank == world - 1, m.lm_head)):
        if not keep:                   # (the host copy freed: an empty tensor, the dtype kept)
            mod.weight.data = torch.empty(0, mod.weight.shape[1], dtype=mod.weight.dtype)
    m.rotary.to(dev)
    tr = PipeTrainer(m, args, dev, rank, world, lo)
    if args.resume and os.path.exists(os.path.join(args.out, "ckpt", "meta.json")):
        tr.load(os.path.join(args.out, "ckpt"))
    run(tr, args, lambda s: QT.kd_batches(args, s, args.micro, args.seq),
        lambda: QT.kd_batches(args, 0, args.val_micro, args.seq, val=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
