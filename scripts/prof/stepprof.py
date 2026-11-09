"""fine-grained step profile + memory breakdown (24 Oct 2026; first used for a 4-arm comparison: A adapters / B plain
conversion / C float AdamW / D TTF from scratch, one GPU). Runs TARGET (ttf.train, or a trainer script) as
__main__ and, by optimizer step (one step each; the autograd engine's multithreading off from the first profiled
step on, so backward Python and ops run on the profiled thread):
  steps < SP_FROM   warm-up; their wall times (unprofiled) are kept
  step SP_FROM      torch.profiler, CPU + CUDA only: the attribution and the op tables (a with_stack / record_shapes
                    profile's events of an adapter step do not fit in 125 GB of host memory when walked in Python)
  next step         torch.profiler with record_shapes, with_stack, profile_memory: the chrome trace only
  next step         torch.cuda.memory._record_memory_history (python stacks), snapshot at its end
then writes OUT.{trace.json.gz, ops_cuda.txt, ops_cpu.txt, groups.json, groups.txt, mem.pickle, mem.html, mem.json} and exits.

Attribution (exclusive, adds up): every op's self CPU time and every kernel's GPU time goes to the innermost named range
around it. Ranges ("g:" + path) come from module forwards (embed, attn.full, attn.lin, mlp, norm, final_norm, head_loss,
tern.fwd.G with G = T (main ternary layer) or ad (an adapter's down / up), qtad), from the ternary layer's backward
(tern.bwd.G) and the TTF update (ttf.G with helpers ttf.orth, ttf.fgmm_r, ttf.fgmm_l, ttf.fire, ttf.codec, ttf.store),
the teacher (teacher: its forward; kd.prep: its log-probs), data (get_batch), opt (optimizer step + clip) and the
backward call. Forward ranges opened while a backward runs are the checkpoint recompute (prefix recompute/). Backward ops
of autograd nodes get the label of the forward op that made the node (by sequence number), prefixed bwd/. Ops inside
ttf.G itself (not in a helper) are split by op name: randn / normal -> sketch.rng, mm -> sketch.mm, the rest -> v+elem
(the factored v, the sketch cross terms, the 1/a 1/b bounds, the spend's elementwise part).
  SP_OUT=out/A SP_FROM=3 python scripts/prof/stepprof.py ttf.train ARGS...
  SP_MEMONLY=1: only the memory step (at step SP_FROM); SP_MEM_ENTRIES: the history's ring size (default 20M: an
  adapter step makes more than 2M allocator events, and a truncated history loses the step's start)
  SP_TRACE=0: stop after the attribution step (no chrome trace, no memory history: those two need the 125 GB of host
  memory on an adapter step; the attribution step alone fits a 32 GB PC; qmem.py gives the memory breakdown)
  SP_TRAIN_EMB=1: --convert random leaves the embedding / head trainable (from-scratch arm)
"""
import collections, gc, gzip, json, os, pickle, runpy, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TTF_CORE", os.getcwd()))
import torch
import torch.nn as nn
from torch.autograd.profiler import record_function as rf
import ttf.layers as L
import ttf.update as U
import ttf.kernel as KN
import ttf.qwen as Q
import ttf.data as D

FROM = int(os.environ.get("SP_FROM", "3")); N = 1; OUT = os.environ["SP_OUT"]
TARGET = sys.argv[1]; sys.argv = [TARGET] + sys.argv[2:]
ST = {"k": 0, "prof": None, "ts": [], "mem": False, "bwd": False, "t0": 0.0}
STACK = []
MOD = []          # student QwenTern models (build_qwen or QwenTern())
OPTS = []


def _push(name):
    STACK.append(name)
    r = rf("g:" + "/".join(STACK)); r.__enter__(); return r


def _pop(r):
    r.__exit__(None, None, None); STACK.pop()


def ranged(name_fn):
    def deco(f):
        def g(*a, **k):
            nm = name_fn(*a, **k)
            if nm is None:
                return f(*a, **k)
            if ST["bwd"] and not nm.startswith(("tern.bwd", "ttf", "bwd")) and not any(s.startswith("recompute") for s in STACK):
                nm = "recompute/" + nm
            r = _push(nm)
            try:
                return f(*a, **k)
            finally:
                _pop(r)
        return g
    return deco


def grp(l):
    return getattr(l, "_sp", "T")


# ---- module forwards
from transformers.models.qwen3_5 import modeling_qwen3_5 as HQ
_CLS = [(HQ.Qwen3_5TextModel, "model.glue"), (HQ.Qwen3_5DecoderLayer, "layer.glue"), (HQ.Qwen3_5Attention, "attn.full"), (HQ.Qwen3_5GatedDeltaNet, "attn.lin"), (HQ.Qwen3_5MLP, "mlp"),
        (HQ.Qwen3_5RMSNorm, "norm")]
if hasattr(HQ, "Qwen3_5RMSNormGated"):
    _CLS.append((HQ.Qwen3_5RMSNormGated, "norm"))
for cls, nm in _CLS:
    cls.forward = ranged(lambda self, *a, _n=nm, **k: None if getattr(self, "_sp_teacher", False) else _n)(cls.forward)
nn.Embedding.forward = ranged(lambda self, *a, **k: None if getattr(self, "_sp_teacher", False) else "embed")(nn.Embedding.forward)
L.KernelTernaryLinear.forward = ranged(lambda self, *a, **k: f"tern.fwd.{grp(self)}")(L.KernelTernaryLinear.forward)
_kb = L._KernelTernFn.backward
L._KernelTernFn.backward = staticmethod(ranged(lambda ctx, gy: f"tern.bwd.{grp(ctx.layer)}")(_kb))
Q.QwenTern._head_loss = ranged(lambda *a, **k: "head_loss")(Q.QwenTern._head_loss)
_tsf = U._ts_layer_factored
U._ts_layer_factored = ranged(lambda l, *a, **k: f"ttf.{grp(l)}")(_tsf)
for fn, nm in (("_orth", "ttf.orth"), ("_fg_mm_r", "ttf.fgmm_r"), ("_fg_mm_l", "ttf.fgmm_l"), ("_load_pair", "ttf.codec"),
               ("_store_pair", "ttf.codec"), ("_host_get", "ttf.store"), ("_host_put", "ttf.store")):
    setattr(U, fn, ranged(lambda *a, _n=nm, **k: _n)(getattr(U, fn)))
KN.ttf_fire = ranged(lambda *a, **k: "ttf.fire")(KN.ttf_fire)
_gb = D.get_batch
D.get_batch = ranged(lambda *a, **k: "data")(_gb)
_cfw = HQ.Qwen3_5ForCausalLM.forward


def _causal_fwd(self, *a, **k):          # the teacher (transformers' own model called directly)
    if any(self is m.hf for m in MOD):
        return _cfw(self, *a, **k)
    for mm in self.modules():
        mm._sp_teacher = True
    r = _push("teacher")
    try:
        return _cfw(self, *a, **k)
    finally:
        _pop(r)


HQ.Qwen3_5ForCausalLM.forward = _causal_fwd
_lsm = torch.nn.functional.log_softmax


def _log_softmax(x, *a, **k):            # the KD prep (teacher logits -> log-probs, outside the student's loss)
    if not STACK and not x.requires_grad:
        r = _push("kd.prep")
        try:
            return _lsm(x, *a, **k)
        finally:
            _pop(r)
    return _lsm(x, *a, **k)


torch.nn.functional.log_softmax = _log_softmax
_bw = torch.Tensor.backward


def _backward(self, *a, **k):
    ST["bwd"] = True; r = _push("backward")
    try:
        return _bw(self, *a, **k)
    finally:
        _pop(r); ST["bwd"] = False


torch.Tensor.backward = _backward
_clip = torch.nn.utils.clip_grad_norm_
torch.nn.utils.clip_grad_norm_ = ranged(lambda *a, **k: "opt.clip")(_clip)

if os.environ.get("SP_TRAIN_EMB") == "1":   # from scratch: --convert random's freeze of the embedding / head undone
    _rg = torch.Tensor.requires_grad_

    def _requires_grad_(self, mode=True):
        if not mode and isinstance(self, nn.Parameter) and self.dim() == 2 and self.shape[0] > 100_000:
            return self
        return _rg(self, mode)
    torch.Tensor.requires_grad_ = _requires_grad_

_bq = Q.build_qwen


def build_qwen(*a, **k):
    m, sw = _bq(*a, **k)
    if not any(m is x for x in MOD):
        MOD.append(m)
    return m, sw


Q.build_qwen = build_qwen
_qti = Q.QwenTern.__init__


def _qt_init(self, *a, **k):
    _qti(self, *a, **k)
    if not any(self is x for x in MOD):
        MOD.append(self)


Q.QwenTern.__init__ = _qt_init


def mark():
    QT = sys.modules.get("qwen_tad")       # (lab) the adapted linear: patched only now (importing qwen_tad first would
    if QT is not None and not getattr(QT.QTad.forward, "_sp", False):   # change its wrappers' order / switch it on)
        QT.QTad.forward = ranged(lambda *a, **k: "qtad")(QT.QTad.forward); QT.QTad.forward._sp = True
    for m in MOD:
        for n, l in m.named_modules():
            if isinstance(l, L.KernelTernaryLinear):
                l._sp = "ad" if ".ad." in n else "T"


# ---- attribution
def attribute(prof):
    ev = prof.events()
    seq_lab = {}
    lab = {}

    def label_of(e):
        if id(e) in lab:
            return lab[id(e)]
        p = e; out = None
        while p is not None:
            if p.name.startswith("g:"):
                out = p.name[2:]; break
            p = p.cpu_parent
        lab[id(e)] = out
        return out
    for e in ev:                                   # forward ops: sequence number -> the forward label
        if e.sequence_nr is not None and e.sequence_nr >= 0 and not e.name.startswith(("autograd::", "g:")):
            l_ = label_of(e)
            if l_ is not None and "backward" not in l_.split("/")[0:1]:
                seq_lab.setdefault((e.fwd_thread or e.thread, e.sequence_nr), l_)
    cpu = collections.Counter(); gpu = collections.Counter(); nk = collections.Counter(); syncs = collections.Counter()
    for e in ev:
        if e.name.startswith("g:"):
            l_ = e.name[2:]
            cpu[l_] += e.self_cpu_time_total       # Python time inside the range not in any op
            continue
        l_ = label_of(e)
        if l_ is not None and l_.split("/")[-1] == "backward" or l_ is None:
            p = e; node = None
            while p is not None:                    # an autograd node's backward: the forward op's label
                if p.name.startswith("autograd::engine::evaluate_function"):
                    node = p; break
                p = p.cpu_parent
            if node is not None:
                f_ = seq_lab.get((node.fwd_thread or node.thread, node.sequence_nr)) or seq_lab.get((node.thread, node.sequence_nr))
                l_ = ("bwd/" + f_) if f_ else ("bwd/?" + node.name.split(":")[-1].strip()[:40])
            elif l_ is None:
                l_ = "other/" + e.name[:60]
        leaf = l_.split("/")[-1]
        if leaf.startswith("ttf.") and leaf in ("ttf.T", "ttf.ad"):
            sub = "sketch.rng" if any(s in e.name for s in ("randn", "normal")) else ("sketch.mm" if e.name in ("aten::mm", "aten::matmul") else "v+elem")
            l_ = l_ + "/" + sub
        cpu[l_] += e.self_cpu_time_total
        for kk in e.kernels:
            gpu[l_] += kk.duration; nk[l_] += 1
        if "Synchronize" in e.name or e.name in ("aten::_local_scalar_dense", "aten::item"):
            syncs[e.name] += 1
    return cpu, gpu, nk, syncs


def summarize(cpu, gpu, nk, wall, nsteps):
    def comp(l_):                                  # component of a full label path
        parts = l_.split("/")
        rec = "recompute/" in l_ or any(p.startswith("recompute") for p in parts)
        bwd = parts[0] == "bwd"
        leaf = [p for p in parts if not p.startswith(("recompute", "bwd", "backward"))]
        if parts[0] == "other":
            return "other: " + "/".join(parts[1:])
        base = leaf[-1] if leaf else parts[-1]
        tg = [p for p in leaf if p in ("ttf.T", "ttf.ad")]
        if tg and base != tg[-1]:                  # a TTF helper / op class: under its layer group
            base = tg[-1] + "/" + base
        if leaf and leaf[0] in ("teacher", "kd.prep", "data", "opt", "opt.clip"):
            base = leaf[0]
        return ("recompute:" if rec else ("bwd:" if bwd else "")) + base
    C = collections.Counter(); G = collections.Counter(); K = collections.Counter()
    for l_ in set(cpu) | set(gpu):
        c_ = comp(l_); C[c_] += cpu[l_]; G[c_] += gpu[l_]; K[c_] += nk[l_]
    tc, tg = sum(C.values()), sum(G.values())
    rows = sorted(set(C) | set(G), key=lambda c_: -(G[c_] + C[c_]))
    lines = [f"wall {wall / nsteps:.3f} s/step profiled; GPU kernels {tg / 1e6 / nsteps:.3f} s/step "
             f"({tg / 1e6 / wall:.0%} busy); CPU in ops + ranges {tc / 1e6 / nsteps:.3f} s/step",
             f"{'component':<40}{'GPU ms/step':>12}{'GPU %':>7}{'CPU ms/step':>12}{'CPU %':>7}{'kernels/step':>13}"]
    for c_ in rows:
        lines.append(f"{c_:<40}{G[c_] / 1e3 / nsteps:>12.1f}{G[c_] / max(tg, 1):>7.1%}{C[c_] / 1e3 / nsteps:>12.1f}"
                     f"{C[c_] / max(tc, 1):>7.1%}{K[c_] / nsteps:>13.0f}")
    return lines, {c_: {"gpu_ms": G[c_] / 1e3 / nsteps, "cpu_ms": C[c_] / 1e3 / nsteps, "kernels": K[c_] / nsteps} for c_ in rows}


# ---- memory
def census():
    """persistent CUDA tensors by category (storages counted once)"""
    seen = {}; cat = collections.Counter()

    def add(t, c_):
        if not torch.is_tensor(t) or not t.is_cuda:
            return
        try:
            s = t.untyped_storage(); k = s.data_ptr()
        except Exception:
            return
        if k in seen or s.nbytes() == 0:
            return
        seen[k] = c_; cat[c_] += s.nbytes()
    lay = {}
    for m in MOD:
        for n, l in m.named_modules():
            if isinstance(l, L.KernelTernaryLinear):
                lay[id(l)] = (n, l)
    for m in MOD:
        for n, b in m.named_buffers():
            add(b, "trits (packed)" if n.endswith("wpacked") else "buffers (" + n.split(".")[-1] + ")")
        for n, p in m.named_parameters():
            g_ = "adapter" if ".ad." in n else "main"
            if n.endswith(("row_scale", "col_scale")):
                c_ = f"scales row/col ({g_})"
            elif "embed_tokens" in n or "lm_head" in n:
                c_ = "float: embedding / head"
            elif n.endswith("norm.weight") or ".norm." in n:
                c_ = "float: norms"
            elif n.endswith("out_bias"):
                c_ = f"float: out_bias ({g_})"
            else:
                c_ = "float: other params (" + n.split(".")[-1] + ")"
            add(p, c_)
            if p.grad is not None:
                add(p.grad, "grads of float params")
    for o in OPTS:
        for p, s in o.state.items():
            for kk, v in s.items():
                add(v, f"optimizer state ({type(o).__name__}.{kk})")
    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict) and "_ts" in obj and isinstance(obj["_ts"], dict):
                for kk, v in obj.items():
                    if kk == "_ts":
                        for k2, v2 in v.items():
                            nm = k2[0] if isinstance(k2, tuple) else "?"
                            ln = lay.get(k2[1], ("?", None))[0] if isinstance(k2, tuple) and len(k2) > 1 else "?"
                            lt = ln.split(".")[-1]; g_ = "ad" if ".ad." in ln else "T"
                            for t in (v2 if isinstance(v2, tuple) else (v2,)):
                                add(t, f"TTF {'v (factored)' if nm == 'v' else ('sketch sums (bf16)' if nm == 'mx' else nm)} {g_}")
                    else:
                        ln = lay.get(kk, ("?", None))[0]; lt = ln.split(".")[-1] if ln != "?" else "?"
                        g_ = "ad" if ".ad." in ln else "T"
                        for t in (v if isinstance(v, tuple) else (v,)):
                            add(t, f"TTF U/V {g_} {lt}")
            elif isinstance(obj, HQ.Qwen3_5ForCausalLM) and not any(obj is m.hf for m in MOD):
                for t in list(obj.parameters()) + list(obj.buffers()):
                    add(t, "KD teacher weights")
        except Exception:
            pass
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                add(obj, "other live tensors")
        except Exception:
            pass
    return cat


def classify_frames(fr):
    names = [f.get("name", "") for f in fr]; files = [f.get("filename", "") for f in fr]
    if "_causal_fwd" in names:
        return "KD teacher forward"
    if any(n in ("_loss_chunk_kd", "_ce_chunk", "_kd_topk_chunk", "_head_loss") for n in names):
        return "logits / loss (head chunks)"
    if "_log_softmax" in names:
        return "KD teacher log-probs"
    if any("update.py" in f for f in files):
        return "TTF update (new U/V, v, sketch sums, workspace)"
    if any("checkpoint.py" in f for f in files) and "_backward" in names:
        return "checkpoint recompute (backward)"
    if "_backward" in names:
        return "backward (grads / workspace)"
    if any(n == "step" for n in names) and any("optim" in f or "opt_lean" in f for f in files):
        return "optimizer step"
    if any("modeling_qwen3_5" in f or "layers.py" in f or "qwen.py" in f or "kernel.py" in f for f in files):
        return "forward activations (saved / live)"
    if "get_batch" in names:
        return "data"
    return "other (" + (names[0] + "@" + os.path.basename(files[0]) if names else "no stack") + ")"


def memory_report(snap, census_cat):
    """replay the trace: the allocation peak, and the live blocks at it by category"""
    out = {}
    for di, tr in enumerate(snap.get("device_traces", [])):
        live = {}; cur = 0; peak = (0, -1)
        for i, e in enumerate(tr):
            a = e.get("action")
            if a == "alloc":
                live[e["addr"]] = (e["size"], e.get("frames", []), i); cur += e["size"]
                if cur > peak[0]:
                    peak = (cur, i)
            elif a in ("free_requested",):
                b = live.pop(e["addr"], None)
                if b:
                    cur -= b[0]
        if peak[1] < 0:
            continue
        live = {}; cur = 0
        for i, e in enumerate(tr[:peak[1] + 1]):
            a = e.get("action")
            if a == "alloc":
                live[e["addr"]] = (e["size"], e.get("frames", []), i)
            elif a == "free_requested":
                live.pop(e["addr"], None)
        cat = collections.Counter()
        for sz, fr, i in live.values():
            cat[classify_frames(fr)] += sz
        pe = tr[peak[1]]
        where = " <- ".join(f"{f.get('name')}@{os.path.basename(f.get('filename', ''))}:{f.get('line')}" for f in pe.get("frames", [])[:8])
        out[di] = {"trace_peak_bytes_above_start": peak[0], "transient_at_peak": dict(cat), "peak_alloc_where": where,
                   "n_events": len(tr)}
    return out


def report_light():
    p = ST["prof"]; wall = time.time() - ST["t0"]
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    ka = p.key_averages()
    open(OUT + ".ops_cuda.txt", "w").write(ka.table(sort_by="self_cuda_time_total", row_limit=80, max_name_column_width=70))
    open(OUT + ".ops_cpu.txt", "w").write(ka.table(sort_by="self_cpu_time_total", row_limit=80, max_name_column_width=70))
    del ka
    cpu, gpu, nk, syncs = attribute(p)
    lines, comp = summarize(cpu, gpu, nk, wall, N)
    dts = [b - a for a, b in zip(ST["ts"], ST["ts"][1:])][1:]
    head = (f"stepprof {TARGET}: profiled step {FROM}; unprofiled s/step before: "
            + (" ".join(f"{x:.2f}" for x in dts) if dts else "n/a"))
    lines = [head] + lines + ["host syncs per step: " + ", ".join(f"{k} {v / N:.0f}" for k, v in syncs.items())]
    json.dump({"head": head, "wall_profiled": wall / N, "unprofiled": dts, "components": comp,
               "labels_cpu_us": dict(cpu), "labels_gpu_us": dict(gpu), "syncs": dict(syncs)}, open(OUT + ".groups.json", "w"))
    open(OUT + ".groups.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines[:60]), flush=True)
    ST["prof"] = None; del p; gc.collect()


def export_full():
    p = ST["prof"]
    p.export_chrome_trace(OUT + ".trace.json")
    with open(OUT + ".trace.json", "rb") as f, gzip.open(OUT + ".trace.json.gz", "wb") as g:
        shutil.copyfileobj(f, g)
    os.remove(OUT + ".trace.json")
    ST["prof"] = None; del p; gc.collect()
    print(f"stepprof: chrome trace -> {OUT}.trace.json.gz", flush=True)


def start_mem():
    ST["mem"] = True
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ST["mem_census0"] = census(); ST["mem_alloc0"] = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=int(os.environ.get("SP_MEM_ENTRIES", "20000000")), stacks="python")


def mem_finish():
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    pickle.dump(snap, open(OUT + ".mem.pickle", "wb"))
    try:
        from torch.cuda import _memory_viz as mv
        open(OUT + ".mem.html", "w").write(mv.trace_plot(snap))
    except Exception as e_:
        print("memory_viz html failed:", e_, flush=True)
    st = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    rep = {"persistent_at_step_start": dict(ST["mem_census0"]), "allocated_at_step_start": ST["mem_alloc0"],
           "peak_allocated": st["allocated_bytes.all.peak"], "peak_reserved": st["reserved_bytes.all.peak"],
           "reserved_now": st["reserved_bytes.all.current"], "allocated_now": st["allocated_bytes.all.current"],
           "device_used_now": total - free, "cuda_context_and_other": (total - free) - st["reserved_bytes.all.current"],
           "num_alloc_retries": st.get("num_alloc_retries"), "trace": memory_report(snap, ST["mem_census0"])}
    json.dump(rep, open(OUT + ".mem.json", "w"), indent=1)
    G = 2 ** 30
    lines = [f"memory: peak allocated {rep['peak_allocated'] / G:.2f} GiB, peak reserved {rep['peak_reserved'] / G:.2f} GiB, "
             f"allocated at step start {rep['allocated_at_step_start'] / G:.2f} GiB, CUDA context + non-PyTorch "
             f"{rep['cuda_context_and_other'] / G:.2f} GiB, alloc retries {rep['num_alloc_retries']}",
             "persistent at step start (tensor census):"]
    for k, v in sorted(ST["mem_census0"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<44}{v / G:8.3f} GiB")
    lines.append(f"  {'(census total)':<44}{sum(ST['mem_census0'].values()) / G:8.3f} GiB")
    for di, t in rep["trace"].items():
        lines.append(f"transient at the peak (live blocks allocated during the step, by stack): peak at {t['peak_alloc_where']}")
        for k, v in sorted(t["transient_at_peak"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<44}{v / G:8.3f} GiB")
        lines.append(f"  {'(transient total)':<44}{sum(t['transient_at_peak'].values()) / G:8.3f} GiB")
    lines.append(f"fragmentation at the peak ~ peak reserved - peak allocated = {(rep['peak_reserved'] - rep['peak_allocated']) / G:.2f} GiB")
    open(OUT + ".mem.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)
    open(OUT + ".done", "w").write("SP_DONE\n")
    os._exit(0)


def wrap(cls):
    _st = cls.step

    def step(self, *a, **k):
        if self not in OPTS:
            OPTS.append(self)
        r = _push("opt")
        try:
            out = _st(self, *a, **k)
        finally:
            _pop(r)
        kk = ST["k"]; ST["k"] += 1
        if ST["mem"]:
            mem_finish()
        if kk < FROM:
            torch.cuda.synchronize(); ST["ts"].append(time.time())
        A_ = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if kk == FROM - 1 and os.environ.get("SP_MEMONLY") == "1":    # SP_MEMONLY=1: the memory step only
            mark(); start_mem()
        elif kk == FROM - 1:                   # the light profile (attribution) over step FROM
            mark(); torch.cuda.synchronize()
            torch.autograd.set_multithreading_enabled(False)
            ST["prof"] = torch.profiler.profile(activities=A_); ST["prof"].start(); ST["t0"] = time.time()
        elif kk == FROM:
            torch.cuda.synchronize(); ST["prof"].stop(); report_light()
            if os.environ.get("SP_TRACE") == "0":   # attribution only: no trace / memory step (fits a 32 GB PC)
                open(OUT + ".done", "w").write("SP_DONE\n"); sys.stdout.flush(); os._exit(0)
            ST["prof"] =torch.profiler.profile(activities=A_, record_shapes=True, with_stack=True, profile_memory=True)
            torch.cuda.synchronize(); ST["prof"].start()
        elif kk == FROM + 1:
            torch.cuda.synchronize(); ST["prof"].stop(); export_full(); start_mem()
        return out
    cls.step = step


from ttf.opt_lean import LeanAdam
for cls in (torch.optim.AdamW, LeanAdam):
    wrap(cls)
if TARGET.endswith(".py"):
    runpy.run_path(TARGET, run_name="__main__")
else:
    runpy.run_module(TARGET, run_name="__main__", alter_sys=True)
