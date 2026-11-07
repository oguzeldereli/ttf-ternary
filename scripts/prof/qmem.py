"""GPU memory breakdown of one training step (23 Oct 2026: where a 27B conversion model's peak goes).
Runs TARGET (e.g. ttf.train, as __main__) with the CUDA caching allocator's history recorded from the start
(python stacks); after the first optimizer step it prints
  - static: every parameter / buffer on the GPU, by group (T1 / T2 / adapter ternary layers, float tail) and dtype
  - peak: the live allocations at the step's peak, by category from their allocation stacks (weights and buffers from
    the load, forward activations kept for backward, backward / dx workspace, TTF update, loss head, optimizer, other),
    and for the forward activations the share made inside each layer group
and exits. QMEM_OUT=FILE also gets the text. Measures existing code only.
(The float optimizer state is part of a real step's memory: run with the optimizer as the recipe has it.)
  QMEM_OUT=mem.txt python scripts/prof/qmem.py ttf.train ARGS... --stop-after 1
"""
import collections, os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import ttf.qwen as Q
from ttf.layers import KernelTernaryLinear
from ttf.opt_lean import LeanAdam

OUT = os.environ.get("QMEM_OUT", "qmem.txt"); MOD = [None]; CNT = [0]
GiB = 2 ** 30
_bq = Q.build_qwen


def build_qwen(*a, **k):
    m, sw = _bq(*a, **k); MOD[0] = m
    return m, sw


def group(name):
    return "T2" if ".lin2" in name else ("adapter" if ".ad." in name else "T1")


def static(m):
    tern = {id(l): group(n) for n, l in m.named_modules() if isinstance(l, KernelTernaryLinear)}
    rows = collections.Counter(); seen = set()
    for n, l in m.named_modules():
        g = tern.get(id(l))
        for k, t in list(l.named_parameters(recurse=False)) + list(l.named_buffers(recurse=False)):
            if t is None or not t.is_cuda or t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr())
            kind = ("trits" if k in ("wpacked",) else "ternary-layer " + ("param" if isinstance(t, torch.nn.Parameter) else "buffer")) if g else ("float param" if isinstance(t, torch.nn.Parameter) else "float buffer")
            rows[(g or "float", kind, str(t.dtype).replace("torch.", ""))] += t.numel() * t.element_size()
    return rows


def cat(frames):
    names = [(os.path.basename(f.get("filename", "")), f.get("name", "")) for f in frames]
    s = " ".join(f"{a}:{b}" for a, b in names)
    if "_ts_layer" in s or "update.py" in s:
        return "TTF update"
    if "step" in [b for _, b in names] and ("opt_lean.py" in s or "optim" in s):
        return "optimizer"
    if "backward" in s or "_grad_x" in s:
        return "backward / dx workspace"
    if any("loss" in b or "head" in b for _, b in names):
        return "loss head"
    if "forward" in s:
        return "forward (kept for backward)"
    if any(k in s for k in ("convert_qwen_streaming", "build_qwen", "load_state", "_apply", "torch/nn/modules/module.py:to")):
        return "weights / buffers (load)"
    return "other"


def fwd_group(frames):
    for f in frames:                     # innermost first: the first layer forward on the stack
        if os.path.basename(f.get("filename", "")) == "layers.py" and f.get("name") == "forward":
            return "in ternary layer"
    return "outside ternary layers"


def peak_report():
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    ev = [e for tr in snap["device_traces"] for e in tr]
    live, tot, best, bi = {}, 0, -1, -1
    for i, e in enumerate(ev):
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = e["size"]; tot += e["size"]
            if tot > best:
                best, bi = tot, i
        elif a == "free_completed" and e["addr"] in live:
            tot -= live.pop(e["addr"])
    live = {}
    for e in ev[:bi + 1]:
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    by = collections.Counter(); fg = collections.Counter(); big = []
    for e in live.values():
        c = cat(e.get("frames", [])); by[c] += e["size"]
        if c == "forward (kept for backward)":
            fg[fwd_group(e.get("frames", []))] += e["size"]
        big.append((e["size"], c, " < ".join(f"{os.path.basename(f['filename'])}:{f['name']}:{f['line']}" for f in e.get("frames", [])[:4])))
    lines = [f"peak in the recorded history: {best / GiB:.2f} GiB (event {bi} of {len(ev)}; torch max_memory_allocated "
             f"{torch.cuda.max_memory_allocated() / GiB:.2f} GiB; history may have dropped early events: "
             f"{'yes' if len(ev) >= 4_000_000 else 'no'})", "at the peak, by category:"]
    for c, b in by.most_common():
        lines.append(f"  {c:<32}{b / GiB:8.2f} GiB")
    for c, b in fg.most_common():
        lines.append(f"    forward: {c:<26}{b / GiB:8.2f} GiB")
    lines.append("largest 15 live blocks at the peak:")
    for s, c, fr in sorted(big, reverse=True)[:15]:
        lines.append(f"  {s / GiB:6.2f} GiB  {c:<28} {fr[:150]}")
    return lines


def wrap(cls):
    _st = cls.step

    def step(self, closure=None):
        out = _st(self, closure)
        CNT[0] += 1
        if CNT[0] == 1:
            lines = [f"qmem {TARGET} {' '.join(sys.argv[1:])[:300]}", "static (GPU tensors of the model):"]
            if MOD[0] is not None:
                rows = static(MOD[0]); tot = 0
                for (g, kind, dt), b in sorted(rows.items()):
                    lines.append(f"  {g:<8}{kind:<24}{dt:<10}{b / GiB:8.2f} GiB"); tot += b
                lines.append(f"  total static {tot / GiB:.2f} GiB")
            lines += peak_report()
            txt = "\n".join(lines); print(txt, flush=True); open(OUT, "w").write(txt + "\n")
            os._exit(0)
        return out
    cls.step = step


if __name__ == "__main__":   # (static / peak_report can be imported)
    torch.cuda.memory._record_memory_history(max_entries=int(os.environ.get("QMEM_ENTRIES", "4000000")), stacks="python")   # (27B 2 x 8: 4M drops the early events)
    TARGET = sys.argv[1]; sys.argv = [TARGET] + sys.argv[2:]
    Q.build_qwen = build_qwen
    for cls in (torch.optim.AdamW, LeanAdam):
        wrap(cls)
    if TARGET.endswith(".py"):
        runpy.run_path(TARGET, run_name="__main__")
    else:                                            # a module (e.g. ttf.train, the core trainer)
        runpy.run_module(TARGET, run_name="__main__", alter_sys=True)
