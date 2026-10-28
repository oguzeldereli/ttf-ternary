"""--ext MODULE[:SPEC] (repeatable): structure plug-ins, loaded by name, in the order given. Replaces the import-order
monkeypatching of ttf.qwen that the experiment scripts relied on (a script imported earlier or later changed which
wrappers were active). A plug-in module defines

    ttf_ext(spec: str) -> an object with any of these methods (both optional):

  build(model, swapped, make_linear, log) -> swapped      after build_qwen, on a fresh start and on resume: wrap or add
                                                          layers (a wrapper keeps its ternary layer as .lin), so that a
                                                          checkpoint written with them loads
  setup(model, log)                                       before the first step, after the weights are loaded (fresh
                                                          start, --init-ckpt and resume alike): what trains, not the
                                                          structure (e.g. layers frozen: their ts_frozen and their
                                                          parameters' requires_grad)

The spec string is the plug-in's own (e.g. "rank=0.2,alt=1"); it is saved with the run's arguments, and a resume must
pass the same --ext list (train.py refuses a different one). Only build may change the structure; plug-ins that add
parameters must do so in build (the optimizer is built after it). The conversion's hooks (begin / weight / refine /
fitted / end) left the core with the conversion on 28 Oct 2026 (the ttf-conversion package)."""
import importlib

_ACTIVE = []


def load(specs):
    """--ext values -> plug-in objects (also kept as the active list)"""
    _ACTIVE.clear()
    for s in specs or ():
        mod, _, spec = s.partition(":")
        m = importlib.import_module(mod)
        if not hasattr(m, "ttf_ext"):
            raise SystemExit(f"--ext {s}: module {mod} has no ttf_ext(spec)")
        _ACTIVE.append((s, m.ttf_ext(spec)))
    return [e for _, e in _ACTIVE]


def active():
    return [e for _, e in _ACTIVE]


def having(name):
    """the active plug-ins' methods called name, in order"""
    return [getattr(e, name) for e in active() if hasattr(e, name)]


def base(q):
    """a swapped module's ternary layer: the module itself, or a wrapper's .lin"""
    while not hasattr(q, "wpacked") and hasattr(q, "lin"):    # (wrappers of wrappers: down to the ternary layer)
        q = q.lin
    return q


def structural(specs):
    """the specs (of a --ext list) whose plug-ins define build: the ones that must match between a run and its resume"""
    out = []
    for s in specs or ():
        mod, _, spec = s.partition(":")
        if hasattr(importlib.import_module(mod).ttf_ext(spec), "build"):
            out.append(s)
    return out
