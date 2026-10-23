"""--ext MODULE[:SPEC] (repeatable): structure and conversion plug-ins, loaded by name, in the order given. Replaces the
import-order monkeypatching of ttf.qwen / ttf.hf_init that the experiment scripts relied on (a script imported earlier
or later changed which wrappers were active). A plug-in module defines

    ttf_ext(spec: str) -> an object with any of these methods (all optional):

  build(model, swapped, make_linear, log) -> swapped      after build_qwen, on a fresh start and on resume: wrap or add
                                                          layers. A wrapper keeps its ternary layer as .lin; the
                                                          conversion rounds .lin and puts the wrapper back at the end
  begin(model, files, dev, log)                           streaming conversion: after the float parts are loaded, before
                                                          the calibration inputs (e.g. a residual-stream fold)
  weight(name, W, layer_idx) -> W                         the fp weight (float32) that is rounded (e.g. the fold)
  refine(name, W, Hc, T, a, c) -> (T, a, c)               after GPTQ (e.g. coordinate descent)
  fitted(name, wrapper, W, Hc, mu, T, a, c)               after the layer's own trits are final: fits on what they left
                                                          (adapters, a second ternary layer)
  end(model, log)                                         after the last layer

The spec string is the plug-in's own (e.g. "rank=0.2,alt=1"); it is saved with the run's arguments, so a resume builds
the same structure. Plug-ins that add parameters must do so in build (the optimizer is built after it)."""
import importlib

_ACTIVE = []


def load(specs):
    """--ext values -> plug-in objects (also kept as the active list for the conversion)"""
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
    return getattr(q, "lin", q) if not hasattr(q, "wpacked") else q
