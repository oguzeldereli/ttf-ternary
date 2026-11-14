"""ttf.export.dumpcheck: teacher-forced per-op check of the runtime against the IR reference.

Run the runtime with TTF_DUMP=DIR (every IR value of one graph call as raw f32 / i32 + DIR/index.json with the ggml ne
of each), then
  python -m ttf.export.dumpcheck MODEL.ttf DIR [--graph step]
Every IR node of the graph is run by ttf.export.ir.OPS on the runtime's own inputs and compared with the runtime's
output: worst rel L2 and the fraction of differing elements per op kind (ttf.linear split by module name).
The dump must cover one call from empty state (a prefill): the reference's state starts empty.
For the head, run the runtime with --dump so it emits all T rows.
"""
import json
import sys

import numpy as np
import torch

from .io import read_ttf
from . import ir as ttf_ir


def _load(ddir, idx, name):
    e = idx[name]
    ne = e["ne"]
    # the torch rank when the dump gives it (at T = 1 ggml drops trailing 1s: trimming ne would guess wrong)
    nd = e.get("nd") or max([i + 1 for i, n in enumerate(ne) if n > 1] + [1])
    a = np.fromfile(f"{ddir}/{name}.bin", np.float32 if e["type"] == "f32" else np.int32)
    return torch.from_numpy(a.reshape(list(reversed(ne[:nd]))))


def check_dump(path, ddir, graph="step", log=print):
    """-> {op kind: (worst rel L2, fraction != 0 at the worst, count, value name)}"""
    kv, W = read_ttf(path)
    ir = json.loads(kv["ttf.ir"])
    D = ttf_ir.DTYPES[kv["ttf.compute_dtype"]]
    run = ttf_ir.Runner(ir, W, D)
    idx = json.load(open(ddir + "/index.json"))
    worst, st = {}, {}
    for n in ir["graphs"][graph]["nodes"]:
        if any(i is not None and i not in idx for i in n["in"]) or n["out"][0] not in idx:
            log("skip", n["op"], n["out"])
            continue
        ins = [_load(ddir, idx, i) if i is not None else None for i in n["in"]]
        ref = ttf_ir.OPS[n["op"]](ins, n, run.W, st, D)[0].float()
        eng = _load(ddir, idx, n["out"][0]).reshape(ref.shape)
        d = (eng - ref).abs()
        rel = (d.norm() / ref.norm().clamp_min(1e-30)).item()
        neq = (d > 0).float().mean().item()
        k = n["op"] + ("" if n["op"] != "ttf.linear" else "." + n["p"].split(".")[-1])
        w = worst.setdefault(k, [0.0, 0.0, 0, ""])
        w[2] += 1
        if rel >= w[0]:
            w[0], w[1], w[3] = rel, neq, n["out"][0]
    return {k: tuple(v) for k, v in worst.items()}


if __name__ == "__main__":
    g = sys.argv[sys.argv.index("--graph") + 1] if "--graph" in sys.argv else "step"
    torch.set_num_threads(4)
    for k, (rel, neq, cnt, where) in sorted(check_dump(sys.argv[1], sys.argv[2], g).items()):
        print(f"{k:22s} n={cnt:3d} worst rel {rel:.2e}  frac!=0 {neq:.4f}  at {where}")
