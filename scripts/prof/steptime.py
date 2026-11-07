"""wall time per optimizer step of a lab trainer (TARGET .py or ttf.train), GPU synchronised, for STEPTIME_N steps after
STEPTIME_SKIP warm-up steps; prints "steptime: mean s/step ..." and exits (23 Oct, the CPU-overhead fix's measurements).
  STEPTIME_N=6 python scripts/prof/steptime.py ttf.train ARGS --resume ...
"""
import os, runpy, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TTF_CORE", os.getcwd()))
import torch
from ttf.opt_lean import LeanAdam

N = int(os.environ.get("STEPTIME_N", "6")); SKIP = int(os.environ.get("STEPTIME_SKIP", "2"))
TARGET = sys.argv[1]; sys.argv = [TARGET] + sys.argv[2:]
TS = []


def wrap(cls):
    _st = cls.step

    def step(self, closure=None):
        out = _st(self, closure)
        torch.cuda.synchronize(); TS.append(time.time())
        if len(TS) == SKIP + N + 1:
            d = [b - a for a, b in zip(TS[SKIP:], TS[SKIP + 1:])]
            print(f"steptime window {TS[SKIP]:.3f} {TS[-1]:.3f}", flush=True)
            print(f"steptime: mean {sum(d) / len(d):.3f} s/step over {len(d)} steps (min {min(d):.3f}, max {max(d):.3f}; "
                  f"first {SKIP} skipped); peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            os._exit(0)
        return out
    cls.step = step


for cls in (torch.optim.AdamW, LeanAdam):
    wrap(cls)
if TARGET.endswith(".py"):
    runpy.run_path(TARGET, run_name="__main__")
else:
    runpy.run_module(TARGET, run_name="__main__", alter_sys=True)
