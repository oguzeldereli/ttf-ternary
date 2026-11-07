"""CPU-side profile of a trainer (23 Oct, the CPU-overhead fix): cProfile over CPUPROF_N optimizer steps after
CPUPROF_SKIP warm-up steps (no GPU event recording: small memory); prints the top functions by own time and by
cumulative time, the step wall times, and exits. CPUPROF_MT=0: the autograd engine's multithreading off, so the backward's
Python (custom Functions, checkpoint recompute, the TTF update) runs on the profiled thread instead of a device thread.   CPUPROF_OUT=prof.txt python scripts/prof/cpuprof.py TARGET ARGS
"""
import cProfile, io, os, pstats, runpy, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TTF_CORE", os.getcwd()))
import torch
from ttf.opt_lean import LeanAdam

N = int(os.environ.get("CPUPROF_N", "3")); SKIP = int(os.environ.get("CPUPROF_SKIP", "2"))
OUT = os.environ.get("CPUPROF_OUT", "cpuprof.txt"); TARGET = sys.argv[1]; sys.argv = [TARGET] + sys.argv[2:]
PR = cProfile.Profile(); TS = []
if os.environ.get("CPUPROF_MT") == "0":
    torch.autograd.set_multithreading_enabled(False)


def wrap(cls):
    _st = cls.step

    def step(self, closure=None):
        out = _st(self, closure)
        torch.cuda.synchronize(); TS.append(time.time()); k = len(TS)
        if k == SKIP:
            PR.enable()
        elif k == SKIP + N:
            PR.disable()
            d = [b - a for a, b in zip(TS[SKIP - 1:], TS[SKIP:])]
            s = io.StringIO(); ps = pstats.Stats(PR, stream=s)
            s.write(f"cpuprof {TARGET}: {N} steps profiled, wall per step {' '.join(f'{x:.2f}' for x in d)} s\n")
            ps.sort_stats("tottime").print_stats(35); ps.sort_stats("cumulative").print_stats(45)
            open(OUT, "w").write(s.getvalue()); print(s.getvalue()[:3000], flush=True)
            os._exit(0)
        return out
    cls.step = step


for cls in (torch.optim.AdamW, LeanAdam):
    wrap(cls)
if TARGET.endswith(".py"):
    runpy.run_path(TARGET, run_name="__main__")
else:
    runpy.run_module(TARGET, run_name="__main__", alter_sys=True)
