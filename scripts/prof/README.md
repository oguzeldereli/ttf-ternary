# Profilers

Each runs a trainer (normally `ttf.train`) as `__main__`, hooks the optimizer step, measures, and exits. They change
nothing in what runs. Multi-step `torch.profiler` traces need a lot of host memory (a 0.8B adapter step's `with_stack`
profile does not fit in 32 GB): run them on a large-memory machine.

| script | what it gives |
|---|---|
| `steptime.py` | wall seconds per optimizer step, GPU synchronised, after warm-up steps |
| `stepprof.py` | exclusive time attribution by named range (embedding, attention, MLP, ternary forward / dx / dw, TTF update, loss head, optimizer), op tables, a chrome trace and a memory snapshot of one step each |
| `qmem.py` | GPU memory of one step: static tensors by group and dtype, live allocations at the peak by category |
| `cpuprof.py` | cProfile of the Python side over a few steps (launch overhead, host syncs) |
| `teacher_time.py` | the bf16 teacher's forward time per layer (KD with an online teacher) |

The trainer's own `--profile N` (torch.profiler over N steps), `--mem-snapshot PATH:STEP` and `--mem-report` cover the
quick cases; `scripts/analysis/mem_snapshot.py` reads a snapshot.
