"""Teacher targets for --kd-cache from a GGUF teacher (e.g. Qwen3.8-27B Q4_K_M, fits a 24 GB card) through llama.cpp
(llama_topk.cpp): the same kd_NNNNN.npz shards as teacher_topk.py (starts [B] int64, ids [B, seq, k] int32,
lp [B, seq, k] fp16, log-probabilities under the full softmax), so the trainer reads either.

  python scripts/quant/teacher_gguf.py MODEL.gguf DATA.bin OUT_DIR --tool ./llama_topk [--sequential] [--part R/N]

Window starts: --sequential = every non-overlapping seq-token window of DATA once, in an order shuffled with --seed
(each training token seen once over the cache); otherwise uniform draws as teacher_topk.py. --part R/N: this machine
writes only the shards with index % N == R (the order and the starts are the same on every machine), so N machines fill
one cache; restarts skip finished shards. OUT_DIR/done is written by whoever finds every shard present.
"""
import argparse, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("gguf"); ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--tool", required=True)
ap.add_argument("--windows", type=int, default=0, help="default: all windows with --sequential")
ap.add_argument("--batch", type=int, default=48); ap.add_argument("--k", type=int, default=32)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--sequential", action="store_true"); ap.add_argument("--part", default="0/1")
ap.add_argument("--ngl", type=int, default=99)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
data = np.memmap(a.data, dtype=np.uint32, mode="r")
rng = np.random.default_rng(a.seed)
if a.sequential:
    nall = (len(data) - 1) // a.seq
    starts_all = rng.permutation(nall).astype(np.int64) * a.seq
    starts_all = starts_all[:a.windows] if a.windows else starts_all
else:
    starts_all = rng.integers(0, len(data) - a.seq - 1, a.windows)
nsh = len(starts_all) // a.batch
r, N = map(int, a.part.split("/"))
mine = [sh for sh in range(nsh) if sh % N == r and not os.path.exists(os.path.join(a.out, f"kd_{sh:05d}.npz"))]
print(f"{len(starts_all)} windows, {nsh} shards; part {r}/{N}: {len(mine)} to do", flush=True)
proc = subprocess.Popen([a.tool, a.gguf, str(a.k), str(a.ngl), str(a.seq)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for sh in mine:
    fn = os.path.join(a.out, f"kd_{sh:05d}.npz")
    t1 = time.time()
    st = starts_all[sh * a.batch:(sh + 1) * a.batch]
    top_i = np.empty((len(st), a.seq, a.k), dtype=np.int32)
    top_l = np.empty((len(st), a.seq, a.k), dtype=np.float16)
    for b, s in enumerate(st):
        w = np.asarray(data[s:s + a.seq], dtype=np.int32)
        proc.stdin.write(np.int32(a.seq).tobytes() + w.tobytes()); proc.stdin.flush()
        top_i[b] = np.frombuffer(proc.stdout.read(a.seq * a.k * 4), dtype=np.int32).reshape(a.seq, a.k)
        top_l[b] = np.frombuffer(proc.stdout.read(a.seq * a.k * 2), dtype=np.float16).reshape(a.seq, a.k)
    tmp = fn + ".tmp.npz"
    np.savez(tmp, starts=st, ids=top_i, lp=top_l)
    os.replace(tmp, fn)
    cov = float(np.exp(top_l.astype(np.float32)).sum(-1).mean())
    print(f"shard {sh}: {len(st)} windows in {time.time() - t1:.0f}s, top-{a.k} mass {cov:.3f}", flush=True)
proc.stdin.write(np.int32(0).tobytes()); proc.stdin.close(); proc.wait()
if all(os.path.exists(os.path.join(a.out, f"kd_{sh:05d}.npz")) for sh in range(nsh)):
    open(os.path.join(a.out, "done"), "w").close()
