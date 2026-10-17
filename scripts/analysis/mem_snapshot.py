"""What holds GPU memory at the peak of a training step (from ttf.train --mem-snapshot PATH:STEP).

    python scripts/analysis/mem_snapshot.py SNAPSHOT.pickle [--top 30]

Replays the recorded allocations and frees, finds the moment the allocated bytes peak (and the reserved bytes, which
add the allocator's unused cached blocks: fragmentation), and groups the blocks alive then by the innermost frame in
ttf/ (file:line function), and also by the outermost ttf/ frame (which phase of the step). Blocks that are still
alive at the end of the trace are counted as persistent (trits, float tail, state), the rest as transient.
"""
import argparse
import pickle
from collections import defaultdict

GiB = 2 ** 30


def site(frames, inner=True):
    fs = [f for f in frames if "ttf/" in f.get("filename", "")]
    if not fs:
        fs = [f for f in frames if "torch/" not in f.get("filename", "")] or frames[:1]
    if not fs:
        return "?"
    f = fs[0] if inner else fs[-1]
    return f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap"); ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    snap = pickle.load(open(args.snap, "rb"))
    tr = snap["device_traces"][0]
    live, cur, res = {}, 0, 0
    peak, peak_i, rpeak, rpeak_i, rpeak_alloc = 0, -1, 0, -1, 0
    for i, e in enumerate(tr):
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = (e["size"], i); cur += e["size"]
        elif a == "free_completed":
            s = live.pop(e["addr"], None)
            if s:
                cur -= s[0]
        elif a == "segment_alloc":
            res += e["size"]
        elif a == "segment_free":
            res -= e["size"]
        elif a == "oom":
            print(f"OOM event at entry {i}: allocated {cur / GiB:.2f} GiB, reserved {res / GiB:.2f} GiB, "
                  f"request {e.get('size', 0) / 2**20:.0f} MiB")
        if cur > peak:
            peak, peak_i = cur, i
        if res > rpeak:
            rpeak, rpeak_i, rpeak_alloc = res, i, cur
    end_live = {v[1] for v in live.values()}
    print(f"{len(tr)} events; peak allocated {peak / GiB:.2f} GiB at event {peak_i}; peak reserved {rpeak / GiB:.2f} GiB "
          f"(allocated then {rpeak_alloc / GiB:.2f}: {(rpeak - rpeak_alloc) / GiB:.2f} GiB cached but unused); "
          f"alive at the end {sum(v[0] for v in live.values()) / GiB:.2f} GiB")
    live = {}
    for i, e in enumerate(tr[:peak_i + 1]):
        if e["action"] == "alloc":
            live[e["addr"]] = (e["size"], i)
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    for label, inner in (("innermost ttf frame", True), ("outermost ttf frame (phase)", False)):
        by = defaultdict(lambda: [0, 0, 0])
        for addr, (s, i) in live.items():
            g = by[site(tr[i].get("frames", []), inner)]
            g[0 if i in end_live else 1] += s; g[2] += 1
        print(f"\n== alive at the peak, by {label}: persistent / transient GiB (blocks) ==")
        tot_p = sum(v[0] for v in by.values()); tot_t = sum(v[1] for v in by.values())
        print(f"   total: persistent {tot_p / GiB:.2f}, transient {tot_t / GiB:.2f}")
        for k, (p, t, n) in sorted(by.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:args.top]:
            print(f"   {p / GiB:6.2f} {t / GiB:6.2f} ({n:5d})  {k}")
    segs = snap.get("segments", [])
    if segs:
        tot = sum(s["total_size"] for s in segs); used = sum(s["allocated_size"] for s in segs)
        print(f"\nat the dump: {len(segs)} segments, reserved {tot / GiB:.2f} GiB, allocated {used / GiB:.2f} GiB")


if __name__ == "__main__":
    main()
