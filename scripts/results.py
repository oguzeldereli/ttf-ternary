"""The results database from the command line (ttf/results.py; default file results/runs.jsonl or $TTF_RESULTS).

  python scripts/results.py status                          runs whose last event is queued / running (and failed lately)
  python scripts/results.py table --metric wiki_fixed [--runs PATTERN] [--metric fw_fixed ...]
  python scripts/results.py show RUN                        a run's record, events, results and notes (RUN: id or name)
  python scripts/results.py find TEXT [--tag T] [--type note]
  python scripts/results.py note "TEXT" [--runs a,b] [--tags x,y]
  python scripts/results.py result RUN METRIC VALUE [--n N] [--eval WHAT]
  python scripts/results.py event RUN STATUS [--step S] [--reason R]
  python scripts/results.py ingest FILE [FILE ...]          merge records files (e.g. a run dir's records.jsonl)
  python scripts/results.py validate
"""
import argparse, fnmatch, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ttf import results as R   # noqa: E402


def runs_matching(recs, key):
    """run ids whose id or name matches key (exact, then glob, then substring)"""
    runs = [r for r in recs.values() if r["type"] == "run"]
    for test in (lambda r: key in (r["id"], r["name"]), lambda r: fnmatch.fnmatch(r["name"], key),
                 lambda r: key in r["name"]):
        m = [r["id"] for r in runs if test(r)]
        if m:
            return m
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    t = sub.add_parser("table"); t.add_argument("--metric", action="append", required=True); t.add_argument("--runs", default="*")
    t.add_argument("--sort", default=None)
    s = sub.add_parser("show"); s.add_argument("run")
    f = sub.add_parser("find"); f.add_argument("text", nargs="?", default=""); f.add_argument("--tag"); f.add_argument("--type")
    n = sub.add_parser("note"); n.add_argument("text"); n.add_argument("--runs", default=""); n.add_argument("--tags", default="")
    r = sub.add_parser("result"); r.add_argument("run"); r.add_argument("metric"); r.add_argument("value", type=float)
    r.add_argument("--n", type=int); r.add_argument("--eval")
    e = sub.add_parser("event"); e.add_argument("run"); e.add_argument("status", choices=R.STATUSES)
    e.add_argument("--step", type=int); e.add_argument("--reason")
    i = sub.add_parser("ingest"); i.add_argument("files", nargs="+")
    sub.add_parser("validate")
    a = ap.parse_args()
    db = a.db
    allrecs = R.load(db)
    cur = R.current(allrecs)

    if a.cmd == "ingest":
        added, skipped = R.ingest(a.files, db)
        print(f"{added} records added, {skipped} already present or invalid")
    elif a.cmd == "validate":
        bad = [(r.get("id"), e) for r in allrecs for e in R.validate(r)]
        ids = [r.get("id") for r in allrecs]
        dup = len(ids) - len(set(ids))
        by = {}
        for r_ in cur.values():
            by[r_.get("type", "?")] = by.get(r_.get("type", "?"), 0) + 1
        print(f"{len(allrecs)} records ({', '.join(f'{v} {k}' for k, v in sorted(by.items()))} in force), "
              f"{len(bad)} malformed, {dup} duplicate ids")
        for x in bad[:20]:
            print("  ", *x)
        sys.exit(1 if bad or dup else 0)
    elif a.cmd == "note":
        print(R.add("note", db, text=a.text, runs=[x for x in a.runs.split(",") if x] or None,
                    tags=[x for x in a.tags.split(",") if x] or None))
    elif a.cmd in ("result", "event"):
        m = runs_matching(cur, a.run)
        if len(m) != 1:
            sys.exit(f"run {a.run!r}: {len(m)} matches {m[:5]}")
        if a.cmd == "result":
            print(R.add("result", db, run=m[0], metric=a.metric, value=a.value, n=a.n, eval=a.eval))
        else:
            print(R.add("event", db, run=m[0], status=a.status, step=a.step, reason=a.reason))
    elif a.cmd == "status":
        last = {}
        for r_ in cur.values():
            if r_["type"] == "event":
                if r_["run"] not in last or r_["time"] >= last[r_["run"]]["time"]:
                    last[r_["run"]] = r_
        for rid, ev in sorted(last.items(), key=lambda kv: kv[1]["time"]):
            if ev["status"] in ("queued", "running"):
                run = cur.get(rid, {})
                print(f"{ev['status']:8s} {ev['time']}  {rid}  step {ev.get('step', '-')}  {run.get('purpose', '')}")
    elif a.cmd == "table":
        ids = runs_matching(cur, a.runs) if a.runs != "*" else [r_["id"] for r_ in cur.values() if r_["type"] == "run"]
        val = {}
        for r_ in cur.values():
            if r_["type"] == "result" and r_["run"] in ids and r_["metric"] in a.metric:
                k = (r_["run"], r_["metric"])
                if k not in val or r_["time"] >= val[k][1]:
                    val[k] = (r_["value"], r_["time"])
        rows = [(rid, [val.get((rid, m), (None,))[0] for m in a.metric]) for rid in ids]
        rows = [x for x in rows if any(v is not None for v in x[1])]
        key = a.sort or a.metric[0]
        j = a.metric.index(key)
        rows.sort(key=lambda x: (x[1][j] is None, x[1][j] if x[1][j] is not None else 0))
        w = max([len(x[0]) for x in rows] + [4])
        print(f"{'run':{w}s}  " + "  ".join(f"{m:>12s}" for m in a.metric))
        for rid, vs in rows:
            print(f"{rid:{w}s}  " + "  ".join(f"{v:12.4f}" if v is not None else f"{'-':>12s}" for v in vs))
    elif a.cmd == "show":
        m = runs_matching(cur, a.run)
        if not m:
            sys.exit(f"no run matches {a.run!r}")
        for rid in m[:5]:
            run = cur[rid]
            print(json.dumps({k: v for k, v in run.items() if k != "config"}, indent=1, default=str))
            if run.get("config"):
                print("config:", " ".join(f"{k}={v}" for k, v in sorted(run["config"].items()) if v not in (None, False, "", 0)))
            for r_ in sorted(cur.values(), key=lambda x: x["time"]):
                if r_["type"] == "event" and r_["run"] == rid:
                    print(f"  event  {r_['time']} {r_['status']} step {r_.get('step', '-')} {r_.get('reason', '')}")
                elif r_["type"] == "result" and r_["run"] == rid:
                    print(f"  result {r_['metric']} = {r_['value']:.4f}" + (f" (n {r_['n']})" if r_.get("n") else ""))
                elif r_["type"] == "note" and (rid in (r_.get("runs") or []) or run["name"] in (r_.get("runs") or [])):
                    print(f"  note   {r_['time']} {r_['text'][:300]}")
    elif a.cmd == "find":
        for r_ in sorted(cur.values(), key=lambda x: x["time"]):
            if a.type and r_["type"] != a.type:
                continue
            if a.tag and a.tag not in (r_.get("tags") or []):
                continue
            blob = json.dumps(r_, default=str)
            if a.text.lower() in blob.lower():
                txt = r_.get("text") or r_.get("name") or f"{r_.get('metric')} {r_.get('value')}"
                print(f"{r_['type']:6s} {r_['time']}  {r_['id']}  {str(txt)[:200]}")


if __name__ == "__main__":
    main()
