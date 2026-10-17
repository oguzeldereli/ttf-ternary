"""Backfill the results database from run directories (one-off, 17 Oct consolidation; moves to ttf-lab afterwards).

Every directory below ROOT holding a metrics.jsonl, train.log or final_eval.json is one run:
  run     name (directory), host, time (first timestamp of train.log, else the files' oldest mtime), group (the path
          above it), config (the flags of a 'start NAME on HOST: FLAGS' log line, else the args saved in ckpt.pt when
          --ckpt_args), steps / tokens (metrics.jsonl), ckpt (HOST:path when a checkpoint file exists)
  event   finished (a FINAL line or final metrics record) or stopped (neither: crashed, killed or still running)
  result  val_train (the final val), val_last (the last val when there is no final one), wiki_fixed / fw_fixed /
          protocol (final_eval.json)
Ids are deterministic (name@host@time; results and events from the run id), so a second run adds nothing.
Qwen arm logs (logs/q08*.txt, no run directory) give run + val_train records too.
  python scripts/backfill/backfill_dirs.py --root checkpoints --host laptop --out /tmp/x.jsonl [--ckpt_args]
"""
import argparse, glob, hashlib, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf import results as R   # noqa: E402

TS = re.compile(r"(20\d\d-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)")
START = re.compile(r"start (\S+) (?:seed \S+ )?on (\S+?):? (--.*?)(?: --resume)?\s*$")
FINAL = re.compile(r"FINAL val loss ([0-9.]+)")


def parse_flags(s):
    toks, cfg, i = s.split(), {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                v = toks[i + 1]; i += 2
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                cfg[k] = v
            else:
                cfg[k] = True; i += 1
        else:
            i += 1
    return cfg


def rid(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def iso(d, t):
    return f"{d}T{t}"


SEEN = set()        # run ids of this backfill: a copy of a run elsewhere gets its group appended


def run_dir_records(d, root, host, ckpt_args, src_prefix):
    name = os.path.basename(d)
    files = {f: os.path.join(d, f) for f in ("train.log", "metrics.jsonl", "final_eval.json") if os.path.exists(os.path.join(d, f))}
    log = open(files["train.log"], errors="ignore").read() if "train.log" in files else ""
    m = TS.search(log)
    if m:
        t0 = iso(*m.groups())
    else:
        mt = min(os.path.getmtime(p) for p in files.values())
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt))
    cfg, purpose = None, None
    sm = [x for x in (START.search(l) for l in log.splitlines()) if x]
    if sm:
        cfg = parse_flags(sm[-1].group(3))
    ck = None
    for c in ("ckpt.pt", "eval.pt", "ckpt_9154.pt"):
        if os.path.exists(os.path.join(d, c)):
            ck = os.path.join(d, c); break
    if cfg is None and ckpt_args and ck:
        try:
            import torch
            blob = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
            a = blob.get("args")
            if a is not None:
                cfg = a if isinstance(a, dict) else vars(a)
                cfg = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
            else:   # older checkpoints keep their settings as top-level fields, the model shape in cfg
                cfg = {k: v for k, v in blob.items() if isinstance(v, (int, float, str, bool)) and k not in ("step",)}
                c_ = blob.get("cfg")
                if c_ is not None:
                    cfg.update({"model." + k: v for k, v in vars(c_).items() if isinstance(v, (int, float, str, bool))})
                cfg["_partial"] = True     # what the checkpoint recorded, not the full command line
            del blob
        except Exception as e:
            print(f"  {d}: ckpt args not read ({type(e).__name__})", file=sys.stderr)
    steps = tokens = None
    final = last = None
    if "metrics.jsonl" in files:
        for line in open(files["metrics.jsonl"], errors="ignore"):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in x:
                steps = max(steps or 0, int(x["step"]))
            if x.get("tokens"):
                tokens = max(tokens or 0, int(x["tokens"]))
            if "val_loss" in x:
                last = (x["val_loss"], x.get("step"))
                if x.get("final"):
                    final = last
    if final is None:
        fm = FINAL.findall(log)
        if fm:
            final = (float(fm[-1]), steps)
    group = os.path.relpath(os.path.dirname(d), root)
    run_id = f"{name}@{host}@{t0.replace('-', '').replace(':', '')}"
    if run_id in SEEN:
        run_id += "~" + (group if group != "." else "top").replace("/", "~")
    SEEN.add(run_id)
    run = R.make("run", id=run_id, time=t0, name=name, host=host,
                 group=None if group == "." else group, config=cfg, steps=steps, tokens=tokens,
                 ckpt=(f"{host}:{src_prefix}{os.path.relpath(ck, root)}" if ck else None),
                 source=f"{host}:{src_prefix}{os.path.relpath(d, root)}")
    out = [run]
    ev = "finished" if final else "stopped"
    out.append(R.make("event", id="e-" + rid(run["id"], ev), time=t0, run=run["id"], status=ev, step=steps,
                      reason=None if final else "no final validation in the logs (backfill)"))
    if final:
        out.append(R.make("result", id="r-" + rid(run["id"], "val_train"), time=t0, run=run["id"], metric="val_train",
                          value=float(final[0]), step=final[1], source=run["source"] + "/metrics.jsonl"))
    elif last:
        out.append(R.make("result", id="r-" + rid(run["id"], "val_last"), time=t0, run=run["id"], metric="val_last",
                          value=float(last[0]), step=last[1], source=run["source"] + "/metrics.jsonl"))
    if "final_eval.json" in files:
        try:
            fe = json.load(open(files["final_eval.json"]))
            for k, met, nk in (("wiki_loss", "wiki_fixed", "wiki_windows"), ("fw_loss", "fw_fixed", "fw_windows"),
                               ("train_protocol_loss", "protocol", None)):
                if isinstance(fe.get(k), (int, float)):
                    out.append(R.make("result", id="r-" + rid(run["id"], met), time=t0, run=run["id"], metric=met,
                                      value=float(fe[k]), n=fe.get(nk) if nk else None, step=fe.get("step"),
                                      source=run["source"] + "/final_eval.json"))
        except Exception:
            pass
    return out


def qwen_log_records(p, host, src_prefix, root):
    name = os.path.splitext(os.path.basename(p))[0]
    txt = open(p, errors="ignore").read()
    fm = FINAL.findall(txt)
    mt = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p)))
    run = R.make("run", id=f"{name}@{host}@{mt.replace('-', '').replace(':', '')}", time=mt, name=name, host=host,
                 group="logs", source=f"{host}:{src_prefix}{os.path.relpath(p, root)}")
    out = [run, R.make("event", id="e-" + rid(run["id"], "x"), time=mt, run=run["id"],
                       status="finished" if fm else "stopped")]
    if fm:
        out.append(R.make("result", id="r-" + rid(run["id"], "val_train"), time=mt, run=run["id"], metric="val_train",
                          value=float(fm[-1]), source=run["source"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt_args", action="store_true")
    ap.add_argument("--src_prefix", default="", help="path prefix for the source / ckpt fields (the origin's path)")
    a = ap.parse_args()
    recs, nd = [], 0
    for dp, dn, fn in os.walk(a.root):
        if any(f in fn for f in ("metrics.jsonl", "train.log", "final_eval.json")):
            recs += run_dir_records(dp, a.root, a.host, a.ckpt_args, a.src_prefix); nd += 1
        for f in fn:
            if re.match(r"q08.*\.txt$", f) and (os.path.basename(dp) == "logs" or dp.rstrip("/").endswith("_logs")):
                recs += qwen_log_records(os.path.join(dp, f), a.host, a.src_prefix, a.root)
    have = {r["id"] for r in R.load(a.out)}
    new = [r for r in recs if r["id"] not in have]
    if new:
        R.append(new, a.out)
    print(f"{a.root} ({a.host}): {nd} run directories, {len(recs)} records, {len(new)} new -> {a.out}")


if __name__ == "__main__":
    main()
