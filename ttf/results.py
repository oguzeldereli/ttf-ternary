"""The results database: one append-only JSON-lines file of typed records (replaces QUEUE.md / RUNS.md / RUN_INDEX.md).

Record types (every record has `id`, `type`, `time`; optional `supersedes` (the id it replaces) and `source` (where a
backfilled record came from, file:line)):
  run     name, purpose, commit, host, config (the full resolved arguments), data, steps, tokens, parent (a resumed or
          continued run's origin), ckpt (path), ckpt_sha (or "deleted")
  event   run, status (queued | running | finished | failed | stopped), step, reason
  result  run, metric (val_train, wiki_fixed, fw_fixed, protocol, a zero-shot task, s_per_step, ...), value, n (windows /
          items), eval (script and commit)
  note    text, runs (linked run ids or names), tags
Nothing is ever rewritten: a correction is a new record whose `supersedes` names the old one.

Where records are written: every training run appends to `<out_dir>/records.jsonl` (it travels with the run, so many
hosts never write one file); `ingest` merges such files (or any records file) into the central database
(`results/runs.jsonl` of this repo, or $TTF_RESULTS), skipping ids it already holds. Notes (free text: plans, reasoning)
are kept apart from the measurements, in $TTF_NOTES (default ../ttf-private/results/notes.jsonl beside this repository
when that folder exists, else results/notes.jsonl, which git ignores); reading without a path sees both files.
"""
from __future__ import annotations
import fcntl
import json
import os
import socket
import subprocess
import time
import uuid

TYPES = ("run", "event", "result", "note")
STATUSES = ("queued", "running", "finished", "failed", "stopped")
REQUIRED = {"run": ("name",), "event": ("run", "status"), "result": ("run", "metric", "value"), "note": ("text",)}
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.environ.get("TTF_RESULTS") or os.path.join(_ROOT, "results", "runs.jsonl")
NOTES = os.environ.get("TTF_NOTES") or (
    os.path.join(_ROOT, "..", "ttf-private", "results", "notes.jsonl") if os.path.isdir(os.path.join(_ROOT, "..", "ttf-private"))
    else os.path.join(_ROOT, "results", "notes.jsonl"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def host():
    return socket.gethostname().split(".")[0]


def git_commit(path=None):
    """the commit of the code that runs (with '+dirty' when the tree has changes), or None outside a git checkout"""
    d = path or os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.run(["git", "-C", d, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
        if not c:
            return None
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return None


def code_commit():
    """git_commit(), or for a copy of the code without .git (lab PCs, rented boxes) the COMMIT file written when it
    was synced (`git rev-parse --short=12 HEAD > COMMIT` before the rsync)"""
    c = git_commit()
    if c:
        return c
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMMIT")
    try:
        return open(f).read().strip() or None
    except OSError:
        return None


def new_id(rec):
    """runs: name@host@time (readable, unique per launch); other records: type-time-random"""
    if rec["type"] == "run":
        return f"{rec['name']}@{rec.get('host') or host()}@{time.strftime('%Y%m%dT%H%M%S')}"
    return f"{rec['type'][0]}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def validate(rec):
    """the record's problems (an empty list when it is well formed)"""
    errs = []
    t = rec.get("type")
    if t not in TYPES:
        return [f"unknown type {t!r}"]
    for k in ("id", "time") + REQUIRED[t]:
        if rec.get(k) in (None, ""):
            errs.append(f"{t} without {k}")
    if t == "event" and rec.get("status") not in STATUSES:
        errs.append(f"event status {rec.get('status')!r}")
    if t == "result" and not isinstance(rec.get("value"), (int, float)):
        errs.append(f"result value {rec.get('value')!r} is not a number")
    return errs


def make(type_, **fields):
    """a complete record of the given type (id and time filled in); raises on a malformed record"""
    rec = {"type": type_, "time": fields.pop("time", None) or now(), **{k: v for k, v in fields.items() if v is not None}}
    if type_ == "run":
        rec.setdefault("host", host())
    rec.setdefault("id", new_id(rec))
    e = validate(rec)
    if e:
        raise ValueError("; ".join(e))
    return rec


def append(rec, path=None):
    """append one record (or a list) to a records file, under an exclusive lock, flushed to disk; returns the id(s).
    Without a path: notes to NOTES, the other records to DB."""
    recs = rec if isinstance(rec, list) else [rec]
    if path is None:
        notes = [r for r in recs if r.get("type") == "note"]
        rest = [r for r in recs if r.get("type") != "note"]
        if notes and rest:
            append(notes, NOTES); append(rest, DB)
            return [r["id"] for r in recs] if isinstance(rec, list) else recs[0]["id"]
        path = NOTES if notes else DB
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for r in recs:
        e = validate(r)
        if e:
            raise ValueError(f"{r.get('id')}: " + "; ".join(e))
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for r in recs:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
        f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    return [r["id"] for r in recs] if isinstance(rec, list) else recs[0]["id"]


def add(type_, path=None, **fields):
    """make + append; returns the new record's id"""
    return append(make(type_, **fields), path)


def load(path=None):
    """every record of a file, in order (a damaged last line from an interrupted write is skipped); without a path the
    database and the notes"""
    if path is None:
        return load(DB) + (load(NOTES) if os.path.abspath(NOTES) != os.path.abspath(DB) else [])
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"results: {path}:{i} is not JSON (skipped)")
    return out


def current(records):
    """the records in force: {id: record}, a record that supersedes another replacing it under the first id of the
    chain (a correction of a correction replaces both)"""
    cur, first = {}, {}
    for r in records:
        s = r.get("supersedes")
        k = first.get(s, s) if s else r.get("id")
        if k is None:            # (no id: malformed, validate reports it)
            continue
        first[r.get("id", k)] = k
        cur[k] = r
    return cur


def ingest(paths, db=None):
    """merge records files into the database, skipping ids it already holds; returns (added, skipped)"""
    have = {r["id"] for r in load(db)}
    new = []
    for p in paths:
        for r in load(p):
            if r.get("id") and r["id"] not in have and not validate(r):
                new.append(r); have.add(r["id"])
    if new:
        append(new, db)
    return len(new), sum(len(load(p)) for p in paths) - len(new)


class RunLog:
    """what a training run writes: its run record at the start, events, results; to <out_dir>/records.jsonl.
    All methods swallow errors (the database must never stop a run)."""
    def __init__(self, out_dir, name, config, purpose=None, data=None, steps=None, parent=None):
        self.path = os.path.join(out_dir, "records.jsonl")
        self.id = None
        self.ended = False       # a finished / failed / stopped event written
        try:
            rec = make("run", name=name, purpose=purpose, commit=code_commit(), config=config, data=data, steps=steps,
                       parent=parent, ckpt=os.path.join(out_dir, "ckpt.pt"))
            self.id = append(rec, self.path)
        except Exception as e:
            print(f"results: run record not written ({e})", flush=True)

    def event(self, status, step=None, reason=None):
        self.ended = self.ended or status in ("finished", "failed", "stopped")
        if self.id:
            try:
                append(make("event", run=self.id, status=status, step=step, reason=reason), self.path)
            except Exception as e:
                print(f"results: event not written ({e})", flush=True)

    def result(self, metric, value, n=None, step=None, eval=None):
        if self.id:
            try:
                append(make("result", run=self.id, metric=metric, value=float(value), n=n, step=step, eval=eval),
                       self.path)
            except Exception as e:
                print(f"results: result not written ({e})", flush=True)


def run_of(out_dir, name=None, create=True):
    """the id of the latest run record in <out_dir>/records.jsonl; for a run directory from before the database, a
    minimal run record (name, checkpoint) is created first when create is set"""
    path = os.path.join(out_dir, "records.jsonl")
    runs = [r for r in load(path) if r.get("type") == "run"]
    if runs:
        return runs[-1]["id"], path
    if not create:
        return None, path
    rec = make("run", name=name or os.path.basename(os.path.normpath(out_dir)), ckpt=os.path.join(out_dir, "ckpt.pt"),
               source="run directory without records (made by an evaluation)")
    return append(rec, path), path
