"""Backfill the results database with the run logs kept as markdown (one-off, 17 Oct consolidation): every entry of
docs/QUEUE.md, every section of docs/RUNS.md, every table row and paragraph of docs/RUN_INDEX.md (docs/RESULTS.md is a
summary, not a log: it stays a document)
becomes a `note` record, verbatim, with
  time    the entry's own "HH:MM (DD Mon)" date, else the entry above it (QUEUE) or the nearest dated header, else
          when the line was written (git blame); year 2026
  runs    the run names it mentions in backticks
  tags    its source (queue / runs / run_index / results) and topics found in the text (27b, conversion, compute, rank,
          qwen, tinyllama, superposition, paper, patent, myriad, lab)
  source  file:line
Ids are deterministic (source + text), so a second run adds nothing. The files themselves then leave the tree (they
remain in tag pre-consolidation-2026-10-17).
  python scripts/backfill/backfill_docs.py --out results/runs.jsonl
"""
import argparse, hashlib, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf import results as R   # noqa: E402

MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
DATE = re.compile(r"\(~?(\d{1,2}) (Sep|Oct|Nov)\b")
HDATE = re.compile(r"(\d{1,2})(?:\s*-\s*(\d{1,2}))? (Sep|Oct|Nov)(?:,? (\d{1,2}):(\d\d))?")
TIME = re.compile(r"^\*\*~?(\d{1,2}):(\d\d)")
RUNREF = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_.{},\-]*)`")
TOPICS = {"27b": r"\b27B\b|qw27", "conversion": r"conver|GPTQ|gptq|init_hf", "compute": r"s/step|s per step|profile|kernel|tok/s",
          "rank": r"\brank\b|\br\d{2,4}\b", "qwen": r"Qwen|q08", "tinyllama": r"TinyLlama|\btl\d", "superposition": r"superposition|decrowd",
          "paper": r"\bpaper\b", "patent": r"patent", "myriad": r"Myriad", "lab": r"\blab\b|lab PC", "master": r"\bmaster\b"}


def rid(*p):
    return hashlib.sha1("|".join(map(str, p)).encode()).hexdigest()[:12]


def blame_time(path, ln):
    """the author time of line ln of path (git blame), as the records' time format"""
    import subprocess
    out = subprocess.run(["git", "blame", "--porcelain", "-L", f"{ln},{ln}", "--", path], capture_output=True, text=True).stdout
    m = re.search(r"^author-time (\d+)", out, re.M)
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(int(m.group(1)))) if m else "2026-09-26T00:00:00"


def stamp(day, mon, hh=0, mm=0):
    return f"2026-{MON[mon]:02d}-{int(day):02d}T{int(hh):02d}:{int(mm):02d}:00"


def blocks(path, mode):
    """(first line number, text, header date) of each entry"""
    lines = open(path).read().split("\n")
    out, cur, start, hdate = [], [], 0, None
    def flush():
        t = "\n".join(cur).strip()
        if t:
            out.append((start, t, hdate))
    for i, l in enumerate(lines, 1):
        is_head = l.startswith("#")
        if mode == "queue":
            new = is_head or bool(TIME.match(l))
        elif mode == "run_index":
            new = is_head or l.startswith("| ") or (l.strip() == "" and cur and not cur[-1].startswith("| "))
        else:
            new = l.startswith("## ") or l.startswith("### ") or l.startswith("# ")
        if new:
            flush(); cur, start = [], i
            if is_head:
                m = HDATE.search(l)
                if m:
                    hdate = (m.group(2) or m.group(1), m.group(3), m.group(4) or 0, m.group(5) or 0)
        if mode == "run_index" and (l.startswith("|---") or l.startswith("| run")):
            continue
        cur.append(l)
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--docs", default="docs")
    a = ap.parse_args()
    have = {r["id"] for r in R.load(a.out)}
    new, n = [], 0
    for fname, mode in (("QUEUE.md", "queue"), ("RUNS.md", "runs"), ("RUN_INDEX.md", "run_index")):   # RESULTS.md stays a doc
        p = os.path.join(a.docs, fname)
        last = None
        if not os.path.exists(p):
            continue
        for ln, text, hdate in blocks(p, mode):
            if text.startswith("#") and "\n" not in text and mode != "queue":
                continue                                  # a bare header line
            m = DATE.search(text[:200])
            tm = TIME.match(text)
            if m:
                last = (m.group(1), m.group(2))
                t = stamp(m.group(1), m.group(2), *(tm.groups() if tm else (0, 0)))
            elif last and mode == "queue":                # an undated entry: the date of the entry above it
                t = stamp(last[0], last[1], *(tm.groups() if tm else (0, 0)))
            elif hdate:
                t = stamp(hdate[0], hdate[1], *(tm.groups() if tm else (hdate[2], hdate[3])))
            elif last:
                t = stamp(last[0], last[1])
            else:
                t = blame_time(p, ln)                     # undated: when the entry was written (git blame)
            runs = sorted({x for x in RUNREF.findall(text) if not x.startswith("--") and len(x) > 2})
            tags = [mode] + [k for k, pat in TOPICS.items() if re.search(pat, text)]
            rec = R.make("note", id="n-" + rid(fname, ln, text), time=t, text=text, runs=runs or None, tags=tags,
                         source=f"docs/{fname}:{ln}")
            n += 1
            if rec["id"] not in have:
                new.append(rec); have.add(rec["id"])
    if new:
        R.append(new, a.out)
    print(f"{n} notes from the docs, {len(new)} new -> {a.out}")


if __name__ == "__main__":
    main()
