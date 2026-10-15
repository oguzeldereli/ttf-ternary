"""Score gen_bench.py outputs.

  python scripts/eval/score_bench.py --task gsm8k OUT.jsonl [OUT2.jsonl ...]

gsm8k: the number after the last "Answer:" (else the last number in the output) against the reference after "####".
humaneval: the first ```python block (else the whole output), with the prompt's import lines put in front, run against
the task's tests in a separate python -I process (10 s limit, temporary directory); pass@1, greedy.
ifeval: lm-eval's IFEval checks (prompt- and instruction-level, strict and loose); needs lm-eval, langdetect, nltk,
immutabledict (lm_eval's package init is skipped: only the ifeval checkers are imported).
Prints accuracy with a standard error per file; with two files, also the paired difference over the items in both.
"""
import argparse, json, math, os, re, subprocess, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["gsm8k", "humaneval", "ifeval"]); ap.add_argument("files", nargs="+")
ap.add_argument("--max_new", type=int, default=0, help="gen_bench.py's limit (default per task), for the at-limit count")
a = ap.parse_args()
LIM = a.max_new or {"gsm8k": 512, "humaneval": 512, "ifeval": 1024}[a.task]
ref = {"gsm8k": "gsm8k_test.jsonl", "humaneval": "humaneval.jsonl", "ifeval": "ifeval.jsonl"}[a.task]
docs = {}
for l in open(os.path.join(ROOT, "data", "bench", ref)):
    r = json.loads(l)
    docs[r.get("id") or r.get("task_id") or str(r["key"])] = r

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def num(s):
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def score_gsm8k(r, d):
    gold = num(d["answer"].split("####")[-1].strip())
    m = re.findall(r"Answer:\s*\**\s*(" + NUM + ")", r["output"])
    m = m or re.findall(NUM, r["output"])
    return float(bool(m) and num(m[-1]) is not None and abs(num(m[-1]) - gold) < 1e-6)


def score_humaneval(r, d):
    m = re.findall(r"```(?:python)?\n(.*?)```", r["output"], re.S)
    code = m[0] if m else r["output"]
    imports = "\n".join(l for l in d["prompt"].splitlines() if l.startswith(("import ", "from ")))
    if f"def {d['entry_point']}" not in code:      # a body only: complete the prompt with it
        code = d["prompt"] + code
    prog = f"{imports}\n{code}\n\n{d['test']}\n\ncheck({d['entry_point']})\n"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.py")
        open(p, "w").write(prog)
        try:
            ok = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    return float(ok)


if a.task == "ifeval":
    import importlib.util, types
    _d = os.path.dirname(importlib.util.find_spec("lm_eval").origin)
    for _n, _p in [("lm_eval", _d), ("lm_eval.tasks", _d + "/tasks")]:
        _m = types.ModuleType(_n); _m.__path__ = [_p]; sys.modules[_n] = _m
    from lm_eval.tasks.ifeval import utils as ifu


def score_ifeval(r, d):
    res = ifu.process_results(d, [r["output"]])
    return {"prompt_strict": float(res["prompt_level_strict_acc"]), "prompt_loose": float(res["prompt_level_loose_acc"]),
            "inst_strict": res["inst_level_strict_acc"], "inst_loose": res["inst_level_loose_acc"]}


def mean_se(xs):
    n = len(xs); m = sum(xs) / n
    return m, math.sqrt(max(m * (1 - m), 0) / n) if n > 1 else 0.0


per = []
for f in a.files:
    rows = [json.loads(l) for l in open(f)]
    sc = {}
    for r in rows:
        d = docs[r["id"]]
        sc[r["id"]] = {"gsm8k": score_gsm8k, "humaneval": score_humaneval, "ifeval": score_ifeval}[a.task](r, d)
    per.append(sc)
    ntok = sum(r["gen_tokens"] for r in rows) / len(rows)
    if a.task == "ifeval":
        ps, pl = mean_se([v["prompt_strict"] for v in sc.values()]), mean_se([v["prompt_loose"] for v in sc.values()])
        ins = [x for v in sc.values() for x in v["inst_strict"]]; inl = [x for v in sc.values() for x in v["inst_loose"]]
        print(f"{f}: n {len(sc)}  prompt strict {ps[0]:.3f} ± {ps[1]:.3f}  prompt loose {pl[0]:.3f} ± {pl[1]:.3f}  "
              f"inst strict {sum(ins) / len(ins):.3f}  inst loose {sum(inl) / len(inl):.3f}  ({ntok:.0f} gen tokens/item, "
              f"{sum(r['gen_tokens'] >= LIM for r in rows)} at the length limit)")
    else:
        m, se = mean_se(list(sc.values()))
        print(f"{f}: n {len(sc)}  acc {m:.3f} ± {se:.3f}  ({ntok:.0f} gen tokens/item, "
              f"{sum(r['gen_tokens'] >= LIM for r in rows)} at the length limit)")
if len(per) == 2:
    key = (lambda v: v["prompt_strict"]) if a.task == "ifeval" else (lambda v: v)
    common = sorted(set(per[0]) & set(per[1]))
    diff = [key(per[1][k]) - key(per[0][k]) for k in common]
    m = sum(diff) / len(diff); sd = math.sqrt(sum((x - m) ** 2 for x in diff) / max(len(diff) - 1, 1))
    print(f"paired, second minus first, {len(common)} items: {m:+.3f} ± {sd / math.sqrt(len(diff)):.3f} "
          f"(first only right {sum(x < 0 for x in diff)}, second only right {sum(x > 0 for x in diff)})")
