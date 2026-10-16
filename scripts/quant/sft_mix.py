"""Post-training mixture for converting an instruction model, rendered with the target model's own chat template:
reasoning (thinking on), tool calling (tools in the system block, tool calls and tool results), chat, safety, plus a
plain-text share for knowledge. Each source becomes a uint32 stream; --mix interleaves them in 8192-token segments by
share (the trainer samples windows uniformly, so the share of tokens is the share of training windows).

  python scripts/quant/sft_mix.py build models/qw27 data/sft --source nemotron:math:7e6 nemotron:code:8e6 ...
  python scripts/quant/sft_mix.py mix data/mix2_train.bin 72e6 data/wikiq8_train.bin:0.30 data/sft_math.bin:0.10 ...

Sources: nemotron:SPLIT (nvidia/Nemotron-Post-Training-Dataset-v1, CC-BY-4.0, responses by Qwen3-235B-A22B /
DeepSeek-R1; SPLIT in chat, code, math, stem, tool), hermes (NousResearch/hermes-function-calling-v1, Apache-2.0),
toolace (Team-ACE/ToolACE, Apache-2.0), tulu:general / tulu:safety (allenai/tulu-3-sft-mixture, ODC-BY; safety = the
WildJailbreak, WildGuardMix and CoCoNot subsets; the hard-coded identity answers are left out), ot3
(open-thoughts/OpenThoughts3-1.2M, Apache-2.0, reasoning), ocr (nvidia/OpenCodeReasoning split_0, CC-BY-4.0, code
reasoning), fineweb (HuggingFaceFW/fineweb-edu sample-10BT, ODC-BY, plain text with end-of-text between documents).
--decon N: drop a conversation sharing any N-word sequence (lower-cased \\w+ words) with the evaluation items
(data/bench/*.jsonl, data/zeroshot/*.jsonl); word n-grams as in GPT-3's and Tulu-3's decontamination (token n-grams
matched code boilerplate and number lists). Tulu-3 (ODC-BY) chat-only streams: chat_data.py."""
import argparse
import json
import random
import sys

import numpy as np

NEMO = "nvidia/Nemotron-Post-Training-Dataset-v1"
SPLIT_FILES = {"chat": "chat", "code": "code", "math": "math", "stem": "stem", "tool": "tool"}


def _files(repo, prefix):
    from huggingface_hub import list_repo_files
    return sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet") and prefix in f)


def nemotron_rows(split, rng, max_files, skip_files=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    fs = _files(NEMO, f"data/{SPLIT_FILES[split]}-")
    rng.shuffle(fs)
    for f in fs[skip_files:skip_files + max_files]:     # --skip_files: held-out files for validation
        t = pq.read_table(hf_hub_download(NEMO, f, repo_type="dataset"), columns=["reasoning", "messages", "metadata"])
        rows = t.to_pylist()
        rng.shuffle(rows)
        for r in rows:
            msgs = []
            for m in r["messages"]:
                x = {"role": m["role"], "content": m["content"] or ""}
                if x["role"] == "assistant" and "</think>" in x["content"]:   # the template wants the reasoning apart
                    r_, c_ = x["content"].split("</think>", 1)
                    x["reasoning_content"] = r_.replace("<think>", "").strip()
                    x["content"] = c_.lstrip("\n")
                if m.get("tool_calls"):
                    x["tool_calls"] = [{"type": "function", "function": {"name": c["function"]["name"],
                                        "arguments": _args(c["function"]["arguments"])}} for c in m["tool_calls"]]
                msgs.append(x)
            meta = json.loads(r["metadata"]) if r.get("metadata") else {}
            yield msgs, meta.get("tools"), r.get("reasoning") == "on"


def _args(a):
    try:
        return json.loads(a) if isinstance(a, str) else a
    except Exception:
        return a


def hermes_rows(rng):
    from datasets import load_dataset
    for cfg in ("func_calling_singleturn", "func_calling"):
        ds = list(load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train"))
        rng.shuffle(ds)
        for r in ds:
            conv, tools = [], None
            for m in r["conversations"]:
                role = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}[m["from"]]
                c = m["value"]
                if role == "system":                      # the tool schemas sit in <tools>...</tools> in the system text
                    if "<tools>" in c:
                        try:
                            tools = json.loads("[" + c.split("<tools>")[1].split("</tools>")[0].strip().replace("}\n{", "},{")
                                               .replace("} {", "},{") + "]")
                            if tools and isinstance(tools[0], list):
                                tools = tools[0]
                        except Exception:
                            tools = None
                    continue                               # the model's own template writes the tool system block
                if role == "tool":
                    c = c.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                conv.append({"role": role, "content": c})
            if tools and conv and conv[-1]["role"] == "assistant":
                yield conv, [t if "type" in t else {"type": "function", "function": t} for t in tools], False


def toolace_rows(rng):
    from datasets import load_dataset
    ds = list(load_dataset("Team-ACE/ToolACE", split="train"))
    rng.shuffle(ds)
    for r in ds:
        conv = [{"role": {"user": "user", "assistant": "assistant", "tool": "tool"}.get(m["from"], m["from"]),
                 "content": m["value"]} for m in r["conversations"]]
        if conv and conv[-1]["role"] == "assistant":
            yield [{"role": "system", "content": r["system"]}] + conv, None, False


SAFETY = ("wildjailbreak", "wildguardmix", "coconot")


def _parquet_rows(repo, files, columns, rng):
    """rows of the given parquet files, file by file, row group by row group (bounded memory), shuffled within a group"""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for f in files:
        pf = pq.ParquetFile(hf_hub_download(repo, f, repo_type="dataset"))
        for g in range(pf.num_row_groups):
            rows = pf.read_row_group(g, columns=columns).to_pylist()
            rng.shuffle(rows)
            yield from rows


def _split_think(c):
    """'<think>R</think>A' -> (R, A); the chat template writes the reasoning block itself"""
    if "</think>" in c:
        r_, c_ = c.split("</think>", 1)
        return r_.replace("<think>", "").strip(), c_.lstrip("\n")
    return None, c


def tulu_rows(kind, rng):
    from huggingface_hub import list_repo_files
    repo = "allenai/tulu-3-sft-mixture"
    fs = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")); rng.shuffle(fs)
    for r in _parquet_rows(repo, fs, ["messages", "source"], rng):
        safe = any(k in r["source"] for k in SAFETY)
        if "hard_coded" in r["source"] or safe != (kind == "safety"):
            continue
        if kind == "if" and "ifdata" not in r["source"]:      # tulu:if: the persona instruction-following set only
            continue
        m = [{"role": x["role"], "content": x["content"]} for x in r["messages"]]
        if m and m[-1]["role"] == "assistant":
            yield m, None, False


def ot3_rows(rng, skip_files):
    from huggingface_hub import list_repo_files
    repo = "open-thoughts/OpenThoughts3-1.2M"
    fs = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")); rng.shuffle(fs)
    for r in _parquet_rows(repo, fs[skip_files:], ["conversations"], rng):
        m = []
        for x in r["conversations"]:
            role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}.get(x["from"])
            if role is None:
                break
            y = {"role": role, "content": x["value"]}
            if role == "assistant":
                rc, y["content"] = _split_think(x["value"])
                if rc is not None:
                    y["reasoning_content"] = rc
            m.append(y)
        if m and m[-1]["role"] == "assistant":
            yield m, None, True


def ocr_rows(rng, skip_files):
    from huggingface_hub import list_repo_files
    repo = "nvidia/OpenCodeReasoning"
    fs = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("split_0/") and f.endswith(".parquet"))
    rng.shuffle(fs)
    for r in _parquet_rows(repo, fs[skip_files:], ["input", "output"], rng):
        if not r["input"] or not r["output"]:
            continue
        rc, ans = _split_think(r["output"])
        y = {"role": "assistant", "content": ans}
        if rc is not None:
            y["reasoning_content"] = rc
        yield [{"role": "user", "content": r["input"]}, y], None, True


def oci_rows(rng, skip_files):
    """nvidia/OpenCodeInstruct (CC-BY-4.0): Python function problems with a solution and unit tests; only solutions that
    passed every test (average_test_score 1). Direct answers, no thinking block."""
    from huggingface_hub import list_repo_files
    repo = "nvidia/OpenCodeInstruct"
    fs = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")); rng.shuffle(fs)
    for r in _parquet_rows(repo, fs[skip_files:], ["input", "output", "average_test_score"], rng):
        try:
            ok = float(r["average_test_score"]) >= 1.0
        except (TypeError, ValueError):
            ok = False
        if ok and r["input"] and r["output"]:
            yield [{"role": "user", "content": r["input"]}, {"role": "assistant", "content": r["output"]}], None, False


def fineweb_rows(rng, skip_files):
    from huggingface_hub import list_repo_files
    repo = "HuggingFaceFW/fineweb-edu"
    fs = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("sample/10BT/")); rng.shuffle(fs)
    for r in _parquet_rows(repo, fs[skip_files:], ["text"], rng):
        yield r["text"], None, None                     # plain text: no chat template


def _ngram_hashes(ids, n):
    """rolling uint64 hashes of every n-token window of ids"""
    x = np.asarray(ids, dtype=np.uint64)
    if len(x) < n:
        return np.empty(0, np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    p = np.power(np.uint64(1000003), np.arange(n, dtype=np.uint64))
    with np.errstate(over="ignore"):
        return (w * p).sum(1)


def _hits(h, sorted_set):
    """any of the hashes h in the sorted unique array"""
    i = np.searchsorted(sorted_set, h).clip(0, len(sorted_set) - 1)
    return bool((sorted_set[i] == h).any())


_WORD = None


def _word_ids(text):
    """lower-cased \\w+ words as stable integer ids (crc32), for word n-gram hashing"""
    import re, zlib
    global _WORD
    _WORD = _WORD or re.compile(r"\w+")
    return [zlib.crc32(w.encode()) for w in _WORD.findall(text.lower())]


def eval_ngrams(tok, n):
    import glob, os
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    texts = []
    for f in glob.glob(os.path.join(root, "bench", "*.jsonl")) + glob.glob(os.path.join(root, "zeroshot", "*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            for k in ("question", "answer", "prompt", "canonical_solution", "test", "context", "target"):
                if isinstance(r.get(k), str):
                    texts.append(r[k])
            for k in ("choices", "contexts"):
                if isinstance(r.get(k), list):
                    texts += [c for c in r[k] if isinstance(c, str)]
    hs = [_ngram_hashes(_word_ids(t), n) for t in texts]
    return np.unique(np.concatenate(hs)), len(texts)


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    if a.decon:
        evh, ne = eval_ngrams(tok, a.decon)
        print(f"decontamination: {len(evh)} word {a.decon}-grams from {ne} evaluation texts", file=sys.stderr, flush=True)
    for spec in a.source:
        parts = spec.split(":")
        name, budget = parts[0], float(parts[-1])
        rng = random.Random(a.seed)
        if name == "nemotron":
            it, tag = nemotron_rows(parts[1], rng, a.max_files, a.skip_files), parts[1]
        elif name == "hermes":
            it, tag = hermes_rows(rng), "hermes"
        elif name == "toolace":
            it, tag = toolace_rows(rng), "toolace"
        elif name == "tulu":
            it, tag = tulu_rows(parts[1], rng), f"tulu_{parts[1]}"
        elif name in ("ot3", "ocr", "fineweb", "oci"):
            it, tag = {"ot3": ot3_rows, "ocr": ocr_rows, "fineweb": fineweb_rows, "oci": oci_rows}[name](rng, a.skip_files), name
        else:
            raise SystemExit(f"unknown source {spec}")
        chunks, n, k, skip, long_, dec = [], 0, 0, 0, 0, 0
        for msgs, tools, think in it:
            if n >= budget:
                break
            try:
                txt = msgs if isinstance(msgs, str) else tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                                                                 enable_thinking=think)
            except Exception:
                skip += 1; continue
            ids = tok(txt, add_special_tokens=False).input_ids + ([eos] if isinstance(msgs, str) else [])
            if len(ids) > a.max_len:
                long_ += 1; continue
            if a.decon and _hits(_ngram_hashes(_word_ids(txt), a.decon), evh):
                dec += 1; continue
            chunks.append(np.asarray(ids, dtype=np.uint32)); n += len(ids); k += 1
            if k % 20000 == 0:
                print(f"  {tag}: {n / 1e6:.1f}M tokens, {k} items, {dec} dropped as contaminated", file=sys.stderr, flush=True)
        arr = np.concatenate(chunks)
        out = f"{a.out}_{tag}.bin"
        arr.tofile(out)
        print(f"{out}: {len(arr) / 1e6:.2f}M tokens, {k} conversations (skipped {skip} errors, {long_} > {a.max_len} "
              f"tokens, {dec} contaminated)", file=sys.stderr, flush=True)
        if a.show:
            print(tok.decode(arr[:a.show].tolist()), file=sys.stderr)


def mix(a):
    seg = 8192
    srcs = []
    for s in a.parts:
        p, f = s.rsplit(":", 1)
        srcs.append((p, float(f)))
    tot = sum(f for _, f in srcs)
    total = int(float(a.total))
    plan = []
    for i, (p, f) in enumerate(srcs):
        arr = np.memmap(p, dtype=np.uint32, mode="r")
        nseg = int(total * f / tot) // seg
        reps = -(-nseg * seg // len(arr))               # a small source repeats (epochs) rather than shrinking its share
        plan.append((arr, nseg, reps))
        print(f"{p}: {nseg * seg / 1e6:.1f}M tokens ({f / tot:.0%}), {len(arr) / 1e6:.1f}M available"
              + (f", repeated {reps}x" if reps > 1 else ""))
    order = np.random.default_rng(a.seed).permutation(np.concatenate([np.full(n, i, np.int16) for i, (_, n, _) in enumerate(plan)]))
    out = np.memmap(a.out_bin, dtype=np.uint32, mode="w+", shape=(len(order) * seg,))
    pos = [0] * len(plan)
    for k, i in enumerate(order):
        arr = plan[i][0]
        st = pos[i] % (len(arr) - seg)
        out[k * seg:(k + 1) * seg] = arr[st:st + seg]
        pos[i] += seg
    out.flush()
    print(f"{a.out_bin}: {len(out) / 1e6:.1f}M tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("model_dir"); b.add_argument("out")
    b.add_argument("--source", nargs="+", required=True); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max_files", type=int, default=2); b.add_argument("--skip_files", type=int, default=0); b.add_argument("--max_len", type=int, default=32768)
    b.add_argument("--show", type=int, default=0); b.add_argument("--decon", type=int, default=0)
    m = sub.add_parser("mix"); m.add_argument("out_bin"); m.add_argument("total"); m.add_argument("parts", nargs="+")
    m.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else mix(a)


if __name__ == "__main__":
    main()
