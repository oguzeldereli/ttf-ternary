"""Post-training mixture for converting an instruction model, rendered with the target model's own chat template:
reasoning (thinking on), tool calling (tools in the system block, tool calls and tool results), chat, safety, plus a
plain-text share for knowledge. Each source becomes a uint32 stream; --mix interleaves them in 8192-token segments by
share (the trainer samples windows uniformly, so the share of tokens is the share of training windows).

  python scripts/quant/sft_mix.py build models/qw27 data/sft --source nemotron:math:7e6 nemotron:code:8e6 ...
  python scripts/quant/sft_mix.py mix data/mix2_train.bin 72e6 data/wikiq8_train.bin:0.30 data/sft_math.bin:0.10 ...

Sources: nemotron:SPLIT (nvidia/Nemotron-Post-Training-Dataset-v1, CC-BY-4.0, responses by Qwen3-235B-A22B /
DeepSeek-R1; SPLIT in chat, code, math, stem, tool), hermes (NousResearch/hermes-function-calling-v1, Apache-2.0),
toolace (Team-ACE/ToolACE, Apache-2.0). Tulu-3 (ODC-BY) is built by chat_data.py."""
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


def nemotron_rows(split, rng, max_files):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    fs = _files(NEMO, f"data/{SPLIT_FILES[split]}-")
    rng.shuffle(fs)
    for f in fs[:max_files]:
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


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    for spec in a.source:
        parts = spec.split(":")
        name, budget = parts[0], float(parts[-1])
        rng = random.Random(a.seed)
        if name == "nemotron":
            it, tag = nemotron_rows(parts[1], rng, a.max_files), parts[1]
        elif name == "hermes":
            it, tag = hermes_rows(rng), "hermes"
        elif name == "toolace":
            it, tag = toolace_rows(rng), "toolace"
        else:
            raise SystemExit(f"unknown source {spec}")
        chunks, n, k, skip, long_ = [], 0, 0, 0, 0
        for msgs, tools, think in it:
            if n >= budget:
                break
            try:
                txt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=think)
            except Exception:
                skip += 1; continue
            ids = tok(txt, add_special_tokens=False).input_ids
            if len(ids) > a.max_len:
                long_ += 1; continue
            chunks.append(np.asarray(ids, dtype=np.uint32)); n += len(ids); k += 1
        arr = np.concatenate(chunks)
        out = f"{a.out}_{tag}.bin"
        arr.tofile(out)
        print(f"{out}: {len(arr) / 1e6:.2f}M tokens, {k} conversations (skipped {skip} errors, {long_} > {a.max_len} "
              f"tokens)", file=sys.stderr, flush=True)
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
    b.add_argument("--max_files", type=int, default=2); b.add_argument("--max_len", type=int, default=32768)
    b.add_argument("--show", type=int, default=0)
    m = sub.add_parser("mix"); m.add_argument("out_bin"); m.add_argument("total"); m.add_argument("parts", nargs="+")
    m.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else mix(a)


if __name__ == "__main__":
    main()
