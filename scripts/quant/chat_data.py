"""Chat-format token streams for converting an instruction model: conversations from an SFT set (default
allenai/tulu-3-sft-mixture, all shards, shuffled) rendered with the target model's own chat template (thinking off, so
every assistant turn carries the empty think block the model sees at inference) and written as uint32 streams.

  python scripts/quant/chat_data.py models/qw27 data/chatq8 --train_tokens 30e6 --val_tokens 1e6
  -> data/chatq8_train.bin, data/chatq8_val.bin (held-out conversations)

--mix WIKI_BIN OUT_BIN FRAC: interleave a chat stream with another stream in 8192-token segments, FRAC of chat
(the trainer samples windows uniformly, so the mix is the share of training windows)."""
import argparse
import os
import sys

import numpy as np


def mix(chat_bin, other_bin, out_bin, frac, total, seg=8192, seed=0):
    a = np.memmap(chat_bin, dtype=np.uint32, mode="r")
    b = np.memmap(other_bin, dtype=np.uint32, mode="r")
    na, nb = int(total * frac) // seg, int(total * (1 - frac)) // seg
    assert na * seg <= len(a) and nb * seg <= len(b), (len(a), len(b), na * seg, nb * seg)
    order = np.random.default_rng(seed).permutation(np.r_[np.zeros(na, np.int8), np.ones(nb, np.int8)])
    out = np.memmap(out_bin, dtype=np.uint32, mode="w+", shape=((na + nb) * seg,))
    ia = ib = 0
    for k, w in enumerate(order):
        if w == 0:
            out[k * seg:(k + 1) * seg] = a[ia * seg:(ia + 1) * seg]; ia += 1
        else:
            out[k * seg:(k + 1) * seg] = b[ib * seg:(ib + 1) * seg]; ib += 1
    out.flush()
    print(f"{out_bin}: {len(out) / 1e6:.1f}M tokens, chat {na * seg / 1e6:.1f}M + other {nb * seg / 1e6:.1f}M")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir"); ap.add_argument("out")
    ap.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--train_tokens", type=float, default=30e6); ap.add_argument("--val_tokens", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", nargs=3, metavar=("OTHER_BIN", "OUT_BIN", "FRAC"))
    ap.add_argument("--mix_total", type=float, default=60e6)
    a = ap.parse_args()
    if a.mix:
        mix(a.out + "_train.bin", a.mix[0], a.mix[1], float(a.mix[2]), int(a.mix_total)); return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    files = sorted(f for f in list_repo_files(a.dataset, repo_type="dataset") if f.endswith(".parquet"))
    convs = []
    for f in files:
        t = pq.read_table(hf_hub_download(a.dataset, f, repo_type="dataset"), columns=["messages"])
        convs += t.column("messages").to_pylist()
        print(f"{f}: {len(convs)} conversations", file=sys.stderr, flush=True)
    order = np.random.default_rng(a.seed).permutation(len(convs))
    need = {"val": int(a.val_tokens), "train": int(a.train_tokens)}   # val first: held-out conversations
    k, skipped = 0, 0
    for split in ("val", "train"):
        chunks, n = [], 0
        while n < need[split]:
            batch = []
            while len(batch) < 512 and k < len(order):
                m = [{"role": x["role"], "content": x["content"]} for x in convs[order[k]]]; k += 1
                if not m or m[-1]["role"] != "assistant":
                    skipped += 1; continue
                try:
                    batch.append(tok.apply_chat_template(m, tokenize=False, enable_thinking=False))
                except Exception:
                    skipped += 1
            if not batch:
                break
            for ids in tok(batch, add_special_tokens=False).input_ids:
                chunks.append(np.asarray(ids, dtype=np.uint32)); n += len(ids)
        arr = np.concatenate(chunks)[:need[split]]
        arr.tofile(f"{a.out}_{split}.bin")
        print(f"{a.out}_{split}.bin: {len(arr) / 1e6:.2f}M tokens from {len(chunks)} conversations "
              f"(skipped {skipped} so far)", file=sys.stderr, flush=True)
    print(tok.decode(np.fromfile(f"{a.out}_train.bin", dtype=np.uint32)[:300].tolist()))


if __name__ == "__main__":
    main()
