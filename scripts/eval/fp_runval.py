"""Full-precision reference on a run's own training-time validation windows.

bitnet/train.py's evaluate() scores eval_iters batches of max(batch_size, 16) windows drawn by get_batch() with a
generator seeded seed + 12345. This script draws the same windows and scores transformers' bf16 Qwen model on them
(layers that do not fit the GPU stay in CPU memory), so a run's val curve can be read against full precision.
    python scripts/eval/fp_runval.py HF_DIR --val data/wikiq8_val.bin --eval_iters 4 [--seed 1337] [--check_first 8]
--check_first N: also the first N contiguous windows (qwen_check.py's reference windows), to check this path.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir"); ap.add_argument("--val", required=True)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--batch_size", type=int, default=1)
ap.add_argument("--eval_iters", type=int, default=4); ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--check_first", type=int, default=0); ap.add_argument("--gpu_gib", type=int, default=20)
ap.add_argument("--out", default="")
a = ap.parse_args()
data = np.memmap(a.val, dtype=np.uint32, mode="r")

gen = torch.Generator().manual_seed(a.seed + 12345)      # as evaluate(): same generator, same draws
bs = max(a.batch_size, 16)
starts = [int(i) for _ in range(a.eval_iters) for i in torch.randint(len(data) - a.seq - 1, (bs,), generator=gen)]
print(f"{len(starts)} run-val windows, first starts {starts[:4]}", flush=True)

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
t = time.time()
m = Qwen3_5ForCausalLM.from_pretrained(a.hf_dir, dtype=torch.bfloat16, device_map="auto",
                                       max_memory={0: f"{a.gpu_gib}GiB", "cpu": "110GiB"}).eval()
print(f"loaded in {time.time() - t:.0f} s", flush=True)


@torch.no_grad()
def wloss(s):
    x = torch.from_numpy(data[s:s + a.seq + 1].astype(np.int64)).to("cuda")[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lg = m(input_ids=x[:, :-1]).logits[0].float()
    return float(F.cross_entropy(lg, x[0, 1:]))


res = {}
for name, ss in ([("first", [i * a.seq for i in range(a.check_first)])] if a.check_first else []) + [("runval", starts)]:
    ls = []
    for k, s in enumerate(ss):
        t = time.time(); ls.append(wloss(s))
        print(f"  {name} {k + 1}/{len(ss)} start {s}: {ls[-1]:.4f} ({time.time() - t:.0f} s)", flush=True)
    res[name] = {"mean": float(np.mean(ls)), "losses": ls, "starts": ss}
    print(f"{name}: mean {np.mean(ls):.4f} over {len(ls)} windows", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
