"""bf16 teacher forward time (24 Oct 2026): the HF model in bf16, as many
decoder layers on the GPU as fit under TT_GPU_GIB, the rest offloaded to host memory (accelerate), no grad; NB windows of
2048 tokens (one per micro-batch, as --teacher runs at 1 x 16) through the whole model. Each decoder layer is timed with
CUDA events (pre / post hooks), so the GPU-resident layers give the compute time per layer without the offload's copies;
the head (lm_head, hooked) is timed too. The call is the trainer's: Qwen3_5ForCausalLM(input_ids=x).logits (train.py). Prints the total
wall per step, the per-layer means (resident, offloaded; linear / full attention) and the compute-only estimate
(resident mean per layer kind x the kind's layer count + head).
  python scripts/prof/teacher_time.py HF_DIR DATA.bin [NB]
"""
import os, sys, time
import numpy as np
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

path, data = sys.argv[1], sys.argv[2]; NB = int(sys.argv[3]) if len(sys.argv) > 3 else 16
GIB = os.environ.get("TT_GPU_GIB", "38")
t0 = time.time()
m = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto",
                                         max_memory={0: f"{GIB}GiB", "cpu": "400GiB"})
m.eval(); print(f"load {time.time() - t0:.0f} s", flush=True)
dm = getattr(m, "hf_device_map", {})
layers = m.model.layers
ev = {}


def where(i):
    for k, v in dm.items():
        if k.endswith(f"layers.{i}") or f"layers.{i}." in k + ".":
            return "gpu" if v in (0, "cuda", "cuda:0") else "cpu"
    return "gpu"


for i, l in enumerate(layers):
    def pre(mod, a, i=i):
        e = torch.cuda.Event(enable_timing=True); e.record(); ev.setdefault(i, []).append([e, None])

    def post(mod, a, o, i=i):
        e = torch.cuda.Event(enable_timing=True); e.record(); ev[i][-1][1] = e
    l.register_forward_pre_hook(pre); l.register_forward_hook(post)
kinds = [getattr(m.config, "layer_types", None) or getattr(getattr(m.config, "text_config", m.config), "layer_types", None)]
kinds = kinds[0] or ["?"] * len(layers)
d = np.memmap(data, dtype=np.uint32, mode="r")
head_t = []


def hpre(mod, a):
    e = torch.cuda.Event(enable_timing=True); e.record(); head_t.append([e, None])


def hpost(mod, a, o):
    e = torch.cuda.Event(enable_timing=True); e.record(); head_t[-1][1] = e
m.lm_head.register_forward_pre_hook(hpre); m.lm_head.register_forward_hook(hpost)


def step(rep):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        for b in range(NB):
            x = torch.from_numpy(np.asarray(d[(rep * NB + b) * 2049:(rep * NB + b) * 2049 + 2048], dtype=np.int64))[None].cuda()
            m(input_ids=x, use_cache=False).logits
    torch.cuda.synchronize(); return time.time() - t


for rep in range(3):
    ev.clear(); head_t.clear()
    w = step(rep)
    per = {i: sum(a.elapsed_time(b) for a, b in v) / 1e3 for i, v in ev.items()}
    res = [per[i] for i in per if where(i) == "gpu"]; off = [per[i] for i in per if where(i) == "cpu"]
    hd = sum(a.elapsed_time(b) for a, b in head_t) / 1e3
    lin = [per[i] for i in per if where(i) == "gpu" and "linear" in str(kinds[i])]
    full = [per[i] for i in per if where(i) == "gpu" and "full" in str(kinds[i])]
    mean = lambda v: sum(v) / max(len(v), 1)
    est = mean(lin) * sum("linear" in str(k) for k in kinds) + mean(full) * sum("full" in str(k) for k in kinds) + hd
    print(f"rep {rep}: wall {w:.2f} s for {NB} x 2048 tokens; layers resident {len(res)} (mean {mean(res):.3f} s), offloaded "
          f"{len(off)} (mean {mean(off):.3f} s); resident linear-attention {mean(lin):.3f} s x {len(lin)}, full {mean(full):.3f} s x "
          f"{len(full)}; head {hd:.2f} s; compute-only estimate (all layers resident) {est:.2f} s / step", flush=True)
print(f"peak allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
