"""Generate text with a ternarized Qwen3.5 / 3.8 checkpoint from our trainer (ttf/qwen.py): transformers' model with the
large linears replaced by our ternary layers (trits, row / column scales, beta, rank-16 float adapter, int8 activations),
exactly as trained. Prompts from --prompts (one per line) or a few defaults; --chat applies the tokenizer's chat template.
    python scripts/quant/qwen_generate.py HF_DIR CKPT [--prompts FILE] [--max_new 128] [--temp 0.7] [--chat]
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ttf.qwen import build_qwen                                        # noqa: E402
from ttf.layers import KernelTernaryLinear, enable_rc_scales            # noqa: E402
from ttf.bitlinear import STATE                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir"); ap.add_argument("ckpt")
ap.add_argument("--prompts", default=""); ap.add_argument("--max_new", type=int, default=128)
ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--chat", action="store_true"); ap.add_argument("--mag", default="add:16")
a = ap.parse_args()
dev = "cuda"
t0 = time.time()
mk = lambda i, o: KernelTernaryLinear(i, o, 8, beta=True, int8=True)
model, _ = build_qwen(a.hf_dir, "kernel", make_linear=mk, grad_checkpoint=False, device=dev)
enable_rc_scales(model)
kind, r = a.mag.split(":")
for m in model.modules():
    if isinstance(m, KernelTernaryLinear):
        m.enable_lowrank_mag(kind, int(r))
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)["model"]
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"loaded {a.ckpt} in {time.time() - t0:.0f}s; missing {len(miss)} {miss[:3]}, unexpected {len(unexp)} {unexp[:3]}",
      flush=True)
del sd
model.eval(); STATE.updates_enabled = False
for p in model.parameters():
    p.requires_grad_(False)
print(f"GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.hf_dir)
prompts = [l.rstrip("\n") for l in open(a.prompts)] if a.prompts else [
    "The capital of France is",
    "Albert Einstein was born in",
    "def fibonacci(n):",
    "Explain in one paragraph why the sky is blue.",
    "List three prime numbers greater than 100:",
]
hf = model.hf
for q in prompts:
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) \
        if a.chat else q
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    t1 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = hf.generate(ids, max_new_tokens=a.max_new, do_sample=a.temp > 0, temperature=max(a.temp, 1e-5),
                          top_p=a.top_p, pad_token_id=tok.eos_token_id)
    n = out.shape[1] - ids.shape[1]
    print(f"\n### {q}\n{tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)}\n"
          f"[{n} tokens, {n / (time.time() - t1):.1f} tok/s]", flush=True)
