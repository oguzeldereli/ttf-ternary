"""Sample from a wiki32k (Llama 32k BPE) checkpoint.

  python3 gen.py --ckpt checkpoints/p3b_acc1/ckpt.pt --prompt "The history of" --tokens 60
"""
import argparse, torch
from tokenizers import Tokenizer
from ttf.flip import build_kernel_transformer
from ttf.master import build_master_transformer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/p3b_acc1/ckpt.pt")
ap.add_argument("--prompt", default="The history of")
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--topk", type=int, default=40)
ap.add_argument("--n", type=int, default=1)
args = ap.parse_args()

dev = "cuda"
tok = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
mc, mode = blob["cfg"], blob.get("mode", "kernel")
if mode == "master":
    model = build_master_transformer(mc, grad_checkpoint=False)
else:
    model = build_kernel_transformer(
        mc, grad_checkpoint=False, evidence=mode == "evidence", ev_bits=3,
        beta=blob.get("beta", False), int8=blob.get("int8", False),
        dw_mode=blob.get("dw_mode", "dense"), int8_dx=blob.get("int8_dx", False))
    for p in model.float_tail_parameters():
        p.data = p.data.to(torch.bfloat16)
model.load_state_dict(blob["model"], strict=False)
model = model.to(dev).eval()
print(f"{args.ckpt}: {mode} | step {blob.get('step')} | {mc.n_params()/1e6:.0f}M params", flush=True)

ids = tok.encode(args.prompt, add_special_tokens=False).ids or [1]
for i in range(args.n):
    idx = torch.tensor([ids], device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, args.tokens, temperature=args.temp, top_k=args.topk)
    print(f"\n--- sample {i+1}\n{tok.decode(out[0].tolist())}", flush=True)
