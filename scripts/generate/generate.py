"""Generate text from a (possibly mid-training) checkpoint + word vocab.

Usage:
  python3 generate.py --ckpt checkpoints/b27/ckpt.pt --vocab data/tech_vocab.json \
      --prompt "technology facilitated abuse" --tokens 80 --temp 0.8 --topk 40
"""
import json, re, argparse, torch
from ttf.config import ModelConfig
from ttf.flip import build_kernel_transformer, build_stateless_transformer
from ttf.model import BitTransformer

PAT = re.compile(r"\w+|[^\w\s]")


def build(mode, cfg, beta=False):
    if mode in ("kernel", "evidence"):
        return build_kernel_transformer(cfg, grad_checkpoint=False, beta=beta,
                                        evidence=mode == "evidence")
    if mode == "stateless":
        return build_stateless_transformer(cfg, grad_checkpoint=False)
    return BitTransformer(cfg, grad_checkpoint=False)


def detok(ids, itos):
    # word-level: join with spaces, but glue punctuation to the previous word
    out = []
    for i in ids:
        w = itos.get(i, "<unk>")
        if w == "<eos>":
            out.append("\n")
        elif re.fullmatch(r"[^\w\s]", w) and out:
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/b27/ckpt.pt")
    ap.add_argument("--vocab", default="data/tech_vocab.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    stoi = json.load(open(args.vocab))
    itos = {v: k for k, v in stoi.items()}

    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    mc: ModelConfig = blob["cfg"]
    model = build(blob.get("mode", "kernel"), mc, blob.get("beta", False))
    for p in model.float_tail_parameters():
        p.data = p.data.to(torch.bfloat16)
    model.load_state_dict(blob["model"])
    model = model.to(dev).eval()
    print(f"loaded step {blob.get('step')} | {mc.n_params()/1e9:.1f}B params", flush=True)

    ids = [stoi.get(t, 0) for t in PAT.findall(args.prompt)] or [1]  # <eos> if empty
    idx = torch.tensor([ids], device=dev)
    with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16):
        out = model.generate(idx, args.tokens, temperature=args.temp, top_k=args.topk)
    print("\n" + detok(out[0].tolist(), itos))


if __name__ == "__main__":
    main()
