"""Conversion check for --init_hf (no training): the loss of a pretrained Llama on the first N fixed Wikipedia windows
(scripts/eval/final_eval.py), as
  hf        the original model in transformers (reference)
  fp32      our model with the copied weights (must match hf: checks the name map and the RoPE permutation)
  master    master mode, latent = the weight (BitNet absmean ternary in the forward)
  kernel:M  the trits and scales of --init_hf_method M, with the run's forward (beta, row / column scales, int8 acts)

    python scripts/quant/hf_check.py HF_DIR [--n 100] [--methods absmean,row,als] [--skip_hf]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from final_eval import window_losses                                            # noqa: E402
from ttf.layers import build_kernel_transformer, enable_rc_scales               # noqa: E402
from ttf.master import build_master_transformer, MasterTernaryLinear         # noqa: E402
from ttf.model import BitTransformer                                          # noqa: E402
from ttf.hf_init import model_config_from_hf, init_from_hf, calib_hessians    # noqa: E402


class _HF(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x).logits, None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--methods", default="absmean,row,als")
    ap.add_argument("--skip_hf", action="store_true")
    ap.add_argument("--data", default="data/wiki32k_val.bin")
    ap.add_argument("--calib", type=int, default=0, help="calibration windows from --calib_data for the input second "
                    "moments (method gptq; with it every method also reports its output-space error)")
    ap.add_argument("--calib_data", default="data/wiki32k_train.bin")
    ap.add_argument("--skip_fp", action="store_true", help="skip the fp32 and master checks")
    a = ap.parse_args()
    dev = "cuda"
    data = np.memmap(a.data, dtype=np.uint16, mode="r")
    mc = model_config_from_hf(a.hf_dir)
    print(f"{a.hf_dir}: dim {mc.dim} layers {mc.n_layers} heads {mc.n_heads}/{mc.n_kv_heads} hidden {mc.hidden_dim} "
          f"vocab {mc.vocab_size} tied {mc.tie_embeddings}", flush=True)

    def run(name, model):
        t = time.time()
        w, _ = window_losses(model.eval(), data, 2048, a.n, dev)
        print(f"{name:14s} loss {w.mean():.4f}  ({len(w)} windows, {time.time() - t:.0f}s)", flush=True)
        return w

    if not a.skip_hf:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(a.hf_dir, torch_dtype=torch.float32).to(dev)
        run("hf", _HF(m)); del m; torch.cuda.empty_cache()
    if not a.skip_fp:
        m = BitTransformer(mc, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
        init_from_hf(m, a.hf_dir, "fp32"); run("fp32", m); del m; torch.cuda.empty_cache()
        m = build_master_transformer(mc, grad_checkpoint=False).to(dev)
        init_from_hf(m, a.hf_dir, "master"); run("master", m); del m; torch.cuda.empty_cache()
    hess = calib_hessians(a.hf_dir, a.calib_data, n_seq=a.calib, dev=dev) if a.calib else None
    for meth in a.methods.split(","):
        m = build_kernel_transformer(mc, grad_checkpoint=False, rate=0.0, beta=True, int8=True, dw_mode="dense").to(dev)
        enable_rc_scales(m)
        t0 = time.time()
        init_from_hf(m, a.hf_dir, "kernel", meth, rc=True, hess=hess)
        print(f"  converted in {time.time() - t0:.0f}s", flush=True)
        run("kernel:" + meth, m); del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
