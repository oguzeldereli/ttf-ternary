#!/usr/bin/env bash
# Continue lm_lowrank256_xb2_100M to the end of its schedule (step 9155, 300M tokens).
# The 100M part ran on the old trainer, whose final checkpoint is labelled end_step (3052)
# although the last completed step is 3051; relabel it so the resume starts exactly at 3052.
set -eu
cd "$(dirname "$0")/../.."
D=checkpoints/lm_lowrank256_xb2_100M
while pgrep -f "out_dir $D " >/dev/null; do sleep 15; done
grep -q "FINAL val" $D/train.log || { echo "100M part did not finish cleanly"; exit 1; }
PYTHONPATH=. python - <<'P'
import torch
p = "checkpoints/lm_lowrank256_xb2_100M/ckpt.pt"
b = torch.load(p, map_location="cpu", weights_only=False)
if b["step"] == 3052:
    b["step"] = 3051; torch.save(b, p + ".tmp"); import os; os.replace(p + ".tmp", p)
print("ckpt step", b["step"])
P
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
exec python -m ttf.train --preset small --mode kernel \
  --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 \
  --lookahead 1 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 9155 --probe 0-40:5,40-320:20 \
  --eval_iters 30 --eval_interval 250 --save_secs 900 \
  --out_dir $D --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
  --lr_diag --track_reversals --resume >> $D/train.log 2>&1
