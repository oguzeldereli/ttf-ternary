#!/usr/bin/env bash
# Full-precision reference: same 110M architecture, plain nn.Linear (no ternary, no activation
# quantization), fp32 weights + AdamW, same data/schedule as master (300M tokens). Waits for the
# master window run if it is still going.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
while pgrep -u "$USER" -f "out_dir checkpoints/curve_master" >/dev/null; do sleep 30; done
rm -rf checkpoints/fp32_baseline; mkdir -p checkpoints/fp32_baseline
python -m ttf.train --preset small --mode fp32 --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 \
  --eval_interval 250 --eval_iters 30 --save_secs 3600 --out_dir checkpoints/fp32_baseline \
  > checkpoints/fp32_baseline/train.log 2>&1
echo "$(date '+%F %T') DONE fp32_baseline $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/fp32_baseline/train.log)"
