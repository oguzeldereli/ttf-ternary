#!/usr/bin/env bash
# Priority test: look-ahead ON for the first 40 steps (1.31M tokens), OFF from there. Does the no-look-ahead unigram
# plateau (2-5M) still happen? To 10M tokens (step 305), snapshots every 20 steps, probes, momentum diagnostics.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
while pgrep -f "out_dir checkpoints/plateau_la$" >/dev/null; do sleep 10; done
D=checkpoints/la40_off; mkdir -p $D
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 \
  --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 \
  --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 10 --save_secs 100000 --snap_every 20 --stop_after 306 \
  --probe 0-300:10 --lookahead 2 --lookahead_xbatch --lookahead_off 40:100000 --lowrank 256 --ckpt_skip 2 \
  --out_dir $D > $D/train.log 2>&1
python -m scripts.analysis.plateau $D 2>&1 | grep "^step" > $D/plateau.txt
