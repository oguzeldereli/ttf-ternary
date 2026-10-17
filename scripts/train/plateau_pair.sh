#!/usr/bin/env bash
# The unigram plateau: first 240 steps (7.9M tokens) with snapshots every 20 steps and per-step momentum diagnostics,
# without look-ahead (plateau_nola, started separately) and with it (plateau_la); then the plateau analysis on both.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
while pgrep -f "out_dir checkpoints/plateau_nola" >/dev/null; do sleep 15; done
D=checkpoints/plateau_la; mkdir -p $D
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 \
  --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 \
  --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 10 --save_secs 100000 --snap_every 20 --stop_after 241 \
  --probe 0-240:10 --lr_diag --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --out_dir $D > $D/train.log 2>&1
for r in plateau_nola plateau_la; do
  echo "== $r"; python -m scripts.analysis.plateau checkpoints/$r 2>&1 | grep "^step"
done > checkpoints/plateau_analysis.txt
