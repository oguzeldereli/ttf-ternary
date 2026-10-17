#!/usr/bin/env bash
# After master_rec: ours (momentum + look-ahead x2) from scratch to 60M with the same 5% per-step
# recording (same weight sample: fixed seed), for any-window comparison with master.
set -u
cd "$(dirname "$0")/../.."
while kill -0 799640 2>/dev/null; do sleep 20; done
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p checkpoints/ours_rec
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips \
  --track_reversals --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 1830 --eval_iters 30 \
  --eval_interval 250 --save_secs 1800 --snap_every 1000 --lookahead 2 --lookahead_xbatch --lowrank 256 \
  --ckpt_skip 2 --record_sample 0.05 --out_dir checkpoints/ours_rec > checkpoints/ours_rec/train.log 2>&1
