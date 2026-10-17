#!/usr/bin/env bash
# First-step moves vs the summed gradient of the next k steps (k up to 100), ours, from 4 checkpoints.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in lm_lowrank256_xb2_diag/ckpt_340.pt r4090_replay_11M_205M/ckpt_2000.pt \
         r4090_replay_11M_205M/ckpt_4000.pt r4090_replay_11M_205M/ckpt_6000.pt; do
  s=$(basename $c .pt); s=${s#ckpt_}; n=curve100_ours_$s
  rm -rf checkpoints/$n; mkdir -p checkpoints/$n; ln -sf "$PWD/checkpoints/$c" checkpoints/$n/ckpt.pt
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
    --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips \
    --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after $((s + 101)) --eval_iters 1 \
    --eval_interval 100000 --save_secs 100000 --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
    --window_curve 1,2,5,10,20,50,100 --resume --out_dir checkpoints/$n > checkpoints/$n/train.log 2>&1
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n"
done
