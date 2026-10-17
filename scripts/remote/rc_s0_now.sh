#!/usr/bin/env bash
# 4090: stop accumulate-then-flip from scratch (and its queue), then the baseline that should have had row/column
# scales all along: rank-256 momentum, no look-ahead, learned row and column scales, from scratch -> 300M.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
pkill -u "$USER" -f "remote/sync_out.sh"       # the old sync shipped checkpoints through the home folder
setsid nohup bash scripts/remote/sync_out.sh > /dev/null 2>&1 < /dev/null &
pkill -u "$USER" -f "remote/accum_now.sh"
pkill -TERM -u "$USER" -f "out_dir checkpoints/accum33_s0$"
while pgrep -u "$USER" -f "[p]ython -m ttf.train" >/dev/null; do sleep 5; done
N=rc_s0; mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --rc_scale $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) \
  --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
