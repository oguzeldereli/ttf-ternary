#!/usr/bin/env bash
# Extend magadd16_wd_qk (additive r16 + adapter weight decay 0.1 + per-head temperature) from 32M to 131M tokens
# (step 4000, where master's induction is 0.44). Waits for the 32M run, then resumes it in place.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
until grep -q "DONE magadd16_wd_qk" checkpoints/remote_queue.log 2>/dev/null; do sleep 30; done
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 20; done
N=magadd16_wd_qk
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 --snap_every 250 \
  --stop_after 4001 --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
  --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --resume --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE ${N}_131M $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
