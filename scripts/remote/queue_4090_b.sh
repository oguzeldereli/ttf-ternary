#!/usr/bin/env bash
# 4090, unattended (~7.5 h): stop accumulate-then-flip from scratch, restart the sync (logs only, protects the
# home quota), then from scratch to 300M, no look-ahead, all with learned row/column scales:
#   rc_s0            plain rank-256 momentum + row/column scales (the baseline that should have had them)
#   la40rc_s0        the same with look-ahead for the first 40 steps only: no 2-5M plateau; does the slope then
#                    match master's all the way?
#   adaptrate_rc_s0  adaptive flip rate (best branch so far) + row/column scales
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
pkill -u "$USER" -f "remote/sync_out.sh"
setsid nohup bash scripts/remote/sync_out.sh > /dev/null 2>&1 < /dev/null &
pkill -u "$USER" -f "remote/accum_now.sh"; pkill -u "$USER" -f "remote/rc_s0_now.sh"
pkill -TERM -u "$USER" -f "out_dir checkpoints/accum33_s0$"
while pgrep -u "$USER" -f "[p]ython -m ttf.train" >/dev/null; do sleep 5; done
run() {   # NAME [flags]
  local N=$1; shift
  mkdir -p checkpoints/$N
  python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
    --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
    --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
    --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
    --lowrank 256 --ckpt_skip 2 --rc_scale "$@" $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) \
    --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
}
run rc_s0 --lookahead 0
run la40rc_s0 --lookahead 2 --lookahead_xbatch --lookahead_off 40:100000
run adaptrate_rc_s0 --lookahead 0 --adapt_rate
