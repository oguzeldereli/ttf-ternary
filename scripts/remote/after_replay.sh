#!/usr/bin/env bash
# Runs after the 11M->205M replay finishes, inside the booking (ends 09:00): each job gets a hard stop
# at 08:40 (SIGTERM = save after the current step), and a job is skipped if it cannot finish by then.
#   1-2. momentum horizon: branches from ckpt_6243 (205M) with beta 0.99 / 0.995 (beta 0.97: val 3.2005)
#   3.   stateless look-ahead x2 at rate 0.013 (the momentum run's flip count), 10M screen from scratch
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
while pgrep -u "$USER" -f "out_dir checkpoints/r4090_replay_11M_205M" >/dev/null; do sleep 30; done
echo "$(date '+%F %T') replay finished; starting after_replay jobs"
STOP=$(date -d "08:40" +%s)
left() { echo $(( STOP - $(date +%s) )); }
job() {  # name, minutes needed, command...
  local n=$1 need=$2; shift 2
  if [ "$(left)" -lt $(( need * 60 )) ]; then echo "$(date '+%F %T') SKIP $n (not enough time)"; return; fi
  mkdir -p checkpoints/$n
  timeout -s TERM -k 120 "$(left)" "$@" --out_dir checkpoints/$n > checkpoints/$n/train.log 2>&1
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
BR="python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips
  --track_reversals --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --stop_after 6544 --eval_iters 30
  --eval_interval 100000 --save_secs 100000 --lookahead 2 --lookahead_xbatch --ckpt_skip 2 --resume
  --lowrank 256"
for b in 0.99 0.995; do
  n=b6243_A_beta$b; mkdir -p checkpoints/$n; ln -sf $S/ckpt_6243.pt checkpoints/$n/ckpt.pt
  job $n 20 $BR --lr_beta $b
done
if [ "$(left)" -ge 1200 ]; then
  timeout -s TERM -k 120 "$(left)" scripts/train/screen_10m.sh r4090_la_xb2_r013 --lookahead 2 --lookahead_xbatch --rate_peak 0.013
  rm -f checkpoints/r4090_la_xb2_r013/ckpt.pt
  echo "$(date '+%F %T') DONE r4090_la_xb2_r013 $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/r4090_la_xb2_r013/train.log)"
else echo "$(date '+%F %T') SKIP r4090_la_xb2_r013"; fi
echo "$(date '+%F %T') after_replay done"
