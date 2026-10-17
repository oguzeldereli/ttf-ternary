#!/usr/bin/env bash
# Priority on the 4090: rank-256 momentum WITHOUT look-ahead at batch 48 = 3 x 16, exactly the text a look-ahead
# step reads (training batch + 2 check batches). Same 9155-step schedule, LR and flip-rate schedule as the
# look-ahead runs, so equal steps = equal text read. Pauses la_sched (checkpoint saved) and resumes it afterwards.
# Snapshots every 500 steps (induction tracker). Stop any time: it resumes with the same command.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
pkill -TERM -u "$USER" -f "out_dir checkpoints/la_sched$"
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 5; done
N=nola_b48
mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 48 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 500 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 \
  $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)"
bash scripts/remote/la_sched.sh
