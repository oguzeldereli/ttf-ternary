#!/usr/bin/env bash
# Lab PC: one --mech run without look-ahead.  mech_run.sh NAME MECH START STOP [extra train flags]
#   START = branch: from nola_lab at 131M (step 4000, warm momentum; its metrics to step 4000 copied for the curve)
#   START = scratch: from step 0.   STOP = --stop_after step (9156 = the full 300M schedule).
# Waits until no other training run of this user is on the GPU (a resumed run continues where it stopped).
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
N=$1; MECH=$2; START=$3; STOP=$4; shift 4
while pgrep -u "$USER" -f "python -m ttf.train|scripts.analysis.momentum_mech" >/dev/null; do sleep 30; done
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $HOME/ternary-LLM/scripts/lab/sync_out.sh >/dev/null 2>&1 &
mkdir -p checkpoints/$N
if [ "$START" = branch ] && [ ! -f checkpoints/$N/ckpt.pt ]; then
  B=$(ls $S/bench_nola*/ckpt_4000.pt 2>/dev/null | head -1)
  [ -n "$B" ] || { mkdir -p $S/bench_nola; cp $HOME/ternary-sync/inbox/bench_nola/ckpt_4000.pt $S/bench_nola/; B=$S/bench_nola/ckpt_4000.pt; }
  cp "$B" checkpoints/$N/ckpt.pt; cp $HOME/ternary-sync/inbox/bench_nola/metrics_4000.jsonl checkpoints/$N/metrics.jsonl
  echo "[branch of nola_lab at step 4000 (131M); --mech $MECH from here, no look-ahead]" > checkpoints/$N/train.log
fi
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --mech $MECH --stop_after $STOP "$@" \
  $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> $HOME/ternary-sync/lab_queue.log
