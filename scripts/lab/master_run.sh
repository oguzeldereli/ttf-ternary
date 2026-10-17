#!/usr/bin/env bash
# Lab PC: one 110M master-weights run from scratch, full 300M schedule, with extra flags (e.g. one --m_* restriction).
#   master_run.sh NAME [train flags]
set -u
L=$HOME/ternary-LLM/scripts/lab
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
N=$1; shift
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $L/sync_out.sh >/dev/null 2>&1 &
mkdir -p checkpoints/$N
python -m ttf.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
  --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 --track_flips \
  "$@" $([ -f checkpoints/$N/ckpt.pt ] && echo --resume) --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> $HOME/ternary-sync/lab_queue.log
