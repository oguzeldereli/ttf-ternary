#!/usr/bin/env bash
# Lab PC: resume a run shipped to ~/ternary-sync/inbox/NAME (ckpt.pt, train.log, metrics.jsonl, READY) with the
# given train flags (the run's own recipe; --resume and --out_dir are added). Waits for READY and a free GPU.
#   bash resume_run.sh NAME [train flags...]
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
N=$1; shift
until [ -f checkpoints/$N/ckpt.pt ] || [ -f $HOME/ternary-sync/inbox/$N/READY ]; do sleep 30; done
if [ ! -f checkpoints/$N/ckpt.pt ]; then
  mkdir -p checkpoints/$N && cp $HOME/ternary-sync/inbox/$N/{ckpt.pt,train.log,metrics.jsonl} checkpoints/$N/ \
    && rm -rf $HOME/ternary-sync/inbox/$N
fi
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 30; done
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $HOME/ternary-LLM/scripts/lab/sync_out.sh >/dev/null 2>&1 &
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 900 --lowrank 256 --ckpt_skip 2 \
  "$@" --resume --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> $HOME/ternary-sync/lab_queue.log
