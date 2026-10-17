#!/usr/bin/env bash
# Lab PC: master weights degraded toward us: branch of curve_master at 131M (step 4000) -> 300M with the latent
# weights stored at K bits after every step (stochastic rounding; K = 2 = 3 levels = a stateless master).
#   master_branch.sh NAME K
set -u
L=$HOME/ternary-LLM/scripts/lab
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
N=$1; B=$2
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $L/sync_out.sh >/dev/null 2>&1 &
mkdir -p checkpoints/$N
if [ ! -f checkpoints/$N/ckpt.pt ]; then
  cp $HOME/ternary-sync/inbox/master4000/ckpt_4000.pt checkpoints/$N/ckpt.pt
  cp $HOME/ternary-sync/inbox/master4000/metrics_4000.jsonl checkpoints/$N/metrics.jsonl
  echo "[branch of curve_master at step 4000 (131M): latent weights stored at $B bits]" > checkpoints/$N/train.log
fi
python -m ttf.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
  --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 --track_flips \
  --master_bits $B --resume --out_dir checkpoints/$N >> checkpoints/$N/train.log 2>&1
echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> $HOME/ternary-sync/lab_queue.log
