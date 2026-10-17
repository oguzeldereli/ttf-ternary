#!/usr/bin/env bash
# goldbug: one arm on one GPU, unattended. Runs NAME with FLAGS until FINAL, resuming from its checkpoint after any
# exit (watchdog: up to 8 restarts, 60 s apart); then the final evaluation on fixed windows (Wikipedia + FineWeb-Edu).
#   bash scripts/goldbug/queue_gpu.sh GPU NAME FLAGS...       (started by scripts/goldbug/start.sh)
set -u
source "${GB:-/scratch0/$USER/gb}/env.sh"; cd "$GB/repo"
g=$1; n=$2; shift 2; D=checkpoints/$n; mkdir -p $D; L=$GB/queue.log
export CUDA_VISIBLE_DEVICES=$g
for try in $(seq 0 8); do
  grep -q "FINAL val loss" $D/train.log 2>/dev/null && break
  R=""; [ -f $D/ckpt.pt ] && R="--resume"
  echo "$(date '+%F %T') $n start (try $try) $R" >> $L
  python -m ttf.train "$@" --out_dir $D $R >> $D/train.log 2>&1
  echo "$(date '+%F %T') $n exited ($?)" >> $L
  grep -q "FINAL val loss" $D/train.log && break
  sleep 60
done
if grep -q "FINAL val loss" $D/train.log && [ ! -f $D/final_eval.json ]; then
  python scripts/eval/final_eval.py $D >> $D/final_eval.log 2>&1
  echo "$(date '+%F %T') $n final eval: $(grep -oE '"wiki_loss": [0-9.]+|"fw_loss": [0-9.]+' $D/final_eval.json | paste -sd' ')" >> $L
fi
