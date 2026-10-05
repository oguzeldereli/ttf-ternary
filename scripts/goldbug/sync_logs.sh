#!/usr/bin/env bash
# goldbug: every 20 min copy the small files (logs, final_eval.json, queue log) to ~/ternary-sync/goldbug so they
# survive the booking and the laptop can read them; checkpoints go to the laptop with scripts/goldbug/pull.sh.
source "${GB:-/scratch0/$USER/gb}/env.sh"
while true; do
  mkdir -p $HOME/ternary-sync/goldbug
  rsync -a --include='*/' --include='train.log' --include='final_eval.*' --include='final_eval.log' --exclude='*' \
    $GB/repo/checkpoints/ $HOME/ternary-sync/goldbug/ 2>/dev/null
  cp $GB/queue.log $GB/smoke.txt $HOME/ternary-sync/goldbug/ 2>/dev/null
  sleep 1200
done
