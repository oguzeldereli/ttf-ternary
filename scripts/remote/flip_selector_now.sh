#!/usr/bin/env bash
# Priority on the 4090, alone on the GPU: pause the nola_b48 filler (checkpoint saved), run the learned flip selectors
# (A aligned, B loss change) at nola_lab @164M, then resume nola_b48.
# Output line by line: ~/ternary-sync/runs/momentum_mech/flip_selector.txt
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
pkill -u "$USER" -f "remote/night_4090.sh"
pkill -u "$USER" -f "remote/nola_b48_now.sh"
pkill -TERM -u "$USER" -f "out_dir checkpoints/nola_b48$"
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 5; done
mkdir -p checkpoints/nola_lab
[ -f checkpoints/nola_lab/ckpt_5000.pt ] || cp $HOME/ternary-sync/inbox/nola5000/ckpt_5000.pt checkpoints/nola_lab/
python -u -m scripts.analysis.flip_selector nola_lab 5000 2>&1 | grep --line-buffered -v Warn \
  > $HOME/ternary-sync/runs/momentum_mech/flip_selector.txt
echo "$(date '+%F %T') DONE flip_selector"
bash scripts/remote/nola_b48_now.sh
