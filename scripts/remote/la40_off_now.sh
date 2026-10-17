#!/usr/bin/env bash
# Priority test on the 4090: pause la_sched (SIGTERM saves its checkpoint), run look-ahead ON for the first 40
# steps (1.31M tokens) then OFF, to 10M tokens (step 305), then resume la_sched where it stopped. The test runs
# outside checkpoints/ so its 15 snapshots aren't shipped; the plateau analysis runs here and only logs + the
# analysis go home (~/ternary-sync/runs/la40_off).
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
pkill -TERM -u "$USER" -f "out_dir checkpoints/la_sched$"
while pgrep -u "$USER" -f "python -m ttf.train" >/dev/null; do sleep 5; done
D=$S/scratch_runs/la40_off; rm -rf "${D:?}"; mkdir -p $D
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 \
  --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 \
  --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 10 --save_secs 100000 --snap_every 20 --stop_after 306 \
  --probe 0-300:10 --lookahead 2 --lookahead_xbatch --lookahead_off 40:100000 --lowrank 256 --ckpt_skip 2 \
  --out_dir $D > $D/train.log 2>&1
mkdir -p $HOME/ternary-sync/runs/la40_off && cp $D/train.log $D/metrics.jsonl $HOME/ternary-sync/runs/la40_off/
echo "$(date '+%F %T') DONE la40_off training"
bash scripts/remote/la_sched.sh &
python -m scripts.analysis.plateau $D 2>&1 | grep "^step" > $D/plateau.txt
cp $D/plateau.txt $HOME/ternary-sync/runs/la40_off/
echo "$(date '+%F %T') DONE la40_off analysis"
wait
