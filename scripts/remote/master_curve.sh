#!/usr/bin/env bash
# Master (latent weights + AdamW, the p2_baseline settings) from scratch to step 6350 with nested windows
# measured after 1,2,5,10,20,50,100 steps from each of the momentum run's checkpoint steps. Waits for the
# sqrt(n) job if it is still running.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
while pgrep -u "$USER" -f "scripts.analysis.filter_sqrt_n" >/dev/null; do sleep 30; done
rm -rf checkpoints/curve_master; mkdir -p checkpoints/curve_master
python -m ttf.train --preset small --mode master --master_dtype fp32 --data data/wiki32k_train.bin \
  --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 --stop_after 6350 \
  --track_flips --window_curve 1,2,5,10,20,50,100 --window_curve_at 340,1000,2000,3000,4000,5000,6000,6243 \
  --out_dir checkpoints/curve_master > checkpoints/curve_master/train.log 2>&1
echo "$(date '+%F %T') DONE curve_master"
