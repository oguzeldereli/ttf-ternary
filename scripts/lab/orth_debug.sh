#!/usr/bin/env bash
# Lab PC: why --ts_orth chol changes the loss (5.52 vs 5.65 at step 200 on the 4090). 300 steps at 110M each with
# Householder QR, Cholesky-QR (fp32 Gram), Cholesky-QR on unit columns, and Cholesky-QR in fp64; for the Cholesky runs
# TERN_ORTHDBG=1 logs, one step in 10: max |Q^T Q - I|, span error vs Householder on the same input, input condition.
# Summary: ~/ternary-sync/orthdbg[_TAG].txt.   bash scripts/lab/orth_debug.sh [TAG "variants"]
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
TAG=${1:-}; VARS=${2:-qr chol chol_cs chol64}
O=checkpoints/orthdbg${TAG:+_$TAG}; mkdir -p $O; R=$HOME/ternary-sync/orthdbg${TAG:+_$TAG}.txt
K="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100 --eval_iters 4
   --save_secs 1000000 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank 512
   --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --stop_after 301"
echo "$(date '+%F %T') start on $(hostname)" > $R
for v in $VARS; do
  TERN_ORTHDBG=1 python -m ttf.train $(echo $K) --ts_orth $v --out_dir $O/$v > $O/$v.log 2>&1
  echo "$v: $(grep 'val loss' $O/$v.log | cut -c1-28 | tr '\n' ' ') $(grep -m1 -E 'Error' $O/$v.log)" >> $R
  grep orthdbg $O/$v.log | sed -n '1p;2p;3p;5p;10p;20p;30p' | sed "s/^/  $v /" >> $R
done
echo "$(date '+%F %T') done" >> $R
