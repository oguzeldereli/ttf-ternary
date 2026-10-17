#!/bin/bash
# Lab PC (16 Oct, user): the 110M TTF final recipe from scratch (rank 512, theta 16, p 0.75 annealed leak, no momentum,
# extras, 300M tokens) at fewer tokens per step. Everything counted in steps is scaled to stay the same in tokens:
#   b4: 4 x 2048 = 8k tokens / step, 36620 steps, warmup 1220, tau 4000, flip-rate warmup 120, eval every 1000 x 120 batches
#   b8: 8 x 2048 = 16k tokens / step, 18310 steps, warmup 610, tau 2000, flip-rate warmup 60, eval every 500 x 60 batches
# Reference (32k / step, same recipe): sw512_t75 fixed 2.6728, seed 2 2.6671. Then final_eval on the fixed windows.
#   setsid nohup bash ~/ternary-LLM/scripts/lab/batch_scan.sh b4 > /tmp/oguzelde/bs_b4.log 2>&1 &
set -u
ARM=$1; S=${S:-/tmp/oguzelde/tern}; source $S/env.sh; cd $S/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $ARM in
  b4) B=4; ST=36620; W=1220; TAU=4000; RW=120; EI=1000; EN=120 ;;
  b8) B=8; ST=18310; W=610; TAU=2000; RW=60; EI=500; EN=60 ;;
  b16) B=16; ST=9155; W=305; TAU=1000; RW=30; EI=250; EN=30 ;;
esac
D=checkpoints/scr_t75_$ARM; mkdir -p $D
R=""; [ -f $D/ckpt.pt ] && R="--resume"
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 \
  --batch_size $B --grad_accum 1 --steps $ST --warmup $W --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup $RW \
  --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval $EI --eval_iters $EN \
  --save_secs 1800 --snap_every 0 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp \
  --ts --ts_theta 16 --ts_tau $TAU --ts_anneal 0.75 --ts_tau_anneal --lowrank 512 --ts_rank_s 512 --ts_nomom --ts_fused \
  --gemm int8mm $R --out_dir $D >> $D/train.log 2>&1
echo "$(date '+%F %T') train end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
python scripts/eval/final_eval.py $D --fw 0 > $D/final_eval.txt 2>&1
echo "$(date '+%F %T') BS_DONE $ARM" >> $D/train.log
