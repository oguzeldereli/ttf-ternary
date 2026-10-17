#!/bin/bash
# --ts_factored at full length (17 Oct, user): the 110M TTF recipe from scratch at rank 320 (p 0.75 annealed leak, no
# momentum, theta 16, extras, 300M tokens = 9155 x 16 x 2048), dense TTF update vs the factored one, same seed.
#   bash /tmp/oguzelde/tern/repo/scripts/lab/fact_scratch.sh dense|fact256|fact1024 [SEED]
set -u
ARM=$1; SEED=${2:-1337}; S=${S:-/tmp/oguzelde/tern}; source $S/env.sh; cd $S/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
X=""; [ $ARM != dense ] && X="--ts_factored ${ARM#fact}"
D=checkpoints/fs320_$ARM${SUF:-}; [ $SEED != 1337 ] && D=${D}_s$SEED; mkdir -p $D
R=""; [ -f $D/ckpt.pt ] && R="--resume"
echo "$(date '+%F %T') start fs320_$ARM seed $SEED on $(hostname -s)" >> $D/train.log
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 \
  --batch_size 16 --grad_accum 1 --steps ${STEPS:-9155} --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 \
  --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 \
  --save_secs 1800 --snap_every 0 --lookahead 0 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp \
  --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --lowrank 320 --ts_rank_s 320 --ts_nomom --ts_fused \
  --gemm int8mm --seed $SEED $X $R --out_dir $D >> $D/train.log 2>&1
echo "$(date '+%F %T') train end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
python scripts/eval/final_eval.py $D --fw 0 > $D/final_eval.txt 2>&1
echo "$(date '+%F %T') FS_DONE $ARM" >> $D/train.log
