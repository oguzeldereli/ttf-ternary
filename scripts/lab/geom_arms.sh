#!/bin/bash
# 110M conversion / geometry arms (15-16 Oct; docs/QUEUE.md 23:00): one arm per lab PC, then the analyses on its result.
#   bash /tmp/oguzelde/tern/repo/scripts/lab/geom_arms.sh ARM
# Needs $S/models/fp32_110m_hf (scripts/analysis/export_hf_llama.py of checkpoints/fp32_baseline), fp32_110m_sp
# (perturb_hf.py, shrink 0.8 noise 0.3) and repo/$MD/fp32_baseline_slim/ckpt.pt (for the comparisons).
# All conversion arms: 50M tokens (1526 steps x 16 x 2048), KD from the fp model at weight 0.5, Wikipedia (wiki32k).
#   conv      GPTQ start, then TTF (trits frozen to 400, theta 64, rank 512)          the current recipe
#   sp        shrink and perturb the fp weights, then as conv                          arm c
#   deout     fp32 fine-tune with the residual outlier penalty (--act_reg 0.05), then GPTQ in the analysis   arm a
#   homo      master mode, homotopy fp -> ternary over 1000 steps (--m_qmix 1000)     arm b
#   kdscratch TTF from scratch (final recipe, theta 16) with the same KD and tokens   arm d (insight only)
#   gptqc     centred GPTQ start (input mean out before rounding, W mu as a float bias), then as conv   arm e
#   gptqs     sequential, error-aware, centred GPTQ start (convert_seq_asym), then as conv
#   convreg   as conv with the residual outlier penalty on the ternary student (--act_reg 0.05)       arm f
#   resann    TTF-native homotopy: GPTQ trits + the fp rounding residual faded 1 -> 0 over 1000 steps (--res_anneal)
#   resann256 the same with the residual at rank 256 (the memory-cheap form for big models); resann_long: 1400 steps
#   homo1500  master homotopy over 1500 steps
#   conv_long / homo_long / resann_150M   the same at 150M tokens (4578 steps; fades over 3000)
#   fpseed2   full precision from scratch, seed 2, 300M tokens, snapshots every 1000 steps (baselines)
set -u
ARM=$1; S=${S:-/tmp/oguzelde/tern}; REPO=${REPO:-$S/repo}      # Myriad: S=~/Scratch/tern REPO=~/Scratch/ternary-LLM
source $S/env.sh; cd $REPO
D=checkpoints/g_$ARM; mkdir -p $D logs
MD=${MD:-$S/models}      # outside the code tree (setup.sh mirrors the code with --delete)
HF=$MD/fp32_110m_hf
DATA="--data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1"
KD="--distill $HF --distill_w 0.5"
TTF="--mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense
  --tail_fp32 --eval_interval 100 --eval_iters 8 --save_secs 1800 --snap_every 0 --lookahead 0 --ckpt_skip 2 --rc_scale
  --lowrank_mag add:16 --mag_wd 0.1 --ts --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm
  --lowrank 512 --ts_rank_s 512"
CONV="--steps 1526 --warmup 300 --lr 5e-4 --min_lr 5e-5 --init_hf_method gptq --ts_theta 64 --ts_start 400"
case $ARM in
  conv)      F="$DATA $TTF $CONV --init_hf $HF $KD" ;;
  sp)        F="$DATA $TTF $CONV --init_hf $MD/fp32_110m_sp $KD" ;;
  gptqc)     F="$DATA $TTF ${CONV/gptq/gptqc} --init_hf $HF $KD" ;;                     # arm e: centred GPTQ start
  gptqs)     F="$DATA $TTF ${CONV/gptq/gptqs} --init_hf $HF $KD" ;;                     # sequential error-aware centred start
  resann)    F="$DATA $TTF ${CONV/--ts_start 400/--ts_start 100} --init_hf $HF $KD --res_anneal 1000" ;;      # TTF homotopy, full residual
  resann256) F="$DATA $TTF ${CONV/--ts_start 400/--ts_start 100} --init_hf $HF $KD --res_anneal 1000:256" ;;  # rank-256 residual
  resann_long) F="$DATA $TTF ${CONV/--ts_start 400/--ts_start 100} --init_hf $HF $KD --res_anneal 1400" ;;   # slower fade
  homo1500)  F="$DATA --mode master --init_hf $HF $KD --m_qmix 1500 --rc_scale --tail_fp32 --steps 1526 --warmup 100
               --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 8 --save_secs 1800 --snap_every 0 --lookahead 0" ;;
  conv_long) F="$DATA $TTF ${CONV/--steps 1526/--steps 4578} --init_hf $HF $KD" ;;                # 150M tokens: does the base matter later?
  gptqc_long) F="$DATA $TTF $(echo $CONV | sed "s/--steps 1526/--steps 4578/; s/method gptq /method gptqc /") --init_hf $HF $KD" ;;
  homo_long) F="$DATA --mode master --init_hf $HF $KD --m_qmix 3000 --rc_scale --tail_fp32 --steps 4578 --warmup 100
               --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 8 --save_secs 1800 --snap_every 0 --lookahead 0" ;;
  resann_150M) F="$DATA $TTF $(echo $CONV | sed "s/--steps 1526/--steps 4578/; s/--ts_start 400/--ts_start 100/") --init_hf $HF $KD --res_anneal 3000" ;;
  convreg)   F="$DATA $TTF $CONV --init_hf $HF $KD --act_reg 0.05" ;;                    # arm f: outlier penalty on the student
  deout)     F="$DATA --mode fp32 --init_hf $HF $KD --act_reg 0.05 --steps 1526 --warmup 100 --lr 3e-4 --min_lr 3e-5
               --eval_interval 100 --eval_iters 8 --save_secs 1800 --snap_every 0 --lookahead 0" ;;
  homo)      F="$DATA --mode master --init_hf $HF $KD --m_qmix 1000 --rc_scale --tail_fp32 --steps 1526 --warmup 100
               --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 8 --save_secs 1800 --snap_every 0 --lookahead 0" ;;
  kdscratch) F="$DATA $TTF --preset small $KD --steps 1526 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 --ts_theta 16 --qk_temp" ;;
  fpseed2)   F="$DATA --preset small --mode fp32 --seed 2 --steps 9155 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4
               --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 1000 --lookahead 0" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
F=$(echo $F)
R=""; [ -f $D/ckpt.pt ] && R="--resume"
echo "$(date '+%F %T') start $ARM on $(hostname -s): $F $R" >> $D/train.log
python -m bitnet.train $F $R --out_dir $D >> $D/train.log 2>&1
echo "$(date '+%F %T') train end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
# analyses on the result (light): geometry against fp; losses of the roundings for the fp32 arms (trit_compare), fixed
# windows for the ternary ones (final_eval)
if [ $ARM = deout ] || [ $ARM = fpseed2 ]; then
  python scripts/analysis/geometry.py --model fp=$MD/fp32_baseline_slim/ckpt.pt --model $ARM=$D/ckpt.pt --gptq_of $ARM \
    --out $D/geometry.json > $D/geometry.txt 2>&1
  python scripts/analysis/trit_compare.py --fp $D/ckpt.pt --n_eval 64 --calib 64 --out $D/roundings.json > $D/roundings.txt 2>&1
else
  python scripts/analysis/geometry.py --model fp=$MD/fp32_baseline_slim/ckpt.pt --model $ARM=$D/ckpt.pt \
    --out $D/geometry.json > $D/geometry.txt 2>&1
  python scripts/eval/final_eval.py $D --fw 0 > $D/final_eval.txt 2>&1
fi
echo "$(date '+%F %T') ARM_DONE $ARM" >> $D/train.log
