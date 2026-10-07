#!/bin/bash -l
# Myriad: step-time test of large models (1.3B, b3 2.9B), TTF and master, at 16 / 32 / 64 sequences of 2048 tokens per
# step. Each configuration runs ~150 s (warm-up + steady steps); the time per step is read from the step log
# (cumulative seconds, steps 20 -> last). Results in logs/speed_big/summary.txt.
#   qsub scripts/myriad/speed_big.sh
#$ -N speed_big
#$ -l h_rt=0:50:00
#$ -l mem=12G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/speed_big; mkdir -p $O
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
hostname > $O/summary.txt; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $O/summary.txt
C="--data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --grad_accum 1 --steps 9155 --warmup 305
   --eval_interval 100000 --eval_iters 30 --save_secs 100000 --snap_every 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1
   --qk_temp --track_flips"
T="--mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense
   --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --lookahead 0 --ckpt_skip 2 --ts --ts_nomom --ts_theta 16 --ts_tau 1000
   --ts_tau_anneal --ts_anneal 0.5 --ts_fused --gemm int8mm --lowrank 1024"
M="--mode master --master_dtype fp32 --lr 5e-4 --min_lr 5e-5"
F="--ts_fast --ts_orth scholqr3"
one() {  # NAME PRESET BATCH FLAGS...
  n=$1; p=$2; b=$3; shift 3
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; timeout 170 python -m bitnet.train --preset $p --batch_size $b $(echo $C) $* --out_dir $TMPDIR/sp_$n" > $O/$n.log 2>&1
  r=$(grep -E '^step' $O/$n.log | awk '{x=$(NF-3); sub("s","",x); if ($2>=20) {if (!s0) {s0=$2; t0=x}; s1=$2; t1=x}} END {if (s1>s0) printf "%.2f s/step (%d steps measured), %.0f tokens/s", (t1-t0)/(s1-s0), s1-s0, '$b'*2048*(s1-s0)/(t1-t0); else print "no steady steps"}')
  pk=$(grep -E '^step' $O/$n.log | tail -1 | grep -o 'peakVRAM [0-9.]*GiB')
  oom=$(grep -c -i "out of memory" $O/$n.log)
  echo "$n  preset $p  batch $b  $r  $pk  ${oom:+oom=$oom}" >> $O/summary.txt
}
one ttf1b_b16      d2048_l24 16 $T
one ttf1b_fast_b16 d2048_l24 16 $T $F
one ttf1b_fast32_b16 d2048_l24 16 $T $F --tf32
one ttf1b_fast32_b32 d2048_l24 32 $T $F --tf32
one ttf1b_fast32_b64 d2048_l24 64 $T $F --tf32
one mx1b_b16       d2048_l24 16 $M
one mx1b_b64       d2048_l24 64 $M
one ttfb3_fast32_b16 b3 16 $T $F --tf32
one ttfb3_fast32_b32 b3 32 $T $F --tf32
one mxb3_b16       b3 16 $M
echo "$(date '+%F %T') done" >> $O/summary.txt
