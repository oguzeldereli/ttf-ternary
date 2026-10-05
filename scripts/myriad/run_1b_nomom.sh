#!/bin/bash -l
# Myriad: 1.3B model (d2048_l24, width 2048), 300M tokens of wiki32k, 1 A100. TTF without the short momentum
# (--ts_nomom: the accumulator takes the full-rank g / sqrt(v)), accumulator rank R, tau 1000, leak and steps x
# (lr ratio)^0.5, theta 16, float tail at peak lr 1.5e-3 (TTF's better lr at 340M). Otherwise the recipe of
# run_1b_ts.sh (x1b_ts512_la1000an05: ranks 512 with momentum, lr 7.5e-4, fixed windows 2.5510; master + the same extras
# 2.4980). Ranks 1024 without momentum has the state of ranks 512 with it.
#   qsub -N tern_1b_nm512 -v R=512 scripts/myriad/run_1b_nomom.sh     (resubmit the same way to resume)
#   qsub -N tern_1b_nm1024 -v R=1024 scripts/myriad/run_1b_nomom.sh
#$ -l h_rt=40:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
R=${R:-512}
N=x1b_nomom${R}_lr15; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
F="--preset d2048_l24 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30
   --save_secs 1800 --snap_every 1000 --lookahead 0 --lowrank $R --ckpt_skip 2 --rc_scale
   --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_nomom --ts_theta 16 --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.5
   --out_dir $D"
F=$(echo $F)
[ -f $D/ckpt.pt ] && F="$F --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m bitnet.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
