#!/bin/bash -l
# Myriad: master weights + the same float extras as TTF (row / column scales, rank-16 additive adapter, attention temperature): the fair reference at 1.3B. Master weights (fp32 latent + STE + AdamW) on the 1.3B model (d2048_l24, width 2048),
# 300M tokens of wiki32k, 1 A100. Peak lr 7.5e-4: at 1.5e-3 (x1b_master) master diverged from ~33M tokens (val 4.24 -> 4.53). Every layer gradient-checkpointed (no --ckpt_skip): peak ~26-28 GB, fits the 40 GB nodes. Checkpoints every 30 min (resubmit the same script to resume).
#   qsub scripts/myriad/run_1b_mx_extras_lr75.sh   (from ~/Scratch/ternary-LLM)
#$ -N tern_1b_mx75
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
N=x1b_mx_extras_lr75; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
F="--preset d2048_l24 --mode master --master_dtype fp32 --data data/wiki32k_train.bin --val data/wiki32k_val.bin
   --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 250
   --eval_iters 30 --save_secs 1800 --snap_every 1000 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --out_dir $D"
F=$(echo $F)
[ -f $D/ckpt.pt ] && F="$F --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m bitnet.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
