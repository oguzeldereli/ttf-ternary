#!/bin/bash -l
# Myriad: per-token losses (scripts/analysis/token_dump.py, 400 Wikipedia windows) of 1.3B TTF ranks 1024 and master +
# extras lr 5e-4 at the matched checkpoints (step 2000 and final), for scripts/analysis/token_gap.py.
#   qsub -N tdump1b scripts/myriad/token_dump_1b.sh
#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/token_dump; mkdir -p $O
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
C=checkpoints
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM
for p in x1b_nomom1024_lr15/ckpt_2000.pt:ttf1024_2000 x1b_mx_extras_lr50/ckpt_2000.pt:mx50_2000 \
         x1b_nomom1024_lr15/ckpt.pt:ttf1024_final x1b_mx_extras_lr50/ckpt.pt:mx50_final \
         x1b_nomom1024_lr15/ckpt_6000.pt:ttf1024_6000; do
  python scripts/analysis/token_dump.py $C/\${p%%:*} $O/\${p##*:}.npz --data wiki --n 400
done" > logs/token_dump_1b_$(date +%m%d_%H%M).txt 2>&1
