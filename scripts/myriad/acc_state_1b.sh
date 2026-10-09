#!/bin/bash -l
# Myriad (CPU): scripts/analysis/acc_state.py on the 1.3B TTF ranks 1024 snapshots, for comparison with goldbug's b3.
#   qsub -N acc1b scripts/myriad/acc_state_1b.sh
#$ -l h_rt=2:00:00
#$ -l mem=6G
#$ -pe smp 8
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUN="apptainer exec -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; for s in 2000 4000; do
  CUDA_VISIBLE_DEVICES= python scripts/analysis/acc_state.py checkpoints/x1b_nomom1024_lr15/ckpt_\$s.pt; done" \
  > logs/acc_state_1b_$(date +%m%d_%H%M).txt 2>&1
