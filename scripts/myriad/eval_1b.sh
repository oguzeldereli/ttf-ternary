#!/bin/bash -l
# Myriad: final evaluation (fixed windows: Wikipedia 974 + FineWeb-Edu 2000) and zero-shot (six tasks) of the 1.3B runs
#   qsub -N eval1b scripts/myriad/eval_1b.sh [run dirs under checkpoints/ ...]
#$ -l h_rt=3:00:00
#$ -l mem=10G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUNS=${*:-x1b_ts512_la1000an05 x1b_master_lr75 x1b_dryspend_r512_lr75}
D=$(for r in $RUNS; do echo checkpoints/$r; done | tr '\n' ' ')
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/eval/final_eval.py $D; python scripts/eval/zeroshot.py $D" \
  > logs/eval_1b_$(date +%m%d_%H%M).txt 2>&1
