#!/bin/bash -l
# Myriad: zero-shot only (six tasks, scripts/eval/zeroshot.py) of run dirs under checkpoints/ (fixed-window eval done).
#   qsub -N zs -v RUNS="gb3_ttf_r1024 gb3_ttf_r1536 gb3_mx_lr4e4" scripts/myriad/zeroshot_runs.sh
#$ -l h_rt=4:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
D=$(for r in $RUNS; do echo checkpoints/$r; done | tr '\n' ' ')
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/eval/zeroshot.py $D" > logs/zeroshot_$(date +%m%d_%H%M).txt 2>&1
