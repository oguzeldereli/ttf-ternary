#!/bin/bash -l
# Myriad: one 110M arm of scripts/lab/geom_arms.sh (training + analyses) in the apptainer image.
#   qsub -N g_gptqc -v ARM=gptqc scripts/myriad/geom_arm.sh
#$ -l h_rt=6:00:00
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
apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c \
  "export OMP_NUM_THREADS=4; S=$S REPO=$HOME/Scratch/ternary-LLM bash scripts/lab/geom_arms.sh $ARM"
