#!/bin/bash -l
# Myriad: 26.5B rank 1024, all state on the GPU, PyTorch capped at 22.4 GiB (a 3090 Ti's 23.54 GiB less its 1.09 GiB
# CUDA context), state written in place, --ckpt_block: does 27B train on a 24 GB card with no host offload?
#   qsub scripts/myriad/run_b27_capped.sh
#$ -N b27_cap
#$ -l h_rt=1:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; WHERE=gpu INPLACE=1 CAP=22.4 bash scripts/lab/b27_memprobe.sh" \
  >> logs/b27_capped.out 2>&1
