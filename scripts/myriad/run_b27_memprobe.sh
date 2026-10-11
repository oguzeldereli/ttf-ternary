#!/bin/bash -l
# Myriad: 26.5B memory probe (scripts/lab/b27_memprobe.sh) on one A100: state on the GPU, uncapped, so the whole
# peak is measured (what a 24 GB card would need); then PyTorch capped at 22.4 GiB (a 3090 Ti: 23.54 GiB less the 1.09 GiB
# its CUDA context took) with the state written in place, all of it and 70% of it on the GPU.
#   qsub scripts/myriad/run_b27_memprobe.sh
#$ -N b27_mem
#$ -l h_rt=2:00:00
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
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
      WHERE=gpu bash scripts/lab/b27_memprobe.sh; WHERE=gpu INPLACE=1 CAP=22.4 bash scripts/lab/b27_memprobe.sh;
      WHERE=split KEEP=0.7 INPLACE=1 CAP=22.4 bash scripts/lab/b27_memprobe.sh" \
  >> logs/b27_memprobe.out 2>&1
