#!/bin/bash -l
# Myriad: gradient SNR / drift / rank capture per layer (scripts/analysis/grad_snr.py) at 1.3B TTF ranks 1024 step 4000
# (wiki32k, its training data) and 2.9B TTF ranks 1024 step 4146 (goldbug snapshot, FineWeb-Edu): is the 2.9B deep-layer
# deficit a weaker / less consistent gradient there? 80 GB A100 (the 2.9B accumulators are ~45 GB).
#   qsub -N gsnr_t -v N=12,WHICH=1b scripts/myriad/grad_snr.sh      (test)
#   qsub -N gsnr -v N=96,WHICH="1b 3b" scripts/myriad/grad_snr.sh
#$ -l h_rt=3:00:00
#$ -l mem=24G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/grad_snr; mkdir -p $O
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
for w in $WHICH; do
  case $w in
    1b) C=checkpoints/x1b_nomom1024_lr15/ckpt_4000.pt; D=data/wiki32k_train.bin ;;
    3b) C=checkpoints/gb3_ttf_r1024/snap_4146.pt; D=data/fwedu32k_train_head.bin ;;
  esac
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/analysis/grad_snr.py $C --data $D --n $N" \
    > $O/${w}_n${N}_$(date +%m%d_%H%M).txt 2>&1
done
