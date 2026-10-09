#!/bin/bash -l
# Myriad: gradient SNR / drift / rank capture per layer (scripts/analysis/grad_snr.py) at two checkpoints ~1000 steps
# apart, then the persistence of each layer's mean gradient between them. 1.3B TTF ranks 1024 (wiki32k, its training
# data) at 4000 / 5000 vs 2.9B TTF ranks 1024 (goldbug snapshots, FineWeb-Edu) at 4146 / 5001: is the 2.9B deep-layer
# deficit a weaker or less persistent gradient there? 80 GB A100.
#   qsub -N gsnr1b -v N=192,TAG=1b,D=data/wiki32k_train.bin,CS="checkpoints/x1b_nomom1024_lr15/ckpt_4000.pt checkpoints/x1b_nomom1024_lr15/ckpt_5000.pt" scripts/myriad/grad_snr.sh
#   qsub -N gsnr3b -v N=192,TAG=3b,D=data/fwedu32k_train_head.bin,CS="checkpoints/gb3_ttf_r1024/snap_4146.pt checkpoints/gb3_ttf_r1024/snap_5001.pt" scripts/myriad/grad_snr.sh
#$ -l h_rt=4:00:00
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
MUS=""
for C in $CS; do
  st=$(basename $C .pt); mu=$O/${TAG}_${st}_mu.pt; MUS="$MUS $mu"
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/analysis/grad_snr.py $C --data $D --n $N --save_mu $mu" \
    > $O/${TAG}_${st}_n${N}.txt 2>&1
done
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/analysis/grad_snr.py --persist $MUS" > $O/${TAG}_persist.txt 2>&1
