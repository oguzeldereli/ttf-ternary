#!/bin/bash -l
# Myriad: rank spectra of TTF's evidence at 1.3B and 2.9B (scripts/analysis/rank_spectrum.py), against the 110M / 340M
# ones from the lab. Uses the mean gradients grad_snr.py saved on 10 Oct (n 192, steps 4000 / 5000 and 4146 / 5001), plus
# three single training batches per size (n 12, micro 4, another seed, every 2nd layer) for the per-step keep share.
#   qsub -N rspec scripts/myriad/rank_spec.sh
#$ -l h_rt=2:00:00
#$ -l mem=24G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/grad_snr
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
P="source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python"
C1=checkpoints/x1b_nomom1024_lr15/ckpt_4000.pt; C3=checkpoints/gb3_ttf_r1024/snap_4146.pt
$RUN "$P scripts/analysis/grad_snr.py $C1 --data data/wiki32k_train.bin --n 12 --micro 4 --seed 99 --every 2 --no_svd \
  --save_mu $O/1b_step4000_mu.pt --save_thirds" > $O/1b_step4000.txt 2>&1
$RUN "$P scripts/analysis/grad_snr.py $C3 --data data/fwedu32k_train_head.bin --n 12 --micro 4 --seed 99 --every 2 \
  --no_svd --save_mu $O/3b_step4146_mu.pt --save_thirds" > $O/3b_step4146.txt 2>&1
$RUN "$P scripts/analysis/rank_spectrum.py $O/spec_1b3b.json \
  1.3B=2048:$O/1b_ckpt_4000_mu.pt:$O/1b_ckpt_5000_mu.pt:$C1 2.9B=3072:$O/3b_snap_4146_mu.pt:$O/3b_snap_5001_mu.pt:$C3 \
  1.3Bkeep=2048:$O/1b_ckpt_4000_mu.pt:keep:$O/1b_step4000_mu.pt 2.9Bkeep=3072:$O/3b_snap_4146_mu.pt:keep:$O/3b_step4146_mu.pt" \
  > $O/spec_1b3b.txt 2>&1
rm -f $O/1b_step4000_mu.pt $O/3b_step4146_mu.pt
