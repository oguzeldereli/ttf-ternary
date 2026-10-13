#!/bin/bash -l
# Myriad: rank spectra of TTF's evidence at 340M on the final recipe (b340_spec512, Wikipedia, rank 512), the third width
# next to 110M (lab) and 1.3B (rank_spec.sh): mean gradients at steps 4000 / 5000 (n 192), one training batch for the
# per-step keep share (n 12, micro 4, another seed), then scripts/analysis/rank_spectrum.py on both.
#   qsub -N rspec340 scripts/myriad/rank_spec340.sh
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
P="source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python"
D=data/wiki32k_train.bin; C=checkpoints/b340_spec512
for st in 4000 5000; do
  $RUN "$P scripts/analysis/grad_snr.py $C/ckpt_$st.pt --data $D --n 192 --save_mu $O/340_ckpt_${st}_mu.pt" \
    > $O/340_ckpt_${st}_n192.txt 2>&1
done
$RUN "$P scripts/analysis/grad_snr.py $C/ckpt_4000.pt --data $D --n 12 --micro 4 --seed 99 --no_svd \
  --save_mu $O/340_step4000_mu.pt --save_thirds" > $O/340_step4000.txt 2>&1
$RUN "$P scripts/analysis/rank_spectrum.py $O/spec_340.json \
  340M=1024:$O/340_ckpt_4000_mu.pt:$O/340_ckpt_5000_mu.pt:$C/ckpt_4000.pt \
  340Mkeep=1024:$O/340_ckpt_4000_mu.pt:keep:$O/340_step4000_mu.pt" > $O/spec_340.txt 2>&1
rm -f $O/340_step4000_mu.pt
