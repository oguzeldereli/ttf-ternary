#!/bin/bash -l
# Myriad: one from-scratch run with the full trainer command in F (everything but --out_dir / --resume), then the
# fixed-window evaluation. Resubmit the same way to resume after the wall-clock limit.
#   qsub -N te_E2 -v N=sw512_t75_ternemb,F="--preset small --mode kernel ..." scripts/myriad/run_arm.sh
#$ -l h_rt=16:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
A="$(echo $F) --out_dir $D"
[ -f $D/ckpt.pt ] && A="$A --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $A" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
if grep -q "FINAL val loss" $D/train.log && [ ! -f $D/final_eval.json ]; then
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/eval/final_eval.py $D" > $D/final_eval.log 2>&1
fi
