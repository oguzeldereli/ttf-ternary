#!/bin/bash -l
# Myriad: step time of the goldbug arms (scripts/goldbug/arms_b16.txt flags on wiki32k data) on an 80 GB A100, ~330 s each.
#$ -N speed_arm
#$ -l h_rt=0:30:00
#$ -l mem=12G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/speed_arm; mkdir -p $O
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
nvidia-smi --query-gpu=name --format=csv,noheader > $O/summary.txt
while read -r g n flags; do
  case "$g" in \#*|"") continue ;; esac
  f=$(echo $flags | sed 's#data/fwedu32k_train.bin#data/wiki32k_train.bin#; s#data/fwedu32k_val.bin#data/wiki32k_val.bin#')
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; timeout 330 python -m bitnet.train $f --out_dir $TMPDIR/$n" > $O/$n.log 2>&1
  r=$(grep -E '^step' $O/$n.log | awk '{x=$(NF-3); sub("s","",x); if ($2>=10) {if (!s0) {s0=$2; t0=x}; s1=$2; t1=x}} END {if (s1>s0) printf "%.2f s/step (%d steps)", (t1-t0)/(s1-s0), s1-s0; else print "no steady steps"}')
  echo "$n $r $(grep -E '^step' $O/$n.log | tail -1 | grep -o 'peakVRAM [0-9.]*GiB')" >> $O/summary.txt
done < scripts/goldbug/arms_b16.txt
echo done >> $O/summary.txt
