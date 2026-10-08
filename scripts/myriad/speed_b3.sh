#!/bin/bash -l
# Myriad: step time and peak memory of the b3 goldbug arms (arms_b3.txt flags, wiki32k data) on an 80 GB A100:
# TTF ranks 1536 alone, master lr 4e-4 alone, then TTF ranks 1536 + ranks 1024 at the same time on the one GPU.
#$ -N speed_b3
#$ -l h_rt=1:00:00
#$ -l mem=24G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 8
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/speed_b3; mkdir -p $O
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
nvidia-smi --query-gpu=name --format=csv,noheader > $O/summary.txt
flags() { grep "^[01+]* $1 " scripts/goldbug/arms_b3.txt | cut -d' ' -f3- | sed 's#data/fwedu32k_train.bin#data/wiki32k_train.bin#; s#data/fwedu32k_val.bin#data/wiki32k_val.bin#'; }
rate() { grep -E '^step' $O/$1.log | awk '{x=$(NF-3); sub("s","",x); if ($2>=10) {if (!s0) {s0=$2; t0=x}; s1=$2; t1=x}} END {if (s1>s0) printf "%.2f s/step (%d steps)", (t1-t0)/(s1-s0), s1-s0; else print "no steady steps"}'; }
one() { $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; timeout $2 python -m bitnet.train $(flags $1) --out_dir $TMPDIR/$1$3" > $O/$1$3.log 2>&1; }
one gb3_ttf_r1536 600 _alone; echo "TTF r1536 alone: $(rate gb3_ttf_r1536_alone) $(grep -E '^step' $O/gb3_ttf_r1536_alone.log | tail -1 | grep -o 'peakVRAM [0-9.]*GiB') oom=$(grep -c -i 'out of memory' $O/gb3_ttf_r1536_alone.log)" >> $O/summary.txt
one gb3_mx_lr4e4 600 _alone; echo "master alone: $(rate gb3_mx_lr4e4_alone) $(grep -E '^step' $O/gb3_mx_lr4e4_alone.log | tail -1 | grep -o 'peakVRAM [0-9.]*GiB') oom=$(grep -c -i 'out of memory' $O/gb3_mx_lr4e4_alone.log)" >> $O/summary.txt
one gb3_ttf_r1536 900 _pair & one gb3_ttf_r1024 900 _pair & wait
for n in gb3_ttf_r1536_pair gb3_ttf_r1024_pair; do echo "$n (sharing the GPU): $(rate $n) $(grep -E '^step' $O/$n.log | tail -1 | grep -o 'peakVRAM [0-9.]*GiB') oom=$(grep -c -i 'out of memory' $O/$n.log)" >> $O/summary.txt; done
echo "$(date '+%F %T') done" >> $O/summary.txt
