#!/bin/bash -l
# Myriad, ~10 min: one configuration at 1.3B on one A100, 25 steps: the step timer (forward+backward vs update,
# allocated / peak GiB) every step, a torch profile of steps 20-24 (top GPU kernels + chrome trace), and nvidia-smi
# sampled every second (utilization, memory, power, SM clock). Output: logs/prof1b_<KIND>/.
#   qsub -N prof_KIND scripts/myriad/profile_1b_one.sh KIND [TAG extra-flags...]   KIND = master | old | ts
#$ -l h_rt=0:15:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; KIND=$1; TAG=${2:-}; shift; [ $# -gt 0 ] && shift; EXTRA="$*"
O=logs/prof1b_$KIND${TAG:+_$TAG}; mkdir -p $O
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
C="--preset d2048_l24 --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 100000 --eval_iters 2 --save_secs 1000000"
K="--mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense
   --track_flips --tail_fp32 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp"
case $KIND in
  master) F="$C --mode master --master_dtype fp32" ;;
  old) F="$C $K --lr_gate --lr_vnorm 0.99 --lr_beta 1 --dry_vec 0.0303 --spend 3" ;;
  ts) F="$C $K --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.5" ;;
esac
F=$(echo $F $EXTRA)
{ hostname; nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv,noheader; nproc; } > $O/node.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > $O/nvsmi.csv &
SMI=$!
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; TERN_TIME=1 python -m ttf.train $F --profile 5 --out_dir $O/run" \
  > $O/train.txt 2>&1
kill $SMI
echo "$(date '+%F %T') done" >> $O/node.txt
