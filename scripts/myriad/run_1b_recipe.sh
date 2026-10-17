#!/bin/bash -l
# Myriad: the recipe (sharp base + dry friction 1/33 + spend 3, rank 512) on the 1.3B model (d2048_l24, width 2048),
# 300M tokens of wiki32k, 1 A100. ~8.3 s/step -> ~21 h. Checkpoints every 30 min (resubmit the same script to resume).
#   qsub scripts/myriad/run_1b_recipe.sh      (from ~/Scratch/ternary-LLM)
#$ -N tern_1b_recipe
#$ -l h_rt=30:00:00
#$ -l mem=8G
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
N=x1b_dryspend_r512_s0; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
F="--preset d2048_l24 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30
   --save_secs 1800 --snap_every 1000 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lr_gate --lr_vnorm 0.99
   --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3 --out_dir $D"
F=$(echo $F)
[ -f $D/ckpt.pt ] && F="$F --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
