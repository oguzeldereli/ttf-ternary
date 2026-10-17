#!/bin/bash -l
# Myriad: shadow comparison at 340M / 1.3B: a TTF run (ranks 512) carrying shadow momentum / accumulator states at ranks
# 64 / 128 / 512 and full rank on the same gradients; compared with full rank every 250 steps (spectrum, cosine, sign
# agreement, which weights would move); q / down matrices of two blocks dumped at steps 500 and 2000.
#   qsub -N shd340 -l h_rt=8:00:00 scripts/myriad/shadows.sh d1024_l24 3001 1.5e-3 x340_shadows
#   qsub -N shd1b -l h_rt=14:00:00 scripts/myriad/shadows.sh d2048_l24 2001 7.5e-4 x1b_shadows
#$ -l mem=12G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; P=$1; STOP=$2; LR=$3; N=$4; D=checkpoints/$N; mkdir -p $D
MID=$([ $P = d2048_l24 ] && echo "84,90" || echo "84,90")
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
F="--preset $P --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr $LR --min_lr $(awk "BEGIN{print $LR/10}") --eval_interval 250
   --eval_iters 30 --save_secs 1000000 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1
   --qk_temp --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --ts_fused --gemm cublas
   --ts_shadows 64,128,512,0 --ts_diag 250 --ts_dump_steps 500,2000 --ts_dump_layers 0,6,$MID --stop_after $STOP --out_dir $D"
F=$(echo $F)
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end" >> $D/train.log
