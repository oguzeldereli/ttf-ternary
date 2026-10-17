#!/bin/bash -l
# Myriad: 1.3B (d2048_l24, width 2048), final recipe (no momentum, steps x (lr ratio)^0.75, annealed leak), rank 1024,
# Wikipedia, 300M tokens, snapshots every 1000 steps: the 1.3B point of the rank spectra on the same recipe and data as
# 110M (sw512_p075_tauan) and 340M (b340_spec512), for the width exponent of the persistent signal (13 Oct).
#   qsub -N x1b_spec scripts/myriad/run_1b_t75_spec.sh (from ~/Scratch/ternary-LLM; resubmit to resume)
#$ -l h_rt=24:00:00
#$ -l mem=32G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
N=x1b_t75_r1024_spec; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
F="--preset d2048_l24 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30
   --save_secs 1800 --snap_every 1000 --lookahead 0 --lowrank 1024 --ckpt_skip 2 --rc_scale
   --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_nomom --ts_fused --gemm int8mm --ts_rank_s 1024 --ts_theta 16
   --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.75 --out_dir $D"
F=$(echo $F)
[ -f $D/ckpt.pt ] && F="$F --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
