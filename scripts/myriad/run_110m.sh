#!/bin/bash -l
# Myriad: one 110M from-scratch run with exactly the lab sweep's command (scripts/lab/scratch_run.sh + the sweep base:
# ranks 192, no momentum, fused update, int8 forward) plus the arm's flags F. Resubmit the same way to resume.
#   qsub -N s110_sp050 -v N=sw192_spend050_myr,F="--ts_spend 0.5" scripts/myriad/run_110m.sh
#   STEPS=60 for a test job
#$ -l h_rt=12:00:00
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
F=${F:-}; STEPS=${STEPS:-9155}; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
A="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps $STEPS --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30
   --save_secs 1800 --snap_every 1000 --lookahead 0 --lowrank 256 --ckpt_skip 2
   --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.5 --lowrank 192
   --ts_rank_s 192 --ts_nomom --ts_fused --gemm int8mm $F --out_dir $D"
A=$(echo $A)
[ -f $D/ckpt.pt ] && A="$A --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $A" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
