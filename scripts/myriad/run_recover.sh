#!/bin/bash -l
# Myriad: a recovery run from a pretrained checkpoint (--init_hf), the lab recipe of scripts/quant/recover_run.sh;
# F replaces the TTF block when it contains --mode master. Model under ~/Scratch/hf/models/MODEL (login-node download).
#   qsub -N tl_mx -v N=tl_master,MODEL=TinyLlama-1.1B-intermediate-step-1431k-3T,STEPS=1000,F="--mode master --lr 5e-4 --min_lr 5e-5" scripts/myriad/run_recover.sh
#$ -l h_rt=12:00:00
#$ -l mem=24G
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
F=${F:-}; STEPS=${STEPS:-1000}; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
TTF="--mode kernel --int8 --dw_mode dense --track_flips --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal
     --ts_nomom --ts_fused --gemm int8mm --lowrank 1024 --ts_rank_s 1024"
case "$F" in *"--mode master"*) B="--master_dtype fp32";; *) B=$TTF;; esac
A="--init_hf $HOME/Scratch/hf/models/$MODEL --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048
   --batch_size 16 --grad_accum 1 --steps $STEPS --warmup 30 --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4
   --eval_interval 50 --eval_iters 10 --save_secs 99999 --snap_every 0 --no_final_save --lookahead 0 --ckpt_skip 2
   --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp $B $F --out_dir $D"
A=$(echo $A)
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $A" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
