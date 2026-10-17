#!/bin/bash -l
# Myriad: 26.5B (b27) TRAINING demo inside a 24 GB card's budget: rank 1024, 70% of the state on the GPU and the rest on
# the host with asynchronous copies (all of it on the GPU, in place, ran out of memory under the cap at step 1; the 70%
# split peaked at 17.3 GiB in the probe; no --track_flips, whose copy of the packed trits is 4.3 GiB at this size), PyTorch capped at 22.4 GiB (a 3090 Ti's 23.54 GiB less its CUDA context), --ckpt_block; 1 x 2048 tokens per step,
# STEPS steps (cosine over them, warmup 50), validation every 100 steps. Resubmit to resume (checkpoint every 30 min).
#   qsub -N b27_train scripts/myriad/run_b27_train.sh        (STEPS=500 default)
#$ -l h_rt=12:00:00
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
STEPS=${STEPS:-500}; D=checkpoints/b27_train; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
A="--preset b27 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048
  --batch_size 1 --grad_accum 1 --steps $STEPS --warmup 50 --int8 --dw_mode dense --lr 7.5e-4 --min_lr 7.5e-5
  --eval_interval 100 --eval_iters 4 --save_secs 1800 --lookahead 0 --ckpt_skip 0 --loss_chunk 256 --rc_scale
  --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lowrank 1024 --ts_rank_s 1024 --ts --ts_nomom --ts_theta 16 --ts_tau 1000
  --ts_anneal 0.75 --ts_tau_anneal --ts_fused --gemm int8mm --ts_store_a fp12s --lean_tail --ts_offload_async --ts_offload_keep 0.7
  --ckpt_block --mem_cap_gib 22.4 --mem_report --out_dir $D"
A=$(echo $A)
[ -f $D/ckpt.pt ] && A="$A --resume"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 >> $D/smi.csv &
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $A" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
