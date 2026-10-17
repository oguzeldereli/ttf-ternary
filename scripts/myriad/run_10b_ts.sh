#!/bin/bash -l
# Myriad: the ~10B preset (b10: width 5120, 36 layers, MLP 13824; 9.9B ternary weights) trained with TTF at the real batch
# (16 x 2048 tokens per step, one pass: the fused update cannot accumulate micro-batches without a full gradient per
# layer), lean: update inside the backward, int8 direction / 12-bit accumulator stored for real, lean float tail
# (bf16 / fp16 values, int8 first moment), every layer checkpointed, loss in chunks, no flip tracking (a mask per weight).
# State on the GPU (an A100 has room; host offload is only for 16 GB cards). Needs an 80 GB A100 (node types U / V): the
# 20-step test on a 40 GB A100 peaked at 37.8 GiB in step 0 and ran out of memory in step 1 (6.5 GiB persistent; the rest
# is activations of 32k tokens); ~39 s per step. Ranks 512, tau 1000, theta 16, leak and steps
# x (lr ratio)^0.5 (the 1.3B recipe), float tail at lr 5e-4.
#   qsub -v MODE=test -l h_rt=1:00:00 scripts/myriad/run_10b_ts.sh    (20 steps: memory and step time)
#   qsub scripts/myriad/run_10b_ts.sh                                  (3000 steps, 98M tokens; resubmit to resume)
#$ -N tern_10b_ts
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
MODE=${MODE:-full}
N=x10b_ts512; [ $MODE = test ] && N=x10b_ts512_test
D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
F="--preset b10 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 3000 --warmup 100 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --gemm cublas --lr 5e-4 --min_lr 5e-5 --eval_interval 250 --eval_iters 10 --save_secs 3600
   --snap_every 0 --lookahead 0 --ckpt_skip 0 --loss_chunk 256 --lean_tail --mem_report --rc_scale --lowrank_mag add:16
   --mag_wd 0.1 --qk_temp --ts --ts_fused --ts_store_m int8 --ts_store_a fp12s --lowrank 512 --ts_rank_s 512 --ts_theta 16
   --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.5 --out_dir $D"
[ $MODE = test ] && F="$F --stop_after 21 --eval_interval 1000000 --eval_iters 1 --save_secs 100000000"
F=$(echo $F)
[ -f $D/ckpt.pt ] && F="$F --resume"
T=""; [ $MODE = test ] && T="TERN_TIME=1 "
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; ${T}python -m ttf.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
