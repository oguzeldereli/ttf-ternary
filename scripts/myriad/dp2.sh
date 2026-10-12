#!/bin/bash -l
# Myriad: data-parallel proof of concept on 2 GPUs (torchrun; each layer's gradient all-reduced before TTF's update,
# the float tail's gradients before its step). 110M final recipe, 300 steps, 8 x 2048 per GPU (= the usual batch of 16),
# against the same run on 1 GPU with batch 16; the ranks print packed-trit checksums every 100 steps.
#   qsub -N dp2 scripts/myriad/dp2.sh
#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l gpu=2
#$ -pe smp 8
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
B="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --steps 300
   --warmup 30 --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100 --eval_iters 10 --save_secs 99999 --snap_every 0
   --no_final_save --lookahead 0 --ckpt_skip 2 --int8 --dw_mode dense --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp
   --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 512 --ts_rank_s 512"
mkdir -p checkpoints/dp2_2gpu checkpoints/dp2_1gpu logs/dp2
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m torch.distributed.run --nproc_per_node 2 -m bitnet.train $B --batch_size 8 --out_dir checkpoints/dp2_2gpu" > logs/dp2/dp2_2gpu.txt 2>&1
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m bitnet.train $B --batch_size 16 --out_dir checkpoints/dp2_1gpu" > logs/dp2/dp2_1gpu.txt 2>&1
grep -hE "data parallel|val loss|^step +(100|200|290) " logs/dp2/dp2_2gpu.txt | head -20 > logs/dp2/summary.txt
grep -hE "val loss|^step +(100|200|290) " logs/dp2/dp2_1gpu.txt >> logs/dp2/summary.txt
