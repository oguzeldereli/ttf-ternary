#!/bin/bash -l
# Myriad: where the 1.3B step time and memory go, on one A100: master, our old rule (dry + spend, rank 512) and --ts
# (ranks 512). Each: 30 steps with the step timer (forward+backward vs update, allocated / peak GiB), then a 5-step
# torch profile (top GPU kernels). Writes logs/profile_1b_<name>.txt.
#   qsub scripts/myriad/profile_1b.sh      (from ~/Scratch/ternary-LLM)
#$ -N tern_prof1b
#$ -l h_rt=2:00:00
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
C="--preset d2048_l24 --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 100000 --eval_iters 2 --save_secs 1000000"
K="--mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense
   --track_flips --tail_fp32 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp"
OLD="--lr_gate --lr_vnorm 0.99 --lr_beta 1 --dry_vec 0.0303 --spend 3"
TS="--ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > logs/profile_1b_gpu.txt; hostname >> logs/profile_1b_gpu.txt
for cfg in "master:--mode master --master_dtype fp32" "old:$K $OLD" "ts:$K $TS"; do
  n=${cfg%%:*}; F=$(echo $C ${cfg#*:}); O=$TMPDIR/prof_$n
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; TERN_TIME=1 python -m ttf.train $F --stop_after 31 --out_dir $O" \
    > logs/profile_1b_$n.txt 2>&1
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $F --profile 5 --out_dir ${O}_p" \
    >> logs/profile_1b_$n.txt 2>&1
done
