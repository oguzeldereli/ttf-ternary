#!/bin/bash -l
# Myriad test job: 1 A100, 30 min. Checks the GPU, the env and the trainer speed on the 110M and 1.3B (width 2048) models.
#   qsub scripts/myriad/test_job.sh      (from ~/Scratch/ternary-LLM)
#$ -N tern_test
#$ -l h_rt=0:30:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=10G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
# Myriad is RHEL 7 (glibc 2.17); torch wheels need glibc >= 2.28, so the Scratch env runs inside an Ubuntu 24.04 container
# (ubuntu24.sif) with the host GPU driver (--nv). RUN = run a command in that container with the env on PATH.
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$RUN "source $S/env.sh; python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
C="--mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 100000 --eval_iters 2 --lookahead 0 --ckpt_skip 2
   --rc_scale --lr_gate --lr_vnorm 0.99 --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3"
C=$(echo $C)                         # one line: it is passed inside a bash -c string
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train --preset small $C --lowrank 512 --stop_after 40 --out_dir $TMPDIR/t110" 2>&1 | grep -E "params|step +(20|30|40) |Error|Traceback"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train --preset d2048_l24 $C --lowrank 512 --stop_after 30 --out_dir $TMPDIR/t1b" 2>&1 | grep -E "params|step +(10|20|30) |Error|Traceback|memory"
