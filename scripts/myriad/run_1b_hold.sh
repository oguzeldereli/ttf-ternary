#!/bin/bash -l
# Myriad: continue a finished 1.3B run past its 300M-token cosine schedule with the lr held at its final value (min_lr;
# for TTF the step scale and leak follow the same held lr ratio, 0.1) for another 9155 steps (300M tokens; wiki32k has
# ~398M training tokens, so windows repeat): does TTF keep closing the gap when master's lr is small?
#   qsub -N h1b_ttf -v ARM=ttf scripts/myriad/run_1b_hold.sh     (x1b_ts512_la1000an05 -> x1b_ts512_hold)
#   qsub -N h1b_mx  -v ARM=mx  scripts/myriad/run_1b_hold.sh     (x1b_mx_extras_lr75  -> x1b_mx_extras_lr75_hold)
#$ -l h_rt=40:00:00
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
C="--data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155
   --warmup 305 --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 250 --eval_iters 30 --save_secs 1800 --snap_every 0
   --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --extend_to 18310 --resume"
if [ "$ARM" = ttf ]; then
  SRC=x1b_ts512_la1000an05; N=x1b_ts512_hold
  F="--preset d2048_l24 --mode kernel $C --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
     --int8 --dw_mode dense --track_flips --tail_fp32 --lookahead 0 --lowrank 512 --ckpt_skip 2
     --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.5"
else
  SRC=x1b_mx_extras_lr75; N=x1b_mx_extras_lr75_hold
  F="--preset d2048_l24 --mode master --master_dtype fp32 $C"
fi
D=checkpoints/$N; mkdir -p $D
[ -e $D/ckpt.pt ] || ln -s ../$SRC/ckpt.pt $D/ckpt.pt
hostname >> $D/train.log; echo "continuation of $SRC at its final lr to step 18310" >> $D/train.log
F=$(echo $F --out_dir $D)
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m ttf.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
