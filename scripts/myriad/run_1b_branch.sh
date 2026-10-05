#!/bin/bash -l
# Myriad: a branch of a 1.3B run from one of its snapshots (ckpt_STEP.pt, full state), continued to the end of the same
# schedule with changed late-phase settings. Recipe of run_1b_nomom.sh (no momentum, lr 1.5e-3, accumulator rank R);
# TAUANNEAL=0 makes the leak constant (tau steps) instead of scaled with the lr; EXTRA adds / overrides flags.
#   qsub -N b1b_p1 -v N=x1b_nm512_b2k_p1,SRC=x1b_nomom512_lr15,STEP=2000,EXTRA="--ts_anneal 1.0" scripts/myriad/run_1b_branch.sh
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
R=${R:-512}; TAUANNEAL=${TAUANNEAL:-1}; EXTRA=${EXTRA:-}
D=checkpoints/$N; mkdir -p $D
[ -e $D/ckpt.pt ] || ln -s ../$SRC/ckpt_$STEP.pt $D/ckpt.pt
hostname >> $D/train.log; echo "branch of $SRC at step $STEP: TAUANNEAL=$TAUANNEAL EXTRA=$EXTRA" >> $D/train.log
TA=""; [ "$TAUANNEAL" = 1 ] && TA="--ts_tau_anneal"
F="--preset d2048_l24 --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0
   --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30
   --save_secs 1800 --snap_every 0 --lookahead 0 --lowrank $R --ckpt_skip 2 --rc_scale
   --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_nomom --ts_theta 16 --ts_tau 1000 $TA --ts_anneal 0.5 $EXTRA
   --resume --out_dir $D"
F=$(echo $F)
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m bitnet.train $F" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
