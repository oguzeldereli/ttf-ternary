#!/usr/bin/env bash
# Lab PC: recovery runs from a pretrained checkpoint (--init_hf), one after another, the 110M recipe otherwise:
#   [AFTER=PID] recover_run.sh MODEL STEPS "NAME|flags" ["NAME|flags" ...]
# MODEL is a directory under /tmp/$USER/tern/hf/models (scripts/quant/lab_hf_check.sh downloads them). Kernel arms get
# the final TTF recipe (rank 256, p 0.75, annealed leak, no momentum); a flags string with --mode master replaces it.
# SAVE=1 keeps the final checkpoint (default: none; a TinyLlama checkpoint is 5 GB).
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
M=$S/hf/models/$1; STEPS=$2; shift 2
mkdir -p logs; NFS="--no_final_save"; [ -n "${SAVE:-}" ] && NFS=""
[ -n "${AFTER:-}" ] && while kill -0 $AFTER 2>/dev/null; do sleep 60; done   # AFTER=PID: start when that chain ends
TTF="--mode kernel --int8 --dw_mode dense --track_flips --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal
     --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256"
for job in "$@"; do
  N=${job%%|*}; F=${job#*|}
  case "$F" in *"--mode master"*) B="--master_dtype fp32";; *) B=$TTF;; esac
  mkdir -p checkpoints/$N
  while pgrep -f "[p]ython -m ttf.train" >/dev/null; do sleep 60; done
  python -m ttf.train --init_hf $M --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 \
    --batch_size 16 --grad_accum 1 --steps $STEPS --warmup 30 --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 \
    --eval_interval 50 --eval_iters 10 --save_secs 99999 --snap_every 0 $NFS --lookahead 0 --ckpt_skip 2 \
    --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp $(echo $B) $F --out_dir checkpoints/$N \
    >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> logs/recover.log
done
