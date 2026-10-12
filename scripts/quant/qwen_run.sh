#!/usr/bin/env bash
# Lab PC: TTF recovery of a converted Qwen3.5 / 3.8 model (bitnet/qwen.py), one job after another:
#   [SAVE=1] qwen_run.sh MODEL STEPS "NAME|flags" ["NAME|flags" ...]
# MODEL is a directory under /tmp/$USER/tern/hf/models; data: Wikipedia re-tokenized with its tokenizer
# (scripts/quant/retok.py -> data/wikiqw_{train,val}.bin, uint32). TTF recipe as for TinyLlama (rank 256, p 0.75, annealed
# leak, no momentum); flags add the conversion method, theta, --ts_start.
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
M=$S/hf/models/$1; STEPS=$2; shift 2
mkdir -p logs; NFS="--no_final_save"; [ -n "${SAVE:-}" ] && NFS=""
TTF="--mode kernel --int8 --dw_mode dense --track_flips --ts --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom
     --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_state_inplace --ts_vdiag"
for job in "$@"; do
  N=${job%%|*}; F=${job#*|}
  mkdir -p checkpoints/$N
  while pgrep -f "[p]ython -m bitnet.train" >/dev/null; do sleep 60; done
  python -m bitnet.train --init_hf $M --data data/wikiqw_train.bin --val data/wikiqw_val.bin --data_u32 --seq_len 2048 \
    --batch_size 8 --grad_accum 2 --steps $STEPS --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 \
    --eval_interval 50 --eval_iters 10 --save_secs 99999 --snap_every 0 $NFS --lookahead 0 --ckpt_skip 0 \
    --rc_scale --lowrank_mag add:16 --mag_wd 0.1 $(echo $TTF) $F --out_dir checkpoints/$N \
    >> checkpoints/$N/train.log 2>&1
  echo "$(date '+%F %T') DONE $N $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$N/train.log | tail -1)" >> logs/qwen.log
done
