#!/usr/bin/env bash
# Momentum fixes, branched at 131M tokens (ckpt_4000 of the momentum + look-ahead replay), 600 steps each,
# same batches: baseline / --lr_spend 2 / --lr_refresh 16 / --lr_gate. The no-look-ahead run is paused
# (graceful save) and resumed afterwards. A tiny smoke test of every option runs first.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
IN=$HOME/ternary-sync/inbox/ckpt_4000.pt
[ -f $S/ckpt_4000.pt ] || { cp $IN $S/ckpt_4000.pt && rm -f $IN; }
SMOKE="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 256
  --batch_size 2 --grad_accum 1 --steps 100 --warmup 5 --rate 0.02 --g_ref 3.0 --int8 --dw_mode dense
  --tail_fp32 --eval_iters 1 --eval_interval 100000 --save_secs 100000 --lowrank 16 --lookahead 2
  --lookahead_xbatch --stop_after 12"
for f in "" "--lr_spend 2" "--lr_refresh 4 --lr_refresh_every 3" "--lr_gate"; do
  python -m ttf.train $SMOKE $f --out_dir $S/smoke_fix > $S/smoke_fix.log 2>&1 \
    || { echo "SMOKE FAILED for '$f'"; tail -20 $S/smoke_fix.log | tee checkpoints/fixes4000_smoke.err; exit 1; }
  rm -rf $S/smoke_fix
done
echo "$(date '+%F %T') smoke ok"
P=$(pgrep -u "$USER" -f "out_dir checkpoints/lowrank256_nola_full")
[ -n "$P" ] && kill -TERM $P && while kill -0 $P 2>/dev/null; do sleep 5; done && echo "no-look-ahead run paused"
BR="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048
  --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02
  --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4
  --stop_after 4601 --eval_iters 30 --eval_interval 200 --save_secs 100000 --lookahead 2 --lookahead_xbatch
  --lowrank 256 --ckpt_skip 2 --resume"
run() {
  local n=$1; shift
  rm -rf checkpoints/$n; mkdir -p checkpoints/$n; ln -s $S/ckpt_4000.pt checkpoints/$n/ckpt.pt
  python -m ttf.train $BR "$@" --out_dir checkpoints/$n > checkpoints/$n/train.log 2>&1
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
run fix4000_base & run fix4000_spend2 --lr_spend 2 & wait
run fix4000_refresh16 --lr_refresh 16 --lr_refresh_every 10 & run fix4000_gate --lr_gate & wait
# resume the no-look-ahead run
python -m ttf.train --preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin \
  --seq_len 2048 --batch_size 16 --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine \
  --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense --track_flips --tail_fp32 \
  --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 \
  --lookahead 0 --lowrank 256 --ckpt_skip 2 --resume --out_dir checkpoints/lowrank256_nola_full \
  >> checkpoints/lowrank256_nola_full/train.log 2>&1
echo "$(date '+%F %T') DONE lowrank256_nola_full $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/lowrank256_nola_full/train.log)"
