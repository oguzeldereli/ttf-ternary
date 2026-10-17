#!/usr/bin/env bash
# Spatial error feedback, unattended:
#   1. short alpha probe (600 steps, ~20M tokens) at alpha 0 / 0.3 / 1.0, same
#      schedule horizon as the full run so the probes see the same flip rate
#   2. full 300M-token run with the better nonzero alpha, directly comparable to
#      armA_cosine (98.56): identical config plus --err_feedback
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
LOG=checkpoints/ef_overnight.log
say() { echo "[$(date +%m-%d\ %H:%M)] $*" | tee -a $LOG; }
final_ppl() { grep -oE "FINAL val loss [0-9.]+ \| ppl [0-9.]+" "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }

BASE="--preset small --mode kernel --data data/wiki32k_train.bin \
--val data/wiki32k_val.bin --lr 3e-4 --min_lr 3e-5 --eval_iters 30 \
--int8 --dw_mode dense --seq_len 2048 --batch_size 16 --grad_accum 1 \
--steps 9155 --warmup 305 --rate_schedule cosine --rate 0.02"

# ---- 1. alpha probe -----------------------------------------------------------
for A in 0 0.1 0.01 0.03; do
  OUT=checkpoints/ef_probe_a$A
  if [ -f $OUT/done ]; then say "probe alpha=$A already done ($(final_ppl $OUT.log))"; continue; fi
  rm -rf $OUT; mkdir -p $OUT
  say "START probe alpha=$A"
  python -m ttf.train $BASE --err_feedback --ef_alpha $A --stop_after 600 \
      --eval_interval 100000 --save_secs 100000 --track_flips --out_dir $OUT > $OUT.log 2>&1 \
    && touch $OUT/done
  SPS=$(awk '/^step/{s=$2; t=$(NF-3)} END{sub("s","",t); if (s>0) printf "%.2f", t/s}' $OUT.log)
  say "probe alpha=$A -> ppl $(final_ppl $OUT.log) | ${SPS} s/step"
done

ALPHA=$(python -c "
import re
def f(x):
    try: return float(x)
    except Exception: return 1e9
def ppl(a):
    try:
        m = re.findall(r'FINAL val loss [0-9.]+ \\| ppl ([0-9.]+)', open(f'checkpoints/ef_probe_a{a}.log').read())
        return float(m[-1]) if m else 1e9
    except Exception: return 1e9
c = {a: ppl(a) for a in ('0.01', '0.03', '0.1', '0.3')}
print(min(c, key=c.get))")
say "probe summary: $(for a in 0 0.01 0.03 0.1 0.3; do printf 'a%s=%s ' $a $(final_ppl checkpoints/ef_probe_a$a.log); done)-> full run uses alpha=$ALPHA"

# ---- 2. full run --------------------------------------------------------------
OUT=checkpoints/armA_cos_ef
mkdir -p $OUT
say "START armA_cos_ef (alpha=$ALPHA, 300M tokens)"
# always --resume (no-op without a checkpoint), so rerunning this script after a
# crash or reboot continues the run instead of restarting it; retry up to 3 times
for TRY in 1 2 3 4; do
  [ $TRY -gt 1 ] && say "RETRY $TRY armA_cos_ef (resuming from last checkpoint)"
  python -m ttf.train $BASE --err_feedback --ef_alpha $ALPHA --eval_interval 1000 \
      --save_secs 600 --track_flips --resume --out_dir $OUT >> $OUT.log 2>&1 && break
  sleep 30
done
if grep -q "^done$" $OUT.log; then
  touch $OUT/done
  say "DONE armA_cos_ef -> ppl $(final_ppl $OUT.log)   (armA_cosine without EF: 98.56)"
  python plot_run.py $OUT --title "armA_cos_ef: cosine anneal + spatial error feedback (alpha=$ALPHA)" >> $LOG 2>&1
  python plot_compare.py checkpoints/armA_cosine $OUT --out checkpoints/cmp_ef.png \
      --title "error feedback vs none (cosine anneal, 300M tokens)" >> $LOG 2>&1
else
  say "FAILED armA_cos_ef (see $OUT.log)"; tail -5 $OUT.log | tee -a $LOG
fi
say "EF OVERNIGHT COMPLETE"
