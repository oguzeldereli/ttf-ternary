#!/bin/bash
# The consolidation gate (lab PC with the 110M HF model and Qwen3.5-0.8B): every kept configuration through the
# reference (ttf.legacy = the code at tag pre-consolidation-2026-10-17) and through the core, same seed and data; prints
# per-step training losses and the final val of each and whether they are identical. REF2=1 runs the reference twice
# (its own run-to-run spread: a configuration whose reference is not deterministic compares within that spread).
# DET=1: both through scripts/tests/det_run.py (deterministic mode; flash attention's backward otherwise differs run
# to run). CORE_FLAGS_FN (a file defining core_flags NAME) maps a configuration to the core's flags once they differ.
#   bash scripts/tests/gate.sh [NAME ...]          (run from the repo copy on the lab PC)
set -u
S=${S:-/tmp/oguzelde/tern}; source $S/env.sh
G=${GATE_DIR:-/tmp/oguzelde/gate}; mkdir -p $G
MD=$S/models; Q=$S/hf/models/Qwen3.5-0.8B-Base
B="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 512 --batch_size 4
   --grad_accum 1 --warmup 5 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 5 --g_ref 3.0 --int8
   --dw_mode dense --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000 --eval_iters 2 --save_secs 99999
   --snap_every 0 --no_final_save --lookahead 0 --seed 7"
TTF="--ts --ts_nomom --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_fused --gemm int8mm --lowrank 64
   --ts_rank_s 64 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp"
MIC="${B/--batch_size 4 --grad_accum 1/--batch_size 2 --grad_accum 2} --ts_micro"
FAST="--ts_factored 64 --ts_fire_fused --fused_scales --lean_tail --lean_batched"
CONV="--init_hf $MD/fp32_110m_hf --init_hf_method gptqc --distill $MD/fp32_110m_hf --distill_w 0.75"
declare -A C
C[scratch]="$B $TTF --steps 60"
C[micro_fast]="$MIC $TTF $FAST --steps 60"
C[conv]="$MIC $TTF $FAST $CONV --ts_theta 64 --steps 60"
C[conv_msvd]="$MIC $TTF $FAST $CONV --ts_theta 64 --lowrank_mag add:64 --init_mag_svd 64 --steps 40"
C[hid_kd]="$B $TTF $CONV --hid_kd 0.3 --steps 40"
C[master]="$B --mode master --rc_scale --steps 40"
C[fp32]="$B --mode fp32 --steps 40"
C[moe]="$B $TTF --moe 4:2:1024 --steps 40"
C[tern_emb]="$B $TTF --tern_emb tied --steps 40"
C[qwen]="--init_hf $Q --init_stream --init_hf_method gptqc --data data/wikiqw_train40.bin --val data/wikiqw_val.bin --data_u32
   --seq_len 512 --batch_size 2 --grad_accum 2 --ts_micro --warmup 5 --tail_fp32 --lr 5e-4 --min_lr 5e-5
   --eval_interval 1000 --eval_iters 2 --save_secs 99999 --snap_every 0 --no_final_save --lookahead 0 --rc_scale
   --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75
   --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 64 --ts_rank_s 64 $FAST --fused_norm --steps 12 --seed 7"
NAMES=${*:-scratch micro_fast conv conv_msvd hid_kd master fp32 moe tern_emb qwen}
[ -n "${CORE_FLAGS_FN:-}" ] && source $CORE_FLAGS_FN
losses() { python3 -c "
import json, sys
d = [json.loads(l) for l in open(sys.argv[1])]
print(' '.join('%.6f' % x['loss'] for x in d if 'loss' in x), '|', ' '.join('%.6f' % x['val_loss'] for x in d if x.get('final')))" $1 2>/dev/null; }
for n in $NAMES; do
  F=$(echo ${C[$n]})
  run() { rm -rf $G/$n.$1; if [ -n "${DET:-}" ]; then L="python scripts/tests/det_run.py $2"; else L="python -m $2"; fi
          $L $3 --out_dir $G/$n.$1 > $G/$n.$1.log 2>&1 || echo "$n.$1: exit $? ($(grep -m1 -E 'Error|error' $G/$n.$1.log | cut -c1-120))"; }
  run ref ttf.legacy.train "$F"
  [ -n "${REF2:-}" ] && run ref2 ttf.legacy.train "$F"
  CF="$F"; type core_flags > /dev/null 2>&1 && CF=$(core_flags $n "$F")
  [ -z "${REFONLY:-}" ] && run core ttf.train "$CF"
  a=$(losses $G/$n.ref/metrics.jsonl); b=$([ -n "${REF2:-}" ] && losses $G/$n.ref2/metrics.jsonl); c=$([ -z "${REFONLY:-}" ] && losses $G/$n.core/metrics.jsonl)
  st="ref ${a:0:60}..."
  [ -n "${REF2:-}" ] && { [ "$a" = "$b" ] && st="$st | ref deterministic" || st="$st | REF NOT DETERMINISTIC"; }
  [ -z "${REFONLY:-}" ] && { [ "$a" = "$c" ] && st="$st | core IDENTICAL" || st="$st | core DIFFERS (final ${c##*|})"; }
  echo "$n: $st"
done
