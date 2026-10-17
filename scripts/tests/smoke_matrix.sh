#!/bin/bash
# Every option path of the core trainer for a few steps on the 110M model (lab PC): each must exit 0 with finite losses.
# Covers what scripts/tests/gate.sh does not (the gate compares the kept configurations with the reference per step).
#   S=/tmp/$USER/tern bash scripts/tests/smoke_matrix.sh [NAME ...]        (from a core copy with data/ and models/)
set -u
S=${S:-/tmp/$USER/tern}; source $S/env.sh; export PYTHONPATH=$PWD
O=${SMOKE_DIR:-/tmp/$USER/smoke}; mkdir -p $O
HF=models/fp32_110m_hf
B="--preset small --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq-len 256 --batch-size 2 --accum 2
   --steps 8 --stop-after 6 --warmup 2 --eval-interval 1000 --eval-iters 1 --save-secs 99999 --ttf-rank 32 --ckpt-skip 0"
declare -A C
C[default]=""
C[after]="--ttf-update after"
C[sum_dense]="--ttf-accum sum --dw dense"
C[micro_dense]="--dw dense"
C[offload]="--ttf-offload host --ttf-offload-keep 0.5"
C[fp12s]="--ttf-store fp12s"
C[int8store]="--ttf-store int8"
C[lean]="--tail lean"
C[bf16tail]="--tail bf16"
C[bitnet]="--mode bitnet"
C[fp32]="--mode fp32"
C[ternemb]="--tern-emb tied --accum 1 --batch-size 4"
C[moe]="--moe 4:2:512"
C[wsd]="--lr-schedule wsd:5 --ttf-theta-sched 16:8"
C[linear]="--lr-schedule linear --ttf-leak fixed"
C[diag]="--eval-first-n 2 --mem-report --track-flips --ttf-vdiag 2"
C[fwdbf16]="--fwd bf16 --dw dense"
C[dx8]="--dx int8"
C[triton]="--gemm triton --no-fused-scales"
C[cublas]="--gemm cublas"
C[ckblock]="--ckpt block --fire torch"
C[svd]="--init-hf $HF --convert gptqc --adapter-init svd --calib 4"
C[als]="--init-hf $HF --convert als --teacher $HF --kd-hidden 0.3"
C[start]="--ttf-start 3 --ttf-spend 2"
NAMES=${*:-${!C[@]}}
ok=0; bad=0
for n in $NAMES; do
  rm -rf $O/$n; F=$(echo $B ${C[$n]})
  python -m ttf.train $F --out-dir $O/$n > $O/$n.log 2>&1; rc=$?
  l=$(python3 -c "import json,math,sys; L=[json.loads(x)['loss'] for x in open('$O/$n/metrics.jsonl') if '\"loss\"' in x]; print(len(L), all(math.isfinite(v) for v in L), round(L[-1],3))" 2>/dev/null)
  if [ $rc = 0 ] && [[ "$l" == *True* ]]; then ok=$((ok+1)); echo "ok   $n ($l)"; else bad=$((bad+1)); echo "FAIL $n rc $rc ($l): $(grep -E 'Error|error' $O/$n.log | tail -1 | cut -c1-160)"; fi
done
# resume: 3 steps, saved; then to 6 with a larger accumulator rank and adapter rank (both grown on resume)
rm -rf $O/resume; F=$(echo $B)
python -m ttf.train ${F/--stop-after 6/--stop-after 3} --out-dir $O/resume > $O/resume.log 2>&1 && \
python -m ttf.train $F --ttf-rank 48 --adapter-rank 24 --ttf-acc-scale 0.5 --resume --out-dir $O/resume >> $O/resume.log 2>&1 \
  && grep -q "grew the accumulators" $O/resume.log && grep -q "adapters grown" $O/resume.log \
  && { ok=$((ok+1)); echo "ok   resume (rank and adapter growth, acc scale)"; } \
  || { bad=$((bad+1)); echo "FAIL resume: $(grep -E 'Error|error' $O/resume.log | tail -1 | cut -c1-160)"; }
echo "smoke: $ok ok, $bad failed"
