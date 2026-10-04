#!/bin/bash -l
# Myriad, ~1 h on one A100: speed of a 1.3B step (batch 16 x 2048) for the compute work of 4-5 Oct.
# (1) GEMM microbenchmark at the model's shapes, per kind and backend (cuBLAS bf16 on unpacked trits / int8 tensor cores):
#     forward (int8 activations x trits), input gradient (bf16 x trits), weight gradient (dense), trit unpacking; summed
#     over the 168 ternary matrices per pass: the time a step spends in matmuls that touch trits (for the projection of
#     ternary hardware).
# (2) 25 steps per configuration (TERN_TIME: forward+backward / update): TTF as timed on 3 Oct (cuBLAS, fused update,
#     fp64 Cholesky-QR), + orthonormalisation every 4 steps, + deferred batched orthonormalisation, both, + int8 forward;
#     master. Output: logs/speed_1b/.
#   qsub scripts/myriad/speed_1b.sh
#$ -N speed_1b
#$ -l h_rt=1:15:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; O=logs/speed_1b; mkdir -p $O; R=$O/summary.txt
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
{ hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > $R
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/analysis/gemm_bench.py" >> $R 2>&1
C="--preset d2048_l24 --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --lr 7.5e-4 --min_lr 7.5e-5 --eval_interval 100000 --eval_iters 2 --save_secs 1000000 --stop_after 26"
K="--mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8 --dw_mode dense
   --track_flips --tail_fp32 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp
   --ts --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_tau_anneal --ts_anneal 0.5 --ts_fused --gemm cublas --ts_orth chol64"
for v in "ts_base" "ts_every4 --ts_orth_every 4" "ts_defer --ts_orth_defer" "ts_defer_every4 --ts_orth_defer --ts_orth_every 4" \
         "ts_int8_defer_every4 --ts_orth_defer --ts_orth_every 4 --gemm int8mm" "master"; do
  set -- $v; n=$1; shift
  if [ $n = master ]; then F="$C --mode master --master_dtype fp32"; else F="$C $K $*"; fi
  F=$(echo $F)
  $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; TERN_TIME=1 python -m bitnet.train $F --out_dir $O/$n" > $O/$n.log 2>&1
  python3 - $O/$n.log $n >> $R <<'PY'
import re, sys
t = [tuple(map(float, m)) for m in re.findall(r"forward\+backward ([\d.]+)s, update \(momentum step / AdamW\) ([\d.]+)s", open(sys.argv[1]).read())][5:]
err = re.search(r"(Error|out of memory).*", open(sys.argv[1]).read())
if t:
    fb = sum(a for a, _ in t) / len(t); up = sum(b for _, b in t) / len(t)
    print(f"{sys.argv[2]:24s} forward+backward {fb:.2f} s, update {up:.2f} s, step {fb + up:.2f} s (mean of steps 5-{4 + len(t)})")
else:
    print(f"{sys.argv[2]:24s} no timings {err.group(0)[:120] if err else ''}")
PY
done
echo "$(date '+%F %T') done" >> $R
