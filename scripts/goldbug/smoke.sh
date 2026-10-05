#!/usr/bin/env bash
# goldbug, first hour: (1) the GPU's architecture is in torch's build; (2) our kernels against dense references on
# each GPU (tests/test_kernel_ref.py; triton and cublas backends); (3) 20 training steps at 110M per kernel path
# (int8 Triton forward, --gemm cublas, --gemm int8mm) so a failing path is known with its fallback; (4) a throughput
# test of each arm in arms.txt (40 steps, TERN_TIME=1) to fix the token budget. Summary: $GB/smoke.txt
#   source $GB/env.sh; bash scripts/goldbug/smoke.sh [arms file]
set -u
source "${GB:-/scratch0/$USER/gb}/env.sh"; cd "$GB/repo"
A=${1:-scripts/goldbug/arms.txt}; R=$GB/smoke.txt; O=checkpoints/smoke; mkdir -p $O
echo "$(date '+%F %T') smoke on $(hostname -s)" > $R
python -c "import torch; a=torch.cuda.get_arch_list(); p=torch.cuda.get_device_properties(0); s=f'sm_{p.major}{p.minor}'; print(s, 'in build' if s in a else 'NOT IN BUILD', a)" >> $R 2>&1
NG=$(python -c "import torch; print(torch.cuda.device_count())")
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g python tests/test_kernel_ref.py > $O/kernel_ref_gpu$g.txt 2>&1
  echo "GPU $g kernels: $(tail -1 $O/kernel_ref_gpu$g.txt)" >> $R
done
C="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000000 --eval_iters 1 --save_secs 100000000 --lookahead 0
   --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_nomom --lowrank 512 --ts_theta 16 --ts_tau 1000
   --ts_anneal 0.5 --ts_fused --stop_after 21"
for g in triton cublas int8mm; do
  G=""; [ $g != triton ] && G="--gemm $g"
  CUDA_VISIBLE_DEVICES=0 TERN_TIME=1 python -m bitnet.train $(echo $C) $G --out_dir $O/path_$g > $O/path_$g.log 2>&1
  echo "path $g: $(grep -E '^step' $O/path_$g.log | tail -1 | cut -c1-60) $(grep -E 'time step' $O/path_$g.log | tail -1 | grep -oE 'forward\+backward [0-9.]+s') $(grep -m1 -E 'Error|error:' $O/path_$g.log | cut -c1-120)" >> $R
done
while read -r g n flags; do
  [ -z "$g" ] || [ "${g:0:1}" = "#" ] && continue
  CUDA_VISIBLE_DEVICES=$g TERN_TIME=1 python -m bitnet.train $flags --stop_after 41 --eval_interval 1000000 \
    --save_secs 100000000 --out_dir $O/tp_$n > $O/tp_$n.log 2>&1
  t=$(grep -E 'time step' $O/tp_$n.log | tail -10 | grep -oE 'forward\+backward [0-9.]+' | awk '{s+=$2} END {print s/NR}')
  echo "arm $n (GPU $g): ~$t s per step fwd+bwd, $(grep -E '^step' $O/tp_$n.log | tail -1 | grep -oE 'peakVRAM [0-9.]+GiB') $(grep -m1 -E 'Error|out of memory' $O/tp_$n.log | cut -c1-120)" >> $R
done < $A
echo "$(date '+%F %T') done" >> $R; cat $R
