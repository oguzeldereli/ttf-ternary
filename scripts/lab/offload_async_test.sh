#!/usr/bin/env bash
# Lab PC: --ts_offload_async against --ts_offload (copies on the compute stream) and the state on the GPU.
#   offload_async_test.sh eq     110M, 40 steps each, the GPU run twice: the loss differences between the variants must be
#                                no larger than between two identical GPU runs (the kernels are not bitwise deterministic)
#   offload_async_test.sh time   1.3B micro-batch 1 x 2048 (the lean_mem.sh setting), 30 steps each: s/step and peak
#   offload_async_test.sh resume 110M async: 21 steps with the final save, then --resume to 41 (the checkpoint holds the
#                                host state; after the resume it must go back into page-locked buffers: same s/step)
# Summary ~/ternary-sync/offload_async_MODE.txt
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
M=${1:-eq}; O=checkpoints/offload_async; mkdir -p $O; R=$HOME/ternary-sync/offload_async_$M.txt
B="--mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --grad_accum 1
   --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
   --dw_mode dense --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000000 --eval_iters 1 --save_secs 1000000 --lookahead 0
   --ckpt_skip 0 --loss_chunk 256 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --ts --ts_theta 16 --ts_tau 1000
   --ts_anneal 0.5 --ts_fused --ts_store_m int8 --ts_store_a fp16 --lean_tail --mem_report --lowrank 512 --ts_rank_s 512"
echo "$(date '+%F %T') start on $(hostname) $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)" > $R
if [ $M = resume ]; then
  P="--preset small --batch_size 2 --ts_offload_async"; rm -rf $O/resume
  python -m bitnet.train $(echo $B $P) --stop_after 21 --out_dir $O/resume > $O/resume_a.log 2>&1
  python -m bitnet.train $(echo $B $P) --stop_after 41 --resume --out_dir $O/resume > $O/resume_b.log 2>&1
  for x in a b; do echo "$x: $(grep -E '^step ' $O/resume_$x.log | head -1 | cut -c1-40) .. $(grep -E '^step ' $O/resume_$x.log | tail -1 | cut -c1-80) $(grep -m1 -E 'Traceback|Error' $O/resume_$x.log)" >> $R; done
  echo "$(date '+%F %T') done" >> $R; exit
fi
if [ $M = eq ]; then P="--preset small --batch_size 2 --stop_after 41"; else P="--preset d2048_l24 --batch_size 1 --stop_after 31"; fi
for v in gpu gpu2 sync async; do
  case $v in gpu|gpu2) F="";; sync) F="--ts_offload";; async) F="--ts_offload_async";; esac
  rm -rf $O/${M}_$v
  TERN_TIME=1 python -m bitnet.train $(echo $B $P) $F --out_dir $O/${M}_$v > $O/${M}_$v.log 2>&1
  grep -E '^step ' $O/${M}_$v.log | sed -E 's/\| +[0-9.]+s \|.*//' > $O/${M}_$v.loss
  echo "$v: $(grep 'memory (MiB)' $O/${M}_$v.log | tail -1 | sed 's/  memory (MiB): //') | $(grep -E '^step ' $O/${M}_$v.log | awk -F'|' '{for (i=2;i<=NF;i++) if ($i ~ /^ *[0-9.]+s *$/) {t=$i; gsub(/[ s]/,"",t)}; n=$1; sub(/step */,"",n); if (n+0==10) {a=t; s0=n} b=t; s1=n} END {printf "%.2f s/step (10-%d)", (b-a)/(s1-s0), s1}') | last $(tail -1 $O/${M}_$v.loss) $(grep -m1 -E 'Error|out of memory' $O/${M}_$v.log | cut -c1-120)" >> $R
done
python - $O/${M}_gpu $O/${M}_gpu2 $O/${M}_sync $O/${M}_async >> $R <<'PY'
import json, sys
L = {d.rsplit("_", 1)[1]: [json.loads(l)["loss"] for l in open(d + "/metrics.jsonl") if '"loss"' in l] for d in sys.argv[1:]}
for a, b in [("gpu", "gpu2"), ("gpu", "sync"), ("gpu", "async"), ("sync", "async")]:
    n = min(len(L[a]), len(L[b])); d = [abs(x - y) for x, y in zip(L[a][:n], L[b][:n])]
    print(f"{a} vs {b}: max |loss diff| {max(d):.2e} over {n} logged steps (first nonzero at {next((i for i, x in enumerate(d) if x), -1)})")
PY
echo "$(date '+%F %T') done" >> $R
