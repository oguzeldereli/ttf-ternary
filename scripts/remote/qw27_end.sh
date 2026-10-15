#!/bin/bash
# End of the qw27_rel booking on beachcomber (booking ends Fri 16 Oct 19:20). Run on beachcomber once the trainer has
# exited after step 1350 (or after a SIGTERM, which saves):
#   nohup bash /scratch0/oguzelde/tern/end/repo/scripts/remote/qw27_end.sh > /scratch0/oguzelde/tern/end/end.log 2>&1 &
# Uses its own code tree (/scratch0/oguzelde/tern/end/repo; synced from the laptop), not the training run's tree.
# Steps (durations from the dry run of 15 Oct on the step-853 checkpoint; 4070 Ti Super timings for the GPU steps):
#   1 wait for the trainer to exit, record its final lines                          -
#   2 pack: 4-bit embedding and head (pack_ttf.py), sha256 of ckpt and pack        ~1-2 min (pack 6 s from page cache)
#   3 fixed windows: Wikipedia 8 (wikiq8_val), post-training mix 12 (mixval12),    ~4 min (81 + 52 s each form)
#     on the packed file and on the trainer's checkpoint (bf16 embedding / head)
#   4 zero-shot, six tasks, <= 2000 each, packed file (as zeroshot_qwen38_27b_fp)  ~60 min on the 4070, ~35-40 on the 4090
#   5 generation benchmarks (gen_bench.py: GSM8K 200, HumanEval 164, IFEval 200)   only if before DEADLINE_GEN; resumable
# Copy to the laptop with scripts/remote/qw27_pull.sh (38 MB/s measured: ckpt 13.1 GB ~6 min, pack 6.6 GB ~3 min).
set -u
E=/scratch0/oguzelde/tern/end; R=$E/repo; C=${C:-/scratch0/oguzelde/tern/repo/checkpoints/qw27_rel}
P=/scratch0/oguzelde/tern/env/bin/python; M=/scratch0/oguzelde/tern/hf/models/Qwen3.8-27B
O=${O:-$E/qw27_rel}; mkdir -p $O
DRY=${DRY:-0}                               # DRY=1 C=... O=...: no wait, CPU steps only (the GPU is training)
DEADLINE_GEN=${DEADLINE_GEN:-17:45}        # start no generation batch after this (local time); copy-off needs ~30 min
cd $R; export PYTHONPATH=$R
ts() { date +%H:%M:%S; }
echo "$(ts) [1] waiting for the trainer to exit"
[ "$DRY" = 1 ] || while pgrep -u "$USER" -f "[p]ython -m bitnet.train" > /dev/null; do sleep 30; done
grep -E "^step|val loss|FINAL|checkpoint saved" $C/train.log | tail -6
cp $C/train.log $C/metrics.jsonl $O/
echo "$(ts) [2] pack + checksums"
ln -f $C/ckpt.pt $O/qw27_rel_ckpt.pt
(cd $O && sha256sum qw27_rel_ckpt.pt > qw27_rel_ckpt.pt.sha256) &
$P scripts/infer/pack_ttf.py $O/qw27_rel_ckpt.pt $O/qw27_rel_e4.pt --emb_bits 4 --head_bits 4 2>&1 | tail -3
(cd $O && sha256sum qw27_rel_e4.pt > qw27_rel_e4.pt.sha256)
wait
cat $O/*.sha256
echo "$(ts) [2] done: ready to pull (qw27_pull.sh)"
[ "$DRY" = 1 ] && { echo "$(ts) dry run: stop before the GPU steps"; exit 0; }
echo "$(ts) [3] fixed windows (fp 1.9034 / 1.2885, Q8 1.9031 / 1.2884)"
for W in qw27_rel_e4.pt qw27_rel_ckpt.pt; do
  for f in wikiq8_val:8 mixval12:12; do
    echo "== $W ${f%:*}"
    $P scripts/infer/ttf_qwen.py $M $O/$W --ppl data/${f%:*}.bin --n_win ${f#*:} 2>&1 | grep -E "mean loss|Error|Traceback"
  done
done | tee $O/fixed_windows.txt
echo "$(ts) [4] zero-shot (packed)"
$P scripts/eval/qwen_zeroshot.py $M --ttf $O/qw27_rel_e4.pt --limit 2000 --out $O/zeroshot_qwen38_27b_ttf_rel.json 2>&1 \
  | grep -vE "Warning|warn|causal_conv|CUDACachingAllocator" | tail -8
echo "$(ts) [5] generation benchmarks (until $DEADLINE_GEN)"
for task in gsm8k:200 humaneval:0 ifeval:200; do
  [[ "$(date +%H:%M)" > "$DEADLINE_GEN" ]] && { echo "$(ts) past $DEADLINE_GEN: generation left for a 3090 (resumable)"; break; }
  $P scripts/eval/gen_bench.py $M --ttf $O/qw27_rel_e4.pt --task ${task%:*} --limit ${task#*:} --bs 50 \
    --out $O/ttf_${task%:*}.jsonl 2>&1 | grep -E "to do|batch|done in|Error|Traceback"
done
echo "$(ts) all done"
