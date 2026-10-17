#!/bin/bash
# Does the conversion gap grow with how long the fp model was trained? (17 Oct) The 110M fp recipe with a complete
# cosine schedule of STEPS steps (1017 = 1/9, 3052 = 1/3 of the 300M-token baseline; warmup scaled), exported, then
# converted with the best 50M-token recipe (geom_arms.sh gptqc_ts0_kd075, KD from this fp model); fixed-window evals of
# both, so the gap is against the model's own fp. The full-length point: fp32_baseline -> g_gptqc_ts0_kd075.
#   bash /tmp/oguzelde/tern/repo/scripts/lab/fp_dose.sh STEPS
set -u
N=$1; S=/tmp/oguzelde/tern; source $S/env.sh; cd $S/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=checkpoints/fpq$N; mkdir -p $D; W=$(( 305 * N / 9155 ))
[ -f $D/ckpt.pt ] && grep -q "FINAL val" $D/train.log || python -m ttf.train --preset small --mode fp32 \
  --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1 --steps $N \
  --warmup $W --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 3600 --lookahead 0 \
  --out_dir $D >> $D/train.log 2>&1
python scripts/eval/final_eval.py $D --fw 0 > $D/final_eval.txt 2>&1
HFD=$S/models/fpq${N}_hf
[ -f $HFD/config.json ] || python scripts/analysis/export_hf_llama.py $D/ckpt.pt $HFD --val data/wiki32k_val.bin >> $D/train.log 2>&1
HF=$HFD SUF=_fpq$N bash scripts/lab/geom_arms.sh gptqc_ts0_kd075
echo "$(date '+%F %T') DOSE_DONE $N" >> $D/train.log
