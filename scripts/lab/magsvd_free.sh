#!/bin/bash
# --init_mag_svd, training-free (17 Oct): the 110M fp model converted by centred GPTQ with the rounding error's best
# rank-R part (input-covariance norm) moved into the float adapters, R = 0 (adapters at 0) / 16 / 64 / 256; loss on the
# first 16 val windows before any training (fp 110M on the same windows: from eval_first_n of --mode fp32).
set -u
S=/tmp/oguzelde/tern; source $S/env.sh; cd $S/repo; MD=$S/models; HF=$MD/fp32_110m_hf
O=checkpoints/magsvd_free; mkdir -p $O
B="--init_hf $HF --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 1 --eval_first_n 16 --eval_interval 100000 --no_final_save --save_secs 99999 --tail_fp32 --lookahead 0"
python -m bitnet.train $(echo $B) --mode fp32 --out_dir $O/fp > $O/fp.log 2>&1
for R in 0 16 64 256; do
  X="--lowrank_mag add:${R/#0/16}"; [ $R != 0 ] && X="$X --init_mag_svd $R"
  python -m bitnet.train $(echo $B) --mode kernel --int8 --dw_mode dense --rc_scale --init_hf_method ${METHOD:-gptqc} \
    --ts --ts_nomom --ts_fused --lowrank 512 --ts_rank_s 512 $X --out_dir $O/r$R > $O/r$R.log 2>&1
done
for f in fp r0 r16 r64 r256; do echo "$f: $(grep -oE 'eval_first_n: loss [0-9.]+' $O/$f.log) $(grep -oE 'adapters take [0-9.]+ .*' $O/$f.log)"; done | tee $O/summary.txt
