#!/bin/bash
# --init_mag_svd on a real pretrained model, training-free (17 Oct): Qwen3.5-0.8B-Base, streaming GPTQ / centred GPTQ,
# the float adapters at 0 or from the rounding error's best rank-R part (R 16 / 64 / 256); loss on the first 16 val
# windows (fp 2.4849, GPTQ 4.5168, centred 4.5538 on these windows: CONVERSION_GEOMETRY_OCT16.md section 6).
#   bash /tmp/oguzelde/tern/repo/scripts/lab/qwen_magsvd_free.sh [gptq|gptqc]
set -u
METH=${1:-gptq}; S=/tmp/oguzelde/tern; source $S/env.sh; cd $S/repo
M=$S/hf/models/Qwen3.5-0.8B-Base; O=checkpoints/qmagsvd_free; mkdir -p $O
python -c "import transformers, fla" 2>/dev/null || pip install --quiet --no-cache-dir "transformers==5.19.0" "flash-linear-attention==0.5.2"
[ -f data/wikiqw_val.bin ] || python scripts/quant/retok.py data/wiki32k_val.bin data/wikiqw_val.bin --target $M
[ -f data/wikiqw_train40.bin ] || python scripts/quant/retok.py data/wiki32k_train.bin data/wikiqw_train40.bin --target $M --max_tokens 4e7
for R in 0 16 64 256; do
  X="--lowrank_mag add:${R/#0/16}"; [ $R != 0 ] && X="$X --init_mag_svd $R"
  python -m bitnet.train --init_hf $M --init_stream --init_hf_method $METH --data data/wikiqw_train40.bin --val data/wikiqw_val.bin \
    --data_u32 --seq_len 2048 --batch_size 1 --grad_accum 1 --steps 1 --eval_first_n 16 --eval_interval 100000 --no_final_save \
    --save_secs 99999 --tail_fp32 --lookahead 0 --rc_scale --mode kernel --int8 --dw_mode dense --ts --ts_nomom --ts_fused \
    --gemm int8mm --lowrank 256 --ts_rank_s 256 $X --out_dir $O/${METH}_r$R > $O/${METH}_r$R.log 2>&1
  echo "$METH r$R: $(grep -oE 'eval_first_n: loss [0-9.]+' $O/${METH}_r$R.log) $(grep -oE 'adapters take [0-9.]+' $O/${METH}_r$R.log) $(grep -c Traceback $O/${METH}_r$R.log) tracebacks" | tee -a $O/summary.txt
done
