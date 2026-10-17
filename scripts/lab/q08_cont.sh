#!/bin/bash
# The 27B continuation question on Qwen3.5-0.8B (17 Oct): a converted run trained 400 steps at theta 64 (trits nearly
# frozen, floats trained), then continued 400 more with the lr re-warmed to half the peak: at theta 64, or at theta 16
# with the accumulators x 0.25 (--ts_acc_scale: no burst). Labels only, the qwen08_arms.sh recipe.
#   bash scripts/lab/q08_cont.sh base | c64 | c16 | c16z64 (c16 with the adapters grown to rank 64, new columns at 0)
set -u
A=$1; S=${S:-/tmp/$USER/tern}; source $S/env.sh || exit 1; cd $S/repo || exit 1; mkdir -p logs
python -c "import transformers, fla" 2>/dev/null || pip install --quiet --no-cache-dir "transformers==5.19.0" "flash-linear-attention==0.5.2"
M=$S/hf/models/Qwen3.5-0.8B-Base
[ -f $M/config.json ] || HF_HOME=$S/hf python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='$M', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])" > /dev/null 2>&1
[ -f data/wikiqw_val.bin ] || python scripts/quant/retok.py data/wiki32k_val.bin data/wikiqw_val.bin --target $M
[ -f data/wikiqw_train40.bin ] || python scripts/quant/retok.py data/wiki32k_train.bin data/wikiqw_train40.bin --target $M --max_tokens 4e7
F="--init_hf $M --init_stream --init_hf_method gptqc --data data/wikiqw_train40.bin --val data/wikiqw_val.bin --data_u32
  --seq_len 2048 --batch_size 4 --grad_accum 4 --ts_micro --warmup 40 --tail_fp32 --lr 5e-4 --min_lr 5e-5
  --eval_interval 100 --eval_iters 8 --save_secs 99999 --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale
  --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --ts --ts_tau 1000 --ts_anneal 0.75
  --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --mem_cap_gib 15 --ts_start 0"
B=checkpoints/q08c_base
case $A in
  base) mkdir -p $B; python -m bitnet.train $F --ts_theta 64 --steps 400 --out_dir $B > logs/q08c_base.txt 2>&1 ;;
  c64|c16|c16z64) D=checkpoints/q08c_$A; mkdir -p $D
    while [ ! -f $B/ckpt.pt ] || ! grep -q "FINAL val" logs/q08c_base.txt 2>/dev/null; do sleep 60; done
    [ -f $D/ckpt.pt ] || cp $B/ckpt.pt $D/ckpt.pt
    X="--ts_theta 64"; [ $A = c16 ] && X="--ts_theta 16 --ts_acc_scale 0.25"
    [ $A = c16z64 ] && X="--ts_theta 16 --ts_acc_scale 0.25 --lowrank_mag add:64"   # + adapters grown 16 -> 64
    python -m bitnet.train $F $X --steps 800 --lr_rewarm 401:20:0.5 --resume --no_final_save --out_dir $D > logs/q08c_$A.txt 2>&1 ;;
esac
echo "$(date '+%T') q08c_$A: $(grep 'val loss' logs/q08c_$A.txt | grep -oE 'val loss [0-9.]+' | cut -d' ' -f3 | tr '\n' ' ')"
