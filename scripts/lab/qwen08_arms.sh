#!/bin/bash
# Lab PC (16 GB+): do the 110M conversion findings transfer to a real model? Qwen3.5-0.8B-Base, 400 steps x 16 x 2048 (13M
# tokens, labels only, Wikipedia retokenised), the 27B recipe's flags (streaming conversion, rank 256, theta 64):
#   q08_gq_s120   GPTQ, trits frozen to 120 (the 27B recipe's 400 / 1350, scaled)
#   q08_gq_ts0    GPTQ, trits moving from step 0
#   q08_gqc_ts0   centred GPTQ, trits moving from step 0
# First a resume check of a centred run (10 steps, saved, resumed to 20).
#   setsid nohup bash ~/ternary-LLM/scripts/lab/qwen08_arms.sh > /tmp/oguzelde/qwen08_arms.log 2>&1 &
S=${S:-/tmp/$USER/tern}; source $S/env.sh || exit 1; cd $S/repo || exit 1; mkdir -p logs
python -c "import transformers, fla" 2>/dev/null || pip install --quiet --no-cache-dir "transformers==5.19.0" "flash-linear-attention==0.5.2"
M=$S/hf/models/Qwen3.5-0.8B-Base
[ -f $M/config.json ] || HF_HOME=$S/hf python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='$M', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])" > /dev/null 2>&1
[ -f data/wikiqw_val.bin ] || python scripts/quant/retok.py data/wiki32k_val.bin data/wikiqw_val.bin --target $M
[ -f data/wikiqw_train40.bin ] || python scripts/quant/retok.py data/wiki32k_train.bin data/wikiqw_train40.bin --target $M --max_tokens 4e7
F="--init_hf $M --init_stream --data data/wikiqw_train40.bin --val data/wikiqw_val.bin --data_u32 --seq_len 2048 --batch_size 4
  --grad_accum 4 --ts_micro --warmup 40 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 8 --save_secs 99999
  --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense
  --ts --ts_theta 64 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 256
  --ts_rank_s 256 --mem_cap_gib 15"
ONLY=${1:-}     # one arm name: run only it (no resume check)
if [ -z "$ONLY" ]; then
rm -rf checkpoints/q08_rescheck
python -m bitnet.train $F --init_hf_method gptqc --ts_start 0 --steps 10 --out_dir checkpoints/q08_rescheck > logs/q08_rescheck_a.txt 2>&1
python -m bitnet.train $F --init_hf_method gptqc --ts_start 0 --steps 20 --resume --out_dir checkpoints/q08_rescheck > logs/q08_rescheck_b.txt 2>&1
echo "RESCHECK: $(grep -E 'centred layers restored|^step +(10|20) ' logs/q08_rescheck_b.txt | tr '\n' ' ') $(grep -c Traceback logs/q08_rescheck_b.txt) tracebacks"
rm -rf checkpoints/q08_rescheck
fi
run() { [ -n "$ONLY" ] && [ "$1" != "$ONLY" ] && return; N=$1; shift; mkdir -p checkpoints/$N; python -m bitnet.train $F --steps 400 --no_final_save "$@" --out_dir checkpoints/$N > logs/$N.txt 2>&1
        echo "$(date '+%T') $N: $(grep 'val loss' logs/$N.txt | tr -s ' ' | cut -d' ' -f4 | tr '\n' ' ') $(grep -c Traceback logs/$N.txt) tracebacks"; }
run q08_gqc_ts0 --init_hf_method gptqc --ts_start 0
# 17 Oct: the float adapters from the rounding error (--init_mag_svd R) vs the same rank at 0 (ONLY=... runs one)
run q08_gqc_ts0_msvd16 --init_hf_method gptqc --ts_start 0 --init_mag_svd 16
run q08_gqc_ts0_msvd64 --init_hf_method gptqc --ts_start 0 --lowrank_mag add:64 --init_mag_svd 64
run q08_gqc_ts0_z64 --init_hf_method gptqc --ts_start 0 --lowrank_mag add:64
# 17 Oct: theta (the 27B moved 1.5e-6 of its trits in 1350 steps at theta 64, this recipe 3.5e-5 in 400: nearly frozen;
# does letting trits move help a real converted model?)
run q08_gqc_ts0_th32 --init_hf_method gptqc --ts_start 0 --ts_theta 32
run q08_gqc_ts0_th16 --init_hf_method gptqc --ts_start 0 --ts_theta 16
run q08_gqc_ts0_th8 --init_hf_method gptqc --ts_start 0 --ts_theta 8
run q08_gqc_ts0_msvd16_th16 --init_hf_method gptqc --ts_start 0 --init_mag_svd 16 --ts_theta 16   # the start gains together
run q08_gq_ts0 --init_hf_method gptq --ts_start 0
run q08_gq_s120 --init_hf_method gptq --ts_start 120
echo Q08_DONE
