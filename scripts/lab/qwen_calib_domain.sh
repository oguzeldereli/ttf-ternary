#!/bin/bash
# Lab PC: GPTQ calibration domain vs forgetting on Qwen3.5-0.8B (scripts/quant/qwen_calib_domain.py). Needs
# ~/qw38tok/tokenizer.json and ~/q38/{chatq8_train16m,chatq8_val,mixval12}.bin (Qwen3.8 tokens) in the home.
S=${S:-/tmp/$USER/tern}; source $S/env.sh || exit 1; cd $S/repo || exit 1; mkdir -p logs docs/research/analysis
M=$S/hf/models/Qwen3.5-0.8B-Base
for f in chatq8_train16m chatq8_val mixval12; do g=${f/q8/qw}; [ $f = mixval12 ] && g=mixvalqw12; [ -f data/$g.bin ] || python scripts/quant/retok_q38_q35.py ~/q38/$f.bin data/$g.bin --src_tok ~/qw38tok/tokenizer.json --dst_tok $M/tokenizer.json; done
python scripts/quant/qwen_calib_domain.py $M --calib wiki=data/wikiqw_train40.bin,chat=data/chatqw_train16m.bin,mix=data/wikiqw_train40.bin+data/chatqw_train16m.bin \
  --eval wiki=data/wikiqw_val.bin,chat=data/chatqw_val.bin,mix12=data/mixvalqw12.bin --n 12 --out docs/research/analysis/qwen08_calib_domain.json
echo CALIB_DOMAIN_DONE
