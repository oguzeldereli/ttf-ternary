#!/bin/bash
# Lab PC: training-free centred GPTQ on Qwen3.5-0.8B-Base (the real offset): streaming gptq vs gptqc, loss on 16 val windows.
#   setsid nohup bash ~/ternary-LLM/scripts/lab/qwen_gptqc_check.sh > /tmp/oguzelde/qwen_gptqc.log 2>&1 &
S=${S:-/tmp/$USER/tern}; source $S/env.sh || exit 1; cd $S/repo || exit 1
python -c "import transformers, fla" 2>/dev/null || pip install --quiet --no-cache-dir "transformers==5.19.0" "flash-linear-attention==0.5.2"
M=$S/hf/models/Qwen3.5-0.8B-Base
[ -f $M/config.json ] || HF_HOME=$S/hf python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='$M', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])" > /dev/null 2>&1
[ -f data/wikiqw_val.bin ] || python scripts/quant/retok.py data/wiki32k_val.bin data/wikiqw_val.bin --target $M
[ -f data/wikiqw_train.bin ] || python scripts/quant/retok.py data/wiki32k_train.bin data/wikiqw_train.bin --target $M --max_tokens 5e6   # calibration only
python scripts/quant/qwen_stream_check.py $M --data data/wikiqw_val.bin --calib_data data/wikiqw_train.bin --methods gptq,gptqc
echo QWEN_GPTQC_DONE
