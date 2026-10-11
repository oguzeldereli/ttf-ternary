#!/usr/bin/env bash
# Lab PC: fetch pretrained Llama checkpoints with the Llama-2 32k tokenizer (the one our data uses) to local disk and run
# the --init_hf conversion check (scripts/quant/hf_check.py) on each. Log: /tmp/$USER/tern/repo/logs/hf_check.log
#   bash ~/ternary-LLM/scripts/quant/lab_hf_check.sh [REPO_ID ...]
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
export HF_HOME=$S/hf          # never the home folder (10 GB quota)
python -c "import safetensors, transformers, huggingface_hub" 2>/dev/null || \
  pip install --quiet --no-cache-dir safetensors transformers huggingface_hub
mkdir -p logs $S/hf/models
for R in ${@:-JackFram/llama-160m TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T}; do
  D=$S/hf/models/${R##*/}
  python -c "
from huggingface_hub import snapshot_download, list_repo_files
fs = list_repo_files('$R'); st = any(f.endswith('.safetensors') for f in fs)
snapshot_download('$R', local_dir='$D', allow_patterns=['config.json', '*.safetensors' if st else 'pytorch_model*.bin'])
print('downloaded $R', 'safetensors' if st else 'bin')"
  echo "== $(date '+%F %T') $R" >> logs/hf_check.log
  python scripts/quant/hf_check.py $D --n 100 >> logs/hf_check.log 2>&1
done
echo "== $(date '+%F %T') done" >> logs/hf_check.log
