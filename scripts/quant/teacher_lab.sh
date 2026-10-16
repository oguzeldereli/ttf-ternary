#!/bin/bash
# One lab 3090 PC's share of the mix3 teacher cache (Qwen3.8-27B bf16 streamed, top-32 log-probs, 48-window shards;
# 80 s per shard on a 3090 Ti, the same speed as the Q4 GGUF through llama.cpp and exact). Shards are claimed through
# ~/kd3_claims (shared home, mkdir is atomic), so PCs can join at any time; each writes to its own /tmp/oguzelde/kd3.
#   nohup bash ~/ternary-LLM/scripts/quant/teacher_lab.sh > /tmp/oguzelde/teacher_lab.log 2>&1 &
set -u
T=/tmp/oguzelde; mkdir -p $T/kd3 $T/hfcache
if $T/tern/env/bin/python -c "import torch, transformers, fla" 2>/dev/null; then S=$T/tern; bash ~/ternary-LLM/scripts/lab/setup.sh
else
  S=$T/tern2
  sed 's#^S=/tmp/$USER/tern$#S=/tmp/$USER/tern2#' ~/ternary-LLM/scripts/lab/setup.sh > $T/setup2.sh; bash $T/setup2.sh   # env once, code always
  $S/env/bin/python -m pip install -q --no-cache-dir transformers==5.19.0 accelerate==1.15.0 safetensors==0.8.0 \
    huggingface_hub==2.2.0 tokenizers==0.23.3 flash-linear-attention==0.5.2
fi
P=$S/env/bin/python; echo "$(date +%T) env $S: $($P -c 'import transformers, fla; print(transformers.__version__, fla.__version__)')"
M=$T/hf27
[ -f $M/config.json ] || HF_HOME=$T/hfcache $P -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B', local_dir='$M', allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])"
echo "$(date +%T) model $(du -sh $M | cut -f1)"
[ -f $T/mix3_train.bin ] || /opt/ucl/bin/rsync -a -e "ssh -o BatchMode=yes" oguzelde@beachcomber.cs.ucl.ac.uk:/scratch0/oguzelde/tern/m3/mix3_train.bin $T/
echo "$(date +%T) mix3 $(sha256sum $T/mix3_train.bin | cut -c1-16) (a9c5a6fe096aabd3 expected)"
cd $S/repo; export PYTHONPATH=$S/repo
$P scripts/quant/teacher_topk.py $M $T/mix3_train.bin $T/kd3 --sequential --windows 0 --claim_dir ~/kd3_claims
echo "$(date +%T) TEACHER_DONE"
