#!/bin/bash
# Rented GPU box (vast.ai / RunPod / ...; any Linux image with an NVIDIA driver and python3): environment for the 27B
# continuation, the same versions as the lab and beachcomber (torch 2.13, transformers 5.19, flash-linear-attention
# 0.5.2). Blackwell cards (B200 / RTX 50xx, sm_100 / sm_120) get the CUDA 12.8 wheels, everything else CUDA 12.6.
#   bash setup_box.sh            (W=/workspace by default: the persistent disk on vast / RunPod)
set -euo pipefail
W=${W:-/workspace}; mkdir -p $W/tern; cd $W/tern
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
IDX=https://download.pytorch.org/whl/cu126; [ "$CAP" -ge 100 ] && IDX=https://download.pytorch.org/whl/cu129   # 2.13 has no cu128 build; cu129 runs on 12.8 drivers (minor-version compatibility)
[ -x env/bin/python ] || python3 -m venv env
env/bin/pip install -q --upgrade pip
env/bin/python -c "import torch" 2>/dev/null || env/bin/pip install -q --no-cache-dir torch==2.13.0 --index-url $IDX
env/bin/pip install -q --no-cache-dir numpy==2.3.5 transformers==5.19.0 accelerate==1.15.0 safetensors==0.8.0 \
  huggingface_hub==2.2.0 tokenizers==0.23.3 flash-linear-attention==0.5.2
cat > env.sh <<EOT
export PATH=$W/tern/env/bin:\$PATH PYTHONPATH=$W/tern/repo HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/tern/triton-cache
EOT
env/bin/python -c "import torch, triton, transformers, fla; print('torch', torch.__version__, 'triton', triton.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; df -h $W | tail -1
