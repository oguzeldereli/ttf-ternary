#!/usr/bin/env bash
# One-time setup on a CS lab GPU PC (lab105 RTX 3090 Ti; use at most 1-2 of these shared machines). Everything
# lives on local disk under /tmp/$USER/tern so the shared home repo (whose data/ and checkpoints/ links belong to
# the booked 4090) is never touched:
#   env/    micromamba python 3.13 + gcc + torch 2.13.0+cu126 (same as the laptop and the 4090)
#   repo/   a copy of ~/ternary-LLM (code only), with its own data/ and checkpoints/
# Re-run to refresh the code copy (keeps env, data and checkpoints).
#   bash ~/ternary-LLM/scripts/lab/setup.sh
set -euo pipefail
S=/tmp/$USER/tern
mkdir -p "$S" && chmod 700 "/tmp/$USER"
cd "$S"
if [ ! -x "$S/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$S" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$S/mamba
if [ ! -x "$S/env/bin/python" ]; then
  "$S/bin/micromamba" create -y -q -p "$S/env" -c conda-forge python=3.13 gcc gxx tmux rsync
fi
"$S/env/bin/python" -c "import torch" 2>/dev/null || \
  "$S/env/bin/pip" install --quiet --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
"$S/env/bin/python" -c "import scipy, matplotlib" 2>/dev/null || \
  "$S/env/bin/pip" install --quiet --no-cache-dir numpy==2.3.5 scipy matplotlib
mkdir -p "$S/repo/data" "$S/repo/checkpoints"
"$S/env/bin/rsync" -a --delete --exclude checkpoints --exclude data --exclude logs --exclude __pycache__ --exclude .git \
  "$HOME/ternary-LLM/" "$S/repo/"
for f in wiki32k_train.bin wiki32k_val.bin; do
  [ -f "$S/repo/data/$f" ] || cp "$HOME/ternary-data/$f" "$S/repo/data/$f"
done
cat > "$S/env.sh" <<EOT
export PATH=$S/env/bin:\$PATH CC=$S/env/bin/gcc PYTHONPATH=$S/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$S/triton-cache
EOT
source "$S/env.sh"
cd "$S/repo"
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name(0))"
echo "lab setup ok on $(hostname -s)"
