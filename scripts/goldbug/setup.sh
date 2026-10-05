#!/usr/bin/env bash
# goldbug (2 x RTX PRO 6000 96 GB, Blackwell sm_120) or any test machine: user-space env + code copy + data.
# torch 2.13.0 built for CUDA 12.9 (the cu126 wheels used elsewhere have no sm_120 kernels; cu129 also runs on
# sm_80 / sm_86 / sm_89, so the same env is tested on the lab PCs first). Everything under $GB (default
# /scratch0/$USER/gb, wiped when the booking ends): env/, repo/ (copy of ~/ternary-LLM, own data/ and checkpoints/).
# Data: wiki32k from ~/ternary-data; FineWeb-Edu (4 GB, too big for the 10 GB home) is pushed from the laptop with
# scripts/goldbug/push_data.sh. Re-run to refresh the code copy.
#   bash ~/ternary-LLM/scripts/goldbug/setup.sh          (GB=/tmp/$USER/gbdry for the dry run on a lab PC)
set -euo pipefail
GB=${GB:-/scratch0/$USER/gb}; CU=${CU:-cu129}
mkdir -p "$GB" && cd "$GB"
[ -x "$GB/bin/micromamba" ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$GB" bin/micromamba
export MAMBA_ROOT_PREFIX=$GB/mamba
[ -x "$GB/env/bin/python" ] || "$GB/bin/micromamba" create -y -q -p "$GB/env" -c conda-forge python=3.13 gcc gxx tmux rsync
"$GB/env/bin/python" -c "import torch" 2>/dev/null || \
  "$GB/env/bin/pip" install --quiet --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/$CU
"$GB/env/bin/python" -c "import scipy, matplotlib" 2>/dev/null || \
  "$GB/env/bin/pip" install --quiet --no-cache-dir numpy==2.3.5 scipy matplotlib
mkdir -p "$GB/repo/data" "$GB/repo/checkpoints"
"$GB/env/bin/rsync" -a --delete --exclude checkpoints --exclude data --exclude __pycache__ --exclude .git \
  "$HOME/ternary-LLM/" "$GB/repo/"
for f in wiki32k_train.bin wiki32k_val.bin; do
  [ -f "$GB/repo/data/$f" ] || cp "$HOME/ternary-data/$f" "$GB/repo/data/$f"
done
cat > "$GB/env.sh" <<EOT
export PATH=$GB/env/bin:\$PATH CC=$GB/env/bin/gcc PYTHONPATH=$GB/repo GB=$GB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$GB/triton-cache
EOT
source "$GB/env.sh"; cd "$GB/repo"
python - <<'P'
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("arch list", torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, sm_{p.major}{p.minor}, {p.total_memory / 2**30:.1f} GiB")
P
ls -la data/ | grep -E "bin$" || true
echo "setup ok on $(hostname -s). next: bash scripts/goldbug/smoke.sh"
