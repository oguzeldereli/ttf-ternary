#!/bin/bash -l
# Myriad: Qwen3.8-27B ternarized with TTF (the release candidate, working name Qwen3.8-27B-TTFT-6GB):
#   1. the full-precision reference loss on our Wikipedia windows (re-tokenized with Qwen's tokenizer)
#   2. streaming, sequential GPTQ conversion (one decoder layer at a time from the safetensors)
#   3. recovery under a 22.4 GiB cap (the 24 GB claim): trits frozen until step 400, then TTF at theta 64, no clip;
#      rank 256, 4 x 4 micro-batches (no dense gradient), TTF state 70% on the GPU, bf16 float tail; final checkpoint kept
#   qsub -N qwen27 [-v STEPS=1000] scripts/myriad/qwen27.sh
#$ -l h_rt=48:00:00
#$ -l mem=32G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern; RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch $S/ubuntu24.sif bash -c"
M=$HOME/Scratch/hf/models/Qwen3.8-27B; N=qw27_gq_s400_th64; STEPS=${STEPS:-1000}
mkdir -p checkpoints/$N logs/qwen27
export RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4   # the job's own 4 cores only
TOK="export HF_HOME=$HOME/Scratch/hf HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=4;"
[ -f data/wikiq8_val.bin ] || $RUN "source $S/env.sh; $TOK cd $HOME/Scratch/ternary-LLM; python scripts/quant/retok.py data/wiki32k_val.bin data/wikiq8_val.bin --target $M"
[ -f data/wikiq8_train.bin ] || $RUN "source $S/env.sh; $TOK cd $HOME/Scratch/ternary-LLM; python scripts/quant/retok.py data/wiki32k_train.bin data/wikiq8_train.bin --target $M"
[ -f logs/qwen27/ref.txt ] || $RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python scripts/quant/qwen_check.py $M --data data/wikiq8_val.bin --n 16 --methods ''" > logs/qwen27/ref.txt 2>&1
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m bitnet.train --init_hf $M --init_hf_method gptq --init_stream \
  --data data/wikiq8_train.bin --val data/wikiq8_val.bin --data_u32 --seq_len 2048 --batch_size 4 --grad_accum 4 --ts_micro \
  --steps $STEPS --warmup 300 --tail_fp32 --lr 5e-4 --min_lr 5e-5 --eval_interval 100 --eval_iters 4 --save_secs 3600 \
  --snap_every 0 --lookahead 0 --ckpt_skip 0 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 \
  --mode kernel --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 400 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal \
  --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag --ts_store_a fp12s --lean_tail \
  --ts_offload_async --ts_offload_keep 0.7 --mem_cap_gib 22.4 --mem_report --out_dir checkpoints/$N" \
  >> checkpoints/$N/train.log 2>&1
