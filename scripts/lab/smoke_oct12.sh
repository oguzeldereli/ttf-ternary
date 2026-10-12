#!/usr/bin/env bash
# Lab PC: short smoke tests of the 12 Oct mechanisms (110M, final recipe), logs in logs/smoke_*.txt:
# sparse residual + exact spend + float-tail lr 3e-3; temporary residual; MoE 8 x top-2; data parallel (2 ranks, gloo, one GPU)
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1; mkdir -p logs
B="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --warmup 5
   --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 1000 --eval_iters 2 --save_secs 99999 --snap_every 0 --no_final_save
   --lookahead 0 --ckpt_skip 2 --int8 --dw_mode dense --track_flips --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp
   --ts --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 128 --ts_rank_s 128"
run() { N=$1; shift; mkdir -p checkpoints/$N; "$@" --out_dir checkpoints/$N > logs/smoke_$N.txt 2>&1;
        echo "$(date '+%T') $N: $(grep -E '^step' logs/smoke_$N.txt | tail -1 | cut -c1-70) $(grep -cE 'Traceback|Error' logs/smoke_$N.txt) errors" >> logs/smoke.log; }
run sm_rsparse python -m bitnet.train $B --steps 40 --batch_size 4 --ts_resid --ts_resid_sparse 0.01 --ts_exact_spend --lr 3e-3 --min_lr 3e-4
run sm_rtemp python -m bitnet.train $B --steps 40 --batch_size 4 --ts_resid --ts_resid_steps 20
run sm_moe python -m bitnet.train $B --steps 40 --batch_size 4 --moe 8:2:1024
run sm_dp python -m torch.distributed.run --nproc_per_node 2 -m bitnet.train $B --steps 101 --batch_size 2
grep -h "data parallel" logs/smoke_sm_dp.txt | tail -4 >> logs/smoke.log
# Qwen streaming conversion check (0.8B): stepping layer by layer = the model's loss; sequential GPTQ loss
python -c "import transformers, fla" 2>/dev/null || pip install --quiet --no-cache-dir "transformers==5.19.0" flash-linear-attention
M=$S/hf/models/Qwen3.5-0.8B-Base
[ -f $M/config.json ] || HF_HOME=$S/hf python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='$M', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])" > /dev/null 2>&1
for f in wikiqw_val.bin wikiqw_train.bin; do [ -f data/$f ] || scp -q -o BatchMode=yes oguzelde@whitebait-l.cs.ucl.ac.uk:$S/repo/data/$f data/; done
python scripts/quant/qwen_stream_check.py $M --data data/wikiqw_val.bin --calib_data data/wikiqw_train.bin > logs/smoke_qwen_stream.txt 2>&1
grep -E "^fp:|streaming|Error|Traceback" logs/smoke_qwen_stream.txt | tail -3 | sed 's/^/qwen_stream: /' >> logs/smoke.log
# the 27B job's training flags on the 0.8B (streaming GPTQ, 4 x 4 micro-batches, offload, bf16 tail, freeze then move)
mkdir -p checkpoints/sm_qw27flags
python -m bitnet.train --init_hf $M --init_hf_method gptq --init_stream --data data/wikiqw_train.bin --val data/wikiqw_val.bin \
  --data_u32 --seq_len 2048 --batch_size 4 --grad_accum 4 --ts_micro --steps 30 --warmup 5 --tail_fp32 --lr 5e-4 --min_lr 5e-5 \
  --eval_interval 15 --eval_iters 2 --save_secs 99999 --snap_every 0 --no_final_save --lookahead 0 --ckpt_skip 0 --rc_scale \
  --lowrank_mag add:16 --mag_wd 0.1 --mode kernel --int8 --dw_mode dense --ts --ts_theta 64 --ts_start 10 --ts_tau 1000 \
  --ts_anneal 0.75 --ts_tau_anneal --ts_nomom --ts_fused --gemm int8mm --lowrank 256 --ts_rank_s 256 --ts_vdiag \
  --ts_store_a fp12s --lean_tail --ts_offload_async --ts_offload_keep 0.7 --mem_cap_gib 15 --mem_report \
  --out_dir checkpoints/sm_qw27flags > logs/smoke_qw27flags.txt 2>&1
echo "$(date '+%T') qw27flags: $(grep -E 'streaming|^step +(0|10|20) |val loss|peakVRAM' logs/smoke_qw27flags.txt | tail -3 | tr '\n' ' ' | cut -c1-300) $(grep -cE 'Traceback|Error' logs/smoke_qw27flags.txt) errors" >> logs/smoke.log
echo "$(date '+%T') smoke done" >> logs/smoke.log
