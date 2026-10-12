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
echo "$(date '+%T') smoke done" >> logs/smoke.log
