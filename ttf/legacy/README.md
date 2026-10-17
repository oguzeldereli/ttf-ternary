# ttf.legacy: the earlier methods, frozen

A byte-for-byte copy of the package as it was at tag `pre-consolidation-2026-10-17` (then called `bitnet`), kept so the
methods that preceded the current TTF recipe stay runnable. It is never edited: fixes and new work go into the core
(`ttf/`). Its trainer is the old one with all of its options (`python -m ttf.legacy.train --help`); checkpoints it writes
pickle `ttf.legacy.config.ModelConfig`.

The 110M setting every method below was run in (Wikipedia, Llama 32k tokens, 300M tokens = 9155 steps x 16 x 2048):
```
B="--preset small --mode kernel --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16
   --grad_accum 1 --steps 9155 --warmup 305 --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30
   --g_ref 3.0 --int8 --dw_mode dense --tail_fp32 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30"
```

| method | command | recorded (training-time val unless noted) |
|---|---|---|
| stateless stochastic flip rule (the first phase: flips sampled from the gradient, no memory) | `python -m ttf.legacy.train $B --lookahead 0 --out_dir runs/flip_rule` | see docs/FORMULAS.md section 5 |
| low-rank momentum rule (dry friction + spend, momentum rank 512) | `python -m ttf.legacy.train $B --lookahead 0 --lowrank 512 --lr_beta 1 --dry_vec 0.0303 --spend 3 --lr_gate --lr_vnorm 0.99 --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --out_dir runs/momentum_rule` | `gvsharp_dryspend_r512_s0` 2.8215 (rank 1024: 2.7998) |
| look-ahead filter (best: two re-check passes on a rank-256 momentum with the float adapter) | `python -m ttf.legacy.train $B --lookahead 2 --lowrank 256 --rc_scale --lowrank_mag add:16 --out_dir runs/lookahead` | 2.992 (25-27 Sep screen) |
| TTF with short momentum (two-timescale rule before `--ts_nomom`) | `python -m ttf.legacy.train $B --lookahead 0 --ts --lowrank 512 --ts_rank_s 512 --ts_theta 16 --ts_tau 1000 --ts_anneal 0.75 --ts_tau_anneal --ts_fused --gemm int8mm --rc_scale --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --out_dir runs/ttf_momentum` | `ts16rs512tau1000an05_s0` 2.7067 (p 0.5; fixed windows of the 3-seed set: 2.6888) |

The September runs did not keep their full command lines: the flags above are those documented for each method in
docs/FORMULAS.md (sections 2, 2a, 4, 5), checked on 17 Oct to run (30 steps each on a lab PC). They reproduce the
method, not necessarily every number to the last digit; the exact code of each historical run is in the git history,
its recorded results in the results database (`python scripts/results.py show NAME`).

Known issue (found 17 Oct, fixed in the core, kept here as it was): with `--ts_fire_fused` the firing kernel writes the
packed trits in place without bumping the tensor's version, so each layer's cached beta (1 / sqrt(K rho)) is not
recomputed after moves until the model is reloaded. Runs with `--ts_fire_fused` trained with the beta of their start.
