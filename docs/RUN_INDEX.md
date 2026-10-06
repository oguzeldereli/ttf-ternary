# Run index

Running, queued, unfinished and never-run runs: [QUEUE.md](QUEUE.md) (kept current). Formulas of every flag:
[FORMULAS.md](FORMULAS.md). Figure of the best runs: [figures/best_runs.png](figures/best_runs.png).


**Final evaluation (3 Oct, fixed windows for every model; quote these):** Wikipedia val (974 windows) / FineWeb-Edu val
(2000 windows, never used for tuning): TTF ranks 512 (3 seeds) **2.6888 / 3.6453**; master + the same float extras
(2 seeds) 2.7166 / 3.6830; master (3 seeds) 2.7359 / 3.7071; old rule 2.7973 / 3.7545. TTF - master + extras
-0.028 [-0.029, -0.027] / -0.038 [-0.039, -0.037]; TTF - master -0.047 / -0.062; seed spread 0.002-0.0035. Storage
(seed 2): int8 / fp16 2.6877, fp8 / fp16 2.6874 vs fp32 2.6868. Bare TTF 2.8070 (+0.071 vs master). 1.3B (one seed each): TTF 2.5510 / 3.5009 vs master 2.5706 / 3.5489 (-0.020 [-0.021, -0.018] / -0.048). **340M** (one seed each): TTF p 0.5 2.5774 / 3.5251 vs master + extras 2.6061 / 3.5654 (-0.029 [-0.030, -0.027] / -0.040), master 2.6461 / 3.6167.
Training-time
val numbers of different seeds use different windows and differ by up to 0.06 for that reason alone (RUNS.md
"3 Oct 04:45").


## 5-6 Oct: no momentum at scale, step schedules (details in RUNS.md "6 Oct")

| run(s) | what | result |
|---|---|---|
| `x1b_nomom512_lr15` (Myriad) | 1.3B, no momentum, ranks 512, lr 1.5e-3 | **fixed windows 2.5012 / 3.4415** (training-time 2.5169); master + extras lr 7.5e-4 2.4980 (+0.003), lr 5e-4 fixed pending (training-time +0.022) |
| `x1b_nomom1024_lr15` (Myriad) | 1.3B, no momentum, ranks 1024 | **2.4956 at 9000, -0.004 vs master + extras lr 5e-4** (training-time); final pending |
| `x1b_mx_extras_lr50` (Myriad) | 1.3B master + extras at lr 5e-4 | **2.4949** training-time (lr 7.5e-4: 2.5142); fixed windows pending |
| `x1b_ts512_hold` / `x1b_mx_extras_lr75_hold` | 1.3B continued at the final lr | gap 0.054 -> 0.036 over 2600 steps (running) |
| `x1b_nm512_b2k_*` | 1.3B branches from 2000: p 1 / theta 10 -> 20 / constant leak / p 1 + floor 0.3 | +0.025 / +0.025 / +0.046 / +0.007 vs best master (running) |
| `big_ts512an05_nomom_s0` | 340M no momentum | 2.5861 training-time (fixed windows pending) |
| `sw512_*` / `sw192_*` (lab) | 110M step-schedule sweep (p, floors, p schedules, piecewise p, theta / tau schedules) | **nothing beats the base (p 0.5)**; mid-training gains are lost late; fixed windows: base 2.6720 (r512), 2.7748 (r192) |
| `x10b_ts512` (Myriad) | 10B, ranks 512, 98M tokens | 2.9911 training-time |

## 3-4 Oct: scale, checks, memory, low rank (details in RUNS.md "3 Oct night" onwards)

| run(s) | what | result |
|---|---|---|
| `x1b_ts512_la1000an05` vs `x1b_master_lr75` (Myriad) | 1.3B, fixed windows | **TTF 2.5510 vs master 2.5706 (-0.020 [-0.021, -0.018]); FineWeb 3.5009 vs 3.5489 (-0.048)**; LAMBADA 0.155 vs 0.138 |
| `mx_big_extras_s0` (harlequin) | 340M master + the same float extras | **2.6222** (TTF ranks 512 2.6268; TTF led by 0.075 at 5000) |
| `big_ts512an05_s0` / `big_ts512la1000an05_s0` (tope / quillback) | 340M TTF ranks 512, steps x ratio^0.5 / + leak scaled | **2.5927** / 2.6081 (training-time; master + extras 2.6222, TTF p 1 2.6268); fixed windows 2.5774 / 2.5916 vs master + extras 2.6061 (**-0.029 [-0.030, -0.027]**, FineWeb -0.040): p 0.5 fixes the late slope at 340M |
| `mx_lr075_s0` / `mx_lr3e3_s0` / `mx_lr6e3_s0` | master lr sweep | 2.7785 / 2.8136 / diverged (lr 1.5e-3 2.7513 is best) |
| `mx_2stage_s0` / `mx_2stage_lr3e3_s0` | master with a BitNet-style two-stage schedule | 2.7577 / 2.8153 (no gain) |
| `bop_g1e3_t02/t05/t15_s0` | ternary Bop baseline (one timescale) | 3.0885 / 3.9898 / 3.2451 (rerun 3.2407) |
| `tsbare_rs512an05_s0` | TTF without float extras | 2.8266 (+0.075 vs bare master) |
| `q*_s0`, `qfp16s_s0`, `qm8a16_seed2`, `qmf8a16_seed2` | storage formats | 8-bit direction + 16-bit accumulator free (seed 2 too); scaled fp16 2.7076 |
| `ts512an05_lean_s0` | full lean recipe (fused, int8 / fp16 state, bf16 / fp16 tail, LeanAdam) | **2.7020** (fp32 2.7067) |
| `ts{128,64}an05_{rescale,refresh}_s0` | low-rank mechanisms: rescaled threshold / basis refresh every 50 steps | rescale 2.8959 / 3.0056 (+0.012 / +0.013), refresh 2.8880 / 3.0007 (+0.004 / +0.008) vs 2.8837 / 2.9929 |
| `ts64an05_sched128to64_s0` | ranks 128 -> 64 over training (peak 128, mean 96) | 2.9148 (-0.078 vs ranks 64, +0.031 vs ranks 128) |
| `ts64an05_sched64to128_s0` | ranks 64 -> 128 | 2.9432 (-0.050 vs ranks 64, +0.059 vs ranks 128) |
| `ts64an05_table_s0` / `_xspend_s0` / `_tiles2_s0` | exact table 2% / exact spend / 2 x 2 tiles of rank 32 | 2.9891 / 2.9915 / 3.0216 (-0.004 / -0.001 / +0.029 vs 2.9929) |
| `ts128an05_xspend_s0` | exact spend at ranks 128 | **2.8727** (-0.011 vs 2.8837; ranks 64: -0.001) |
| `ts512an05_init10_s0` | ranks 512 + sparse trit init 10% | 2.7779 (+0.071 vs 2.7067) |
| `ts64an05_init10_s0` / `_init02_s0` / `_decay1e4_s0` / `_decay3e4_s0` | ranks 64 + sparse trit init (`--init_density` 0.1 / 0.02) / trit decay (`--ts_decay` 1e-4 / 3e-4) | 3.0452 / 3.0953 / 3.0282 / 3.0780 (+0.052 / +0.102 / +0.035 / +0.085 vs 2.9929) |
| `ts64an05_gate_s0` / `_sgate1_s0` / `_sgate2_s0` | ranks 64 + `--lr_gate` / per-weight sign counter K 1 / K 2 (`--ts_sgate`) | 2.9942 / 2.9937 / 2.9950 (+0.001 / +0.001 / +0.002 vs 2.9929: neutral) |
| `ts512an05_fast_s0` | 110M ranks 512 + `--ts_fused --ts_orth_defer --ts_orth_every 4 --gemm int8mm` | 2.7721 (+0.065 vs 2.7067: the speed modes cost quality) |
| `x1b_mx_extras_lr75` (Myriad 49713) | 1.3B master + the same float extras, lr 7.5e-4 | fixed windows **2.4980** (TTF ranks 512 2.5510: **TTF +0.053 [+0.051, +0.055]**); FineWeb 3.4584 (TTF +0.043); LAMBADA 0.143 (TTF 0.155) |
| `sw512_base` / `_p075` / `_tau2000` / `_tauanneal` / `_th1218` / `_p1` / `_th1020` / `_p025` | 110M sweep, ranks 512 no momentum | 2.6896 / 2.6940 / 2.6980 / 2.7011 / 2.7067 / 2.7147 / 2.7258 / 2.7658 (training-time; base best) |
| `x1b_mx_extras_lr50` (Myriad 53666) | 1.3B master + extras, lr 5e-4 | **2.4949** training-time (-0.019 vs lr 7.5e-4); fixed windows pending (56630) |
| `x10b_ts512` (Myriad 49814) | ~10B, ranks 512, batch 16, 3000 steps (98M tokens) | 2.9911 training-time |
| `sw192_base` / `_p1` / `_th1020` / `_th1218` / `_tauanneal` | 110M proxy sweep, ranks 192 no momentum (fused, int8 forward) | 2.7940 / 2.8312 / 2.8176 / 2.8014 / 2.7965 (training-time) |
| `ts512an05_nomom_s0` / `_seed2` | no short momentum, ranks 512 | fixed windows **2.6707 / 2.6693** (-0.019 [-0.020, -0.018] vs TTF ranks 512, 3 seeds; **-0.047 vs master + extras**; FineWeb -0.026 / -0.064); half the state |
| `ts512an05_fused_s0` / `_oe4_s0` / `_defer_s0` | speed ablation | 2.7082 / 2.7642 / 2.7619 (fused free; every 4 +0.058; deferred +0.055) |
| `x1b_master_lr50` | 1.3B plain master lr 5e-4 | fixed windows **2.4986** (TTF ranks 512 +0.052) |
| `ts64an05_nomom_s0` / `ts128an05_nomom_s0` | no short momentum, ranks 64 / 128 | **2.9724 / 2.8575** (-0.021 vs 2.9929 / -0.026 vs 2.8837; ranks 128 without momentum -0.135 vs ranks 64 at the same memory) |
| `x1b_ts256_la1000an05` (Myriad 49129) | 1.3B ranks 256 | fixed windows **2.6167** (vs ranks 512 2.5510 +0.066 [+0.064, +0.067], master 2.5706 +0.046 [+0.044, +0.048]); FineWeb 3.5739 (+0.073 / +0.025); LAMBADA 0.137 (0.155 / 0.138) |
| `ts128an05_xspend_seed2` | exact spend at ranks 128, seed 2 | 2.8534 (seed-2 training windows; fixed-window comparison pending) |
| `ts64an05_rsketch16_s0` / `_spendtab_s0` / `_sched512to64_s0` | residual-sketch refresh / spend only in a table / ranks 512 -> 64 | 3.0808 / 3.0009 / **2.7777** (+0.088 / +0.008 / -0.215 vs 2.9929; 512 -> 64 peaks at the ranks-512 state) |
| `ts512an05_global_s0` / `ts512an05_fuse_s0` / `ts128an05_fuse_s0` | one basis for the whole model (64% state) / q-k-v and gate-up share an input basis (84%) | 2.7505 / 2.7200 / 2.9172 (+0.044 / +0.013 vs 2.7067, +0.034 vs 2.8837); vs per-layer at the same memory -0.011 / -0.011 / -0.001 |
| `big_ts512an05_seed2` | 340M TTF p 0.5, seed 2 | 2.5866 (training-time); fixed windows 2.5765 / 3.5250 (seed 1337 2.5774 / 3.5251) |
| `mx_big_extras_lr1e3_s0` / `mx_big_extras_seed2` | 340M master + extras at lr 1e-3 / lr 1.5e-3 seed 2 | fixed windows 2.5837 / 3.5405 and 2.6009 / 3.5607: **TTF - master + extras lr 1e-3 -0.006 [-0.008, -0.005], FineWeb -0.015** |
| `big_ts512an05_lr1e3_{s0,seed2}`, `mx_big_extras_lr1e3_seed2` | 340M at lr 1e-3: TTF / master + extras seed 2 | fixed windows TTF 2.6133 / 2.6180 (worse than at 1.5e-3), master + extras 2.5919 / 3.5510; **each at its better lr, 2 seeds each: TTF - master + extras -0.0108 [-0.0120, -0.0097], FineWeb -0.0207** |
| `ts512an05_spend0_s0` (lamprey, 3 Oct) | TTF without spending after a move (`--ts_spend 0`) | 2.7748 (+0.068 vs spend theta 2.7067) |
| `ts512an05_kv128_s0` / `ts512an05_sharedV_s0` | k / v at rank 128 / shared basis | +0.023 / +0.047 (fixed windows); k, v / down stop moving |
| `ts{128,64}an05_*` fixes | sketch / residual / theta 8 / theta 11 / lr 3e-3 / ranks by input width | +0.17 / -0.03 / +0.14 / +0.05 / -0.02 / 0.00 vs the same-rank baseline (2.8837 / 2.9928) |
| `fwc_*` | ranks 512 / 128 / 64 at 1x / 2x / 4x compute, FineWeb-Edu | ranks 512 1x 3.2265; ranks 128 3.4156 / 3.2747 / **3.1793**; ranks 64 3.5391 / 3.4105 / 3.3090: ranks 128 matches ranks 512 at ~2.8x compute, ranks 64 ~8x |
| `w{110m,340m,1b,10b,15b}_r{64,256,512}` (15B running) | width sweep, batch 2, 1000 steps, shadows | top-64 energy share 0.57 / 0.54 / 0.53 / 0.58 at width 768 / 1024 / 2048 / 5120 (flat); 10B ranks 64 / 256 at 1000: 5.4370 / 5.4984; at batch 16 ranks 512 captures 0.96 / 0.91 / 0.77 of the energy at 110M / 340M / 1.3B |
| `lean_mem.sh`, `b27_smoke.sh` | measured memory | 110M 0.38-0.63 GiB, 340M 0.68-1.35, 1.3B 1.61-2.97, 10B 12.3, 15B 17.0, 26.5B 14.1 (ranks 256) |

## Best so far (110M, 300M tokens, from scratch, final validation loss / perplexity)

| run | what | val loss | ppl |
|---|---|---|---|
| `fp32_baseline` | full precision, fp32 + AdamW (reference) | 2.683 | 14.6 |
| `master_tracked` | master weights: fp32 latent + STE + AdamW (reference) | 2.751 | 15.7 |
| **`ts16fullla1000_s0`** | `--ts` (2 Oct): short momentum + accumulator (full rank), tau 1000 with the leak scaled by the lr, fire at theta 16 | **2.6746** | **14.5** |
| `ts16rs1024la1000_s0` | `--ts`, accumulator rank 512 + full-rank short momentum, annealed leak | 2.6968 | 14.8 |
| `ts16full_tau1000_s0` | `--ts`, full rank, tau 1000 | 2.6982 | 14.9 |
| **`ts16rs512tau1000an05_s0`** | `--ts`, both ranks 512, tau 1000, steps x (lr ratio)^0.5 | **2.7067** | **15.0** |
| `ts16rs512la1000_s0` / `_gate` | `--ts`, both ranks 512, tau 1000, annealed leak (/ + sign gate) | 2.7115 / 2.7116 | 15.1 |
| `ts16rs512tau1000_s0` / `_seed2` | `--ts`, both ranks 512, tau 1000, seeds 1 / 2 | 2.7328 / 2.7191 | 15.4 / 15.2 |
| `ts16rs512tau1000an05_seed2` | `--ts`, ranks 512, steps x ratio^0.5, seed 2 (seed 1 2.7067) | 2.6978 | 14.8 |
| `ts16full_tau1000_seed2` / `ts16fulltau1000an05_s0` | `--ts` full rank, seed 2 / with steps x ratio^0.5 | 2.6865 / 2.6804 | 14.7 / 14.6 |
| **`gvsharp_dryspend_r1024_s0`** | sharp base + beta 1 + dry friction 1/33 + spend 3 + rank 1024 (= full rank: every 110M matrix has a smaller side of 768) | **2.7998** | **16.4** |
| `gvsharp_dry_r1024_s0` / `_seed2` | sharp base + beta 1 + dry friction 1/33 + rank 1024, seeds 1 / 2 | 2.8198 / 2.8088 | 16.8 / 16.6 |
| `gvsharp_dryspend_r512_s0` | sharp base + beta 1 + dry friction 1/33 + spend 3 + rank 512 | 2.8215 | 16.8 |
| `gvsharp_b0995_r512_s0` | sharp base + decay 0.995 + rank 512 | 2.8339 | 17.0 |
| `gvsharp_dry_r512_s0` | sharp base + dry friction + rank 512 | 2.8356 | 17.0 |
| `gvsharp_dryspend_s0` | sharp base + dry friction + spend (rank 256) | 2.8699 | 17.6 |
| `gvsharp_rc_s0` | sharp base: gate + Adam step + row/col scales + additive r16 + adapter wd + head temperature | 2.9875 | 19.8 |

The sections below keep each phase's own unit: the first phases report perplexity, from 27 Sep on validation loss.

Fair reference (master + the same float extras, two seeds): **2.730** (2.7321 / 2.7279). `--ts` ranks 512 (two seeds, mean 2.702) is -0.026 below it.
Caveat on `--ts` vs master: our runs carry float extras master does not (row / column scales, a rank-16 additive float
adapter, per-head attention temperature). Alone (no flips) they reach only 3.1725, but master with the same extras
(`mx_extras_s0` / `_seed2`, running) is the fair reference. The `--ts` rows are one seed each except where noted.

Every directory under `checkpoints/` (gitignored). Unless noted: 110M (`small`), wiki32k,
seq 2048, 16 x 2048 = 32,768 tokens/step, same seed and batch order, kernel mode with
int8 forward. "val" = final validation perplexity. Details and discussion: [RUNS.md](RUNS.md).
Each dir has `metrics.jsonl` (+ `ckpt.pt`); logs are `train.log` in the dir, or
`checkpoints/<name>.log` for older runs.

## Ceiling

| run | what | tokens | val |
|---|---|---|---|
| `p2_baseline` | **master weights** (fp32 latent + STE + AdamW, LR 1.5e-3) | 300M | **15.8** |
| `probe_master` | master, 161 steps with probes (`--probe`) | 5M | 326.9 |
| `master_tracked` | master weights, the reference of every later section (val loss 2.751) | 300M | 15.7 |
| `curve_master` | master with window-curve tracking and snapshots every 1000 steps (stopped at 6350) | 208M | - |
| `fp32_baseline` | full precision fp32 + AdamW (val loss 2.683) | 300M | 14.6 |

## Stateless flips, full length (all bf16 float tail = frozen norm gains)

| run | what | tokens | val |
|---|---|---|---|
| `armA_cosine` | **reference**: flip rate 0.02 cosine-annealed to 0 | 300M | **98.6** |
| `armA_cos_600M` | reference, 600M-token schedule | 600M | 92.7 |
| `armA_cos_r005` | cosine from 4x lower rate (0.005) | 300M | 101.0 |
| `armA_linear` | linear anneal | 300M | 101.4 |
| `armA_cos_lockout` | cosine + per-weight flip lockout | 300M | 102.7 |
| `armA_cos_floor` | cosine to a 0.02% floor | 300M | 104.0 |
| `p3b_acc1` | constant rate | 300M | 135.8 |
| `p3b_acc4` | constant rate, 4x batch | 400M | 135.5 |
| `armA_cos_ef` | cosine + spatial error feedback | 300M | 156.0 |
| `armB_absscale` | frozen absolute threshold | 300M | 160.7 |
| **`armA_cos_la1`** | **look-ahead filter** (`--lookahead 1`), reference schedule; stopped at step 6171, resumable | 202M | **65.3** |

## Short probes and screens (1000 steps or fewer)

| run | what | tokens | val |
|---|---|---|---|
| `lr_ctl` | control for the 1000-step probes | 33M | 156.8 |
| `armA_cos_lr15` | 1.5x float-tail LR | 34M | 153.3 |
| `lo_once_t200`, `lo_once_t1000`, `lo_norev_t1000` | lockout variants | 33M | 175-191 |
| `ef_probe_a0*` | error-feedback strength probes | 20M | 256-828 |
| `gref1/7/10/25/100`, `gref10_r3x` | g_ref sweep (300-step alpha screens) | 10M | 350-490 |
| `rownorm`, `invmag`, `hump` | flip-rule shape screens | 10M | 350-431 |
| `ev3_screen` | 3-bit evidence counter (not stateless) | 10M | 554 |
| `rs_smoke` | greedy flip-rate search | 5M | 633 |
| `la1`, `la2` | look-ahead 1 and 2 passes, alpha screens | 10M | 280, 254 |
| `la_rwarm` | look-ahead, rate warmup 0.02 -> 0.04 (stopped at 790; snapshots every 100) | 26M | - |
| `la_r0ramp` | look-ahead, rate ramp 0 -> 0.02 over 305 steps (stopped at 313) | 10M | - |
| `la_fastramp_lr15` | look-ahead, rate ramp over 30 steps + master's tail LR (snapshots 100/200/300) | 10M | 180.7 |
| `la_fast_fp32tail` | same + **fp32 float tail** (`--tail_fp32`), probes | 10M | 179.2 |
| `la_fast_fp32tail_r04` | same, peak rate 0.04 | 10M | (running) |
| `probe_fast` | `la_fastramp_lr15` config, 161 steps with probes | 5M | 317.3 |

## Scratch / smoke tests (safe to delete)

`acc_a`, `acc_b`, `p2_master`, `sanity_master`, `smoke`, `t_armA`, `t_armB`,
`la_r0ramp_lr15` (stopped at step 20), `p3b_acc16`, `p3b_acc64` (never ran),
`wiki_ev`, `wiki_ev3`, `wiki_mf` (early 51M/GPT-2-vocab runs from RESULTS.md, checkpoint only).


## Low-rank momentum and cross-batch look-ahead (24-27 Sep)

Screens at 10M tokens report val loss; the MLP testbed is a ternary MLP language model (`scripts/mlp/mlp_lab.py`).

| run | what | tokens | val (loss) |
|---|---|---|---|
| `lm_frozen` | ternary layers frozen at their random init (flip rate 0) | 10M | 5.698 |
| `lm_plain` | stateless flips from the gradient | 10M | 5.684 |
| `la_xb2` | cross-batch look-ahead x2 (proposals from g) | 10M | 5.002 |
| `la_xb2_rc` | + row/column scales | 10M | 4.974 |
| `lm_lowrank256` | rank-256 momentum, no look-ahead (stalls at the unigram level) | 10M | 6.047 |
| `lm_lowrank256_r04` / `_adapt` | rate 0.04 / angle-scaled decay | 10M | 6.072 / 6.293 |
| `lm_lowrank256_xb2` | rank-256 momentum proposes, look-ahead x2 keeps | 10M | 4.892 |
| **`lm_lowrank256_xb2_100M`** | the same, full 300M schedule | 300M | **3.127** |
| `overnight_full` | `la_xb2` full schedule (stopped) | 130M | - |
| `magadd_full` | look-ahead + momentum + additive r4 float term | 300M | 3.082 |
| `mlp/mlp_master` / `mlp_lowrank64` / `mlp_lowrank256` / `mlp_sketch64` / `mlp_frozen` | MLP testbed (`checkpoints/mlp/`): master / rank-64 momentum / rank 256 / count sketch / frozen | 6000 steps | 4.351 / 4.670 / 4.577 / 4.804 / 5.066 |

## No look-ahead, momentum mechanisms, step size (27-28 Sep)

All no look-ahead unless noted; "branch" = continued from `nola_lab` at step 4000 (131M) with its momentum.
val = final validation loss (not perplexity) at the given tokens. Details: [RUNS.md](RUNS.md), status of running
ones: [QUEUE.md](QUEUE.md).

| run | what | tokens | val |
|---|---|---|---|
| `nola_lab` | rank-256 momentum, no look-ahead (4090 to 30M, lab 3090 to 300M) | 300M | 3.158 |
| `nola_then_la` | `nola_lab` to 131M, then look-ahead x2 on | 300M | 3.1225 |
| `la_sched` | look-ahead 0-30M, off to 131M, on after | 300M | 3.1108 |
| `la_sched98` | look-ahead 0-30M, off to 98M, on after | 300M | 3.1135 |
| `la40_off30` | look-ahead the first 40 steps, then off | 30M | 4.506 |
| `nola_b48` | no look-ahead, batch 48 (3 x 16: the text a look-ahead step reads) | 9155 steps (900M read) | 2.889 |
| `nola_add16` | no look-ahead + additive r16 (stopped: the adapter takes over q/k) | 112M | ~4.6 |
| `magadd16_qk_lab` | look-ahead + additive r16 + per-head temperature (to its planned 200M) | 200M | 3.1212 |
| **`magadd16_wd_qk_lab`** | look-ahead + additive r16 + adapter weight decay 0.1 + per-head temperature | 300M | **2.9925** |
| `mech_v1_b131` | V1 target point (`--mech v1`), branch | 300M | 3.1215 |
| `mech_user_b131` | your design (`--mech user`, gain 1), branch | 300M | 3.1202 |
| `mech_user_g0_b131` | your design without the move correction (gain 0), branch | 300M | 3.1325 |
| `mech_v1_s0` | V1 from scratch | 131M | 3.582 |
| `mech_user_s0` | your design (gain 1) from scratch | 131M | 3.505 |
| `mech_user_g0_s0` | your design (gain 0) from scratch | 131M | 3.548 |
| `small_step_b131` | 1/4 of the flip rate, branch | 300M | 3.1551 |
| `small_step_s0` | 1/4 of the flip rate, from scratch | 300M | 3.1933 |
| `accum33_b131` | accumulate 33 steps without flips, then flip a loss-chosen subset, branch | 300M | 3.2795 |
| `adaptrate_b131` | adaptive flip rate (`--adapt_rate`, keeps cos(g, M) near 0.03), branch | 300M | 3.1604 |
| `multibeta_b131` | adaptive momentum decay (`--multibeta 0.8,0.95,0.99`), branch | 300M | 3.1926 |
| `select_b131` | online flip selector (`--select`), branch; stopped | 229M | 3.366 |
| `rc_b131` | plain momentum + learned row/column scales (`--rc_scale`), branch | 300M | 3.1480 |
| `evid3_b131` | 3-bit evidence counter per weight (`--evidence_bits 3`) + row/column scales, branch (not stateless) | 300M | 3.0970 |
| `master_q4_b131` | master, latent stored at 4 bits (`--master_bits 4`), branch of `master_tracked` at 131M | 300M | 2.9231 |
| `master_q3_b131` | master, latent at 3 bits | 300M | 3.1222 |
| `master_q2_b131` | master, latent at 2 bits (3 levels: a stateless master) | 300M | 3.4686 |
| `rc_s0` | plain momentum + learned row/column scales, from scratch (4090) | 300M | 3.1307 |
| `rc_s0_seed2` | the same, seed 2 | 300M | 3.1273 |
| `speedref_rc_s0` | + per-layer slow speed reference (`--speed_ref 0.995`) | 300M | 3.1315 |
| `gate_rc_s0` / `gate_rc_s0_seed2` | + sign gate (`--lr_gate`, now also without look-ahead) | 300M | 3.1194 / 3.1138 |
| `rate085_rc_s0` | plain at 0.85x peak flip rate (the gate's rate control) | 300M | 3.1208 |
| `vnorm_rc_s0` | + factored Adam step (`--lr_vnorm 0.99`) | 300M | 3.1063 |
| **`gatevnorm_rc_s0`** / `gatevnorm_rc_s0_seed2` | + gate + Adam step | 300M | **3.0773 / 3.0699** |
| `rate064_rc_s0` | plain at 0.64x peak flip rate (gate + Adam step's rate control) | 300M | 3.1290 |
| `vnorm064_rc_s0` | Adam step at 0.64x peak rate, no gate | 300M | 3.1224 |
| `user_cap3_s0` | user's rule: `--lr_beta 1 --mom_ncap 3 --pfun_tanh 10` | 300M | 3.2168 |
| `user_cap1_s0` | user's rule, cap 1 (stopped: too few flips, loses no-context statistics) | ~115M | ~3.83 |
| `user_cap1_flip15_s0` | user's rule, cap 1, flip scale 1.5 (stopped) | 197M | 3.635 |
| `usersweep_cap{1,2,3,10,30}` | user's rule, cap sweep, stop at step 1500 | 49M | 4.132 / - / 3.991 / 4.946 / 5.305 |
| `grav05_rc_s0` | asymmetric gravity 0.5 (`--grav_up 0.5`) | 300M | 3.5013 |
| `grav05_gatevnorm_rc_s0` | asymmetric gravity 0.5 + gate + Adam step | 300M | 3.4973 |
| `gvundo_rc_s0` | gate + Adam step + undo (`--undo`); stopped (undo reverses 2% of flips: inert) | 66M | 3.585 |
| **`gvsharp_rc_s0`** | gate + Adam step + sharp (`--lowrank_mag add:16 --mag_wd 0.1 --qk_temp`): best without look-ahead, beats look-ahead + sharp (2.992) | 300M | **2.9875** |
| `gvsharp_b099_s0` | sharp base + `--lr_beta 0.99` | 300M | 2.9108 |
| `gvsharp_b1_s0` | sharp base + `--lr_beta 1` (relative flip rule, no cap) | 300M | 3.0466 |
| `gvsharp_r512_s0` | sharp base + `--lowrank 512` | 300M | 2.9627 |
| `gvsharp_slowgate_s0` | sharp base + `--slow_gate 64 --slow_beta 0.999` | 300M | 2.9615 |
| `gvsharp_dither_s0` | sharp base + `--dither_ld` (low-discrepancy flip draw) | 300M | 2.9908 |
| **`gvsharp_dry_s0`** | sharp base + `--lr_beta 1 --dry_vec 0.0303` | 300M | **2.8812** |
| `gvsharp_b1spend_s0` | sharp base + `--lr_beta 1 --spend 3` | 300M | 2.9426 |
| `gvsharp_b1spend_slowgate_s0` | sharp base + beta 1 + spend 3 + slow gate | 300M | 2.9708 |
| `gvsharp_b0995_s0` | sharp base + `--lr_beta 0.995` | 300M | 2.8783 |
| `gvsharp_dry01_s0` | sharp base + beta 1 + `--dry_vec 0.01` | 300M | 2.9558 |
| `gvsharp_dry1_s0` | sharp base + beta 1 + `--dry_vec 0.1` | 300M | 3.0031 |
| **`gvsharp_dryspend_s0`** | sharp base + beta 1 + dry 0.0303 + spend 3 | 300M | **2.8699** |
| `gvsharp_b099spend_s0` | sharp base + beta 0.99 + spend 3 | 300M | 2.9031 |
| `gvsharp_dryw_s0` | sharp base + beta 1 + `--dry_w 0.0303` (per weight) | 300M | 2.9409 |
| `gvsharp_b099slow_s0` | sharp base + beta 0.99 + slow gate | 300M | 2.9441 |
| **`gvsharp_dry_r512_s0`** | sharp base + beta 1 + dry 0.0303 + rank 512 | 300M | **2.8356** |
| `gvsharp_b0995spend_r512_s0` | sharp base + beta 0.995 + spend 3 + rank 512 | 300M | 2.8398 |
| **`gvsharp_dryspend_r512_s0`** | sharp base + beta 1 + dry 0.0303 + spend 3 + rank 512 | 300M | **2.8215** |
| `gvsharp_b0995_r512_s0` | sharp base + beta 0.995 + rank 512 | 300M | 2.8339 |
| `gvsharp_b0998_s0` | sharp base + beta 0.998 | 300M | 2.8910 |
| **`gvsharp_dry_r1024_s0`** | sharp base + beta 1 + dry 0.0303 + rank 1024 | 300M | **2.8198** |
| `gvsharp_b0995spend_s0` | sharp base + beta 0.995 + spend 3 | 300M | 2.8807 |
| `gvsharp_dry05_s0` | sharp base + beta 1 + dry 0.05 | 300M | 2.8875 |
| `gvsharp_dry02_s0` | sharp base + beta 1 + dry 0.02 | 300M | 2.8970 |

## What master cannot do without (master from scratch with one ingredient removed, 1-2 Oct; master 2.7513)

| run | what | tokens | val |
|---|---|---|---|
| `mx_factv` (Myriad) | factored (row x column) second moment | 300M | 2.7513 (+0.000) |
| `mx_rank512` (Myriad) | first moment at rank 512 | 300M | 2.7710 (+0.020) |
| `mx_snap` (Myriad) | a latent whose trit changes is set to the new trit's centre | 300M | **2.8225 (+0.071; = our rule r512)** |
| `mx_leak300` (Myriad) | the latent's offset forgets with tau 300 | 300M | **2.8479 (+0.097)** |
| `mx_leak1000` (lab121) | ... tau 1000 | 300M | 2.7298 (-0.021) |
| `mx_lag02` (4090) | trits follow the latent with prob. 0.02 / step (our rate-limited firing) | 300M | **3.5214 (+0.770)** |
| `mx_b1997b2` (lab121) | beta1 0.997 (beta2 0.999) | 300M | **3.2446 (+0.493)** |
| `mx_b2999` (lab121) | beta2 0.999 alone (control) | 300M | 2.7685 (+0.017) |
| `mx_b1997` (lab121) | beta1 0.997, beta2 0.95 | diverged | (beta1 > beta2 confound) |
| `mx_gate` (lab121) | our sign gate on master's update | 300M | **2.7325 (-0.019)** |

## Two timescales, threshold firing (`--ts`, 2 Oct; lab121 4070 Ti Super)

| run | what | tokens | val |
|---|---|---|---|
| `ts16_s0` / `ts8_s0` / `ts32_s0` | theta 16 / 8 / 32 (short momentum rank 128, accumulator rank 512, tau 300, spend theta) | 300M | 2.9415 / +0.107 @9000 / 3.1545 (vs master +0.190 / +0.107 / +0.403) |
| `ts16gate_s0` | theta 16 + sign gate | 300M | 2.9455 (+0.194) |
| `ts16tau1000_s0` | theta 16, tau 1000 | 300M | 2.8561 (+0.105) |
| `ts16r1024_s0` | theta 16, full-rank accumulator | 300M | 2.9292 (+0.178) |
| `ts16rs512_s0` | theta 16, short momentum rank 512 | 300M | **2.8054** (+0.054 vs master; -0.016 vs our rule r512) |
| `ts16sp2_s0` / `ts16rs512sp2_s0` | spend 2 theta (stay at the crossed boundary) / + short rank 512 | 300M | 2.9472 (+0.196) / 2.8138 (+0.063) |
| **`ts16rs512tau1000_s0`** | short rank 512 + tau 1000 | 300M | **2.7328 (-0.019 vs master; -0.089 vs our rule r512)** |
| `ts16rs1024tau1000_s0` (harlequin) / **`ts16rs512tau1000an05_s0`** (pintail) | full-rank short momentum + tau 1000 / short rank 512 + tau 1000 + milder annealing (lr ratio^0.5) | 300M | 2.7203 (-0.031) / **2.7067 (-0.045 vs master)** |
| **`ts16full_tau1000_s0`** (4090) | everything full rank, tau 1000 | 300M | **2.6982 (-0.053 vs master; -0.102 vs our full rank 2.7998)** |
| `ts16rs512tau1000an0_s0` (hake) | short rank 512 + tau 1000, no annealing (steps not scaled by the lr) | stopped @3360 | +0.108 @3250 |
| `ts16rs512la300_s0` / `ts16rs512la1000_s0` (albacore / dory) | short rank 512, leak annealed with the lr (`--ts_tau_anneal`), tau 300 / 1000 | 300M | 2.7495 (-0.002) / **2.7115 (-0.040)** |
| `ts16fullla1000_s0` / `ts16rs1024la1000_s0` (elver / quillback) | all full rank / full-rank short momentum, annealed leak tau 1000 | 300M | **2.6746 (-0.077)** / 2.6968 (-0.055) |
| `ts8rs512la1000_s0` / `ts16rs512la1000gate_s0` (flounder / inanga) | theta 8 / + gate, short rank 512, annealed leak tau 1000 | 300M | 2.7354 (-0.016) / 2.7116 (-0.040) |
| `ts16r256rs256tau1000_s0` / `ts16r128rs128tau1000_s0` (lamprey / goldeye) | best setting with both ranks 256 / 128 | 300M | 2.8136 (+0.062) / 2.9088 (+0.158) |
| `ts16r256rs512tau1000_s0` / `ts16r512rs256tau1000_s0` (tope / koi) | accumulator 256 + short 512 / accumulator 512 + short 256 | 300M | 2.7825 (+0.031) / 2.7815 (+0.030) |
| `ts16rs512tau3000_s0` / `ts24rs512tau1000_s0` / `ts16rs512tau1000_seed2` (barbel / hake / plaice) | tau 3000 / theta 24 / seed 2 | 300M | 2.7270 (-0.024) / 2.8103 (+0.059) / 2.7191 (-0.032) |
| **`big_ts16rs512tau1000_s0`** (4090) | the best setting at 340M (master 2.6626, our rule r512 2.7246) | 300M | **2.6268 (-0.036 vs 340M master)** |
| `ts16r64rs64tau1000_s0` / `ts16r128rs512tau1000_s0` (rudd / skate) | both ranks 64 / accumulator 128 + short 512 | 300M / running | 3.0183 (+0.267) / +0.026 @6000 |
| `ts16fulltau1000an05_s0` / `ts16rs512tau1000an075_s0` / `ts16r256rs256tau1000an05_s0` (harlequin / pintail / uaru) | full rank + annealing ^0.5 / rank 512 + ^0.75 / ranks 256 + ^0.5 | 300M | 2.6804 / 2.7138 / 2.7824 |
| `ts16full_tau1000_seed2` / `ts16rs512tau1000an05_seed2` (dory / flounder) | seed 2 of the two headline runs | 300M | 2.6865 / **2.6978** |
| `ts16fullla1000_seed2` / `ts16rs512la1000_seed2` (goldeye / barbel) | seed 2 of the annealed-leak runs | 300M | **2.6645** / 2.7002 |
| `ts16rs512la1000an05int8_s0` / `ts16r256rs256la1000an05int8_s0` (plaice / hake) | the same in int8 (`--ts_int8`) with the fused update (`--ts_fused`) | 300M | 2.7631 / 2.8170 |
| `ts16rs512la1000an05_s0` / `ts16r256rs256la1000an05_s0` (albacore / elver) | ranks 512 / 256, leak and steps scaled by (lr ratio)^0.5 | 300M | 2.7224 / 2.7827 |
| `x1b_ts512_la1000an05` (Myriad, running) | `--ts` at 1.3B, ranks 512, leak and steps x ratio^0.5, lr 7.5e-4 (1.3B master 2.5868; old rule final 2.7089, +0.122) | 164M (5000) | 2.802 (-0.032 vs master at 5000) |
| `mx_extras_s0` / `mx_extras_seed2` (tope / lamprey) | **fair reference**: master + the same float extras (row/col scales, additive rank-16 adapter, qk temperature) | 300M | **2.7321 / 2.7279 (mean 2.730)** |
| storage formats (`q*_s0`, 19 runs, fused, vs fp32 twin 2.7067; noise ~+-0.005) | free: direction int8 / fp8 with a per-column scale, accumulator fp16 / bf16 / fp16 with a per-column scale, bf16 everywhere; costs: accumulator int8 (+0.012 to +0.021), plain fp16 on the direction (+0.011, underflow: scaled fp16 +0.001), accumulator V in int8 with U fp16 (+0.014 at 7250); broken: per-row int8 +0.050, fp8 everywhere +0.255 (RUNS.md "3 Oct night") | 300M | int8 / fp16 **2.7087**, fp8 / fp16 **2.7052**, scaled fp16 2.7076 |
| `tsnoflip_s0` | control: theta 1e9 (no flips; float extras only) | 300M | 3.1725 (+0.421) |

## Recipe tweaks (1-2 Oct, finished; vs the rule r512 2.8215)

| run | what | tokens | val |
|---|---|---|---|
| `undo512_dryspend_s0` | + undo | 300M | 2.8360 (+0.015) |
| `ours_vfull_r1024` (Myriad) | full rank + per-weight v (`--lr_vfull`) | 300M | 2.8019 (full rank factored 2.7998: no gain) |
| `drywarm512_dryspend_s0` | memory that starts short and lengthens | 300M | 2.8199 (-0.002) |
| `undog512_dryspend_s0` | undo on the batch gradient alone | diverged | stopped @1700 |
| `fast512_dryspend_s0` | saturated weights fire 10x faster | 300M | 2.8201 (-0.001) |

## Flip selection (1 Oct)

| run | what | tokens | val |
|---|---|---|---|
| `flat512_dryspend_s0` | dry + spend, rank 512, `--pshape flat` (sign only, same flip count) | 300M | 2.8924 (+0.071 vs the rule 2.8215) |
| `inv512_dryspend_s0` | the same, `--pshape inv` (prefer small push) | 300M | 2.9867 (+0.165) |
| `cheap512_dryspend_s0` | the same, `--pshape cheap` (prefer small gradient second moment) | 300M | 2.9186 (+0.097) |
| `cheap256_dryspend_s0` | rank 256, `--pshape cheap` | 300M | 2.9469 (+0.077 vs 2.8699) |
| `cheap128_dryspend_s0` / `cheap64_dryspend_s0` | rank 128 / 64, `--pshape cheap` | 300M | 3.0045 / 3.0800 (+0.054 vs 2.9501 / +0.040 vs 3.0401) |
| `cheapcos512_dryspend_s0` | rank 512, flip selection blended from the rule to cheap on a cosine (`--pshape cheap --pshape_sched cos`, 4090) | 300M | 2.8634 |
| `drywarm512_dryspend_s0` | dry + spend, rank 512, memory that starts short and lengthens (`--dry_start 0.12 --dry_warm 1500`: friction 0.12 easing to 1/33 over 49M tokens) | 300M | 2.8199 |
| `undo512_dryspend_s0` | dry + spend, rank 512 + undo (`--undo`): undo with the long-memory recipe | 300M | 2.8360 |

Every selection that moves flips off the largest momentum loses in training, at every rank; see RUNS.md "1 Oct 19:40".

## Running (30 Sep, 14:40)

| run | what | tokens | val |
|---|---|---|---|
| `gvsharp_dry_r1024_seed2` | dry + rank 1024, seed 2 (`--seed 2`) | 300M | 2.8088 |
| **`gvsharp_dryspend_r1024_s0`** | sharp base + beta 1 + dry 0.0303 + spend 3 + rank 1024 | 300M | **2.7998** |
| `gvsharp_dryspend_r1024_seed2` | the best run, seed 2 | 300M | 2.7864 |
| `gvsharp_dryspend_r512_seed2` | dry + spend + rank 512, seed 2 | 300M | 2.8093 |
| `gvsharp_dryspend_seed2` | dry + spend + rank 256, seed 2 | 300M | 2.8639 |
| `rk64_refresh_dryspend_s0` | dry + spend, rank 64 + subspace refresh (`--lr_refresh 8`) | 300M | 3.0434 |
| `rk128_int8_dryspend_s0` | dry + spend, rank 128, momentum rounded to int8 each step (`--mom_int8`) | 300M | 2.9541 |
| `gvsharp_dry04spend_r1024_s0` | dry 0.04 + spend + rank 1024 | 300M | 2.8156 |
| `rare256sum_dryspend_s0` | dry + spend, rank 256 + a rare-pattern momentum (rank 256, fed the gradient outside the main subspace, dry 0.01), summed into the flip signal (`--rare_rank 256 --rare_mode sum`) | 300M | 3.0748 |
| `rare256w03_dryspend_s0` | rank 256 + rare rank 256 summed at 0.3x the main one's size (`--rare_weight 0.3`) | 300M | 2.8726 |
| `own3000_q_dryspend_r512` | our rank-512 run's step-3000 checkpoint continued at 1/4 of the flip rate | 300M | 2.8410 |
| `rk256_int8_dryspend_s0` | dry + spend, rank 256, int8 momentum | 300M | 2.8884 |
| `tier4x64flip_dryspend_s0` / `tier4x64sum_dryspend_s0` | dry + spend, rank 64 + 3 rank-64 tiers (`--tiers 64:0.016,64:0.009,64:0.005`), own flips at 0.33x / summed | 300M | 3.0795 / 3.1726 |
| `rare256flip_dryspend_s0` | the same, but the rare momentum proposes its own flips at 0.5x the rate (`--rare_mode flip`) | 300M | 2.9354 |
| `rare256same_dryspend_s0` | rank 256 + a second rank 256 on the residual with the same memory (dry 1/33), summed: the equal-memory split vs rank 512 | 300M | 2.9657 |
| `branch_m3000_dryspend_r512` | master @3000 converted to trits (`master_to_kernel.py`), continued with dry + spend, rank 512 | 300M | 2.8091 |
| `branch_m3000_q_dryspend_r512` | the same at 1/4 of the flip rate (`--rate_peak 0.005`) | 300M | 2.8197 |
| `rk512_int8_dryspend_s0` | dry + spend, rank 512, int8 momentum | 300M | 2.8346 |

## Rank sweep and model width (30 Sep, running)

Recipe = sharp base + beta 1 + dry 0.0303 + spend 3. Does it survive a rank that fits 27B in 12-16 GB, and does the
rank needed grow with width? 340M = `--preset d1024_l24`.

| run | what | tokens | val |
|---|---|---|---|
| `rk32_dryspend_s0` / `rk64_dryspend_s0` / `rk128_dryspend_s0` | 110M, rank 32 / 64 / 128 (rank 256: 2.8699, 512: 2.8215) | 300M | 3.1369 / 3.0401 / 2.9501 |
| `big_rk64_dryspend_s0` / `big_rk128_dryspend_s0` | 340M, rank 64 / 128 (lab) | 300M | 2.9121 / 2.8677 |
| **`big_dryspend_r512_s0`** | 340M, dry + spend, rank 512 (4090) | 300M | **2.7246** (ppl 15.3) |
| **`big_master`** | 340M master weights (4090) | 300M | **2.6626** (ppl 14.3) |
| **`big_dryspend_r1024_s0`** | 340M, dry + spend, rank 1024 (= full rank) (4090) | 300M | **2.6905** (ppl 14.7) |
