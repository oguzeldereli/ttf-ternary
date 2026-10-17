"""The trainer's command line (python -m ttf.train --help)."""
import argparse

from .config import PRESETS, DEFAULT_PRESET


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--mode", default="kernel",
                    choices=["kernel", "master", "fp32"],
                    help="kernel = packed trits on the Triton GEMM (trained by TTF with --ts); master = BitNet b1.58 "
                         "(latent weights + STE + AdamW); fp32 = the full-precision model")
    ap.add_argument("--master_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="master mode: latent weight dtype (bf16 rounds away small "
                         "AdamW steps; fp32 is the honest ceiling)")
    ap.add_argument("--lowrank", type=int, default=0,
                    help="flip signal = rank-r momentum of each layer's gradient (M ~ U V^T, "
                         "r*(N+K) floats per layer) instead of the current gradient")
    ap.add_argument("--profile", type=int, default=0,
                    help="profile this many steps (after 20 warm-up steps), print the tables, exit")
    ap.add_argument("--qk_temp", action="store_true",
                    help="learnable per-head attention temperature (log-scale per head, no weight decay)")
    ap.add_argument("--lowrank_mag", default="",
                    help="low-rank float magnitude beyond the trits: 'add:R' (W = bT + AB^T) or 'mul:R' "
                         "(W = bT o (1 + AB^T))")
    ap.add_argument("--mag_wd", type=float, default=0.0, help="weight decay of the --lowrank_mag parameters")
    ap.add_argument("--mag_lr_mult", type=float, default=1.0, help="LR multiplier of the --lowrank_mag parameters")
    ap.add_argument("--ts", action="store_true",
                    help="two timescales (ts_step): short low-rank momentum for the direction, long low-rank accumulator "
                         "of Adam-normalized steps, a trit moves when the accumulator crosses theta (no flip rate)")
    ap.add_argument("--ts_theta", type=float, default=16.0, help="--ts: firing threshold (latent half-bin / peak lr)")
    ap.add_argument("--mem_cap_gib", type=float, default=0.0, help="cap PyTorch's memory on the GPU at this many GiB "
                    "(memory-matched comparisons; 0: no cap)")
    ap.add_argument("--init_hf", default="", help="start from a pretrained Llama checkpoint in Hugging Face format (a "
                    "directory with config.json and safetensors); the model shape comes from its config (--preset is ignored). "
                    "See ttf/hf_init.py")
    ap.add_argument("--init_hf_method", default="als", choices=["absmean", "row", "als", "gptq", "gptqc", "gptqs", "zero", "scratch", "native", "random"],
                    help="--init_hf, kernel mode: how the weights become trits and scales (ttf/hf_init.py)")
    ap.add_argument("--distill", default="", help="a full-precision teacher (Hugging Face Llama dir): the loss is "
                    "(1 - W) CE + W KL(teacher || student) per token (--distill_w W)")
    ap.add_argument("--distill_w", type=float, default=1.0, help="--distill: weight of the KL term (1: KL only)")
    ap.add_argument("--kd_cache", default="", help="distil from precomputed teacher targets (scripts/quant/teacher_topk.py: fixed windows of --data with the teacher's top-k log-probabilities): each micro-batch takes the next window in order; the loss is (1 - --distill_w) CE + --distill_w x cross-entropy to the renormalised top k (Qwen models, batch_size 1)")
    ap.add_argument("--kd_mode", default="renorm", choices=["rest", "renorm"], help="--kd_cache: rest = the top k plus one bucket for the teacher's remaining mass (the KL on that distribution); renorm = the top k renormalised (drops the tail)")
    ap.add_argument("--kd_cache_start", type=int, default=0, help="--kd_cache: the training step that reads window 0 (the step a run is resumed at when the cache is introduced)")
    ap.add_argument("--moe", default="", help="E:K[:H]: mixture-of-experts feed-forward, E ternary SwiGLU experts (hidden H, default "
                    "the dense one), top-K routing (ttf/model.py MoEFeedForward)")
    ap.add_argument("--moe_aux", type=float, default=0.01, help="--moe: weight of the load-balancing loss")
    ap.add_argument("--init_stream", action="store_true", help="--init_hf (Qwen): convert layer by layer, sequentially, reading "
                    "each layer's weights from the safetensors (27B: no full-precision copy in memory)")
    ap.add_argument("--data_u32", action="store_true", help="--data / --val are uint32 token streams (vocabularies above 65536)")
    ap.add_argument("--hid_kd", type=float, default=0.0, help="--distill: W > 0 adds W x the error of every block's "
                    "update against the teacher block's (mean and varying parts, each relative), along the student's "
                    "own path, to the global loss (more targets per token than the logits)")
    ap.add_argument("--init_calib", type=int, default=64, help="--init_hf_method gptq: calibration windows "
                    "(of --seq_len tokens) for the input second moments")
    ap.add_argument("--ts_theta_sched", default="", help="--ts: T0:T1[:START], the threshold is T0 until fraction START of training (default 0), then moves "
                    "linearly to T1 at the end")
    ap.add_argument("--ts_start", type=int, default=0, help="--ts: the trits stay frozen until step S (no evidence is kept "
                    "and nothing fires; the floats and the second moment train): calibrate a converted model first")
    ap.add_argument("--ts_vdiag_every", type=int, default=1, help="--ts_vdiag on every N-th step only (it is "
                    "logging alone, but its full-size passes and host reads cost ~28k GPU stalls per 27B step)")
    ap.add_argument("--ts_vdiag", action="store_true", help="--ts: log per step the gradient's size against the second "
                    "moment it is divided by (g^2 / v, mean and max over layers) and the |u| tail")
    ap.add_argument("--ts_nomom", action="store_true", help="--ts: no short momentum; the accumulator takes the "
                    "full-rank g / sqrt(v) each step (the momentum's state freed)")
    ap.add_argument("--ts_spend", type=float, default=1.0,
                    help="--ts: a move spends this x theta (1: to the new trit's centre, like master --m_snap; 2: stays at "
                         "the boundary it crossed, like master)")
    ap.add_argument("--gemm", default="triton", choices=["triton", "cublas", "int8mm"],
                    help="ternary GEMMs: our Triton kernels, or cuBLAS bf16 on trits unpacked per call (faster on A100)")
    ap.add_argument("--ts_store_a", default="", help="--ts: REAL storage of the accumulator (same formats, and fp12s: "
                    "fp16s rounded to 6 mantissa bits, two values in 3 bytes)")
    ap.add_argument("--tern_emb", default="", help="tied | untied: ternary token embedding (and output head) with "
                    "per-row / per-column float scales; kernel mode trains its trits with TTF after the backward, master "
                    "mode keeps a float latent (STE) (ttf/tern_emb.py)")
    ap.add_argument("--lr_linear", action="store_true", help="after warmup, a linear decay from the peak lr to --min_lr "
                    "instead of the cosine (TTF's step scale follows it through the lr ratio)")
    ap.add_argument("--lr_wsd", type=int, default=0, help="warmup-stable-decay: the peak lr until step D (this), then a "
                    "cosine to --min_lr at the last step (0 = off); branch a decay off any stable-phase checkpoint")
    ap.add_argument("--lr_rewarm", default="", help="S:W, continue a decayed run: from step S a linear warmup over W steps "
                    "from --min_lr back to the peak, then a cosine to --min_lr at the last step")
    ap.add_argument("--lean_batched", action="store_true", help="--lean_tail: the low-precision 1-D parameters as one "
                    "flat pass and the 2-D ones stacked by shape (the same arithmetic; ~10x fewer kernel launches)")
    ap.add_argument("--lean_tail", action="store_true", help="float tail in bf16 (embedding, adapter) / fp16 (gains, "
                    "scales) with LeanAdam: int8 first moment, factored second moment, stochastic-rounding updates")
    ap.add_argument("--init_mag_svd", type=int, default=0, help="--init_hf gptq / gptqc with --lowrank_mag add:R: the "
                    "float adapters start from the best rank-R part of each layer's rounding error (in the input "
                    "covariance's norm) instead of 0: the lost precision moved into the float part (hf_init._mag_svd_init)")
    ap.add_argument("--ts_fire_fused", action="store_true", help="--ts_factored: the firing (A = U V^T, legal moves, "
                    "repack, the spend's D V) in one kernel (ttf.kernel.ttf_fire): no [N, K] temporaries")
    ap.add_argument("--ts_acc_scale", type=float, default=1.0, help="--resume: multiply every TTF accumulator by F (a "
                    "run continued at theta F x its old theta keeps each entry's distance to the bar: no burst of moves)")
    ap.add_argument("--fused_norm", action="store_true", help="Qwen: the RMSNorms (not the gated ones) through one fused "
                    "kernel forward and one backward (the same formula in fp32; reduction order differs)")
    ap.add_argument("--fused_scales", action="store_true", help="kernel layers with row / column scales and the 'add' "
                    "adapter: the column scale's backward and the adapter add + row scale (forward and backward) in fused "
                    "kernels (the same per-element roundings; the scale gradients' sums in another order)")
    ap.add_argument("--ts_factored", type=int, default=0, help="--ts --ts_nomom --ts_fused: K > 0: the TTF update from "
                    "the factors of the weight gradient (g = gy^T x is never formed; the sums of g^2 from a K-column "
                    "sketch; see _ts_layer_factored); --ts_vdiag keeps only g^2 / v")
    ap.add_argument("--ts_micro", action="store_true", help="--ts_fused with --grad_accum M: the TTF update sees each "
                    "micro-batch's gradient (momentum and v step per micro-batch with beta^(1/M), v / M; accumulator, firing "
                    "and spend once per step), so no dense gradient is kept between micro-batches: activations shrink M-fold")
    ap.add_argument("--ckpt_block", action="store_true", help="checkpoint each block once (its input) instead of attention "
                    "and MLP separately: half the saved activations (kernel mode; master builds dense weights per sublayer)")
    ap.add_argument("--no_final_save", action="store_true", help="no checkpoint at the end of the run (memory and "
                    "speed probes of large models)")
    ap.add_argument("--mem_snapshot", default="", help="PATH:STEP: record every CUDA allocation (with its Python stack) "
                    "from the start and dump the history to PATH (a pickle) at logged step STEP; scripts/analysis/mem_snapshot.py "
                    "attributes the peak")
    ap.add_argument("--eval_first_n", type=int, default=0, help="before training: the loss over the first N val windows in "
                    "order (qwen_check.py's windows), e.g. a conversion against the full-precision reference")
    ap.add_argument("--mem_report", action="store_true", help="log the MiB of trits, float tail, its gradients and "
                    "optimizer state, TTF state on GPU / host, allocated and peak, at steps 10 / 50 / 100 / 1000")
    ap.add_argument("--ts_offload", action="store_true", help="--ts: keep the state in pinned host memory; each layer's "
                    "state is copied to the GPU for its update and back (same stream: ordered, no host sync)")
    ap.add_argument("--ts_offload_keep", type=float, default=0.0, help="--ts_offload(_async): the state of the first "
                    "share of the ternary layers (module order) stays on the GPU, the rest in host memory (fill a GPU's "
                    "spare memory with state; 0 = all on the host)")
    ap.add_argument("--ts_fused", action="store_true",
                    help="--ts: each layer's update runs inside the backward, right after the layer's input gradient, so "
                         "the fp32 gradient of only one layer exists at a time (saves N*K*4 bytes over all layers)")
    ap.add_argument("--ts_anneal", type=float, default=1.0,
                    help="--ts: the accumulator's steps scale with (lr_t / lr_peak)^P (1: like master's latent; 0.5 milder)")
    ap.add_argument("--ts_tau_anneal_pow", type=float, default=1.0, help="--ts_tau_anneal: the leak per step is "
                    "a^Q / tau (Q = this; a = the step scale), so the memory grows as a^-Q")
    ap.add_argument("--ts_tau_anneal", action="store_true",
                    help="--ts: the leak shrinks with the accumulator's steps (decay 1 - ratio / tau): memory counted in "
                         "distance travelled, so the accumulator's level does not sink below theta as the lr decays")
    ap.add_argument("--ts_tau", type=float, default=300.0, help="--ts: the accumulator's leak time constant (steps)")
    ap.add_argument("--ts_b2", type=float, default=0.95, help="--ts: decay of the factored second moment")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for fp32 matmuls (the low-rank momentum's products); A100 fp32 is 19.5 TFLOPs, TF32 156")
    ap.add_argument("--int8", action="store_true",
                    help="kernel/evidence: s8 tensor-core path (exact int32 forward, "
                         "8-bit gy in dx, int8 dw)")
    ap.add_argument("--int8_dx", action="store_true",
                    help="also run grad_x on the s8 kernel (default: bf16 dx kernel)")
    ap.add_argument("--stop_after", type=int, default=None,
                    help="stop after this many steps WITHOUT changing the LR / flip "
                         "schedules (which still span --steps); evaluates on exit")
    ap.add_argument("--snap_every", type=int, default=0,
                    help="also keep a copy of the checkpoint every N steps (ckpt_<step>.pt)")
    ap.add_argument("--tail_fp32", action="store_true",
                    help="float tail (embeddings, norm gains) in fp32 + AdamW, as in master "
                         "mode (default bf16 + 8-bit Adam freezes the norm gains at 1.0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="run seed (init, data order, look-ahead batches); default TrainConfig.seed = 1337")
    ap.add_argument("--track_flips", action="store_true",
                    help="log per-layer flip rate + never-changed fraction each step")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--extend_to", type=int, default=0, help="train until this step: after --steps the schedule's lr stays "
                    "at --min_lr (continuing a finished run from its checkpoint with --resume)")
    ap.add_argument("--weight_decay", type=float, default=None, help="weight decay of the float tail's embedding / adapter group (default 0.1)")
    ap.add_argument("--min_lr", type=float, default=None,
                    help="cosine floor (default: lr/10)")
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--eval_iters", type=int, default=None)
    ap.add_argument("--loss_chunk", type=int, default=2048,
                    help="tokens per output-head chunk (peak VRAM knob; math is identical)")
    ap.add_argument("--save_secs", type=float, default=900.0,
                    help="wall-clock seconds between checkpoint saves")
    ap.add_argument("--max_temp", type=int, default=86,
                    help="save + stop if GPU temp (C) reaches this (crash guard)")
    ap.add_argument("--resume_temp", type=int, default=85,
                    help="after a thermal pause, wait until GPU temp (C) is at or below "
                         "this before resuming")
    ap.add_argument("--rc_scale", action="store_true",
                    help="learned per-row and per-column scales on every ternary layer "
                         "(W_ij = beta * r_i * c_j * q_ij; N + K floats per layer, fp32, no decay)")
    ap.add_argument("--rc_lr", type=float, default=0.0,
                    help="constant LR for the row/column scales (no warmup/decay); 0 = use the "
                         "float-tail LR schedule")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the elementwise parts (RMSNorm, RoPE, SwiGLU)")
    ap.add_argument("--pipe", action="store_true", help="Qwen, under torchrun --nproc_per_node G: pipeline parallel over G "
                    "GPUs (stage r runs a contiguous block of decoder layers; only hidden states cross stages, nothing is "
                    "all-reduced); every stage reads the same batches. Checkpoints: one per stage (OUT_DIR, OUT_DIR_rank1, "
                    "...: scripts/pp_merge.py joins them); a resume reads its own stage's file, else OUT_DIR/ckpt.pt (a "
                    "single-GPU checkpoint or stage 0's)")
    ap.add_argument("--pipe_split", default="", help="--pipe: decoder layers per stage, e.g. 17,16,16,15 (default even)")
    ap.add_argument("--ckpt_skip", type=int, default=0,
                    help="leave the last K layers without gradient checkpointing (faster, "
                         "more VRAM; ~1.3 GiB per layer at 16x2048 tokens)")
    ap.add_argument("--temp_check", type=int, default=4,
                    help="check GPU temp every N steps")
    return ap
