"""The trainer's command line: python -m ttf.train [--recipe NAME] [options] (python -m ttf.train --help).

One option per concept, the measured best as its default. A recipe (recipes/NAME.toml, or a path) sets the defaults for
a setting (model size, conversion, scale); options given on the command line override it. The resolved options are
written to the run's records (results database) and to <out_dir>/records.jsonl.

from_legacy() translates a command line of the trainer before 17 Oct 2026 (the flags of tag pre-consolidation-2026-10-17,
e.g. a config in the results database) into these options; docs/OPTIONS_MAP.md lists the correspondence.
"""
import argparse
import os
import tomllib

from .config import PRESETS, DEFAULT_PRESET

RECIPES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")
ON = argparse.BooleanOptionalAction


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m ttf.train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--recipe", default="", help="recipes/NAME.toml or a path: defaults for a setting")

    g = ap.add_argument_group("run")
    g.add_argument("--mode", default="ttf", choices=["ttf", "bitnet", "fp32"],
                   help="ttf: packed trits trained by TTF; bitnet: BitNet b1.58 (latent weights, STE, AdamW); "
                        "fp32: the full-precision model")
    g.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS), help="model shape (without --init-hf)")
    g.add_argument("--out-dir", default=None)
    g.add_argument("--resume", action="store_true", help="continue from <out-dir>/ckpt.pt if it exists")
    g.add_argument("--seed", type=int, default=None, help="init and data order (default TrainConfig.seed, 1337)")
    g.add_argument("--steps", type=int, default=None, help="the schedule's length (default TrainConfig)")
    g.add_argument("--stop-after", type=int, default=None, help="stop at this step without changing the schedules")

    g = ap.add_argument_group("data")
    g.add_argument("--data", default=None, help="flat token-id file (train)")
    g.add_argument("--val", default=None, help="flat token-id file (validation)")
    g.add_argument("--data-dtype", default="u16", choices=["u16", "u32"], help="u32: vocabularies above 65536 (Qwen)")
    g.add_argument("--seq-len", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=None, help="sequences per micro-batch")
    g.add_argument("--accum", type=int, default=None, help="micro-batches per step")

    g = ap.add_argument_group("optimisation (the float parameters; TTF's steps follow the same schedule)")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--min-lr", type=float, default=None, help="the schedule's floor (default lr / 10)")
    g.add_argument("--warmup", type=int, default=None)
    g.add_argument("--lr-schedule", default="cosine",
                   help="cosine | linear | wsd:D (warmup, the peak lr until step D, then a cosine to --min-lr)")
    g.add_argument("--lr-rewarm", default="", help="S:W[:F]: from step S a linear warmup over W steps from --min-lr to "
                   "F x the peak, then a cosine (continuing a decayed run)")
    g.add_argument("--weight-decay", type=float, default=None, help="the embedding / head group (default 0.1)")
    g.add_argument("--tail", default="fp32", choices=["fp32", "lean"],
                   help="the float parameters: fp32 + AdamW; lean: bf16 / fp16 + LeanAdam (int8 first moment, factored "
                        "second moment)")

    g = ap.add_argument_group("float extras of the ternary layers")
    g.add_argument("--rc-scale", action=ON, default=True, help="per-row and per-column scales (W = beta r_i c_j q_ij)")
    g.add_argument("--rc-lr", type=float, default=0.0, help="constant lr of the scales (0: the schedule's)")
    g.add_argument("--qk-temp", action=ON, default=True, help="learnable per-head attention temperature")
    g.add_argument("--adapter-rank", type=int, default=16, help="rank of the float adapter A B^T beside the trits (0: none)")
    g.add_argument("--adapter-init", default="zero", choices=["zero", "svd"],
                   help="svd (with --init-hf, --convert gptq / gptqc): A B^T starts as the best rank-R part of each "
                        "layer's rounding error in output space")
    g.add_argument("--adapter-schedule", default="", help="R0:R1[:START:END[:STEP]]: the adapter rank goes from R0 "
                   "(--adapter-rank) down to R1 between the fractions START and END of training (default 0.125:0.75), "
                   "STEP at a time (default 8), each time to the best lower-rank part of A B^T (its optimizer state "
                   "restarts)")
    g.add_argument("--adapter-wd", type=float, default=0.1)
    g.add_argument("--adapter-lr-mult", type=float, default=1.0)

    g = ap.add_argument_group("TTF (--mode ttf)")
    g.add_argument("--ttf-rank", type=int, default=512, help="rank of each layer's accumulator A = U V^T")
    g.add_argument("--ttf-step-fp32", action=ON, default=False, help="--accum M > 1: keep U in fp32 between a step's "
                   "micro-batches (encoded with --ttf-store once per step, not per micro-batch; more memory, not bit-identical)")
    g.add_argument("--ttf-orth", default="qr", choices=["qr", "cholqr2", "spanqr"],
                   help="the accumulator basis' orthonormalisation: Householder QR, Cholesky QR twice (QR where it would "
                        "fail), or spanqr (the spanqr package on PYTHONPATH; above rank 512 its block Gram-Schmidt, needs rank % 256 "
                        "== 0; other shapes Householder)")
    g.add_argument("--ttf-theta", type=float, default=16.0, help="firing threshold (in units of the step)")
    g.add_argument("--ttf-theta-sched", default="", help="T0:T1[:START]: theta T0 until fraction START, then linear to T1")
    g.add_argument("--ttf-tau", type=float, default=1000.0, help="the accumulator's leak time constant (steps at the "
                   "peak step scale; the leak per step shrinks with the step scale: the memory is a distance)")
    g.add_argument("--ttf-step-p", type=float, default=0.75, help="the step scale (lr / peak lr)^P")
    g.add_argument("--ttf-b2", type=float, default=0.95, help="decay of the factored second moment")
    g.add_argument("--ttf-spend", type=float, default=1.0, help="a move spends this x theta of the accumulator")
    g.add_argument("--ttf-band", type=float, default=0.0, help="the exact tally on at most this share of the entries, those "
                   "near the threshold (update._band_fire; 0: off; one batch per step)")
    g.add_argument("--ttf-hspend", type=float, default=0.0, help="a move also spends theta x F x the input correlation "
                   "from the row's correlated entries (update._hspend_basis; 0: off; --dw factored)")
    g.add_argument("--ttf-band-lo", type=float, default=0.5, help="--ttf-band: an entry is in the band while its |A| >= "
                   "this x theta")
    g.add_argument("--ttf-start", type=int, default=0, help="trits frozen until this step (floats train)")
    g.add_argument("--ttf-acc-scale", type=float, default=None, help="--resume: multiply the accumulators by F "
                   "(continuing at theta F x the old one); required when --ttf-theta differs from the checkpoint's")
    g.add_argument("--ttf-store", default="fp32", choices=["fp32", "bf16", "fp12s"],
                   help="storage of U and V (fp12s: 1.5 bytes per value)")
    g.add_argument("--ttf-offload", default="none", choices=["none", "async"],
                   help="async: the state in pinned host memory, its copies on side streams, prefetched in the update's "
                        "order (overlapping the backward; the same values)")
    g.add_argument("--ttf-offload-keep", type=float, default=0.0, help="--ttf-offload async: this share of the layers "
                   "(module order) keeps its state on the GPU")
    g.add_argument("--ttf-vdiag", type=int, default=0, help="every N steps log g^2 / v (0: off)")
    g.add_argument("--dw", default="factored", choices=["factored", "dense"],
                   help="factored: the update from the gradient's factors (gy, x), the dense weight gradient never "
                        "formed (equal loss to dense at 110M)")
    g.add_argument("--dw-sketch", type=int, default=256, help="--dw factored: sketch size for the sums of g^2")
    g.add_argument("--dw-thin", default="factored", choices=["factored", "dense"],
                   help="--dw factored: dense = layers with N K < (N + K)(sketch + 2 rank) (thin adapter factors) take "
                        "the dense weight gradient instead (fewer flops and launches; exact g^2 sums, not bit-identical)")

    g = ap.add_argument_group("kernels and memory")
    g.add_argument("--dx", default="int8", choices=["bf16", "int8x2", "int8"], help="the input gradient's GEMM "
                   "(int8: gy in one int8 slice per row, 7 bits, on the int8 tensor cores where the summed width is "
                   ">= 4096, else bf16; 0.8B 1000 steps, 26 Oct: wiki / chat / mix +0.0003 / +0.0002 / "
                   "+0.0023 vs int8x2, the paired train-loss gap -0.003..+0.007 throughout; 27B slice 5.77 -> 5.27 "
                   "s/step on the laptop. int8x2: two slices, as accurate as bf16, the 110M conversion recipe 2.6675 "
                   "vs 2.6672)")
    g.add_argument("--tf32", action=ON, default=False, help="the fp32 matrix products on the TF32 tensor cores (10-bit "
                   "mantissa products, fp32 sums: the accumulator's basis turn, the int-k head's products, the sketch "
                   "cross terms); the orthonormalisation stays in full fp32. 0.8B 1000 steps, 26 Oct: level with "
                   "fp32; off by default (on in the 27B recovery runs)")
    g.add_argument("--fused-layer", action=ON, default=True,
                   help="(default on; --no-fused-layer: the unfused path) int8 ternary layers with row / column scales: the centring, column scale and act-quant in one "
                        "kernel, the GEMM's epilogue with the row scale, bias and the sum of a converted layer's parts, "
                        "the backward's elementwise work in two kernels (output, input gradient and the TTF factors "
                        "bit-identical; the scale / bias gradients summed in fp32)")
    g.add_argument("--ckpt", default="layer", choices=["layer", "block"],
                   help="activation checkpointing: attention and MLP separately, or each block once (half the memory)")
    g.add_argument("--ckpt-skip", default="auto",
                   help="the last K blocks without checkpointing; auto: as many as the measured memory allows, then the "
                        "int8 trit cache (ttf/memplan.py)")
    g.add_argument("--loss-chunk", type=int, default=2048, help="tokens per output-head chunk (memory only)")
    g.add_argument("--mem-cap-gib", type=float, default=0.0, help="cap PyTorch's GPU memory (0: no cap)")
    g.add_argument("--pipe", action=ON, default=False, help="Qwen under torchrun: pipeline parallel over the ranks")
    g.add_argument("--pipe-split", default="", help="--pipe: decoder layers per stage, e.g. 17,16,16,15")

    g = ap.add_argument_group("conversion and distillation")
    g.add_argument("--init-hf", default="", help="start from a Hugging Face checkpoint (Llama or Qwen3.5 / 3.8)")
    g.add_argument("--convert", default="gptqc",
                   choices=["gptqc", "gptq", "als", "absmean", "scratch"],
                   help="how the weights become trits and scales (gptqc: GPTQ on centred inputs)")
    g.add_argument("--convert-stream", action=ON, default=True, help="Qwen: convert layer by layer from the files")
    g.add_argument("--ext", action="append", default=[], metavar="MODULE[:SPEC]",
                   help="a structure / conversion plug-in (ttf/ext.py), e.g. ttfconv.adapters:rank=0.2,alt=1; repeatable, in order")
    g.add_argument("--calib", type=int, default=64, help="calibration windows for gptq / gptqc")
    g.add_argument("--teacher", default="", help="full-precision teacher (Hugging Face dir): loss (1 - W) CE + W KL")
    g.add_argument("--kd-weight", type=float, default=0.75, help="W, the KL term's weight")
    g.add_argument("--train-emb", default="none", choices=["none", "head", "both"],
                   help="Qwen: train the output head (head) or the head and the token embedding (both) in the float tail; "
                        "the conversion leaves them frozen")
    g.add_argument("--emb-wd", type=float, default=0.0, help="weight decay of the trained embedding / head and the "
                   "int-k group parameters (their own group; 24 Oct: they shared the 0.1 of the float tail)")
    g.add_argument("--emb-lr-mult", type=float, default=1.0, help="their learning rate x this; --intk-emb codes: "
                   "theta / this (the codes move at the embedding / head rate; 0: frozen)")
    g.add_argument("--init-ckpt", default="", help="start a run from a checkpoint's weights (e.g. a pre-converted "
                   "model, saved with --stop-after 0): step 0, a fresh optimizer and TTF state, the same --ext list")
    g.add_argument("--intk-emb", default="", help="E:H[:G]: token embedding / output head as E- / H-bit codes in "
                   "groups of G (64), trained by TTF (ttf/intk_emb.py), e.g. 2:3 (a tied matrix: E bits)")
    g.add_argument("--intk-update", default="factored", choices=["factored", "dense"],
                   help="--intk-emb: factored (no dense matrix or gradient: the head's loss in chunks from the codes, "
                   "the embedding's update on its rows) or dense (the bf16 matrix and an fp32 gradient sum per step)")
    g.add_argument("--kd-cache", default="", help="precomputed teacher top-k targets (scripts/quant/teacher_topk.py)")
    g.add_argument("--kd-cache-mode", default="sample", choices=["renorm", "rest", "sample"],
                   help="sample (needs a cache with tail samples, teacher_topk.py --tail): top k plus the cache's tail "
                        "samples (unbiased; 0.8B at 1k: within 0.0025-0.0045 of the live teacher); rest: top k plus one "
                        "bucket for the remaining mass; renorm: KL on the top k renormalised")
    g.add_argument("--kd-cache-start", type=int, default=0, help="the step that reads the cache's window 0")
    g.add_argument("--moe", default="", help="E:K[:H]: mixture of E ternary experts, top-K routing")
    g.add_argument("--moe-aux", type=float, default=0.01, help="load-balancing loss weight")
    g.add_argument("--tern-emb", default="none", help="none | tied | untied | tied_wide[:M]: ternary token embedding")

    g = ap.add_argument_group("evaluation, checkpoints, diagnostics")
    g.add_argument("--eval-interval", type=int, default=None)
    g.add_argument("--eval-iters", type=int, default=None)
    g.add_argument("--save-secs", type=float, default=900.0, help="seconds between checkpoint saves")
    g.add_argument("--snap-every", type=int, default=0, help="also keep ckpt_<step>.pt every N steps")
    g.add_argument("--final-save", action=ON, default=True, help="checkpoint at the end")
    g.add_argument("--track-flips", action=ON, default=False, help="log the share of trits moved / never moved")
    g.add_argument("--profile", type=int, default=0, help="profile N steps after 20, print the tables, exit")
    g.add_argument("--mem-report", action=ON, default=False, help="log the memory by kind at steps 10, 50, 100, 1000")
    g.add_argument("--mem-snapshot", default="", help="PATH:STEP: CUDA allocation history to PATH at STEP")
    g.add_argument("--max-temp", type=int, default=86, help="pause while the GPU is at or above this (C)")
    g.add_argument("--resume-temp", type=int, default=85, help="after a pause, resume at or below this (C)")
    g.add_argument("--temp-check", type=int, default=4, help="check the GPU temperature every N steps")
    return ap


def recipe_path(name):
    return name if os.path.sep in name or name.endswith(".toml") else os.path.join(RECIPES, name + ".toml")


def parse(argv=None):
    """the options: the recipe's values as defaults, the command line over them; then the derived settings"""
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    if pre.recipe:
        with open(recipe_path(pre.recipe), "rb") as f:
            rec = tomllib.load(f)
        known = {a.dest for a in ap._actions}
        rec = {k.replace("-", "_"): v for k, v in rec.items()}
        bad = sorted(set(rec) - known)
        if bad:
            raise SystemExit(f"recipe {pre.recipe}: unknown options {bad}")
        for a_ in ap._actions:   # argparse checks choices on the command line only, not on defaults (a removed value
            if a_.dest in rec and a_.choices is not None and rec[a_.dest] not in a_.choices:   # was taken silently)
                raise SystemExit(f"recipe {pre.recipe}: {a_.option_strings[0]} {rec[a_.dest]!r}: one of "
                                 f"{', '.join(map(str, a_.choices))}")
        ap.set_defaults(**rec)
    a = ap.parse_args(argv)
    return resolve(a)


def resolve(a):
    """settings derived from several options (read by the trainer)"""
    a.factored_k = a.dw_sketch if (a.mode == "ttf" and a.dw == "factored") else 0
    a.svd_rank = a.adapter_rank if a.adapter_init == "svd" else 0
    a.u32 = a.data_dtype == "u32"
    if a.adapter_schedule:
        f = a.adapter_schedule.split(":")
        if len(f) < 2 or int(f[0]) != a.adapter_rank or int(f[1]) >= int(f[0]) or int(f[1]) < 1:
            raise SystemExit(f"--adapter-schedule {a.adapter_schedule}: R0:R1 with R0 = --adapter-rank > R1 >= 1")
    if a.lr_schedule not in ("cosine", "linear") and not a.lr_schedule.startswith("wsd:"):
        raise SystemExit(f"--lr-schedule {a.lr_schedule}: cosine, linear or wsd:D")
    return a


# ---- the trainer's flags before 17 Oct 2026 -> these (docs/OPTIONS_MAP.md) -------------------------------------------
# old flag: (new flag, value map) for a renamed option; a callable f(value) -> list of new arguments for the rest;
# None: removed with no effect at the values the recipes passed (or a feature only in the trainer at tag pre-consolidation-2026-10-17)
_SAME = lambda new: (lambda v: [new] if v is True else [new, v])
_FLAG = lambda new, val: (lambda v: [new, val])
_TAG = "the old trainer only: tag pre-consolidation-2026-10-17, python -m bitnet.train there"


def _gone(what):
    """a feature removed from the core (25 Oct 2026): a run that used it cannot be repeated here"""
    def f(v):
        raise ValueError(f"{what}: removed ({_TAG})")
    f.note = "removed 25 Oct 2026"
    return f


def _keep_if(ok, then, what):
    """the value translates by then(v) when ok(v), else the run used a removed setting"""
    def f(v):
        if not ok(v):
            raise ValueError(f"{what} {v}: removed ({_TAG})")
        return then(v)
    f.note = "other values: removed 25 Oct 2026"
    return f


_LEGACY = {
    "--preset": _SAME("--preset"), "--data": _SAME("--data"), "--val": _SAME("--val"), "--steps": _SAME("--steps"),
    "--batch_size": _SAME("--batch-size"), "--grad_accum": _SAME("--accum"), "--seq_len": _SAME("--seq-len"),
    "--mode": lambda v: ["--mode", {"kernel": "ttf", "master": "bitnet", "fp32": "fp32"}[v]],
    "--master_dtype": _keep_if(lambda v: v == "fp32", lambda v: [], "--master_dtype"), "--lowrank": _SAME("--ttf-rank"), "--profile": _SAME("--profile"),
    "--qk_temp": _FLAG("--qk-temp", None), "--lowrank_mag": lambda v: ["--adapter-rank", v.split(":")[1]],
    "--mag_wd": _SAME("--adapter-wd"), "--mag_lr_mult": _SAME("--adapter-lr-mult"), "--ts": None,
    "--ts_theta": _SAME("--ttf-theta"), "--mem_cap_gib": _SAME("--mem-cap-gib"), "--init_hf": _SAME("--init-hf"),
    "--init_hf_method": _keep_if(lambda v: v in ("gptqc", "gptq", "als", "absmean", "scratch"), _SAME("--convert"),
                                 "--init_hf_method"), "--distill": _SAME("--teacher"), "--distill_w": _SAME("--kd-weight"),
    "--kd_cache": _SAME("--kd-cache"), "--kd_mode": _keep_if(lambda v: v != "windows", _SAME("--kd-cache-mode"), "--kd_mode"),
    "--kd_cache_start": _SAME("--kd-cache-start"), "--moe": _SAME("--moe"), "--moe_aux": _SAME("--moe-aux"),
    "--init_stream": _FLAG("--convert-stream", None), "--data_u32": _FLAG("--data-dtype", "u32"),
    "--hid_kd": _keep_if(lambda v: float(v) == 0, lambda v: [], "--hid_kd"), "--init_calib": _SAME("--calib"), "--ts_theta_sched": _SAME("--ttf-theta-sched"),
    "--ts_start": _SAME("--ttf-start"), "--ts_vdiag": None, "--ts_vdiag_every": None, "--ts_nomom": None,
    "--ts_spend": _SAME("--ttf-spend"), "--gemm": None,
    "--ts_store_a": _keep_if(lambda v: v in ("", "none", "fp32", "bf16", "fp12s"),
                             lambda v: ["--ttf-store", v if v in ("bf16", "fp12s") else "fp32"], "--ts_store_a"),
    "--tern_emb": _SAME("--tern-emb"), "--lr_linear": _FLAG("--lr-schedule", "linear"),
    "--lr_wsd": lambda v: ["--lr-schedule", f"wsd:{v}"] if int(v) else [], "--lr_rewarm": _SAME("--lr-rewarm"),
    "--lean_batched": None, "--lean_tail": _FLAG("--tail", "lean"),
    "--init_mag_svd": lambda v: ["--adapter-init", "svd"] if int(v) else [], "--ts_fire_fused": None,
    "--ts_acc_scale": _SAME("--ttf-acc-scale"), "--fused_norm": None,
    "--fused_scales": None, "--ts_factored": lambda v: ["--dw", "factored", "--dw-sketch", v] if int(v) else ["--dw", "dense"],
    "--ts_micro": None, "--ckpt_block": _FLAG("--ckpt", "block"),
    "--no_final_save": _FLAG("--no-final-save", None), "--mem_snapshot": _SAME("--mem-snapshot"),
    "--eval_first_n": None, "--mem_report": _FLAG("--mem-report", None),
    "--ts_offload": _FLAG("--ttf-offload", "async"), "--ts_offload_async": _FLAG("--ttf-offload", "async"),
    "--dw_fp32": _gone("--dw_fp32 (the dense weight gradient's GEMM in fp32)"), "--w_cache": None, "--ts_offload_keep": _SAME("--ttf-offload-keep"),
    "--ts_fused": None, "--ts_anneal": _SAME("--ttf-step-p"),
    "--ts_tau_anneal_pow": _keep_if(lambda v: float(v) == 1, lambda v: [], "--ts_tau_anneal_pow"), "--ts_tau_anneal": None,
    "--ts_tau": _SAME("--ttf-tau"), "--ts_b2": _SAME("--ttf-b2"), "--tf32": None,
    "--int8": None, "--int8_dx": _FLAG("--dx", "int8"), "--stop_after": _SAME("--stop-after"),
    "--snap_every": _SAME("--snap-every"), "--tail_fp32": _FLAG("--tail", "fp32"), "--seed": _SAME("--seed"),
    "--track_flips": _FLAG("--track-flips", None), "--resume": _FLAG("--resume", None), "--out_dir": _SAME("--out-dir"),
    "--lr": _SAME("--lr"), "--warmup": _SAME("--warmup"), "--extend_to": _keep_if(lambda v: int(v) == 0, lambda v: [], "--extend_to"),
    "--weight_decay": _SAME("--weight-decay"), "--min_lr": _SAME("--min-lr"), "--eval_interval": _SAME("--eval-interval"),
    "--eval_iters": _SAME("--eval-iters"), "--loss_chunk": _SAME("--loss-chunk"), "--save_secs": _SAME("--save-secs"),
    "--max_temp": _SAME("--max-temp"), "--resume_temp": _SAME("--resume-temp"), "--rc_scale": _FLAG("--rc-scale", None),
    "--rc_lr": _SAME("--rc-lr"), "--compile": None, "--pipe": _FLAG("--pipe", None),
    "--pipe_split": _SAME("--pipe-split"), "--ckpt_skip": _SAME("--ckpt-skip"), "--temp_check": _SAME("--temp-check"),
    # no effect in the TTF recipes (the stateless flip rule's knobs, the dense weight gradient, look-ahead off, the
    # short momentum's rank under --ts_nomom); above, None also marks a speed switch whose path is now the only one
    # (the int8 GEMMs, the fused kernels, the update inside the backward, the batched LeanAdam, TF32, compile, the
    # trit cache) or a diagnostic that went (--eval_first_n)
    "--rate": None, "--rate_peak": None, "--rate_warmup": None, "--rate_schedule": None, "--g_ref": None,
    "--dw_mode": None, "--lookahead": None, "--ts_rank_s": None,
}
# flags whose old meaning is off when absent but whose new option defaults to on (or to another value)
_OLD_OFF = [("--qk_temp", ["--no-qk-temp"]), ("--rc_scale", ["--no-rc-scale"]), ("--lowrank_mag", ["--adapter-rank", "0"]),
            ("--ts_factored", ["--dw", "dense"]), ("--init_stream", ["--no-convert-stream"])]
# old defaults that differ from the new ones
_OLD_DEFAULTS = [("--mode", ["--mode", "ttf"]), ("--ckpt_skip", ["--ckpt-skip", "0"]), ("--ts_tau", ["--ttf-tau", "300"]),
                 ("--ts_anneal", ["--ttf-step-p", "1.0"]), ("--lowrank", ["--ttf-rank", "0"]),
                 ("--init_hf_method", ["--convert", "als"]), ("--distill_w", ["--kd-weight", "1.0"]),
                 ("--mag_wd", ["--adapter-wd", "0.0"]), ("--kd_mode", ["--kd-cache-mode", "renorm"]),
                 ("--int8_dx", ["--dx", "int8x2"])]
_OLD_BOOL = {"--qk_temp", "--ts", "--init_stream", "--data_u32", "--ts_vdiag", "--ts_nomom", "--lr_linear",
             "--lean_batched", "--lean_tail", "--ts_fire_fused", "--fused_norm", "--fused_scales", "--ts_micro",
             "--ckpt_block", "--no_final_save", "--mem_report", "--ts_offload", "--ts_fused", "--ts_tau_anneal", "--tf32",
             "--int8", "--int8_dx", "--tail_fp32", "--track_flips", "--resume", "--rc_scale", "--compile", "--pipe",
             "--ts_offload_async", "--dw_fp32"}


def from_legacy(argv):
    """an old trainer command line (list of strings) -> the same run in these options. Raises on a flag whose feature
    is only in the old trainer (run that at tag pre-consolidation-2026-10-17: python -m bitnet.train there)."""
    out, seen, i, vd = [], set(), 0, [None, None]
    while i < len(argv):
        f = argv[i]
        if f in _OLD_BOOL:
            v, i = True, i + 1
        else:
            if i + 1 >= len(argv):
                raise ValueError(f"{f}: missing value")
            v, i = argv[i + 1], i + 2
        seen.add(f)
        if f == "--ts_vdiag":
            vd[0] = True; continue
        if f == "--ts_vdiag_every":
            vd[1] = v; continue
        if f not in _LEGACY:
            raise ValueError(f"{f}: not in the core trainer (the old trainer only: tag pre-consolidation-2026-10-17)")
        m = _LEGACY[f]
        if m is not None:
            out += [x for x in m(v) if x is not None]
    if vd[0]:
        out += ["--ttf-vdiag", str(vd[1] or 1)]
    for f, new in _OLD_OFF + _OLD_DEFAULTS:
        if f not in seen:
            out += new
    mode = ([argv[j + 1] for j, x in enumerate(argv[:-1]) if x == "--mode"] or ["kernel"])[-1]   # (the last one counts)
    if mode == "kernel" and "--tail_fp32" not in seen and "--lean_tail" not in seen:   # (master: always fp32 AdamW)
        raise ValueError(f"the bf16 float tail (no --tail_fp32 / --lean_tail): removed 26 Oct 2026 ({_TAG})")
    if mode == "kernel" and "--ts" not in seen:
        raise ValueError("kernel mode without --ts (a flip rule): the old trainer only (tag pre-consolidation-2026-10-17)")
    if mode == "kernel" and "--ts_nomom" not in seen:
        raise ValueError("--ts without --ts_nomom (TTF with a short momentum): the old trainer only (tag pre-consolidation-2026-10-17)")
    if mode == "kernel":    # off when absent in the old trainer, removed from the core since (25 Oct 2026)
        acc = int(([argv[j + 1] for j, x in enumerate(argv[:-1]) if x == "--grad_accum"] or ["1"])[-1])
        for f, what in (("--int8", "the bf16 forward (no --int8)"), ("--ts_tau_anneal", "the fixed leak (no --ts_tau_anneal)")) + \
                ((("--ts_micro", "one update from the summed micro-batch gradients (no --ts_micro)"),) if acc > 1 else ()):
            if f not in seen:
                raise ValueError(f"{what}: removed ({_TAG})")
    return out
