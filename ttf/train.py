"""The trainer: python -m ttf.train (options: ttf/args.py, python -m ttf.train --help).

Data: a flat binary file of token ids (uint16, or uint32 with --data-dtype u32) for train and another for val.
Kernel mode: the ternary layers update inside their backward (TTF, ttf/update.py), so only the float tail (embedding,
norms, scales, adapters) has an optimizer. Also the baselines: --mode master (BitNet b1.58) and --mode fp32.
"""
from __future__ import annotations
import os
import shutil
import json
import math
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gpu_temp():
    """Current GPU temperature in Celsius via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None

from .config import ModelConfig, TrainConfig, PRESETS
from .model import BitTransformer
from .bitlinear import STATE
from .master import build_master_transformer, split_params
from .layers import KernelTernaryLinear, build_kernel_transformer, enable_flip_tracking, collect_flip_stats, _rf
from .args import parse
from .data import get_batch, _kd_cache_batch
from .eval import evaluate
from .schedule import lr_at, _ts_scale, resume_acc_scale, LR_LINEAR, LR_REWARM, LR_WSD
from .update import (ts_layer, ts_step, FIRE_FUSED, TS_FUSED_STATS, _load_pair, _orth, _store_codec, _store_pair,
                     _ts_on_host, _ts_state_bytes)


def _tern_lays(model):
    from .tern_emb import tern_emb_layers
    return tern_emb_layers(model)


def _split_emb_group(model, groups, wd, mult):
    """the embedding / head (--train-emb) and the int-k matrices' group mins / steps (--intk-emb) out of the decayed
    float-tail group into their own: weight decay wd (default 0: decay shrinks every row each step, seen or not, and
    squeezes int-k values toward their group minimum; 24 Oct Q4b / Q5 ran with 0.1), lr x mult"""
    ids = {id(p) for n, p in model.named_parameters()
           if "embed_tokens" in n or "lm_head" in n or n.endswith((".m.lo", ".m.s"))}
    moved = []
    for g in groups:
        if g.get("weight_decay", 0.0) and not g.get("rc_lr"):
            keep = [p for p in g["params"] if id(p) not in ids]
            moved += [p for p in g["params"] if id(p) in ids]
            g["params"] = keep
    if moved:
        groups.append({"params": moved, "weight_decay": wd, "emb_mult": mult})
        print(f"embedding / head group: {sum(p.numel() for p in moved) / 1e6:.1f}M, weight decay {wd}, lr x{mult}",
              flush=True)
    return groups


def _intk_lays(model):
    from .intk_emb import intk_layers
    return intk_layers(model)


def mem_report(model, tail_opt, lr_state):
    """--mem-report: MiB held by each part of training (tensors counted by their own size; CUDA context excluded)"""
    MB = 2 ** 20
    nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts if torch.is_tensor(t))
    trits = nbytes([b for n, b in model.named_buffers() if n.endswith("wpacked")])
    params = list(model.parameters())
    opt = nbytes([v for st in tail_opt.state.values() for v in st.values()])
    gpu, host = 0, 0
    def walk(x):
        nonlocal gpu, host
        if torch.is_tensor(x):
            if x.is_cuda: gpu += x.numel() * x.element_size()
            else: host += x.numel() * x.element_size()
        elif isinstance(x, (tuple, list)):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for y in x.values(): walk(y)
    walk(lr_state)
    return {"mem_trits_MB": trits / MB, "mem_tail_params_MB": nbytes(params) / MB,
            "mem_tail_grads_MB": nbytes([p.grad for p in params if p.grad is not None]) / MB,
            "mem_tail_opt_MB": opt / MB, "mem_ts_gpu_MB": gpu / MB, "mem_ts_host_MB": host / MB,
            "mem_alloc_MB": torch.cuda.memory_allocated() / MB, "mem_peak_MB": torch.cuda.max_memory_allocated() / MB}


def main():
    args = parse()
    from . import ext as _X
    _X.load(args.ext)   # --ext plug-ins (ttf/ext.py), before the model is built
    if any(_X.having(h_) for h_ in ("begin", "weight", "refine", "fitted", "end")) and \
            (args.convert != "gptqc" or not args.convert_stream):
        raise SystemExit(f"--ext {args.ext}: these plug-ins hook the conversion, which runs them only in the centred "
                         f"streaming conversion (--convert gptqc --convert-stream); got --convert {args.convert}"
                         f"{'' if args.convert_stream else ' --no-convert-stream'}")
    kmode = {"ttf": "kernel", "bitnet": "master", "fp32": "fp32"}[args.mode]   # the checkpoint's and hf_init's names
    ttf = args.mode == "ttf"
    te = args.tern_emb if args.tern_emb != "none" else ""            # the ternary embedding's kind ("": none)
    mag = f"add:{args.adapter_rank}" if args.adapter_rank else ""     # the adapter ("": none)
    store = args.ttf_store if args.ttf_store != "fp32" else ""      # the accumulator's codec ("": fp32)
    i8, i8dx = args.fwd == "int8", args.dx == "int8"
    from .layers import DX_MODE as _DXM
    _DXM[0] = args.dx
    if os.environ.get("TERN_SYNCDBG"):   # report every host sync (location in the warning)
        torch.cuda.set_sync_debug_mode(1)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.gemm in ("cublas", "int8mm"):
        from . import kernel as _kmod
        _kmod.GEMM_BACKEND[0] = args.gemm
    LR_WSD[0] = int(args.lr_schedule.split(":")[1]) if args.lr_schedule.startswith("wsd:") else 0
    LR_LINEAR[0] = args.lr_schedule == "linear"
    if args.lr_rewarm:
        _rw = args.lr_rewarm.split(":")
        LR_REWARM[:] = [int(_rw[0]), int(_rw[1])] + ([float(_rw[2])] if len(_rw) > 2 else [])
    TS_FUSED_STATS[0] = os.environ.get("TERN_FUSED_STATS", "1") != "0"
    TIME = os.environ.get("TERN_TIME") == "1"   # time the forward/backward vs the momentum step (synchronizes)

    mc: ModelConfig = PRESETS[args.preset]
    _QWEN = False
    if args.init_hf:
        from .qwen import is_qwen35
        _QWEN = is_qwen35(args.init_hf)
        if _QWEN:              # Qwen3.5 / 3.8 (Gated DeltaNet + gated attention): transformers' model, ternary linears
            from .qwen import model_config as _qmc
            mc = _qmc(args.init_hf)
        else:
            from .hf_init import model_config_from_hf
            mc = model_config_from_hf(args.init_hf)
    tc = TrainConfig()
    if args.seed is not None: tc.seed = args.seed
    if args.data: tc.data_path = args.data
    if args.val: tc.val_path = args.val
    if args.steps: tc.max_steps = args.steps
    if args.batch_size: tc.batch_size = args.batch_size
    if args.accum is not None: tc.grad_accum = args.accum
    if args.seq_len: mc.max_seq_len = tc.seq_len = args.seq_len
    if args.out_dir: tc.out_dir = args.out_dir
    if args.lr: tc.lr = args.lr
    if args.warmup is not None: tc.warmup_steps = args.warmup
    if args.weight_decay is not None: tc.weight_decay = args.weight_decay   # float tail (embedding) decoupled weight decay
    tc.min_lr = args.min_lr if args.min_lr is not None else tc.lr / 10
    if args.eval_interval: tc.eval_interval = args.eval_interval
    if args.eval_iters: tc.eval_iters = args.eval_iters
    for k_, v_ in dict(steps=tc.max_steps, batch_size=tc.batch_size, accum=tc.grad_accum, seq_len=tc.seq_len, lr=tc.lr,
                       min_lr=tc.min_lr, warmup=tc.warmup_steps, weight_decay=tc.weight_decay,
                       eval_interval=tc.eval_interval, eval_iters=tc.eval_iters, seed=tc.seed, data=tc.data_path,
                       val=tc.val_path, out_dir=tc.out_dir).items():
        setattr(args, k_, v_)        # the resolved values (the run's records and checkpoint hold what was run)

    BitTransformer.loss_chunk = args.loss_chunk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _WS = int(os.environ.get("WORLD_SIZE", "1"))      # data-parallel proof of concept (torchrun --nproc_per_node G)
    _RK = int(os.environ.get("RANK", "0"))
    _PP = (_RK, _WS) if args.pipe else None           # --pipe: (stage, stages); the data is then that of one GPU
    assert not args.pipe or _WS > 1, "--pipe: run under torchrun with several processes"
    assert not args.pipe or not (args.teacher or te), "--pipe: none of the per-micro-batch extras yet"
    _base_out = args.out_dir
    if _WS > 1:
        import torch.distributed as _dist
        _lr_ = int(os.environ.get("LOCAL_RANK", "0"))
        _ng_ = torch.cuda.device_count()
        _lws_ = int(os.environ.get("LOCAL_WORLD_SIZE", str(_WS)))   # ranks on this machine (several machines: < WORLD_SIZE)
        torch.cuda.set_device(_lr_ % max(_ng_, 1))
        _be_ = "nccl" if _ng_ >= _lws_ else "gloo"     # gloo: several ranks share one GPU (a correctness test)
        _dist.init_process_group(_be_)
        print(f"{'pipeline stage' if _PP else 'data parallel: rank'} {_RK} of {_WS} ({_lws_} on this machine), GPU "
              f"{_lr_ % max(_ng_, 1)} of {_ng_}, {_be_}", flush=True)
        if _RK > 0 and args.out_dir:
            args.out_dir = args.out_dir.rstrip("/") + f"_rank{_RK}"
            tc.out_dir = args.out_dir
    if _WS > 1 and not _PP and args.dw == "factored":    # data parallel, factored: the ranks average the thin products
        from .update import DP_REDUCE as _DPR              # the update reads from g (sketches, basis projections)
        _dp_bf16 = os.environ.get("TTF_DP_BF16", "") == "1"   # TTF_DP_BF16=1: sent in bf16 (half the traffic)

        def _dp_mean(t_):
            if _dp_bf16:
                b_ = t_.to(torch.bfloat16); _dist.all_reduce(b_); t_.copy_(b_)
            else:
                _dist.all_reduce(t_)
            t_.div_(_WS)
        _DPR[0] = _dp_mean
    if _PP:
        _WS, _RK = 1, 0       # --pipe: no data-parallel reduction, every stage reads the same windows
    if args.mem_cap_gib and device == "cuda":   # --mem-cap-gib G: PyTorch may use at most G GiB of this GPU (a 96 GB card
        # standing in for a 24 GB one in memory-matched comparisons; allocations beyond it raise out-of-memory)
        tot_ = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib / tot_))
        print(f"--mem-cap-gib: PyTorch capped at {args.mem_cap_gib:.1f} of {tot_:.1f} GiB", flush=True)
    if args.mem_snapshot and device == "cuda":   # --mem-snapshot PATH:STEP: every allocation with its Python stack
        torch.cuda.memory._record_memory_history(enabled="all", stacks="python", max_entries=2_000_000)
    torch.manual_seed(tc.seed)
    print(mc.report())
    use_beta = args.mode == "ttf"

    _qsw = {}
    if args.head_fused and not _QWEN:
        print("--head-fused: no effect on this model (Qwen only)", flush=True)
    if _QWEN:
        assert args.mode == "ttf" and not te, "Qwen: kernel mode, no --tern-emb yet"
        from .qwen import build_qwen, QwenTern
        QwenTern.loss_chunk = args.loss_chunk
        QwenTern.head_fused = args.head_fused
        _own = None
        if _PP:
            from .qwen import text_config, pp_split
            _rng = pp_split(text_config(args.init_hf).num_hidden_layers, _PP[1], args.pipe_split)
            _own = (*_rng[_PP[0]], _PP[0] == 0, _PP[0] == _PP[1] - 1)
            print(f"--pipe: stage {_PP[0]} of {_PP[1]}: decoder layers {_own[0]}..{_own[1] - 1} of {_rng[-1][1]}"
                  f"{' + embedding' if _own[2] else ''}{' + norm, head and loss' if _own[3] else ''}", flush=True)
        _mkl = lambda i_, o_: KernelTernaryLinear(i_, o_, mc.act_bits, beta=use_beta, int8=i8, int8_dx=i8dx)
        model, _qsw = build_qwen(args.init_hf, "kernel", grad_checkpoint=tc.grad_checkpoint, device=device, own=_own,
                                 make_linear=_mkl)
        from . import ext as _X
        for f_ in _X.having("build"):    # --ext plug-ins: wrappers / extra layers (fresh start and resume alike)
            _qsw = f_(model, _qsw, _mkl, lambda m_: print(m_, flush=True))
        if _PP:
            model.pp = {"rank": _PP[0], "n": _PP[1], "lo": _own[0], "hi": _own[1]}
    elif args.mode == "ttf":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint, beta=use_beta, int8=i8,
                                         int8_dx=i8dx)
    elif args.mode == "fp32":
        # full-precision reference (no ternary, no activation quantization): plain nn.Linear,
        # fp32 weights + AdamW, bf16 autocast in the forward like every other mode
        model = BitTransformer(mc, grad_checkpoint=tc.grad_checkpoint,
                               make_linear=lambda i, o: nn.Linear(i, o, bias=False))
    elif args.mode == "bitnet":
        model = build_master_transformer(
            mc, grad_checkpoint=tc.grad_checkpoint,
            dtype=torch.float32 if args.bitnet_dtype == "fp32" else torch.bfloat16)
    model = model.to(device) if not _PP else model._apply(lambda t_: t_ if t_.is_meta else t_.to(device))   # --pipe: others' stay meta
    try:   # counted from the modules (mc.report() above is a formula: approximate for hybrid / HF models)
        n_t_ = sum(m_.N * m_.K for m_ in model.modules() if getattr(m_, "wpacked", None) is not None and hasattr(m_, "N"))
        n_f_ = sum(p_.numel() for p_ in model.parameters())
        print(f"counted       : ternary {n_t_ / 1e6:,.1f} M + float {n_f_ / 1e6:,.1f} M", flush=True)
    except Exception:
        pass
    if args.moe:      # --moe E:K[:H]: every feed-forward becomes a mixture of E ternary experts, top-K routing
        from .model import enable_moe
        from .master import MasterTernaryLinear as _MTL   # main() imports the name locally later on
        _mk = (lambda i_, o_: KernelTernaryLinear(i_, o_, mc.act_bits, beta=use_beta, int8=i8,
                                                  int8_dx=i8dx).to(device)) \
            if args.mode == "ttf" else \
            (lambda i_, o_: _MTL(i_, o_, mc.act_bits, dtype=torch.float32 if args.bitnet_dtype == "fp32"
                                                else torch.bfloat16).to(device)) if args.mode == "bitnet" else \
            (lambda i_, o_: nn.Linear(i_, o_, bias=False).to(device))   # master: ternary experts with latents, as its layers
        E_, k_, H_ = enable_moe(model, args.moe, _mk)
        BitTransformer.moe_aux_w = args.moe_aux
        print(f"--moe: {E_} experts of hidden {H_}, top-{k_} routing, load-balancing weight {args.moe_aux}", flush=True)
    model.train()
    if te:
        from .tern_emb import enable_tern_emb
        n_te = enable_tern_emb(model, te, "master" if args.mode == "bitnet" else "kernel")
        assert args.mode == "bitnet" or (ttf and args.ttf_update == "fused" and tc.grad_accum == 1), \
            "--tern-emb in kernel mode: TTF --ttf-update fused, grad_accum 1"
        print(f"--tern-emb {te}: {n_te / 1e6:.1f}M ternary embedding weights (+ row / column scales)", flush=True)
    if args.ckpt_skip:
        model.ckpt_skip = args.ckpt_skip
    if args.ckpt == "block":
        for b_ in model.layers:
            b_.ckpt_block = True
    if args.qk_temp:
        from .model import Attention
        for a in model.modules():
            if isinstance(a, Attention):
                a.qk_logscale = nn.Parameter(torch.zeros(a.n_heads, device=device))
        print("per-head attention temperature on (log-scale, init 0)", flush=True)
    if args.rc_scale:
        from .layers import enable_rc_scales
        if args.mode == "bitnet":
            from .master import MasterTernaryLinear
            nrc = 0
            for m_ in model.modules():
                if isinstance(m_, MasterTernaryLinear): m_.enable_rc_scale(); nrc += m_.N + m_.K
            print(f"row/column scales (master): {nrc / 1e3:.1f}k floats", flush=True)
        else:
            print(f"row/column scales: {enable_rc_scales(model) / 1e3:.1f}k floats", flush=True)
    _resume_path = os.path.join(tc.out_dir, "ckpt.pt")
    if (_PP or _WS > 1) and not os.path.exists(_resume_path) and os.path.exists(os.path.join(_base_out, "ckpt.pt")):
        _resume_path = os.path.join(_base_out, "ckpt.pt")    # --pipe: a single-GPU checkpoint (or stage 0's) for all;
        # data parallel: every rank resumes from rank 0's (the only one saved: the ranks' states are identical)
    _resuming = args.resume and os.path.exists(_resume_path)
    _init_ck = bool(args.init_ckpt) and not _resuming   # (a run started with --init-ckpt resumes from its own ckpt.pt)
    _from_ckpt = _resuming or _init_ck
    if _init_ck:   # --init-ckpt: a converted checkpoint's weights, a fresh run (step 0, no optimizer / TTF state)
        _resume_path = args.init_ckpt
    if args.init_hf and _from_ckpt:   # the checkpoint holds the converted model (trits, scales, float parts)
        print(f"--init-hf: {'resuming from' if _resuming else '--init-ckpt'} {_resume_path}, conversion skipped", flush=True)
        if _QWEN:   # as the conversion leaves them: embedding and head frozen (the optimizer is built below)
            model.hf.model.embed_tokens.weight.requires_grad_(False); model.hf.lm_head.weight.requires_grad_(False)
        if args.convert == "gptqc":   # centred layers: their buffers and bias
            _ck = torch.load(_resume_path, map_location="cpu", weights_only=False, mmap=True)["model"]
            _nc = 0                                             # exist before the optimizer is built
            for n_, m_ in model.named_modules():
                _dv = "meta" if isinstance(m_, KernelTernaryLinear) and m_.wpacked.is_meta else device   # --pipe
                if isinstance(m_, KernelTernaryLinear) and f"{n_}.in_mean" in _ck:
                    m_.enable_centering(_ck[f"{n_}.in_mean"].to(_dv), _ck[f"{n_}.out_bias"].to(_dv)); _nc += 1
            print(f"resume: {_nc} centred layers restored", flush=True)
            del _ck
        if _init_ck:   # the weights now, before the optimizer and the embedding / head options (--intk-emb
            _blob = torch.load(_resume_path, map_location="cpu", weights_only=False, mmap=True)   # quantizes these)
            model.load_state_dict(_blob["model"])
            print(f"--init-ckpt: weights of {_resume_path} (its step {_blob.get('step')}), a fresh run from step 0",
                  flush=True)
            del _blob
    if args.init_hf and not _from_ckpt:
        from .hf_init import init_from_hf, GPTQ_OPTS as _GO
        _GO["mag_svd"] = args.svd_rank
        _hess = None
        if _QWEN and args.convert == "random":   # speed measurements without the weights (no download):
            with torch.no_grad():                         # the trits as built (random), float parts N(0, 0.02)
                for p_ in model.parameters():
                    if not p_.is_meta:
                        p_.normal_(0, 0.02)
                model.hf.model.embed_tokens.weight.requires_grad_(False); model.hf.lm_head.weight.requires_grad_(False)
            print("--convert random: random trits and float parts (speed tests only)", flush=True)
        elif _QWEN and args.convert_stream:   # --convert-stream: layer by layer, sequential (models too big for one fp copy)
            from .qwen import convert_qwen_streaming
            convert_qwen_streaming(model, args.init_hf, _qsw, tc.data_path, method=args.convert,
                                   n_seq=args.calib, seq=tc.seq_len, dev=device, dtype_u32=args.u32,
                                   log=lambda m_: print(m_, flush=True))
        elif _QWEN:
            from .qwen import load_qwen, calib_hessians_qwen
            if args.convert in ("gptq", "gptqc"):
                _hess = calib_hessians_qwen(args.init_hf, tc.data_path, n_seq=args.calib, seq=tc.seq_len,
                                            dev=device, dtype_u32=args.u32, log=lambda m_: print(m_, flush=True))
            load_qwen(model, args.init_hf, _qsw, method=args.convert, rc=args.rc_scale, hess=_hess,
                      log=lambda m_: print(m_, flush=True))
            _hess = None
        elif args.convert == "gptqs":    # sequential, error-aware, centred GPTQ (training-free conversion)
            from .hf_init import convert_seq_asym
            _hess = convert_seq_asym(args.init_hf, tc.data_path, n_seq=min(args.calib, 32), dev=device,
                                     log=lambda m_: print(m_, flush=True))
        elif args.convert in ("gptq", "gptqc"):     # input second moments of the full-precision model (not the run's windows)
            from .hf_init import calib_hessians
            _hess = calib_hessians(args.init_hf, tc.data_path, n_seq=args.calib, dev=device,
                                   log=lambda m_: print(m_, flush=True))
        if not _QWEN:
            init_from_hf(model, args.init_hf, kmode, args.convert, rc=args.rc_scale, hess=_hess,
                         log=lambda m_: print(m_, flush=True))
    _teacher = None
    if args.kd_cache:  # precomputed teacher targets (scripts/quant/teacher_topk.py)
        assert hasattr(model, "kd_topk"), "--kd-cache: Qwen models"
        model.kd_w = args.kd_weight
        model.kd_mode = "renorm" if args.kd_cache_mode == "sample" else args.kd_cache_mode   # (sample: renorm's KL on
        # the k + S ids with the loader's log-weights, which already sum to 1)
        assert args.kd_cache_mode != "windows" or args.teacher, "--kd-cache-mode windows: the online teacher (--teacher)"
        print(f"--kd-cache ({args.kd_cache_mode})", flush=True)
        print(f"--kd-cache {args.kd_cache}: from window 0 at step {args.kd_cache_start}, loss (1 - {args.kd_weight}) CE "
              f"+ {args.kd_weight} KD to the teacher's top k", flush=True)
    if args.teacher and _QWEN:   # Qwen: transformers' own model in bf16 as the teacher (it must fit next to the student)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as _QT
        _qt = _QT.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device).eval()
        for p_ in _qt.parameters():
            p_.requires_grad_(False)
        _teacher = lambda x_: (_qt(input_ids=x_).logits, None)
        model.kd_w = args.kd_weight
        print(f"--distill {args.teacher} (Qwen): teacher in bf16, loss (1 - {args.kd_weight}) CE + {args.kd_weight} KL", flush=True)
    elif args.teacher:   # --distill HF_DIR: the run learns the full-precision model's distribution (KL), not only the labels
        from .hf_init import model_config_from_hf as _mch, init_from_hf as _ifh
        from .model import BitTransformer as _BT
        _teacher = _BT(_mch(args.teacher, max_seq_len=tc.seq_len), grad_checkpoint=False,
                       make_linear=lambda i_, o_: nn.Linear(i_, o_, bias=False)).to(device)
        _ifh(_teacher, args.teacher, "fp32", log=lambda *_: None)
        _teacher = _teacher.eval()
        for p_ in _teacher.parameters():   # parameters only: Module.to(bf16) would also cast the complex RoPE table
            p_.data = p_.data.to(torch.bfloat16); p_.requires_grad_(False)   # to real (teacher CE 5.20 instead of 2.34)
        model.kd_w = args.kd_weight
        print(f"--distill {args.teacher}: teacher in bf16, loss (1 - {args.kd_weight}) CE + {args.kd_weight} KL", flush=True)
    if args.compile:
        from .model import enable_compile
        enable_compile()
    if args.intk_emb:                        # --intk-emb: int-k code matrices for the embedding and head, TTF-trained
        from .intk_emb import enable_intk_emb, enable_intk_emb_bt, INTK_FACTORED as _IKF
        _IKF[0] = args.intk_update == "factored"
        assert args.train_emb == "none", "--intk-emb trains the embedding / head itself"
        assert args.mode == "ttf", "--intk-emb: the codes are TTF-trained (--mode ttf)"
        assert not (_IKF[0] and tc.grad_accum > 1 and not args.micro), \
            "--intk-update factored with --accum > 1 needs --ttf-accum micro (no gradient is kept across micro-batches)"
        _en = enable_intk_emb if _QWEN else enable_intk_emb_bt
        assert args.intk_theta_mult == 1.0 or not args.ttf_theta_sched, "--intk-theta-mult with a theta schedule"
        for m_ in _en(model, args.intk_emb, log=lambda m_: print(m_, flush=True)):
            m_.dp_reduce = _WS > 1 and not _PP          # (data parallel only: --pipe stages are not replicas)
            m_.theta_mult = args.intk_theta_mult
    if _QWEN and args.train_emb != "none":   # --train-emb: the head (and the embedding) join the float tail (the
        _te = [model.hf.lm_head.weight] + ([model.hf.model.embed_tokens.weight] if args.train_emb == "both" else [])
        for p_ in _te:                        # conversion and the resume freeze them; 0.8B, 22 Oct, theta 64: -0.085 wiki)
            p_.requires_grad_(True)
        print(f"--train-emb {args.train_emb}: {sum(p_.numel() for p_ in _te) / 1e6:.0f}M embedding / head weights train"
              f" (tied: {model.hf.lm_head.weight is model.hf.model.embed_tokens.weight})", flush=True)

    if args.mode in ("bitnet", "fp32"):
        # the baseline keeps everything in fp32 with a standard AdamW: no bf16
        # rounding anywhere, so the ceiling is not limited by storage precision.
        master, emb, norms = split_params(model)
        # the float extras (--rc-scale, --qk-temp) without weight decay, as in the kernel path's tail groups
        xid = {id(p) for n, p in model.named_parameters() if n.endswith(("row_scale", "col_scale", "qk_logscale"))}
        norms = norms + [p for p in emb if id(p) in xid]; emb = [p for p in emb if id(p) not in xid]
        tail = master + emb + norms
        tail_opt = torch.optim.AdamW(
            [{"params": master, "weight_decay": tc.weight_decay},
             {"params": emb, "weight_decay": tc.weight_decay},
             {"params": norms, "weight_decay": 0.0}],
            lr=tc.lr, betas=(tc.beta1, tc.beta2))
        print(f"master mode: {sum(p.numel() for p in master)/1e6:.1f}M latent "
              f"({args.bitnet_dtype}) + {sum(p.numel() for p in emb+norms)/1e6:.1f}M tail, "
              f"AdamW fp32 states", flush=True)
    elif args.tail == "lean":
        # --tail lean: embedding (and the adapter, added later) in bf16, norm gains / row-column scales / attention
        # temperature in fp16, LeanAdam (int8 first moment, factored second moment, stochastic-rounding write-back)
        from .opt_lean import LeanAdam
        tail = model.float_tail_parameters()
        names = {id(p): n for n, p in model.named_parameters()}
        small = [p for p in tail if names[id(p)].endswith(("norm.weight", "qk_logscale", "row_scale", "col_scale"))]
        sid = {id(p) for p in small}
        emb = [p for p in tail if id(p) not in sid]
        for p in emb:
            p.data = p.data.to(torch.bfloat16)
        for p in small:
            p.data = p.data.to(torch.float16)
        rc = [p for p in small if names[id(p)].endswith(("row_scale", "col_scale"))]
        rcid = {id(p) for p in rc}
        norms = [p for p in small if id(p) not in rcid]
        tail_groups = [{"params": emb, "weight_decay": tc.weight_decay}, {"params": norms, "weight_decay": 0.0}]
        if rc:
            tail_groups.append({"params": rc, "weight_decay": 0.0, "rc_lr": args.rc_lr})
        tail_groups = _split_emb_group(model, tail_groups, args.emb_wd, args.emb_lr_mult)
        tail_opt = LeanAdam(tail_groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))
        LeanAdam.BATCHED = bool(args.tail_batched)
        print(f"lean float tail: {sum(p.numel() for p in emb)/1e6:.1f}M bf16 + {sum(p.numel() for p in small)/1e3:.1f}k fp16, "
              f"LeanAdam (int8 m, factored v, stochastic rounding)", flush=True)
    elif args.tail == "fp32":
        # float tail kept in fp32 with a standard AdamW, exactly like master mode. In
        # bf16 an Adam step (~lr <= 1.5e-3) on a norm gain of 1.0 is below half the
        # representable gap (2^-7) and always rounds away: the gains never train.
        tail = model.float_tail_parameters()
        for p in tail:
            p.data = p.data.float()
        tids = {id(p) for p in tail}
        rc = [p for n, p in model.named_parameters() if n.endswith(("row_scale", "col_scale"))]
        rcid = {id(p) for p in rc}
        norms = [p for n, p in model.named_parameters() if id(p) in tids and id(p) not in rcid
                 and (n.endswith("norm.weight") or n.endswith("qk_logscale"))]
        nids = {id(p) for p in norms} | rcid
        emb = [p for p in tail if id(p) not in nids]
        tail_groups = [{"params": emb, "weight_decay": tc.weight_decay},
                       {"params": norms, "weight_decay": 0.0}]
        if rc:
            tail_groups.append({"params": rc, "weight_decay": 0.0, "rc_lr": args.rc_lr})
        tail_groups = _split_emb_group(model, tail_groups, args.emb_wd, args.emb_lr_mult)
        tail_opt = torch.optim.AdamW(tail_groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))
        print(f"fp32 float tail: {sum(p.numel() for p in emb)/1e6:.1f}M emb + "
              f"{sum(p.numel() for p in norms)/1e3:.1f}k norm gains, AdamW fp32 states", flush=True)
    else:
        # float tail (embeddings + norms + flip predictor): bf16 params + 8-bit Adam.
        # NOTE: norm gains cannot train in bf16 (see --tail fp32).
        from .opt8 import Adam8bit
        rc = [p for n, p in model.named_parameters() if n.endswith(("row_scale", "col_scale"))]
        rcid = {id(p) for p in rc}
        for p in model.float_tail_parameters():
            if id(p) not in rcid:                 # scales stay fp32: bf16 would freeze them
                p.data = p.data.to(torch.bfloat16)
        tail = model.float_tail_parameters()
        groups = [{"params": [p for p in tail if id(p) not in rcid],
                   "weight_decay": tc.weight_decay}]
        if rc:
            groups.append({"params": rc, "weight_decay": 0.0, "rc_lr": args.rc_lr})
        groups = _split_emb_group(model, groups, args.emb_wd, args.emb_lr_mult)   # (--emb-wd / --emb-lr-mult were ignored
        tail_opt = Adam8bit(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2),          # on this tail before 25 Oct)
                            weight_decay=tc.weight_decay)

    STATE.weight_decay = tc.weight_decay

    if args.track_flips:
        n = enable_flip_tracking(model)
        print(f"flip tracking on for {n} ternary layers", flush=True)

    sampler = torch.Generator().manual_seed(tc.seed + 1_000_003 * _RK)   # data parallel: each rank its own batches
    lr_state = {}
    _ddt = np.uint32 if args.u32 else np.uint16      # --data-dtype u32: token ids above 65535 (Qwen's 248k vocabulary)
    train_data = np.memmap(tc.data_path, dtype=_ddt, mode="r")
    val_data = np.memmap(tc.val_path, dtype=_ddt, mode="r") if os.path.exists(tc.val_path) else train_data

    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    metrics_path = os.path.join(tc.out_dir, "metrics.jsonl")
    metrics_f = open(metrics_path, "a")

    def log_metrics(rec):
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    def _opt_sd():    # --pipe: without the other stages' optimizer state (on meta after a resume from a full checkpoint)
        sd_ = tail_opt.state_dict()
        if _PP:
            sd_["state"] = {k_: v_ for k_, v_ in sd_["state"].items()
                            if not any(torch.is_tensor(t_) and t_.is_meta for t_ in v_.values())}
        return sd_

    def save_ckpt(step):
        if _WS > 1 and _RK > 0:     # data parallel: rank 0 saves for all (same trits, state and optimizer on every rank)
            return
        # atomic: write tmp then rename, so a kill mid-save never corrupts ckpt.pt
        if args.ttf_offload == "async" and torch.cuda.is_available():
            torch.cuda.synchronize()      # the host copies of the state are written on a side stream
        tmp = ckpt_path + ".tmp"
        # flip-tracking masks are plain attributes (not state_dict buffers), so they
        # are saved alongside: otherwise "never flipped since init" silently resets
        # to 100% on resume.
        touched = {n: m.touched for n, m in model.named_modules()
                   if hasattr(m, "touched") and m.touched is not None}
        # step the mask started accumulating from: never-flipped is a "since init"
        # number only when this is 0.
        touched_origin = _touched_origin["v"]
        torch.save({"model": {k_: v_ for k_, v_ in model.state_dict().items() if not v_.is_meta},   # --pipe: own stage's
                    "opt": _opt_sd(), "pipe": (model.pp if _PP else None),
                    "touched": touched, "touched_origin": touched_origin,
                    "cfg": mc, "step": step, "mode": kmode, "beta": use_beta, "args": vars(args),
                    "int8": i8, "int8_dx": i8dx,
                    "dw_mode": "dense",    # (read by the evaluation scripts of checkpoints before 17 Oct 2026)
                    # low-rank momentum (U, V) per ternary layer, in module order
                    "lowrank": [lr_state.get(id(m)) for m in model.modules()
                                if isinstance(m, KernelTernaryLinear)] if lr_state else None,
                    # TTF: the rest of the per-layer state (update count, factored v, short momentum), module order
                    "ts_state": [{k[0]: st_[k] for k in (("t", id(m)), ("v", id(m)), ("m", id(m))) if k in st_}
                                 for m in model.modules() if isinstance(m, KernelTernaryLinear)]
                                if (st_ := lr_state.get("_ts")) else None,
                    # --tern-emb: the ternary embedding matrices' TTF state (U V, update count, factored v)
                    "tern_emb_state": [(lr_state.get(id(m)), {k[0]: lr_state.get("_ts", {})[k] for k in
                                        (("t", id(m)), ("v", id(m))) if k in lr_state.get("_ts", {})})
                                       for m in _tern_lays(model)] if te and lr_state else None,
                    # --intk-emb: the int-k embedding / head matrices' TTF state, the same form
                    "intk_state": [(lr_state.get(id(m)), {k[0]: lr_state.get("_ts", {})[k] for k in
                                    (("t", id(m)), ("v", id(m)), ("f", id(m))) if k in lr_state.get("_ts", {})})
                                   for m in _intk_lays(model)] if args.intk_emb and lr_state else None}, tmp)
        os.replace(tmp, ckpt_path)

    mag_on = [False]
    def enable_mag(step0):
        # the new parameters join the float tail (the look-ahead passes and the gradient clip treat them like the
        # tail) as their own AdamW group. Enabled after the resume when branching a checkpoint without them (so it
        # loads unchanged), before it when the checkpoint already has them (so they and their AdamW state load)
        kind, r = mag.split(":")
        newp = []
        from .master import MasterTernaryLinear
        for m in model.modules():
            if isinstance(m, (KernelTernaryLinear, MasterTernaryLinear)):
                newp += m.enable_lowrank_mag(kind, int(r))
        if not newp:        # --mode fp32: no ternary layers to adapt (an empty group would break a later resume)
            mag_on[0] = True
            return
        if args.tail == "lean":
            for p_ in newp:
                p_.data = p_.data.to(torch.bfloat16)
        tail.extend(newp)
        tail_opt.add_param_group({"params": newp, "weight_decay": args.adapter_wd, "lr_mult": args.adapter_lr_mult,
                                  "lr": lr_at(step0, tc) * args.adapter_lr_mult})
        mag_on[0] = True
        print(f"low-rank magnitude '{kind}' rank {r}: {sum(p.numel() for p in newp) / 1e6:.2f}M floats "
              f"(wd {args.adapter_wd}, lr x{args.adapter_lr_mult})", flush=True)

    def adapter_rank_at(step):   # --adapter-schedule R0:R1[:START:END[:STEP]]: the target rank at this step
        f = args.adapter_schedule.split(":") + [None] * 5
        r0, r1 = int(f[0]), int(f[1])
        s0, s1 = (float(f[2]) if f[2] else 0.125), (float(f[3]) if f[3] else 0.75)
        st = int(f[4]) if f[4] else 8
        u = min(max((step / max(1, tc.max_steps) - s0) / max(s1 - s0, 1e-9), 0.0), 1.0)
        return max(r1, r0 - int(u * (r0 - r1) / st + 1e-9) * st)

    @torch.no_grad()
    def shrink_adapters(r_new):
        """every adapter A B^T (rank above r_new) replaced by its best rank-r_new part (QR of A and B, SVD of the r x r
        core, the singular values split evenly); the adapters' optimizer state restarts"""
        from .master import MasterTernaryLinear
        gi = next(i for i, g in enumerate(tail_opt.param_groups) if "lr_mult" in g)
        old, new, err = tail_opt.param_groups[gi]["params"], [], []
        for m in model.modules():
            if isinstance(m, (KernelTernaryLinear, MasterTernaryLinear)) and getattr(m, "mag_A", None) is not None:
                if m.mag_A.shape[1] > r_new:
                    A, B = m.mag_A.float(), m.mag_B.float()
                    Qa, Ra = torch.linalg.qr(A); Qb, Rb = torch.linalg.qr(B)
                    U_, S_, Vh_ = torch.linalg.svd(Ra @ Rb.T)
                    s_ = S_[:r_new].sqrt()
                    err.append(float((S_[r_new:] ** 2).sum() / (S_ ** 2).sum().clamp_min(1e-30)))
                    m.mag_A = nn.Parameter((Qa @ U_[:, :r_new] * s_).to(m.mag_A.dtype))
                    m.mag_B = nn.Parameter((Qb @ Vh_[:r_new].T * s_).to(m.mag_B.dtype))
                new += [m.mag_A, m.mag_B]
        ids = {id(p) for p in old}
        for p in old:
            tail_opt.state.pop(p, None)
        tail_opt.param_groups[gi]["params"] = new
        tail[:] = [p for p in tail if id(p) not in ids] + new
        if hasattr(tail_opt, "_flatbufs"):
            tail_opt._flatbufs.clear()          # LeanAdam's batched buffers are keyed by the old parameters
        return (sum(err) / len(err)) if err else 0.0

    start_step = 0
    _touched_origin = {"v": 0}
    _load_path = _resume_path
    if _PP and args.resume:
        print(f"--pipe: stage {_PP[0]} resumes from {_load_path}", flush=True)
    if _resuming and os.path.exists(_load_path):
        # to host memory: every piece is copied to its device below (on the GPU the whole checkpoint would sit next
        # to the model it is loaded into: the 27B's 10.5 GB did not fit)
        blob = torch.load(_load_path, map_location="cpu", weights_only=False, mmap=True)
        _acc = 1.0                     # the accumulators' --ttf-acc-scale (set with the ternary layers' state below)
        if blob.get("beta", False) != use_beta:
            raise SystemExit(f"checkpoint beta={blob.get('beta', False)} but run beta="
                             f"{use_beta}: forward differs, refusing to resume")
        _cx = (blob.get("args") or {}).get("ext")     # --ext: the structural plug-ins must match the checkpoint's
        if _cx is not None and _X.structural(_cx) != _X.structural(args.ext):   # (None: written before --ext existed)
            raise SystemExit(f"checkpoint was trained with --ext {_X.structural(_cx)} but this run has "
                             f"{_X.structural(args.ext)}: the structure would differ, refusing to resume")
        if mag and any(k.endswith("mag_A") or ".mag_At." in k for k in blob["model"]):
            if args.adapter_schedule:   # the schedule continues from the checkpoint's rank (no growth back to R0)
                mag = f"add:{next(v_.shape[1] for k_, v_ in blob['model'].items() if k_.endswith('mag_A'))}"
            _R = int(mag.split(":")[1]); _grown = 0
            for k_ in [k_ for k_ in blob["model"] if k_.endswith("mag_A")]:   # a larger 'add' rank than the checkpoint's:
                A_, B_ = blob["model"][k_], blob["model"][k_[:-1] + "B"]      # new columns of A at 0 (the layer is
                if A_.shape[1] < _R:                                           # unchanged), of B random as at init
                    g_ = torch.Generator().manual_seed(17_000_000 + _grown)
                    blob["model"][k_] = torch.cat([A_, torch.zeros(A_.shape[0], _R - A_.shape[1], dtype=A_.dtype)], 1)
                    blob["model"][k_[:-1] + "B"] = torch.cat([B_, (torch.randn(B_.shape[0], _R - B_.shape[1], generator=g_)
                                                                  * (1.0 / B_.shape[0]) ** 0.5).to(B_.dtype)], 1)
                    _grown += 1
            enable_mag(int(blob.get("step", 0)) + 1)
            if _grown and "opt" in blob:   # the adapters' optimizer state starts fresh (its shapes are the old rank's)
                _gi = len(tail_opt.param_groups) - 1
                for i_ in blob["opt"]["param_groups"][_gi]["params"]:
                    blob["opt"]["state"].pop(i_, None)
                print(f"--adapter-rank {mag}: {_grown} adapters grown from the checkpoint's rank "
                      f"(new columns of A at 0; their optimizer state restarts)", flush=True)
        new_rc = args.rc_scale and not any(k.endswith("row_scale") for k in blob["model"])
        if new_rc:
            # branching a checkpoint trained without row/column scales: they start at 1 (the layer is unchanged)
            # and their optimizer group starts fresh; everything else loads as saved
            missing, unexpected = model.load_state_dict(blob["model"], strict=False)
            assert all(k.endswith(("row_scale", "col_scale")) for k in missing) and not unexpected, (missing, unexpected)
            rc_groups = [g for g in tail_opt.param_groups if "rc_lr" in g]
            tail_opt.param_groups[:] = [g for g in tail_opt.param_groups if "rc_lr" not in g]
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
            for g in rc_groups:
                tail_opt.add_param_group(g)
            print(f"row/column scales added to a checkpoint without them ({len(missing)} tensors, start at 1)", flush=True)
        elif _PP:    # --pipe: this stage's tensors must all be there; the other stages' (meta) are skipped
            missing, unexpected = model.load_state_dict(blob["model"], strict=False)
            _md = dict(model.state_dict())
            _miss = [k_ for k_ in missing if not _md[k_].is_meta]
            assert not _miss and not unexpected, (f"--pipe: {_load_path} lacks this stage's {_miss[:4]}", unexpected[:4])
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
        else:
            model.load_state_dict(blob["model"])
            if "opt" in blob:
                tail_opt.load_state_dict(blob["opt"])
        start_step = int(blob.get("step", 0)) + 1
        # fast-forward the data streams, so a resumed run sees the batches the uninterrupted
        # run would have (not the step-0 batches again)
        for _ in range(start_step * tc.grad_accum):
            torch.randint(len(train_data) - tc.seq_len - 1, (tc.batch_size,), generator=sampler)
        _lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
        _sdev = {id(m): ("cpu" if _ts_on_host(args, i, len(_lays)) else device) for i, m in enumerate(_lays)}
        if blob.get("lowrank"):
            lays = _lays
            for m, uv in zip(lays, blob["lowrank"]):
                if uv is not None and not m.wpacked.is_meta:
                    lr_state[id(m)] = tuple(t.to(_sdev[id(m)]) for t in uv)
            print(f"resumed low-rank momentum for {len(lr_state)} layers", flush=True)
            if ttf and args.ttf_rank:   # a larger --lowrank than the checkpoint's: grow each accumulator (the --ttf-accum micro
                # path never resizes): V gains orthonormal columns outside its span, U zeros there, so A = U V^T is
                # unchanged and the new directions start empty
                _sa = _store_codec(store)
                _ng = 0
                for i_, m in enumerate(lays):
                    if id(m) not in lr_state or m.wpacked.is_meta:
                        continue
                    U_, V_ = _load_pair(lr_state[id(m)])
                    ra_ = min(getattr(m, "ts_rank", None) or args.ttf_rank, m.N, m.K)
                    if V_.shape[1] >= ra_:
                        continue
                    dv_ = V_.device; U_, V_ = U_.to(device).float(), V_.to(device).float()
                    g_ = torch.Generator(device=device).manual_seed(9_000_001 + 2 * i_)
                    X_ = torch.randn(m.K, ra_ - V_.shape[1], device=device, generator=g_)
                    X_ = X_ - V_ @ (V_.T @ X_); X_ = X_ - V_ @ (V_.T @ X_)
                    V2 = torch.cat([V_, _orth(X_)], 1); U2 = torch.cat([U_, torch.zeros(m.N, ra_ - V_.shape[1], device=device)], 1)
                    pr_ = _store_pair(_sa, U2, V2) if _sa else (U2, V2)
                    lr_state[id(m)] = tuple(t.to(dv_) for t in pr_); _ng += 1
                if _ng:
                    print(f"--lowrank {args.ttf_rank}: grew the accumulators of {_ng} layers from the checkpoint's rank",
                          flush=True)
            _ba = blob.get("args") or {}   # (checkpoints before 17 Oct: the old option names)
            _acc, _amsg = resume_acc_scale(_ba.get("ttf_theta", _ba.get("ts_theta")), args.ttf_theta, args.ttf_acc_scale,
                                           _ba.get("ttf_theta_sched", _ba.get("ts_theta_sched")),
                                           args.ttf_theta_sched) if ttf else (1.0, None)
            if ttf and _acc != 1.0:   # --ttf-acc-scale F (a new theta = F x the run's): every accumulator x F,
                _sa = _store_codec(store)   # so each entry keeps its distance to the bar (no burst of moves)
                for m in lays:
                    if id(m) in lr_state and not m.wpacked.is_meta:
                        U_, V_ = _load_pair(lr_state[id(m)]); dv_ = U_.device
                        U_ = U_.to(device).float() * _acc; V_ = V_.to(device).float()
                        pr_ = _store_pair(_sa, U_, V_) if _sa else (U_, V_)
                        lr_state[id(m)] = tuple(t.to(dv_) for t in pr_)
            if _amsg:
                print(f"*** {_amsg} ***", flush=True)
        for _key, _lays_of in (("tern_emb_state", _tern_lays), ("intk_state", _intk_lays)):
          if not blob.get(_key):
            continue
          if ttf and _acc != 1.0:     # (their accumulators are not rescaled: a changed theta would fire them in a burst)
            raise SystemExit(f"--ttf-acc-scale {_acc}: not implemented for the {_key} accumulators; resume at the "
                             f"checkpoint's theta")
          st_ = lr_state.setdefault("_ts", {})
          for m, (uv, d_) in zip(_lays_of(model), blob[_key]):
            if uv is not None:
                lr_state[id(m)] = tuple(t.to(device) for t in uv)
            for k_, v_ in d_.items():
                st_[(k_, id(m))] = (tuple(x.to(device) if torch.is_tensor(x) else x for x in v_) if isinstance(v_, tuple)
                                    else v_)    # (("f": the int-k store's chunking, ints)
          print(f"resumed the {_key} TTF state of {len(blob[_key])} matrices", flush=True)
        if blob.get("ts_state"):
            lays = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
            st_ = lr_state.setdefault("_ts", {})
            for m, d_ in zip(lays, blob["ts_state"]):
                for k_, v_ in (d_.items() if not m.wpacked.is_meta else ()):
                    st_[(k_, id(m))] = tuple(x.to(_sdev[id(m)]) for x in v_) if isinstance(v_, tuple) else v_
            print(f"resumed TTF state (short momentum, v, step counts) for {len(lays)} layers", flush=True)
        tmask = blob.get("touched") or {}
        nres = 0
        for n, m in model.named_modules():
            if n in tmask and getattr(m, "touched", None) is not None:
                m.touched.copy_(tmask[n].to(m.touched.device)); nres += 1
        if nres:
            _touched_origin["v"] = int(blob.get("touched_origin", 0))
            print(f"resumed from {ckpt_path} at step {start_step} (flip history for "
                  f"{nres} layers, accumulated since step {_touched_origin['v']})",
                  flush=True)
        else:
            _touched_origin["v"] = start_step
            print(f"resumed from {ckpt_path} at step {start_step}", flush=True)
            if args.track_flips:
                print(f"  WARNING: checkpoint has no flip history -> 'never flipped' "
                      f"counts only from step {start_step}, NOT since init. The true "
                      f"since-init value is <= the last value logged before this "
                      f"resume (it is monotone non-increasing).", flush=True)
        del blob
        torch.cuda.empty_cache()

    # save on SIGTERM/SIGINT so `kill` (or Ctrl-C) leaves a fresh resumable ckpt
    import signal
    # the checkpoint's "step" is the last COMPLETED step (resume starts at step + 1), so the
    # handler only sets a flag and the loop saves and exits at the end of the current step
    _last = {"stop": False}
    def _graceful(signum, frame):
        print(f"signal {signum}: saving ckpt after the current step ...", flush=True)
        _last["stop"] = True
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    if _RK == 0 and (not _PP or _PP[0] == 0):   # the results database: this run's records in <out_dir>/records.jsonl
        from .results import RunLog
        RUNLOG[0] = RunLog(tc.out_dir, os.path.basename(os.path.normpath(tc.out_dir)), vars(args), data=tc.data_path,
                           steps=tc.max_steps, parent=(ckpt_path if start_step > 0 else None))
        RUNLOG[0].event("running", start_step)

    # after any resume, so 'net' counts from the weights this run starts at
    if mag and not mag_on[0]:
        enable_mag(start_step)
    t0 = time.time()
    last_save = time.time()
    last_ = max(tc.max_steps, args.extend_to or 0)   # --extend-to N: keep training past the schedule (lr held at min_lr)
    end_step = last_ if args.stop_after is None else min(last_, args.stop_after)
    prof = None
    ts_cfg = None
    if ttf:
        _lays = [m_ for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)]
        for i_, m_ in enumerate(_lays):   # --ttf-offload host(_keep): which layers' state lives in host memory
            m_._ts_host = _ts_on_host(args, i_, len(_lays))
        if args.ttf_offload_keep and args.ttf_offload != "none":
            print(f"--ttf-offload-keep: state of {sum(not m_._ts_host for m_ in _lays)} of {len(_lays)} layers on the GPU",
                  flush=True)
        ts_cfg = dict(r=args.ttf_rank, lr_ratio=1.0, theta=args.ttf_theta, tau=args.ttf_tau, b2=args.ttf_b2, spend=args.ttf_spend,
                      tau_anneal=(args.ttf_leak_pow if args.ttf_leak == "annealed" else False), store_a=store,
                      offload={"none": False, "host": True, "async": "async"}[args.ttf_offload],
                      theta_sched=(tuple(float(x) for x in args.ttf_theta_sched.split(":")) if args.ttf_theta_sched else None),
                      progress=0.0, factored=args.factored_k)
    if ttf and args.intk_emb and args.intk_update == "factored":   # the int-k matrices update inside their backward
        for m_ in _intk_lays(model):
            m_._ts_state, m_._ts_cfg = lr_state, ts_cfg
            m_._ts_host = args.ttf_offload != "none"
    if ttf and args.ttf_update == "fused":         # --ttf-update fused: each layer's update inside its backward (ts_layer)
        for i_, m_ in enumerate([m_ for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)]):
            m_._ts_idx = i_
            if _WS > 1:   # data parallel: every rank's TTF update sees the mean gradient, so all ranks move the same trits
                def _og(layer, g, _c=ts_cfg):
                    if torch.is_tensor(g):     # dense: the mean gradient; factored (gy, x): the update averages its products
                        _dist.all_reduce(g); g.div_(_WS)
                    ts_layer(layer, g, lr_state, _c)
                m_.on_grad = _og
            else:
                def _og1(layer, g, _c=ts_cfg):
                    with _rf("ttf.update"):    # (a range per layer call: only while a profiler runs)
                        ts_layer(layer, g, lr_state, _c)
                m_.on_grad = _og1
            m_.ts_micro = bool(args.micro)
        print("--ttf-update fused: the update runs inside each layer's backward", flush=True)
        FIRE_FUSED[0] = args.fire == "fused"
        if args.fused_save_x:
            from .layers import FUSED_SAVE_X as _FX
            _FX[0] = True
            print("--fused-save-x: fused layers keep their input, the centring redone in the backward", flush=True)
        if args.ttf_step_fp32:
            from .update import STEP_FP32 as _SF
            _SF[0] = True
            print("--ttf-step-fp32: U in fp32 between micro-batches, encoded once per step", flush=True)
        if args.ttf_orth != "qr":
            from .update import ORTH as _OR
            _OR[0] = args.ttf_orth
            print(f"--ttf-orth {args.ttf_orth}", flush=True)
        if args.factored_k:
            assert args.micro or tc.grad_accum == 1, "--dw factored: 16 x 1 or --ttf-accum micro"
            print(f"--dw factored {args.factored_k}: TTF from the gradient's factors (no dense weight gradient)", flush=True)
            if args.dw_thin == "dense":   # thin layers (adapter factors): the dense gradient is cheaper than the factors
                _thin = [m_ for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)
                         and m_.N * m_.K < (m_.N + m_.K) * (args.factored_k
                                                          + 2 * min(getattr(m_, "ts_rank", None) or args.ttf_rank, m_.N, m_.K))]
                for m_ in _thin:
                    m_._dw_dense = True
                print(f"--dw-thin dense: {len(_thin)} thin layers updated from the dense gradient "
                      f"(N K < (N + K)(k + 2 r))", flush=True)
    if args.w_cache != "off" or args.dw_precision == "fp32":
        from .layers import WCACHE as _WC, DW_FP32 as _DF
        _WC["mode"] = args.w_cache; _DF[0] = args.dw_precision == "fp32"
        print(f"--w-cache {args.w_cache}, --dw-precision {args.dw_precision}", flush=True)
    if args.fused_scales:
        from .layers import FUSED_SCALES as _FSC
        _FSC[0] = True
    if args.fused_layer:
        from .layers import FUSED_LAYER as _FLY
        _FLY[0] = True
        print("--fused-layer: each ternary layer's elementwise work fused around its int8 GEMMs", flush=True)
    if args.fused_norm and _QWEN:
        from .qwen import enable_fused_norm
        enable_fused_norm()
        print("--fused-norm: Qwen RMSNorm in one kernel each way", flush=True)
    if args.eval_first_n:   # --eval-first-n N: the model as built or resumed (e.g. right after --init-hf), scored on
        model.eval(); STATE.updates_enabled = False   # the first N val windows in order (qwen_check.py's windows)
        ls_ = []
        with torch.no_grad():
            for i_ in range(args.eval_first_n):
                w_ = torch.from_numpy(val_data[i_ * tc.seq_len:(i_ + 1) * tc.seq_len + 1].astype(np.int64)).to(device)[None]
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    ls_.append(float(model(w_[:, :-1], w_[:, 1:])[1]))
        print(f"eval_first_n: loss {np.mean(ls_):.4f} over the first {args.eval_first_n} val windows ("
              + " ".join(f"{x:.4f}" for x in ls_) + ")", flush=True)
        log_metrics({"step": 0, "first_n_loss": float(np.mean(ls_))})
        model.train(); STATE.updates_enabled = True
    for step in range(start_step, end_step):
        if args.profile and step == start_step + 20:         # after autotune / warm-up
            from torch.profiler import profile, ProfilerActivity
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__(); t_prof = time.time()
        if prof is not None and step == start_step + 20 + args.profile:
            torch.cuda.synchronize(); prof.__exit__(None, None, None)
            wall = (time.time() - t_prof) / args.profile
            ka = prof.key_averages()
            gpu = sum(e.self_device_time_total for e in ka) / 1e6 / args.profile
            print(f"PROFILE {args.profile} steps: wall {wall:.3f} s/step, GPU kernel time {gpu:.3f} s/step "
                  f"({gpu / wall * 100:.0f}% busy)", flush=True)
            print(ka.table(sort_by="self_device_time_total", row_limit=30), flush=True)
            print(ka.table(sort_by="self_cpu_time_total", row_limit=30), flush=True)
            prof.export_chrome_trace(os.path.join(tc.out_dir, "profile_trace.json"))
            raise SystemExit(0)
        # thermal guard: PAUSE (not stop) while GPU is at/above max_temp; resume
        # in place once it cools. No exit, no save — just wait it out.
        if step % args.temp_check == 0:
            temp = gpu_temp()
            if temp is not None and temp >= args.max_temp:
                print(f"GPU {temp}C >= {args.max_temp}C -> pausing to cool "
                      f"(step {step})", flush=True)
                resume_at = min(args.resume_temp, args.max_temp - 1)
                while temp is not None and temp > resume_at:
                    time.sleep(5)
                    temp = gpu_temp()
                print(f"cooled to {temp}C -> resuming", flush=True)
        lr = lr_at(step, tc)
        if ts_cfg is not None:
            ts_cfg["lr_ratio"] = _ts_scale(args, step, tc, lr)
            ts_cfg["progress"] = step / max(1, tc.max_steps)
            ts_cfg["step"] = step; ts_cfg["start"] = args.ttf_start
            ts_cfg["vdiag"] = bool(args.ttf_vdiag) and step % args.ttf_vdiag == 0   # diagnostics only
            ts_cfg["micro"] = tc.grad_accum if args.micro else 1
            if args.factored_k:
                from .layers import FACTORED as _FCT
                _FCT[0] = True          # (--ttf-vdiag reads its g^2 / v from the sketch)
        if args.adapter_schedule and mag_on[0]:
            r_t = adapter_rank_at(step)
            r_c = next((m_.mag_A.shape[1] for m_ in model.modules() if getattr(m_, "mag_A", None) is not None), r_t)
            if r_t < r_c:
                e_ = shrink_adapters(r_t)
                log_metrics({"step": step, "adapter_rank": r_t, "adapter_dropped_energy": e_})
                print(f"--adapter-schedule: step {step}: adapters {r_c} -> {r_t} (dropped {e_:.4f} of their energy)",
                      flush=True)
        STATE.lr = lr
        for g in tail_opt.param_groups:
            g["lr"] = (g["rc_lr"] if g.get("rc_lr") else lr) * g.get("lr_mult", 1.0) * g.get("emb_mult", 1.0)
        STATE.updates_enabled = True
        STATE.grad_scale = tc.grad_accum
        tail_opt.zero_grad(set_to_none=True)
        last_loss = 0.0
        if args.ttf_rank:          # TTF: every ternary layer hands its gradient to the update (ts_layer)
            for l in model.modules():
                if isinstance(l, KernelTernaryLinear): l.capture = True
        if TIME: torch.cuda.synchronize(); _t0 = time.time()
        _prof = None      # TERN_PROF=DIR: one profiled step (the 3rd of the run), GPU time by operation into DIR
        if os.environ.get("TERN_PROF") and step == start_step + 2:
            from torch.profiler import profile, ProfilerActivity
            _prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True)
            _prof.__enter__(); torch.cuda.synchronize(); _pt0 = time.time()
        if _PP:   # --pipe: the step's micro-batches (the windows the single-GPU loop reads) through the pipeline
            _bt = []
            for micro in range(tc.grad_accum):
                if args.kd_cache:
                    x, y, kd_ = _kd_cache_batch(args, (step - args.kd_cache_start) * tc.grad_accum + micro, train_data,
                                                tc.seq_len, device, bs=tc.batch_size)
                else:
                    (x, y), kd_ = get_batch(train_data, tc.batch_size, tc.seq_len, device, sampler), None
                _bt.append((x, y, kd_))

            def _pre_bwd(i_):
                if ttf and tc.grad_accum > 1:
                    for l_ in model.modules():
                        if isinstance(l_, KernelTernaryLinear):
                            l_.cap_accum = True; l_.cap_final = i_ == tc.grad_accum - 1
                    for l_ in (_intk_lays(model) if args.intk_emb else ()):
                        l_._ts_final = i_ == tc.grad_accum - 1
            last_loss = model.pp_train(_bt, _pre_bwd, lambda: torch.autocast(device_type=device.split(":")[0],
                                                                             dtype=torch.bfloat16)).item()
            del _bt
        for micro in range(0 if _PP else tc.grad_accum):
            if ttf and tc.grad_accum > 1:      # TTF with accumulation: sum the captured micro-batch gradients
                for l_ in model.modules():
                    if isinstance(l_, KernelTernaryLinear):
                        l_.cap_accum = True; l_.cap_final = micro == tc.grad_accum - 1
                for l_ in (_intk_lays(model) if args.intk_emb else ()):
                    l_._ts_final = micro == tc.grad_accum - 1
            if args.kd_cache:          # --kd-cache: the next precomputed window, its tokens and the teacher's top k
                # data parallel: each rank its own windows (rank 0 of 1 reads what a single run always read)
                x, y, model.kd_topk = _kd_cache_batch(args, ((step - args.kd_cache_start) * tc.grad_accum + micro) * _WS + _RK,
                                                      train_data, tc.seq_len, device, bs=tc.batch_size)
                if args.kd_cache_mode == "windows":      # the cache's windows, the online teacher's targets
                    model.kd_topk = None
            else:
                x, y = get_batch(train_data, tc.batch_size, tc.seq_len, device, sampler)
            if _teacher is not None:   # --distill: the full-precision model's next-token distribution for this batch
                with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    if args.kd_hidden:      # --kd-hidden: the teacher's stream at every block boundary too (one forward)
                        model.hid_t = _teacher.hiddens(x); model.hid_w = args.kd_hidden
                        tz_ = F.linear(_teacher.norm(model.hid_t[-1]),
                                       _teacher.tok_emb.weight if _teacher.lm_head is None else _teacher.lm_head.weight)
                    else:
                        tz_, _ = _teacher(x)
                tz_ = tz_.reshape(-1, tz_.shape[-1])   # in row chunks of 512: one chunk's fp32 copy is 0.47 GiB at a 248k
                # vocabulary (4096-row chunks made a 1.9 GiB transient, the 0.8B adapter step's memory peak); per-row ops: the same values
                model.kd_logp = torch.cat([F.log_softmax(c_.float(), -1).to(torch.bfloat16) for c_ in tz_.split(512)]); del tz_
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16), \
                    torch.profiler.record_function("model.fwd"):
                _, loss = model(x, y)
            with torch.profiler.record_function("model.bwd"):   # includes the fused TTF updates (ttf.update)
                (loss / tc.grad_accum).backward()
            model.kd_logp = None
            if args.kd_hidden:
                model.hid_t = None
                if micro == 0 and step % 50 == 0:
                    print(f"hid_kd: step {step} block-update error {model.hid_last:.4f}", flush=True)
            if args.kd_cache:
                model.kd_topk = None
            if te and args.mode != "bitnet":   # the ternary embedding's TTF update (both of its gradients)
                from .tern_emb import tern_emb_step
                tern_emb_step(model, lr_state, ts_cfg, ts_layer)
            if args.intk_emb and args.intk_update == "dense" and micro == tc.grad_accum - 1:   # after the last backward
                from .intk_emb import intk_layers
                for m_ in intk_layers(model):
                    m_.step(ts_layer, lr_state, ts_cfg)
            last_loss = loss.detach()          # read once after the last micro-batch (a .item() here synced every one)
        if torch.is_tensor(last_loss):
            last_loss = last_loss.item()
        if TIME:
            torch.cuda.synchronize(); _t1 = time.time()
            _m1, _p1 = torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30
        _rfu = torch.profiler.record_function("step.update"); _rfu.__enter__()   # TERN_PROF: after the backward
        if ttf:
            with torch.profiler.record_function("ttf.step"):
                rs_info = ts_step(model, lr_state, ts_cfg, fused=args.ttf_update == "fused")
        else:
            rs_info = None
        if _WS > 1:       # data parallel: the float tail's gradients are averaged before the clip and the step
            for p_ in tail:
                if p_.grad is not None:
                    _dist.all_reduce(p_.grad); p_.grad.div_(_WS)
        _clip_ps = tail
        _rfc = torch.profiler.record_function("tail.clip"); _rfc.__enter__()
        if _PP:   # --pipe: the clip's norm over all stages' float tails (each stage holds its own gradients)
            _gs = [p_.grad for p_ in _clip_ps if p_.grad is not None]
            _n2 = (torch.stack([torch.linalg.vector_norm(g_, 2).float() for g_ in _gs]) ** 2).sum() if _gs else \
                torch.zeros((), device=device)
            _n2 = _n2.reshape(1).cpu() if _dist.get_backend() == "gloo" else _n2.reshape(1)
            _dist.all_reduce(_n2)
            _cf = (tc.grad_clip / (_n2.to(device).sqrt() + 1e-6)).clamp(max=1.0)[0]
            for g_ in _gs:
                g_.mul_(_cf.to(g_.dtype))
        else:
            torch.nn.utils.clip_grad_norm_(_clip_ps, tc.grad_clip)
        _rfc.__exit__(None, None, None)
        if _WS > 1 and step % 100 == 0:   # data parallel: the ranks' trits must stay identical (same g, same rule)
            _ck = torch.tensor([float(sum(int(m_.wpacked.sum()) for m_ in model.modules() if isinstance(m_, KernelTernaryLinear)))],
                               device=device, dtype=torch.float64)
            _cks = [torch.zeros_like(_ck) for _ in range(_WS)]; _dist.all_gather(_cks, _ck)
            print(f"  data parallel step {step}: packed-trit checksums {[int(x.item()) for x in _cks]} "
                  f"({'in sync' if len({int(x.item()) for x in _cks}) == 1 else 'OUT OF SYNC'})", flush=True)
        with torch.profiler.record_function("tail.step"):
            tail_opt.step()
        _rfu.__exit__(None, None, None)
        if _prof is not None:
            torch.cuda.synchronize(); _pdt = time.time() - _pt0; _prof.__exit__(None, None, None)
            _pd = os.environ["TERN_PROF"]; os.makedirs(_pd, exist_ok=True)
            _ka = _prof.key_averages()
            with open(os.path.join(_pd, f"prof_step{step}.txt"), "w") as _f:
                _f.write(f"step {step} wall {_pdt:.2f}s\n")
                _f.write(_ka.table(sort_by="self_cuda_time_total", row_limit=60, max_name_column_width=70))
                _f.write("\n\nby CPU time\n" + _ka.table(sort_by="self_cpu_time_total", row_limit=30, max_name_column_width=70))
                _rng = [e for e in _ka if e.key.split(".")[0] in ("tern", "ttf", "model", "loss")]   # the named ranges
                _f.write("\n\nnamed ranges (CUDA total = everything inside them)\n")
                _ct = lambda e: max(getattr(e, "device_time_total", 0) or 0, getattr(e, "self_device_time_total", 0) or 0)
                for e in sorted(_rng, key=lambda e: -_ct(e)):   # (the GPU-side annotation of a range carries its time)
                    ct = _ct(e)
                    _f.write(f"  {e.key:28s} calls {e.count:7d}  CUDA {ct / 1e6:9.3f} s  CPU {e.cpu_time_total / 1e6:9.3f} s\n")
            with open(os.path.join(_pd, f"prof_step{step}_stack.txt"), "w") as _f:   # by the Python line that launched it
                if os.environ.get("TERN_PROF_NOSTACK"):   # (that table takes ~20 min on a 27B step)
                    _f.write("skipped (TERN_PROF_NOSTACK)\n")
                else:
                    _f.write(_prof.key_averages(group_by_stack_n=6).table(sort_by="self_cuda_time_total", row_limit=50,
                                                                           max_name_column_width=60))
            if os.environ.get("TERN_PROF_TRACE"):
                _prof.export_chrome_trace(os.path.join(_pd, f"trace_step{step}.json"))
            print(f"TERN_PROF: step {step} profiled ({_pdt:.1f}s wall) -> {_pd}", flush=True)
        if TIME:
            torch.cuda.synchronize(); _t2 = time.time()
            print(f"time step {step}: forward+backward {_t1 - _t0:.3f}s, update (momentum step / AdamW) {_t2 - _t1:.3f}s"
                  f" | GiB allocated after backward {_m1:.2f} (peak in fwd+bwd {_p1:.2f}), after update "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} (peak {torch.cuda.max_memory_allocated() / 2**30:.2f})",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()

        rec = {"step": step, "loss": last_loss, "lr": lr, "touched_origin": _touched_origin["v"],
               "tokens": (step + 1) * tc.grad_accum * tc.batch_size * tc.seq_len}
        fs = collect_flip_stats(model) if args.track_flips else {}
        if rs_info is not None:
            # device counters (look-ahead) become numbers here: one read per step, at logging
            rs_info = {k: (v.item() if torch.is_tensor(v) else v) for k, v in rs_info.items()}
            rec.update(rs_info)
        if fs:
            rec.update(flip_frac=fs["flip_frac_total"], never_frac=fs["never_frac_total"],
                       flip_frac_layers=fs["flip_frac"], never_frac_layers=fs["never_frac"])
        log_metrics(rec)

        if step % tc.log_interval == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            extra = ""
            if fs:
                extra += (f"| flip {fs['flip_frac_total']*100:.3f}% "
                          f"| never {fs['never_frac_total']*100:.1f}% ")
            print(f"step {step:6d} | loss {last_loss:6.3f} | lr {lr:.2e} {extra}"
                  f"| {dt:6.1f}s | peakVRAM {mem:.2f}GiB", flush=True)
            if ttf and step in (10, 1000):
                print(f"  TTF state: {_ts_state_bytes(lr_state) / 2**20:.1f} MiB stored", flush=True)
            if args.mem_snapshot and device == "cuda" and step == int(args.mem_snapshot.rsplit(":", 1)[1]):
                torch.cuda.memory._dump_snapshot(args.mem_snapshot.rsplit(":", 1)[0])    # scripts/analysis/mem_snapshot.py
                torch.cuda.memory._record_memory_history(enabled=None)
                print(f"--mem-snapshot: allocation history to step {step} in {args.mem_snapshot.rsplit(':', 1)[0]}", flush=True)
            if args.mem_report and step in (10, 50, 100, 1000):
                mr = mem_report(model, tail_opt, lr_state)
                log_metrics({"step": step, **mr})
                print("  memory (MiB): " + ", ".join(f"{k[4:-3]} {v:.1f}" for k, v in mr.items()), flush=True)
        if step > 0 and step % tc.eval_interval == 0:
            vl = evaluate(model, val_data, tc, device)
            log_metrics({"step": step, "val_loss": vl, "val_ppl": math.exp(vl)})
            print(f"  ---- val loss {vl:.3f} | ppl {math.exp(vl):.1f}", flush=True)
        if args.snap_every and step > 0 and step % args.snap_every == 0:
            save_ckpt(step)
            if not (_WS > 1 and _RK > 0):
                shutil.copyfile(ckpt_path, os.path.join(tc.out_dir, f"ckpt_{step}.pt"))
        if time.time() - last_save >= args.save_secs:
            save_ckpt(step)
            last_save = time.time()
            print(f"  ---- checkpoint saved at step {step}", flush=True)
        if _last["stop"]:
            save_ckpt(step); print(f"saved at step {step}. exiting.", flush=True)
            if RUNLOG[0] is not None:
                RUNLOG[0].event("stopped", step, "signal")
            raise SystemExit(0)

    if args.final_save:
        save_ckpt(end_step - 1)
    vl = evaluate(model, val_data, tc, device)
    log_metrics({"step": end_step, "val_loss": vl, "val_ppl": math.exp(vl),
                 "final": True})
    print(f"FINAL val loss {vl:.4f} | ppl {math.exp(vl):.2f}", flush=True)
    if RUNLOG[0] is not None:
        RUNLOG[0].result("val_train", vl, n=tc.eval_iters * max(tc.batch_size, 16), step=end_step)
        RUNLOG[0].event("finished", end_step)
    print("done", flush=True)


RUNLOG = [None]          # the run's results.RunLog (rank 0)


def cli():
    """python -m ttf.train / the ttf-train command: main(), with a 'failed' event in the run's records on an error"""
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        if RUNLOG[0] is not None:
            RUNLOG[0].event("failed", reason=f"{type(e).__name__}: {str(e)[:300]}")
        raise


if __name__ == "__main__":
    cli()
