"""Schedules: the learning rate (cosine, linear, WSD, rewarm), the TTF step scale and the theta schedule."""
from __future__ import annotations
import math

from .config import TrainConfig


LR_LINEAR = [False]               # --lr_linear: linear decay after warmup instead of the cosine
LR_WSD = [0]                      # --lr_wsd D: warmup, constant peak lr until step D, then a cosine to min_lr at the end
LR_REWARM = [0, 0]                # --lr_rewarm S:W: from step S, linear warmup over W steps from min_lr to the peak, then cosine


def _ts_scale(args, step, tc, lr):
    """the TTF step scale (lr / peak lr)^p, p = --ts_anneal"""
    return max((lr / tc.lr) ** args.ts_anneal, 0.0)


def _theta_at(sched, prog):
    """--ts_theta_sched T0:T1[:START]: T0 until fraction START of training (default 0), then linear to T1 at the end"""
    th0_, th1_, st_ = (tuple(sched) + (0.0,))[:3]
    prog = min(max(prog, 0.0), 1.0)
    return th0_ if prog <= st_ else th0_ + (th1_ - th0_) * (prog - st_) / max(1e-9, 1.0 - st_)


def lr_at(step, tc: TrainConfig):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step > tc.max_steps:
        return tc.min_lr
    if LR_REWARM[1] and step >= LR_REWARM[0]:
        # continued training of a decayed checkpoint: re-warm from min_lr to the peak, then a fresh cosine to the end
        s0, w = LR_REWARM[:2]
        pk = tc.lr * (LR_REWARM[2] if len(LR_REWARM) > 2 else 1.0)   # S:W:F re-warms to F x the peak (TTF's step too)
        if step < s0 + w:
            return tc.min_lr + (pk - tc.min_lr) * (step - s0 + 1) / w
        r = (step - s0 - w) / max(1, tc.max_steps - s0 - w)
        return tc.min_lr + 0.5 * (pk - tc.min_lr) * (1 + math.cos(math.pi * r))
    if LR_WSD[0]:
        # warmup-stable-decay: the peak lr until step D, then a cosine down to min_lr at the last step
        if step < LR_WSD[0]:
            return tc.lr
        r = (step - LR_WSD[0]) / max(1, tc.max_steps - LR_WSD[0])
        return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))
    r = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    if LR_LINEAR[0]:              # --lr_linear: a straight line from the peak to min_lr (to zero with --min_lr 0)
        return tc.min_lr + (tc.lr - tc.min_lr) * (1.0 - r)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))
