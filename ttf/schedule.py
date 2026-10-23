"""Schedules: the learning rate (cosine, linear, WSD, rewarm), the TTF step scale and the theta schedule."""
from __future__ import annotations
import math

from .config import TrainConfig


LR_LINEAR = [False]               # --lr-schedule linear: linear decay after warmup instead of the cosine
LR_WSD = [0]                      # --lr-schedule wsd D: warmup, constant peak lr until step D, then a cosine to min_lr at the end
LR_REWARM = [0, 0]                # --lr-rewarm S:W: from step S, linear warmup over W steps from min_lr to the peak, then cosine


def _ts_scale(args, step, tc, lr):
    """the TTF step scale (lr / peak lr)^p, p = --ttf-step-p"""
    return max((lr / tc.lr) ** args.ttf_step_p, 0.0)


def _theta_at(sched, prog):
    """--ttf-theta-sched T0:T1[:START]: T0 until fraction START of training (default 0), then linear to T1 at the end"""
    th0_, th1_, st_ = (tuple(sched) + (0.0,))[:3]
    prog = min(max(prog, 0.0), 1.0)
    return th0_ if prog <= st_ else th0_ + (th1_ - th0_) * (prog - st_) / max(1e-9, 1.0 - st_)


def resume_acc_scale(old_theta, new_theta, acc_scale, old_sched=None, new_sched=None):
    """the accumulator scale of a resume (--ttf-acc-scale): resuming at a threshold different from the checkpoint's
    without rescaling fires, in its first update, every entry between the two thresholds at once (27B, theta 64 ->
    16: val 2.25 -> 7.50 in one step; 17 Oct and 23 Oct). So a changed theta needs an explicit --ttf-acc-scale (new /
    old keeps each entry's distance to the bar; 1 keeps the accumulators as they are); with a schedule on either side
    the check is skipped. Returns (scale, message or None); raises SystemExit when the scale is missing."""
    if acc_scale is not None:
        msg = (f"--ttf-acc-scale {acc_scale:g}: accumulators x {acc_scale:g} (checkpoint theta {old_theta}, now "
               f"{new_theta})" if acc_scale != 1.0 else None)
        return float(acc_scale), msg
    if old_sched or new_sched or old_theta is None or new_theta is None or float(old_theta) == float(new_theta):
        return 1.0, None
    raise SystemExit(f"resuming at --ttf-theta {new_theta:g} from a checkpoint trained at {float(old_theta):g}: the first "
                     f"update would fire every accumulator entry between the two at once. Pass --ttf-acc-scale "
                     f"{float(new_theta) / float(old_theta):g} (keeps each entry's distance to the bar) or "
                     f"--ttf-acc-scale 1 (keeps the accumulators as they are).")


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
    if LR_LINEAR[0]:              # --lr-schedule linear: a straight line from the peak to min_lr (to zero with --min-lr 0)
        return tc.min_lr + (tc.lr - tc.min_lr) * (1.0 - r)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))
