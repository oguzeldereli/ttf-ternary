"""Schedules: the learning rate (cosine, linear, WSD, rewarm), the TTF step scale and the theta schedule."""
from __future__ import annotations
import math

from .config import TrainConfig


LR_LINEAR = [False]               # --lr-schedule linear: linear decay after warmup instead of the cosine
LR_WSD = [0]                      # --lr-schedule wsd D: warmup, constant peak lr until step D, then a cosine to min_lr at the end
LR_REWARM = [0, 0]                # --lr-rewarm S:W: from step S, linear warmup over W steps from min_lr to the peak, then cosine


def theta_at(prog, base, start, end, warm):
    """the theta schedule (one default, three knobs; no p, no lr coupling): theta START at progress 0, BASE at the end of
    the lr warmup (progress WARM), END at progress 1, with cosines between them in the step BASE / theta (the step scale
    of _ts_scale), not in theta. start = end = base: constant theta.
    Why the step (8 Nov 2026): a cosine in theta puts the step on a hyperbola that drops early (110M from scratch, 64 ->
    16 -> 80: step 0.35 at mid-run, 2.7320 on the 974 Wikipedia windows); the cosine in the step gives 0.62 there, close
    to paper 1's lr-tied step (lr / peak)^0.75 (0.66), and reproduces its 2.6728 (2.6741, one seed; log theta 2.6980,
    end 40 in theta 2.6976). At 0.8B (st08 recovery, 1000 steps) the theta-space form had ended at 3.1069 / 1.8125 /
    1.5031 (wiki / chat / mix), level with p 0.75 (3.1115 / 1.8123 / 1.5084); a hold at 16 until 60% then a rise ended at
    3.1335 / 3.1321, constant 16 at 3.1751."""
    prog = min(max(float(prog), 0.0), 1.0)
    if warm > 0 and prog < warm:
        return _step_interp(start, base, 0.5 * (1 - math.cos(math.pi * prog / warm)))
    if warm >= 1:
        return base
    return _step_interp(base, end, 0.5 * (1 - math.cos(math.pi * (prog - warm) / (1 - warm))))


def _step_interp(a, b, w):
    """theta from a (w 0) to b (w 1), linear in 1 / theta (so the step base / theta is linear in w); exact end points"""
    if w <= 0 or w >= 1 or a == b:
        return a if w <= 0 else b
    return 1.0 / ((1 - w) / a + w / b)


def _ts_scale(args, step, tc, lr):
    """the TTF step scale of the theta schedule, in rescale form: the bar stays at --ttf-theta (BASE) and the step is
    BASE / theta(t), so each accumulator entry keeps its position A / theta when theta moves (no burst of moves when
    theta falls, as a moving bar gives; the leak 1 - 1 / tau is not touched). Theta alone sets the step (since 6 Nov
    2026); the profilers in scripts/prof hook this per-step call."""
    th = theta_at(step / max(1, tc.max_steps), args.ttf_theta, args.ttf_theta_start, args.ttf_theta_end,
                  tc.warmup_steps / max(1, tc.max_steps))
    return args.ttf_theta / th


def resume_leak_check(ck_leak, ck_args, allow=False):
    """a resume must keep the checkpoint's TTF rule. This core: step scale 1, leak 1 - 1 / tau. Checkpoints before 6 Nov
    2026 may have a step scale (lr / peak lr)^p (--ttf-step-p, the old --ts_anneal; the pre-17-Oct default was nonzero)
    and the annealed leak 1 - rho^p / tau (pre-17-Oct only with --ts_tau_anneal): the same rule only at p 0. A change
    raises SystemExit unless allow (then the message of what changed is returned); None when nothing changes."""
    if ck_leak == "fixed":
        return None
    op = ck_args.get("ttf_step_p", ck_args.get("ts_anneal"))
    if op is None and ck_args and "ttf_theta" not in ck_args:
        op = "1.0"     # (the pre-17-Oct trainer without --ts_anneal: p 1)
    if op is None or float(op) == 0:
        return None
    ann = "ttf_step_p" in ck_args or bool(ck_args.get("ts_tau_anneal"))
    ch = [f"step scale (lr / peak lr)^{op} (checkpoint) vs 1 (this core: theta alone)"]
    if ann:
        ch.append(f"leak 1 - rho^{op} / tau (checkpoint, annealed) vs 1 - 1 / tau")
    if not allow:
        raise SystemExit("resume: " + "; ".join(ch) + ": the TTF rule would change mid-run, refusing "
                         "(--allow-leak-change to accept)")
    return "resume: --allow-leak-change: " + "; ".join(ch)


def resume_theta_check(ck_args, base, start, end, allow=False, ck_shape=None):
    """a resume must keep the checkpoint's TTF rule. A checkpoint from before the theta schedule (7 Nov 2026: no
    ttf_theta_start in its args) ran at constant theta; one with a schedule but no theta_shape "step" (7-8 Nov 2026) ran
    the cosines in theta, not in the step. Resuming either under this core's schedule would change the step mid-run:
    refused (SystemExit) unless the schedule is off on both sides (start = end = base) or allow; returns the message of an
    accepted change, else None."""
    if not ck_args:
        return None
    if "ttf_theta_start" not in ck_args:
        if start == end == base:
            return None
        ch = (f"theta schedule {start:g} -> {base:g} -> {end:g} (this run) vs constant theta (checkpoint from before 7 "
              f"Nov 2026)")
    else:
        cb = float(ck_args.get("ttf_theta", base))
        cs, ce = float(ck_args["ttf_theta_start"]), float(ck_args.get("ttf_theta_end", cb))
        if ck_shape == "step" or cs == ce == cb:
            return None
        ch = (f"the checkpoint's theta schedule {cs:g} -> {cb:g} -> {ce:g} ran its cosines in theta (7-8 Nov 2026), this "
              f"core in the step base / theta")
    if not allow:
        raise SystemExit(f"resume: {ch}: the step would change mid-run, refusing (--allow-leak-change accepts it)")
    return "resume: --allow-leak-change: " + ch


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
