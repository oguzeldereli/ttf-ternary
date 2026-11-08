"""The theta schedule (ttf.schedule.theta_at / _ts_scale): start -> base over the lr warmup, then a rise to end from
the warmup's end, cosines in the step base / theta (8 Nov 2026); in rescale form (bar at base, step base / theta(t));
resumes of constant-theta and theta-space checkpoints."""
import math
from types import SimpleNamespace

import pytest

from ttf.schedule import theta_at, _ts_scale, resume_theta_check
from ttf.args import from_legacy


def test_shape():
    """the step 1 / theta follows the cosines: half-way in each phase it is the mean of the end points' steps"""
    b, s, e, w = 16.0, 64.0, 80.0, 0.1
    assert theta_at(0.0, b, s, e, w) == s
    assert math.isclose(1 / theta_at(w / 2, b, s, e, w), (1 / s + 1 / b) / 2)
    assert theta_at(w, b, s, e, w) == b
    assert math.isclose(1 / theta_at((w + 1) / 2, b, s, e, w), (1 / b + 1 / e) / 2)
    assert theta_at(1.0, b, s, e, w) == e
    xs = [i / 1000 for i in range(1001)]
    th = [theta_at(x, b, s, e, w) for x in xs]
    assert all(th[i] >= th[i + 1] for i in range(100)) and all(th[i] <= th[i + 1] for i in range(100, 1000))


def test_rise_from_warmup_end_and_no_warmup():
    assert math.isclose(1 / theta_at(0.55, 16, 64, 80, 0.1), (1 / 16 + 1 / 80) / 2)
    assert theta_at(0.0, 16, 64, 80, 0.0) == 16.0   # (no lr warmup: no start phase, the rise starts at once)
    assert math.isclose(1 / theta_at(0.5, 16, 64, 80, 0.0), (1 / 16 + 1 / 80) / 2)


def test_step_near_the_lr_tied_rule():
    """the step 16 / theta of 64 -> 16 -> 80 against paper 1's (lr / peak)^0.75 on the 110M recipe (9155 steps, warmup
    305, lr cosine to 0.1 of peak): within 0.04 after the warmup, sum 5501 against 5620 (7 Nov's theta space: 4110)"""
    T, W = 9155, 305
    def lr(t):
        return t / W if t < W else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - W) / (T - W)))
    st = [16 / theta_at(t / T, 16, 64, 80, W / T) for t in range(T)]
    p75 = [lr(t) ** 0.75 for t in range(T)]
    assert max(abs(a - b) for a, b in zip(st[W:], p75[W:])) < 0.04
    assert abs(sum(st) - 5501) < 3 and abs(sum(p75) - 5620) < 3


def test_constant_when_off():
    a = SimpleNamespace(ttf_theta=16.0, ttf_theta_start=16.0, ttf_theta_end=16.0)
    tc = SimpleNamespace(max_steps=1000, warmup_steps=100)
    assert all(_ts_scale(a, s, tc, 1e-3) == 1.0 for s in range(0, 1001, 7))


def test_rescale_step():
    a = SimpleNamespace(ttf_theta=16.0, ttf_theta_start=64.0, ttf_theta_end=80.0)
    tc = SimpleNamespace(max_steps=1000, warmup_steps=100)
    assert _ts_scale(a, 0, tc, 0.0) == 0.25
    assert _ts_scale(a, 100, tc, 0.0) == 1.0          # (the warmup's end: base; the rise starts here)
    assert math.isclose(_ts_scale(a, 550, tc, 0.0), 0.6)   # (half-way: the mean of the steps 1 and 0.2)
    assert math.isclose(_ts_scale(a, 1000, tc, 0.0), 0.2)


def test_resume_of_constant_theta_checkpoint():
    old = {"ttf_theta": 16.0}
    assert resume_theta_check(old, 16.0, 16.0, 16.0) is None
    with pytest.raises(SystemExit):
        resume_theta_check(old, 16.0, 64.0, 80.0)
    assert "allow-leak-change" in resume_theta_check(old, 16.0, 64.0, 80.0, allow=True)
    assert resume_theta_check({"ttf_theta": 16.0, "ttf_theta_start": 64.0}, 16.0, 64.0, 80.0, ck_shape="step") is None
    assert resume_theta_check({}, 16.0, 64.0, 80.0) is None


def test_resume_of_theta_space_checkpoint():
    """a 7-8 Nov checkpoint (schedule, no theta_shape) ran the cosines in theta: refused under the step shape"""
    ck = {"ttf_theta": 16.0, "ttf_theta_start": 64.0, "ttf_theta_end": 80.0}
    with pytest.raises(SystemExit):
        resume_theta_check(ck, 16.0, 64.0, 80.0)
    assert "in theta" in resume_theta_check(ck, 16.0, 64.0, 80.0, allow=True)
    assert resume_theta_check(ck, 16.0, 64.0, 80.0, ck_shape="step") is None
    const = {"ttf_theta": 16.0, "ttf_theta_start": 16.0, "ttf_theta_end": 16.0}
    assert resume_theta_check(const, 16.0, 16.0, 16.0) is None     # (constant theta: no shape)


def test_legacy_linear_schedule_refused():
    with pytest.raises(ValueError):
        from_legacy(["--ts_theta_sched", "8:16"])
