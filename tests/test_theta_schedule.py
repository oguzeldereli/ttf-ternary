"""The theta schedule (ttf.schedule.theta_at / _ts_scale, 7 Nov 2026): start -> base over the lr warmup, then a
cosine rise to end from the warmup's end; in rescale form (bar at base, step base / theta(t)); resumes of constant-theta checkpoints."""
import math
from types import SimpleNamespace

import pytest

from ttf.schedule import theta_at, _ts_scale, resume_theta_check
from ttf.args import from_legacy


def test_shape():
    b, s, e, w = 16.0, 64.0, 80.0, 0.1
    assert theta_at(0.0, b, s, e, w) == s
    assert math.isclose(theta_at(w / 2, b, s, e, w), (s + b) / 2)
    assert theta_at(w, b, s, e, w) == b
    assert math.isclose(theta_at((w + 1) / 2, b, s, e, w), (b + e) / 2)
    assert math.isclose(theta_at(1.0, b, s, e, w), e)
    xs = [i / 1000 for i in range(1001)]
    th = [theta_at(x, b, s, e, w) for x in xs]
    assert all(th[i] >= th[i + 1] for i in range(100)) and all(th[i] <= th[i + 1] for i in range(100, 1000))


def test_rise_from_warmup_end_and_no_warmup():
    assert math.isclose(theta_at(0.55, 16, 64, 80, 0.1), 48.0)
    assert theta_at(0.0, 16, 64, 80, 0.0) == 16.0   # (no lr warmup: no start phase, the rise starts at once)
    assert math.isclose(theta_at(0.5, 16, 64, 80, 0.0), 48.0)


def test_matches_the_ab_arm():
    """the 0.8B A/B arm rw5 (ttf-lab tsched_train THQ_TS 64:16:80:0.1, 7 Nov) logged theta 40 / 20.3 / 42.4 / 68.6 / 80
    at progress 0.05 / 0.25 / 0.5 / 0.75 / 1"""
    for x, t in ((0.05, 40.0), (0.25, 20.3), (0.5, 42.4), (0.75, 68.6), (1.0, 80.0)):
        assert abs(theta_at(x, 16, 64, 80, 0.1) - t) < 0.06


def test_constant_when_off():
    a = SimpleNamespace(ttf_theta=16.0, ttf_theta_start=16.0, ttf_theta_end=16.0)
    tc = SimpleNamespace(max_steps=1000, warmup_steps=100)
    assert all(_ts_scale(a, s, tc, 1e-3) == 1.0 for s in range(0, 1001, 7))


def test_rescale_step():
    a = SimpleNamespace(ttf_theta=16.0, ttf_theta_start=64.0, ttf_theta_end=80.0)
    tc = SimpleNamespace(max_steps=1000, warmup_steps=100)
    assert _ts_scale(a, 0, tc, 0.0) == 0.25
    assert _ts_scale(a, 100, tc, 0.0) == 1.0          # (the warmup's end: base; the rise starts here)
    assert 0.5 < _ts_scale(a, 300, tc, 0.0) < 1.0
    assert math.isclose(_ts_scale(a, 1000, tc, 0.0), 0.2)


def test_resume_of_constant_theta_checkpoint():
    old = {"ttf_theta": 16.0}
    assert resume_theta_check(old, 16.0, 16.0, 16.0) is None
    with pytest.raises(SystemExit):
        resume_theta_check(old, 16.0, 64.0, 80.0)
    assert "allow-leak-change" in resume_theta_check(old, 16.0, 64.0, 80.0, allow=True)
    assert resume_theta_check({"ttf_theta": 16.0, "ttf_theta_start": 64.0}, 16.0, 64.0, 80.0) is None
    assert resume_theta_check({}, 16.0, 64.0, 80.0) is None


def test_legacy_linear_schedule_refused():
    with pytest.raises(ValueError):
        from_legacy(["--ts_theta_sched", "8:16"])


def test_interp_forms():
    """--ttf-theta-interp: the same end points; step = the cosine on 1 / theta, log = on log theta"""
    for it in ("theta", "step", "log"):
        assert math.isclose(theta_at(0.0, 16, 64, 80, 0.1, it), 64) and theta_at(0.1, 16, 64, 80, 0.1, it) == 16
        assert math.isclose(theta_at(1.0, 16, 64, 80, 0.1, it), 80)
    assert math.isclose(1 / theta_at(0.55, 16, 64, 80, 0.1, "step"), (1 / 16 + 1 / 80) / 2)
    assert math.isclose(theta_at(0.55, 16, 64, 80, 0.1, "log"), (16 * 80) ** 0.5)
