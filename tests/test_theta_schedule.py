"""The theta schedule (ttf.schedule.theta_at / _ts_scale, user 7 Nov 2026): start -> base over the lr warmup, base,
then a cosine rise to end; in rescale form (bar at base, step base / theta(t)); resumes of constant-theta checkpoints."""
import math
from types import SimpleNamespace

import pytest

from ttf.schedule import theta_at, _ts_scale, resume_theta_check, THETA_RISE
from ttf.args import from_legacy


def test_shape():
    b, s, e, w = 16.0, 64.0, 80.0, 0.1
    assert theta_at(0.0, b, s, e, w) == s
    assert math.isclose(theta_at(w / 2, b, s, e, w), (s + b) / 2)
    assert theta_at(w, b, s, e, w) == b
    assert theta_at((w + THETA_RISE) / 2, b, s, e, w) == b
    assert theta_at(THETA_RISE, b, s, e, w) == b
    assert math.isclose(theta_at((THETA_RISE + 1) / 2, b, s, e, w), (b + e) / 2)
    assert math.isclose(theta_at(1.0, b, s, e, w), e)
    xs = [i / 1000 for i in range(1001)]
    th = [theta_at(x, b, s, e, w) for x in xs]
    assert all(th[i] >= th[i + 1] for i in range(100)) and all(th[i] <= th[i + 1] for i in range(100, 1000))


def test_rise_from_warmup_end_and_no_warmup():
    assert math.isclose(theta_at(0.55, 16, 64, 80, 0.1, rise=0.1), 48.0)
    assert theta_at(0.0, 16, 64, 80, 0.0) == 16.0   # (no lr warmup: no start phase)


def test_constant_when_off():
    a = SimpleNamespace(ttf_theta=16.0, ttf_theta_start=16.0, ttf_theta_end=16.0)
    tc = SimpleNamespace(max_steps=1000, warmup_steps=100)
    assert all(_ts_scale(a, s, tc, 1e-3) == 1.0 for s in range(0, 1001, 7))


def test_rescale_step():
    a = SimpleNamespace(ttf_theta=16.0, ttf_theta_start=64.0, ttf_theta_end=80.0)
    tc = SimpleNamespace(max_steps=1000, warmup_steps=100)
    assert _ts_scale(a, 0, tc, 0.0) == 0.25
    assert _ts_scale(a, 300, tc, 0.0) == 1.0
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
