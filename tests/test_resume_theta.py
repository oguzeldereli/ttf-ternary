"""A resume at a new firing threshold must say what to do with the accumulators (ttf.schedule.resume_acc_scale)."""
import pytest

from ttf.schedule import resume_acc_scale


def test_same_theta_needs_nothing():
    assert resume_acc_scale(16.0, 16.0, None) == (1.0, None)


def test_changed_theta_without_scale_refuses():
    with pytest.raises(SystemExit) as e:
        resume_acc_scale(64.0, 16.0, None)
    assert "--ttf-acc-scale 0.25" in str(e.value) and "--ttf-acc-scale 1" in str(e.value)


def test_changed_theta_with_scale_is_applied_and_logged():
    acc, msg = resume_acc_scale(64.0, 16.0, 0.25)
    assert acc == 0.25 and "64" in msg and "16" in msg


def test_explicit_one_keeps_the_accumulators():
    assert resume_acc_scale(64.0, 16.0, 1.0) == (1.0, None)


def test_schedules_and_unknown_thetas_skip_the_check():
    assert resume_acc_scale(64.0, 16.0, None, old_sched=(8.0, 16.0)) == (1.0, None)
    assert resume_acc_scale(64.0, 16.0, None, new_sched=(8.0, 16.0)) == (1.0, None)
    assert resume_acc_scale(None, 16.0, None) == (1.0, None)


def test_resume_leak_check():
    """a resume keeps the checkpoint's TTF rule: this core has step scale 1 and the leak 1 - 1 / tau (6 Nov 2026)"""
    from ttf.schedule import resume_leak_check as R
    assert R("fixed", {}) is None                                           # this core
    assert R(None, {"ttf_step_p": 0.0}) is None                             # an old p 0 run: the same rule
    with pytest.raises(SystemExit, match="annealed"):
        R(None, {"ttf_step_p": 0.75})                                       # the old default: step and leak annealed
    assert "annealed" in R(None, {"ttf_step_p": 0.75}, allow=True)
    assert R(None, {"ts_anneal": 0, "ts_theta": 16}) is None                # pre-17-Oct at p 0
    with pytest.raises(SystemExit, match=r"\^1.0"):
        R(None, {"ts_theta": 16})                                           # pre-17-Oct without --ts_anneal: p 1
    m = R(None, {"ts_anneal": 0.5, "ts_theta": 16}, allow=True)             # pre-17-Oct, fixed leak: the step only
    assert "^0.5" in m and "annealed" not in m
