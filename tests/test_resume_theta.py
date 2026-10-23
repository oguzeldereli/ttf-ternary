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
