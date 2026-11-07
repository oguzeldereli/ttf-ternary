"""The script checks in tests/checks (each exits non-zero on a failure): on a GPU, except the ones in ANY (they pick
the device themselves, or are CPU only) which run in the CPU suite too."""
import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANY = {"q38_model", "q38_train", "q38_pipe"}


def _params():
    for p in sorted(glob.glob(os.path.join(ROOT, "tests", "checks", "*.py"))):
        n = os.path.basename(p)[:-3]
        yield pytest.param(p, id=n, marks=() if n in ANY else pytest.mark.gpu)


@pytest.mark.parametrize("script", _params())
def test_check(script):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
