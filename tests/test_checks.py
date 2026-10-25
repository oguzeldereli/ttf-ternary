"""The script checks in tests/checks (each exits non-zero on a failure), on a GPU."""
import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.gpu
@pytest.mark.parametrize("script", sorted(glob.glob(os.path.join(ROOT, "tests", "checks", "*.py"))),
                         ids=lambda p: os.path.basename(p)[:-3])
def test_check(script):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
