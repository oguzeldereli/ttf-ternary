"""Run a module (e.g. ttf.train or ttf.legacy.train) with PyTorch in deterministic mode, for the consolidation gate:
deterministic algorithms (index_add_ and friends), a fixed cuBLAS workspace, attention through the math kernel (flash
attention's backward accumulates with atomics), TF32 off. Two runs of the same arguments must then agree bit for bit.
  python scripts/tests/det_run.py ttf.train ARGS...
"""
import os, runpy, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
sys.path.insert(0, os.getcwd())     # the package of the directory it is run from (not one on PYTHONPATH)
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
import importlib  # noqa: E402
print(f"det_run: {mod} from {importlib.import_module(mod.split('.')[0]).__file__}", flush=True)
runpy.run_module(mod, run_name="__main__", alter_sys=True)
