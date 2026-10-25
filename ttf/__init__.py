import sys as _sys

from . import config as _config
from .config import ModelConfig, TrainConfig, PRESETS, DEFAULT_PRESET
from .model import BitTransformer

# Checkpoints written before the rename of the package (bitnet -> ttf, 17 Oct 2026) pickle bitnet.config.ModelConfig:
# the old module names resolve to this package so they load unchanged.
_sys.modules.setdefault("bitnet", _sys.modules[__name__])
_sys.modules.setdefault("bitnet.config", _config)

__all__ = ["ModelConfig", "TrainConfig", "PRESETS", "DEFAULT_PRESET",
           "BitTransformer"]
