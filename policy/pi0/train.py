"""Register RoboPRO TrainConfigs, then run openpi's training script."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_GLUE = Path(__file__).resolve().parent
_OPENPI = _GLUE / "openpi"
_TRAIN = _OPENPI / "scripts" / "train.py"

if str(_GLUE) not in sys.path:
    sys.path.insert(0, str(_GLUE))

from train_configs import register  # noqa: E402

register()

if not _TRAIN.is_file():
    raise FileNotFoundError(
        f"openpi train script not found: {_TRAIN}\n"
        "Initialize the submodule: git submodule update --init policy/pi0/openpi"
    )

sys.argv[0] = str(_TRAIN)
runpy.run_path(str(_TRAIN), run_name="__main__")
