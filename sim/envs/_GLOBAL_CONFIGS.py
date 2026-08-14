# global configs
import os
from pathlib import Path

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/")]
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/") + 1]

_WORKSPACE = os.path.abspath(os.path.join(ROOT_PATH, ".."))
BENCH_ROOT = os.environ.get("BENCH_ROOT", os.path.join(_WORKSPACE, "benchmark"))
ASSETS_PATH = os.environ.get("ASSETS_ROOT", os.path.join(_WORKSPACE, "assets"))
if not ASSETS_PATH.endswith(os.sep):
    ASSETS_PATH = ASSETS_PATH + os.sep
os.environ.setdefault("ASSETS_ROOT", ASSETS_PATH.rstrip(os.sep))
EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, "embodiments/")
TEXTURES_PATH = os.path.join(ASSETS_PATH, "background_texture/")
CONFIGS_PATH = os.path.join(ROOT_PATH, "task_config/")
SCRIPT_PATH = os.path.join(ROOT_PATH, "script/")
DESCRIPTION_PATH = os.path.join(BENCH_ROOT, "bench_description/")


def resolve_embodiment_path(file_path: str) -> str:
    """Resolve an embodiment file_path from _embodiment_config.yml against ASSETS_ROOT."""
    p = Path(os.path.expandvars(file_path))
    if not p.is_absolute():
        p = Path(ASSETS_PATH) / p
    return str(p)

# Euler angles in world coordinates
# t3d.euler.quat2euler(quat) returns (theta_x, theta_y, theta_z)
# theta_y controls the pitch, and theta_z controls rotation around the axis perpendicular to the tabletop plane
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}

WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],  # -z  -y  -x
    "front": [0.5, -0.5, 0.5, 0.5],  # y   z   -x
    "right": [0.707, 0, 0.707, 0],  # z   y   -x
    "top_down": [0, 0.707, -0.707, 0],  # -x  -y  -z
}

ROTATE_NUM = 10
