#  Add paths to the system path
import sys
from pathlib import Path
import os


def setup_paths():
    # SIM_ROOT / BENCH_ROOT are normally exported by sourcing set_env.sh.
    # If they are absent (e.g. set_env.sh wasn't sourced in this shell), derive
    # them from this file's location so the bench scripts still run. Explicit env
    # vars always win -- we only fill in the gaps.
    #   this file:  <sim_root>/script/bench_script/setup_paths.py
    #   sim_root    = parents[2]
    #   bench_root  = <sim_root>/../benchmark
    here = Path(__file__).resolve()
    if "SIM_ROOT" not in os.environ:
        os.environ["SIM_ROOT"] = str(here.parents[2])
        print(f"[setup_paths] SIM_ROOT not set; derived {os.environ['SIM_ROOT']}")
    if "BENCH_ROOT" not in os.environ:
        os.environ["BENCH_ROOT"] = str(Path(os.environ["SIM_ROOT"]).parent / "benchmark")
        print(f"[setup_paths] BENCH_ROOT not set; derived {os.environ['BENCH_ROOT']}")

    sim_root = Path(os.environ["SIM_ROOT"])
    bench_root = Path(os.environ["BENCH_ROOT"])
    for p in [sim_root, bench_root]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
