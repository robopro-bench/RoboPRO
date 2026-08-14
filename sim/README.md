# RoboPRO simulation runtime

This directory is the SAPIEN + CuRobo runtime used by RoboPRO. It is a modified
fork of [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).

RoboPRO benchmark tasks live in `../benchmark/`, not here. From this directory:

```bash
source set_env.sh   # exports SIM_ROOT + BENCH_ROOT
bash collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
```

See the [root README](../README.md) for install, eval, and citation.
