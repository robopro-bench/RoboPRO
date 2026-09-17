# RoboPRO simulation runtime

This directory is the SAPIEN + CuRobo runtime used by RoboPRO. It is a modified
fork of [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).

RoboPRO benchmark tasks live in `../benchmark/`. Collection, eval, and policies
are siblings of this directory and run from the repo root:

```bash
source set_env.sh   # repo-root file
bash collect/collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean ...
```

Scene smoke tests (`script/bench_script/visualize_task_scene.py`) still run from
this directory after `source ../set_env.sh`.

License: [`LICENSE`](LICENSE) (RoboTwin 2.0 MIT, plus RoboPRO modifications). See the [root README](../README.md) for install, eval, and citation.
