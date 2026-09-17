# Policies

| Directory | What it is |
|---|---|
| `pi0/` | RoboPRO deploy / process glue for π0. The openpi library is a git submodule at `pi0/openpi/`. |
| `pi05/` | Same glue for π0.5, plus dual-env eval/collect wrappers. Submodule at `pi05/openpi/`. |

Clone with `--recurse-submodules` (see the top-level README). If the checkouts are missing:

```bash
git submodule update --init policy/pi0/openpi policy/pi05/openpi
cd policy/pi05/openpi && uv sync && cd -
```

LeRobot conversion is openpi’s example script, invoked by `generate.sh` (not copied into this repo).

Run from the repo root after `source set_env.sh`.

```bash
bash policy/pi05/eval.sh <task> <config> <train_config> <model> <ckpt_id> <ckpt_setting> <seed> <gpu>
bash policy/pi05/eval_double_env.sh <task> <config> <train_config> <model> <ckpt_id> <seed> <gpu_spec>
bash policy/pi05/collect_rollout.sh <task> <config> <train_config> <model> <ckpt_id> <seed> <gpu_spec>
```

The eval harness is [`../eval/`](../eval/). Checkpoints go under `policy/<name>/checkpoints/`. pi05’s isolated env is `policy/pi05/openpi/.venv/`.
