# Policies

| Directory | What it is | License |
|---|---|---|
| `pi0/` | OpenPI π0 library + RoboTwin `deploy_policy` / `eval.sh` glue | Library: [Apache 2.0](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE) ([`pi0/LICENSE`](pi0/LICENSE)). Wrappers: MIT / RoboTwin-derived. |
| `pi05/` | OpenPI π0.5 + the same glue | Same split. [`pi05/LICENSE`](pi05/LICENSE) is the official Apache text for the library only. |

Run from the repo root after `source set_env.sh`.

```bash
bash policy/pi05/eval.sh <task> <config> <train_config> <model> <ckpt_id> <ckpt_setting> <seed> <gpu>
bash policy/pi05/eval_double_env.sh <task> <config> <train_config> <model> <ckpt_id> <seed> <gpu_spec>
bash policy/pi05/collect_rollout.sh <task> <config> <train_config> <model> <ckpt_id> <seed> <gpu_spec>
```

The eval harness is [`../eval/`](../eval/). Checkpoints go under `policy/<name>/checkpoints/`. pi05’s isolated env is `policy/pi05/.venv/`.
