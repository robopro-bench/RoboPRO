# RoboPRO Makefile Entry Points

The main `make` targets, grouped by stage. `make help` prints most targets with their main variables, and `make show-config` prints the values currently in effect. Variables are passed on the command line, e.g. `make verify-rollout TASK_NAME=put_mouse_on_pad`.

## Setup and environment

| Target | What it does |
|---|---|
| `make check-prereqs` | Checks that `uv`, `git`, `nvcc`, and `ffmpeg` are on `PATH`. |
| `make bootstrap` | First-time setup: creates or refreshes `.venv` with uv, syncs `pyproject.toml` / `uv.lock`, applies the sapien and mplib patches, and installs CuRobo v0.7.8 (`scripts/install/bootstrap_uv.sh`). |
| `make sync` | Re-runs `uv sync` after a dependency change, without the rest of bootstrap. |
| `make show-config` | Prints the effective task, config, Python, GPU, and policy variables. |

## Assets and CuRobo configuration

| Target | What it does |
|---|---|
| `make download-assets` | Downloads the asset bundle (~15 GB) into repo-root `assets/`, or into `ASSETS_DEST`. |
| `make link-assets` | Points repo-root `assets` at `ASSETS_DEST`. Needed only when the assets live elsewhere. |
| `make configure-curobo-assets` | Renders `curobo_{left,right}.yml` from the shipped templates for the local asset path. |
| `make patch-curobo-config` | Adds the `attached_object` entries CuRobo needs to attach grasped objects (`scripts/install/patch_aloha_curobo.py`). |
| `make setup` | Runs `link-assets`, `configure-curobo-assets`, and `patch-curobo-config` in order. |

## Smoke tests and scene checks

| Target | What it does |
|---|---|
| `make render-test` | Minimal SAPIEN renderer test. |
| `make check-env-split` | Checks that the dual-env policy server imports nothing that needs sapien. |
| `make verify-scene` | Loads one benchmark scene without running the task. |
| `make verify-rollout` | Runs one headless scripted rollout and saves its video under repo-root `data/`. |

```bash
make verify-rollout TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean BENCH_SUBDIR=office
```

## Data generation

| Target | What it does |
|---|---|
| `make collect-data` | Collects scripted demonstrations for one task/config with `collect/collect_data.sh`. Episodes land under repo-root `data/`. |
| `make precollect-seeds` | Searches for valid evaluation seeds and writes `benchmark/eval_seeds/<task>/<config>.txt`. No demos are saved. |
| `make precollect-envs` | Writes frozen evaluation environment packs to `benchmark/eval_envs/<task>/<config>/`. |
| `make collect-rollout-pi05` | Collects policy rollouts from a pi05 checkpoint with the dual-env setup. |

```bash
make collect-data TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean GPU_ID=0
```

## Policy evaluation

| Target | What it does |
|---|---|
| `make eval-direct` | Single-process eval through `eval/eval_policy.py`. Results land under repo-root `eval_result/`. |
| `make policy-server` | Starts `eval/policy_model_server.py`, the policy half of a split eval. |
| `make eval-client` | Starts `eval/eval_policy_client.py`, the simulator half that talks to a running policy server. |
| `make eval-pi05-single` | Runs `policy/pi05/eval.sh` (policy and simulator in one process). |
| `make eval-pi05-double` | Runs `policy/pi05/eval_double_env.sh` (policy server and simulator client in separate environments). Use this when the pi05 dependencies need their own venv. |

```bash
make eval-pi05-double TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```
