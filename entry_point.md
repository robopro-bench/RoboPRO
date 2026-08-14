# RoboPRO Makefile Entry Points

This file lists the main `make` entry points exposed by the repo and gives a short description of when to use each one. Flag details already live in the `Makefile`, so this page focuses on purpose and a sample command shape.

## Setup and Environment

### `make show-config`

```bash
make show-config
```

Use this to print the current effective Makefile variables before running anything else. It is helpful for sanity-checking the active task, config, Python path, GPU, and policy settings.

### `make bootstrap`

```bash
make bootstrap
```

Use this for first-time environment setup. It creates or refreshes the project `.venv`, syncs the uv-managed dependencies, and runs the post-install patch/setup steps needed by RoboPRO.

### `make sync`

```bash
make sync
```

Use this after dependency changes when you only want to resync the uv environment without rerunning the full bootstrap flow.

## Assets and CuRobo Configuration

### `make download-assets`

```bash
make download-assets
```

Use this to fetch the benchmark asset bundle into the configured asset destination.

### `make link-assets`

```bash
make link-assets
```

Use this to wire the benchmark asset directory and the `sim/assets` path to the destination you want to use locally.

### `make configure-curobo-assets`

```bash
make configure-curobo-assets
```

Use this to generate the CuRobo config files from the shipped templates after assets are in place.

### `make patch-curobo-config`

```bash
make patch-curobo-config
```

Use this to patch the generated CuRobo config files so they work with the RoboPRO asset/layout setup.

## Smoke Tests and Scene Checks

### `make render-test`

```bash
make render-test
```

Use this as the smallest renderer smoke test to confirm the simulator and rendering stack are alive.

### `make verify-scene`

```bash
make verify-scene TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean BENCH_SUBDIR=office
```

Use this to load a benchmark scene and inspect whether the task/environment initializes correctly, without rolling out the task policy.

### `make verify-rollout`

```bash
make verify-rollout TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean BENCH_SUBDIR=office
```

Use this to run a single rollout for a benchmark task and verify end-to-end task execution. This is the easiest entry point for producing one saved sample rollout from a chosen task/config pair.

### `make diag-kitchen-curobo`

```bash
make diag-kitchen-curobo
```

Use this for focused debugging of CuRobo behavior in the kitchen benchmark environments.

## Data Generation

### `make collect-data`

```bash
make collect-data TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean GPU_ID=0
```

Use this to run the dataset collection pipeline for one task/config setting. This is the main entry point for producing benchmark training-style episodes, videos, and scene metadata for a task configuration.

### `make precollect-seeds`

```bash
make precollect-seeds TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean
```

Use this to generate and store successful seeds ahead of a full collection run. It is useful when you want deterministic, reusable rollout seeds before collecting data.

### `make collect-rollout-pi05`

```bash
make collect-rollout-pi05 TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```

Use this to collect policy rollouts from the `pi05` model using the dual-environment setup. This is the right path when you want policy-generated trajectories rather than scripted benchmark demonstrations.

## Policy Evaluation

### `make eval-direct`

```bash
make eval-direct POLICY_NAME=pi05 TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```

Use this for direct single-process evaluation through `script/eval_policy.py`. It is the most straightforward entry point when you want to evaluate a checkpoint against a task/config pair.

### `make policy-server`

```bash
make policy-server POLICY_NAME=pi05 TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```

Use this when you want to run the policy model as a standalone server process, typically as one half of a split evaluation or rollout pipeline.

### `make eval-client`

```bash
make eval-client POLICY_NAME=pi05 TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```

Use this as the client-side evaluation process that talks to a separately running policy server.

### `make eval-pi05-single`

```bash
make eval-pi05-single TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```

Use this for the single-process `pi05` evaluation flow. It is helpful when your environment can run both the simulator and the policy stack together in one process.

### `make eval-pi05-double`

```bash
make eval-pi05-double TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean TRAIN_CONFIG_NAME=my_office_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000
```

Use this for the dual-process `pi05` evaluation flow, where the model server and simulator/client run separately. This is the safer entry point when `pi05` dependencies need isolation.

## Quick Guidance

- Use `bootstrap` when setting up the repo for the first time.
- Use `verify-scene` when checking environment initialization only.
- Use `verify-rollout` when checking one end-to-end task execution or saving one sample rollout.
- Use `collect-data` when building benchmark data for a task/config.
- Use `eval-*` targets when benchmarking learned policies.
- Use `collect-rollout-pi05` when collecting trajectories from the `pi05` policy stack.
