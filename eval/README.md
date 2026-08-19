# Eval

Policy evaluation harness. Run from the repo root after `source set_env.sh`.
Task configs are YAML **names** under [`benchmark/bench_task_config/`](../benchmark/bench_task_config/).

Results land in `eval_result/<task>/<policy>/<task_config>/<ckpt_setting>/<timestamp>/`
(`_result.txt` plus rollout videos when `eval_video_log` is on).

## Usage

Single-process (policy + sim share one Python env):

```bash
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean \
    my_office_train pi05_ckpt 30000 pi05_ckpt_30000 0 0
```

Dual-process (pi05 server in `policy/pi05/openpi/.venv`, sim client separate):

```bash
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean \
    my_office_train pi05_ckpt 30000 0 0
```

Or `make eval-direct` / `make eval-pi05-double`. Direct Python:

```bash
python eval/eval_policy.py --config policy/pi05/deploy_policy.yml --overrides \
    --task_name put_mouse_on_pad --task_config bench_demo_office_clean \
    --train_config_name my_office_train --model_name pi05_ckpt \
    --checkpoint_id 30000 --ckpt_setting my_office_train_pi05_ckpt_30000 \
    --policy_name pi05 --seed 0 --instruction_type seen --test_num 10
```

Policies live in [`../policy/`](../policy/). The simulator is [`../sim/`](../sim/).

## Scripts

| File | Role |
|---|---|
| `eval_policy.py` | Single-process eval |
| `eval_policy_client.py` | Client half of dual-env eval |
| `policy_model_server.py` | Policy inference server (policy venv; no sapien) |
| `eval_seeds.py` | Resolves the fixed eval seed list for a `(task, config)` |
| `_env.py` | Path bootstrap |

## Eval seeds

Eval does **not** scan random seeds at run time when a seed file exists. The benchmark
uses a fixed, expert-validated list per `(task, config)`:

```
benchmark/eval_seeds/<task>/<config>.txt
```

Space-separated integers. Precollect (CuRobo plan + `check_success` only — no hdf5):

```bash
python collect/precollect_eval_seeds.py <task> <config>
```

**Design**

- Seeds start at **40000** so they never overlap training collection seeds (`0` … tens).
- Counts: **20** seeds for `*_clean` configs, **2** for every other config (clutter /
  perturbation). A clutter file is typically a prefix of that task’s clean list
  (e.g. clean `40001 40002 40004 …`, d10/d15 `40001 40002`).
- A seed is kept only if the expert planner both plans and succeeds on that
  `(task, config)` — so a seed that is valid on clean may be dropped at high clutter.

**Same seed, same target pose.** Scene setup seeds `np.random` then calls `load_actors()`
(target / destination) **before** spawning clutter. Obstacle density only draws extra
objects afterward. Therefore the **same seed integer** on `bench_demo_office_clean` and
`bench_demo_office_d15` (or d6…d14) places the **target at the same pose**; only the
clutter around it changes. Compare clutter levels by reusing the same seed ids.

**How eval picks seeds** (first match wins):

1. `--use_eval_seeds false` / `USE_EVAL_SEEDS=0` → scan from `st_seed = 100000 * (1 + seed)` with a live expert check
2. `--eval_seed_file PATH`
3. `benchmark/eval_seeds/<task>/<config>.txt` if present (live expert check skipped — already validated)
4. Scan mode, as in (1)

`--test_num N` / `EVAL_TEST_NUM` caps how many seeds are rolled; with a seed file the
default is the full list.

## Notices

- Precollected files are the intended eval set. Do not treat a live scan from
  `st_seed` as the benchmark protocol unless you explicitly opt out.
- After changing scene / clutter / expert-success criteria, re-run
  `precollect_eval_seeds.py` or eval will keep using the old list.
- `--seed` on the eval CLI is **not** the episode seed when a file is loaded; it only
  sets the scan-mode offset `st_seed`.
- Dual-env: `policy_model_server.py` must run in the policy venv (no sapien). Client
  uses the sim env.
