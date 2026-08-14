# RoboPRO

**P**erturbation-**R**esilient **O**bstacle-awareness — a bimanual manipulation benchmark for policy robustness evaluation.

RoboPRO is a modified fork of [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). The simulation runtime (SAPIEN + CuRobo, Aloha-AgileX) is based on their stack; the 80 tasks, realistic scenes, and perturbation suite are ours.

**Project page:** https://anonymous.4open.science/w/RoboPRO-EDE0/index.html

RoboPRO adds:
- **Realistic scenes** across office, study, kitchen (small & large) domains
- **Systematic perturbation suite** — Language, Vision, and Object axes for evaluating policy robustness
- **Aloha-Agilex** bimanual embodiment with CuRobo motion planning

## Installation

System prereqs (one-time): `libvulkan1 mesa-vulkan-drivers vulkan-tools` (apt), `ffmpeg`, and an NVIDIA driver with CUDA 12.x.

```bash
git clone https://anonymous.4open.science/r/RoboPRO-EDE0
cd RoboPRO
```

### 1. Choose an environment manager

#### Option A. Conda env

```bash
conda create -n robopro python=3.10 -y
# Keep the env isolated from ~/.local/lib site-packages (otherwise sapien/torch
# may resolve there instead of in robopro):
conda env config vars set -n robopro PYTHONNOUSERSITE=1
conda activate robopro
```

#### Option B. uv env

```bash
uv venv --python 3.10
source .venv/bin/activate
# Keep the env isolated from ~/.local/lib site-packages:
export PYTHONNOUSERSITE=1
```

### 2. Install Python dependencies

#### Option A. Conda workflow

```bash
cd sim
pip install -r script/requirements.txt
pip install setuptools==69.5.1       # provides pkg_resources for sapien
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
bash script/_install.sh              # patches sapien urdf_loader + mplib planner
cd ..
```

> **aarch64 (GB10 / DGX Spark):** PyPI has no aarch64 wheel for `sapien==3.0.0b1`, so `requirements.txt` will fail to resolve it. Build the SAPIEN wheel from source first — see [docs/setup_sapien_aarch64.md](docs/setup_sapien_aarch64.md) — then re-run the requirements install (pip will treat sapien as satisfied).

> **SAPIEN version matters:** the benchmark is pinned to `sapien==3.0.0b1`. A different SAPIEN version can change physics and rendering behavior, which shifts evaluation results — success rates from mismatched versions are not comparable. Verify with `python -c "import sapien; print(sapien.__version__)"` before collecting data or running evals.

`script/_install.sh` also clones CuRobo v0.7.8 into `envs/curobo/` and pip-installs it editable, then re-pins `warp-lang==1.12.0` and `setuptools==69.5.1`. If you keep `scipy==1.10.1` from `requirements.txt`, `scikit-image` will print a version-conflict warning — harmless.

#### Option B. uv workflow

```bash
bash scripts/install/bootstrap_uv.sh
```

This bootstraps `.venv` from the root `pyproject.toml` and `uv.lock`, then runs the post-install patches and clones CuRobo v0.7.8 into `sim/envs/curobo/` as an editable install. The uv path does not install `sim/script/requirements.txt` directly, so any dependency added there must also be mirrored in `pyproject.toml`. If you keep `scipy==1.10.1`, `scikit-image` may print a version-conflict warning during install — harmless.

### 3. Assets (~15 GB)

```bash
python scripts/install/download_assets.py
```

This fetches the HuggingFace asset bundle (repo id in `scripts/install/download_assets.py`) into `benchmark/assets/` (objects, embodiments, background_texture, backgrounds). The bundle already includes the large `aloha-agilex/.../meshes/box2_Link.dae` mesh — no separate fetch needed.

The shipped `task_config/_embodiment_config.yml` uses relative paths. RoboPRO keeps assets under `benchmark/assets/`, so add a one-line symlink so those paths resolve:

```bash
ln -sfn ../benchmark/assets sim/assets
```

Generate the local-path curobo configs from the shipped templates, and patch them so CuRobo can attach grasped objects (the shipped configs lack the `attached_object` link entries):

```bash
ASSETS_PATH="$(pwd)/benchmark"
cd benchmark/assets/embodiments/aloha-agilex
ASSETS_PATH="$ASSETS_PATH" python -c 'from pathlib import Path; import os; assets_path = os.environ["ASSETS_PATH"]; [Path(f"curobo_{side}.yml").write_text(Path(f"curobo_{side}_tmp.yml").read_text(encoding="utf-8").replace("${ASSETS_PATH}", assets_path), encoding="utf-8") for side in ("left", "right")]'
cd -
python scripts/install/patch_aloha_curobo.py
```

### 4. CuRobo cache patch

In `sim/envs/curobo/src/curobo/geom/sdf/world_mesh.py`, replace `clear_cache` with:

```python
def clear_cache(self):
    self._wp_mesh_cache = {}
    if self._mesh_tensor_list is not None:
        self._mesh_tensor_list[2][:] = 0
    if self._env_n_mesh is not None:
        self._env_n_mesh[:] = 0
    if self._env_mesh_names is not None:
        for i in range(self.n_envs):
            for j in range(len(self._env_mesh_names)):
                self._env_mesh_names[i][j] = None
    super().clear_cache()
```

### 5. Verify (headless rollout)

```bash
cd sim
source ../set_env.sh
python script/bench_script/visualize_task_scene.py \
    put_mouse_on_pad bench_demo_office_clean \
    --bench-subdir office --rollout --no-render --seed 0 --save_data
```

Expected on success: a `Success: True` line and an MP4 at `sim/data/bench_data/video/episode_put_mouse_on_pad_0.mp4` (~176 frames @ 320×240).

## Usage

Collection and eval run from the repo root. Scene smoke tests still run from `sim/`.

```bash
source set_env.sh                  # exports WORKSPACE_ROOT, SIM_ROOT, BENCH_ROOT, DATA_ROOT, POLICY_ROOT
```

### Collect demonstrations

```bash
bash collect/collect_data.sh <task_name> <task_config> <gpu_id>
# Example:
bash collect/collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
```

Episodes land in `data/<save_path>/<task_name>/<task_config>/` (YAML `./data/dataset` → `data/dataset/...`). Schema and grounding: [`collect/README.md`](collect/README.md).

### Convert HDF5 to LeRobot

[`sim/script/lerobot_convert/`](sim/script/lerobot_convert/) turns a scene-organised RoboPRO dump (`<tier>/seedN/data/episode*.hdf5`) into a LeRobot v2.1 dataset (parquet + `countertop`/`left`/`right` videos, 1:1 at 30 fps). The task prompt is the HDF5 `task_name` looked up in `benchmark/bench_description/plain_instructions.json` (or `--task-text`).

From the repo root (env with `cv2`, `av`, `h5py`, `pandas`, `numpy`):

```bash
PYTHONPATH=sim/script python -m lerobot_convert.convert_scenes \
    --src /path/to/<task>_38scene_... \
    --out /path/to/lerobot_out \
    --limit 2 --overwrite
```

### Run inference (policy eval)

From the repo root after `source set_env.sh`. Eval rolls a trained checkpoint out against a `(task, config)` pair and writes a per-rollout success log under `eval_result/`. Two modes depending on whether your policy fits in the same Python env as the simulator.

**Pretrained checkpoints:**

| Policy | Weights |
|---|---|
| pi05 | TODO: HuggingFace checkpoint |
| X-VLA | TODO: HuggingFace checkpoint |

For pi05, symlink the downloaded `jax_30000/` dir to `policy/pi05/checkpoints/<train_config_name>/<model_name>/30000/`.

**Args (shared by both modes):**

| Arg | Meaning |
|---|---|
| `task_name` | Bench env class, e.g. `put_mouse_on_pad` (file at `benchmark/bench_envs/<scene>/<task>.py`) |
| `task_config` | Perturbation YAML name, e.g. `bench_demo_office_clean` (in `benchmark/bench_task_config/`) |
| `train_config_name` | Training config used to fine-tune the checkpoint |
| `model_name` | Subdir name under `checkpoints/<train_config_name>/` |
| `checkpoint_id` | Step number, e.g. `30000` |
| `seed` | RNG seed for episode initialisation |
| `gpu_id` | CUDA device, or `<server_gpu>:<client_gpu>` for dual-env |

**Mode A — single-process** (policy + sim share one Python env, e.g. when openpi is conda-installable alongside SAPIEN):

```bash
bash policy/pi05/eval.sh <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <ckpt_setting> <seed> <gpu_id>
# Example:
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 pi05_ckpt_30000 0 0
```

**Mode B — dual-env / dual-process** (recommended for pi05 since openpi+jax need an isolated uv venv at `policy/pi05/.venv/`):

```bash
bash policy/pi05/eval_double_env.sh <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <gpu_spec>
# Example (single GPU):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0
# Example (split: server on GPU 0, sim client on GPU 1):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0:1
```

The script spawns a `policy_model_server.py` in the pi05 venv and an `eval_policy_client.py` in the RoboPRO sim env, communicating over a free socket port.

**Eval seeds.** When `BENCH_ROOT` is set and `benchmark/eval_seeds/<task>/<task_config>.txt` exists, eval loads that fixed seed list (skips live expert scanning). Override with `--eval_seed_file /path/to.txt`, or fall back to scanning other seeds with `--use_eval_seeds false`. Cap episodes with `--test_num N` (capped by the file length). Precollect seeds via `python collect/precollect_eval_seeds.py <task> <task_config>` (also used by `scripts/slurm/slurm_precollect_then_eval.sh`).

**Faster evaluation (optional).** Two opt-in settings cut the time per episode. Neither changes the scene a seed builds, the RNG streams, or the result. Leave them unset and eval behaves exactly as before.

| Setting | Where | Effect |
|---|---|---|
| `ROBOTWIN_SKIP_CUROBO_WARMUP=1` | environment | Skips CuRobo's CUDA-graph warmup when the planner is built (~5 s per arm, ~10 s per process). The planner still works; it compiles on its first real plan instead. A policy eval never plans a motion, so it never pays that cost. |
| `render_static_cameras: [countertop_camera]` | task config YAML, under `camera:` | Renders only the listed static cameras each step. The others are still built, so the scene is unchanged; they just take no pictures. Unset renders every camera. Wrist cameras stay under `collect_wrist_camera`. |

List every static camera your policy reads in `render_static_cameras`. A camera left off the list still appears in the observation, but with no `rgb` image.

**Direct Python invocation** (bypassing the shell wrappers):

```bash
python eval/eval_policy.py \
    --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name put_mouse_on_pad \
    --task_config bench_demo_office_clean \
    --train_config_name my_office_train \
    --model_name pi05_ckpt \
    --checkpoint_id 30000 \
    --ckpt_setting "my_office_train_pi05_ckpt_30000" \
    --policy_name pi05 \
    --seed 0 \
    --instruction_type seen \
    --test_num 10
```

**Where results land:**

```
sim/eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
    _result.txt        # success count, per-seed pass/fail
    *.mp4              # rollout videos (if eval_video_save is enabled)
```

### Batch eval on SLURM

For sweeping many `(task, config)` pairs across nodes:

```bash
sbatch scripts/slurm/slurm_eval_bench.sh \
    <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <test_num>
```

Set `--chdir` and `--output` in the sbatch header to your local checkout (see comments at the top of `scripts/slurm/slurm_eval_bench.sh`). Pin a specific Python with `export PI05_PYTHON=/path/to/miniconda3/envs/pi05/bin/python`.

## Perturbation configs

Drop-in YAMLs under `benchmark/bench_task_config/`:

| Config | What it perturbs |
|---|---|
| `bench_demo_*_clean.yml` | Baseline (no perturbation) |
| `bench_demo_language.yml` | Per-episode instruction sampled from `instruction_bank.json` |
| `bench_demo_vision.yml` | Lighting (L1–L4) + blur (cycle 5 types) + per-frame pixel shake |
| `bench_demo_vision_lighting.yml` | Lighting only |
| `bench_demo_vision_blur.yml` | Blur only |
| `bench_demo_vision_shake.yml` | Pixel shake only |
| `bench_demo_object.yml` | Target texture swap + unseen obstacles + background_plus |
| `bench_compositional_object_d{6,10,15}.yml` | OOD obstacles + OOD targets |
| `bench_compositional_object_vision_d{6,10,15}.yml` | OOD objects + vision (lighting, blur, pixel shake) |
| `bench_compositional_full_d{6,10,15}.yml` | All axes: OOD objects + vision + background randomization |

See the YAMLs in `benchmark/bench_task_config/` for parameter-level details, and `docs/videos/` (`vision*.mp4`, `language.mp4`, `object.mp4`) for sample outputs.

## Scenes and tasks

| Scene | Tasks |
|---|---|
| Office | `put_mouse_on_pad`, `put_phone_on_holder`, `put_book_on_book`, `put_book_in_fileholder`, `put_milktea_on_shelf`, `put_stapler_in_drawer`, `open_drawer`, `close_drawer`, ... |
| Study | `put_book_on_stand`, `put_pen_in_cup`, ... |
| Kitchen (Small) | `put_dish_in_rack`, `place_in_sink`, ... |
| Kitchen (Large) | `microwave_heat`, `fridge_store`, ... |

Full list in [`benchmark/TASKS.md`](benchmark/TASKS.md) and `benchmark/bench_envs/`. Grounding and episode schema: [`collect/README.md`](collect/README.md).

## New tasks

1. Write the task env under `benchmark/bench_envs/<scene>/<task>.py`.
2. Add `_eval_step_lim.yml` entry under `benchmark/bench_task_config/`.
3. Add a description template under `benchmark/bench_description/task_instructions/`.

Start from an analogous sibling task (`kitchenl/`, `office/`, `study/`) — copying a proven recipe is faster than inventing from scratch.

## Troubleshooting

Common setup problems and where their fixes live:

| Symptom | Fix |
|---|---|
| `pip install -r sim/script/requirements.txt` can't find a `sapien==3.0.0b1` wheel (aarch64 / ARM machines) | Build SAPIEN from source: [docs/setup_sapien_aarch64.md](docs/setup_sapien_aarch64.md) |
| Eval success rates differ unexpectedly from reported numbers | Check `sapien.__version__` — must be `3.0.0b1`; other versions change physics/rendering and skew results (Installation step 2) |
| `ModuleNotFoundError: pkg_resources` when importing sapien | `pip install setuptools==69.5.1` (Installation step 2) |
| CuRobo fails to attach grasped objects during planning | The shipped curobo configs lack the `attached_object` link entries — run `python scripts/install/patch_aloha_curobo.py` (Installation step 3) |
| CuRobo keeps stale collision meshes across episodes | Apply the `clear_cache` patch to `world_mesh.py` (Installation step 4) |

## License

This repository is a modified fork of RoboTwin 2.0, released under the MIT license. See [`LICENSE`](LICENSE) (Copyright 2025 Tianxing Chen; Copyright 2025–2026 RoboPRO authors). `sim/`, `eval/`, `collect/`, and the policy wrappers are RoboTwin-derived; the 80-task benchmark and grounding extras are RoboPRO. Vendored openpi (`policy/pi0/src`, `policy/pi05/src`) is [Apache 2.0](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE) (Physical Intelligence). Details: [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

If you use RoboPRO, please cite RoboTwin (the simulation platform this work is based on) and this project:

```
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}

@article{TODO_robopro,
  title={TODO: RoboPRO paper title},
  author={TODO},
  journal={TODO},
  year={2026}
}
```
