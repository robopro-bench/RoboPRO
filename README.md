# RoboPRO

**P**erturbation-**R**esilient **O**bstacle-awareness — a bimanual manipulation benchmark for policy robustness evaluation.

**Project page:** https://robopro-bench.github.io/RoboPRO/

RoboPRO extends the RoboTwin simulation framework with:
- **Realistic scenes** across office, study, kitchen (small & large) domains
- **Systematic perturbation suite** — Language, Vision, and Object axes for evaluating policy robustness
- **Aloha-Agilex** bimanual embodiment with CuRobo motion planning

## Installation

System prereqs (one-time): `libvulkan1 mesa-vulkan-drivers vulkan-tools` (apt), `ffmpeg`, and an NVIDIA driver with CUDA 12.x.

```bash
git clone https://github.com/robopro-bench/RoboPRO.git
cd RoboPRO
```

### 1. Conda env

```bash
conda create -n robopro python=3.10 -y
# Keep the env isolated from ~/.local/lib site-packages (otherwise sapien/torch
# may resolve there instead of in robopro):
conda env config vars set -n robopro PYTHONNOUSERSITE=1
conda activate robopro
```

### 2. Python deps

```bash
cd customized_robotwin
pip install -r script/requirements.txt
pip install setuptools==69.5.1       # provides pkg_resources for sapien
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
bash script/_install.sh              # patches sapien urdf_loader + mplib planner
```

`script/_install.sh` also clones CuRobo v0.7.8 into `envs/curobo/` and pip-installs it editable, then re-pins `warp-lang==1.12.0` and `setuptools==69.5.1`. If you keep `scipy==1.10.1` from `requirements.txt`, `scikit-image` will print a version-conflict warning — harmless.

### 3. Assets (~15 GB)

```bash
cd ..                                # back to repo root
python scripts/install/download_assets.py
```

This fetches the HuggingFace bundle (`RoboPRO/RoboPRO_assets`) into `benchmark/assets/` (objects, embodiments, background_texture, backgrounds). The bundle already includes the large `aloha-agilex/.../meshes/box2_Link.dae` mesh — no separate fetch needed.

The shipped `task_config/_embodiment_config.yml` uses upstream-relative paths (`./assets/embodiments/...`). RoboPRO keeps assets under `benchmark/assets/`, so add a one-line symlink so the upstream paths resolve:

```bash
ln -sfn ../benchmark/assets customized_robotwin/assets
```

Generate the local-path curobo configs from the shipped templates, and patch them so CuRobo can attach grasped objects (the shipped configs lack the `attached_object` link entries):

```bash
ASSETS_PATH="$(pwd)/benchmark"
cd benchmark/assets/embodiments/aloha-agilex
for side in left right; do
  sed "s|\${ASSETS_PATH}|$ASSETS_PATH|g" curobo_${side}_tmp.yml > curobo_${side}.yml
done
cd -
python scripts/install/patch_aloha_curobo.py
```

### 4. CuRobo cache patch

In `customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py`, replace `clear_cache` with:

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
cd customized_robotwin
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench
python script/bench_script/visualize_task_scene.py \
    put_mouse_on_pad bench_demo_office_clean \
    --bench-subdir office --rollout --no-render --seed 0 --save_data
```

Expected on success: a `Success: True` line and an MP4 at `customized_robotwin/data/bench_data/video/episode_put_mouse_on_pad_0.mp4` (~176 frames @ 320×240).

## Usage

All commands run from `customized_robotwin/` with the bench env exported:

```bash
cd customized_robotwin
source set_env.sh                  # exports BENCH_ROOT + ROBOTWIN_ROOT
export ROBOTWIN_BENCH_TASK=bench   # routes loaders to BENCH_ROOT/{bench_task_config, bench_envs}
```

### Collect demonstrations

```bash
bash collect_data.sh <task_name> <task_config> <gpu_id>
# Example:
bash collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
```

Episodes land in `customized_robotwin/data/<task_name>/<task_config>/`.

### Run inference (policy eval)

Eval rolls a trained checkpoint out against a `(task, config)` pair and writes a per-rollout success log. Two modes depending on whether your policy fits in the same Python env as the simulator.

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
bash policy/pi05/eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
# Example:
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 0 0
```

**Mode B — dual-env / dual-process** (recommended for pi05 since openpi+jax need an isolated uv venv at `policy/pi05/.venv/`):

```bash
bash policy/pi05/eval_double_env.sh <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <gpu_spec>
# Example (single GPU):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0
# Example (split: server on GPU 0, sim client on GPU 1):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0:1
```

The script spawns a `policy_model_server.py` in the pi05 venv and an `eval_policy_client.py` in the RoboTwin conda env, communicating over a free socket port.

**Direct Python invocation** (bypassing the shell wrappers):

```bash
python script/eval_policy.py \
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
customized_robotwin/eval_result/bench_eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
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

For arrayed sweeps over a `(tasks × configs)` grid, see `customized_robotwin/robotwin_*.sbatch` for templates.

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

See the YAMLs in `benchmark/bench_task_config/` for parameter-level details, and `docs/videos/` (`vision*.mp4`, `language.mp4`, `object.mp4`) for sample outputs.

## Scenes and tasks

| Scene | Tasks |
|---|---|
| Office | `put_mouse_on_pad`, `put_phone_on_holder`, `put_book_on_book`, `put_book_in_fileholder`, `put_milktea_on_shelf`, `put_stapler_in_drawer`, `open_drawer`, `close_drawer`, ... |
| Study | `put_book_on_stand`, `put_pen_in_cup`, ... |
| Kitchen (Small) | `put_dish_in_rack`, `place_in_sink`, ... |
| Kitchen (Large) | `microwave_heat`, `fridge_store`, ... |

Full list in `benchmark/bench_envs/`.

## New tasks

1. Write the task env under `benchmark/bench_envs/<scene>/<task>.py`.
2. Add `_eval_step_lim.yml` entry under `benchmark/bench_task_config/`.
3. Add a description template under `benchmark/bench_description/task_instructions/`.

Naming tip: never reuse an existing RoboTwin task name. Start from an analogous sibling task (`kitchenl/`, `office/`, `study/`) — copying a proven recipe is faster than inventing from scratch.

## Citation

If you use RoboPRO, please cite our RSS 2026 Workshop (RoboData) paper *Data and Evaluation Protocols for Robust Robotic Manipulation*:

```bibtex
@inproceedings{li2026robopro,
  title     = {Data and Evaluation Protocols for Robust Robotic Manipulation},
  author    = {Li, Zhiyuan and Eret, Charles and Zhao, Xuan and Wu, Yangzheng and Rasouli, Amir},
  booktitle = {Robotics: Science and Systems (RSS) 2026 Workshop on RoboData},
  year      = {2026},
  url       = {https://robopro-bench.github.io/RoboPRO/}
}
```

## License

See `LICENSE`.
