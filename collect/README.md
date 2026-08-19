# Collect

This folder collects RoboPRO episodes (CuRobo demonstrations and policy rollouts),
writes grounding sidecars, replays saved trajectories, precollects eval seeds, and
converts HDF5 dumps to LeRobot.

Run from the **repo root** after `source set_env.sh`. Task configs are YAML **names**
under [`benchmark/bench_task_config/`](../benchmark/bench_task_config/). Copy-me example
(do not run as-is): [`datagen_template.yml`](../benchmark/bench_task_config/datagen_template.yml).

Episodes land in `data/<task>/<config>/` (YAML `save_path: ./data`).

## Usage

### CuRobo demonstrations

```bash
source set_env.sh
bash collect/collect_data.sh <task> <config> <gpu>
# Example:
bash collect/collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
# Multi-GPU (comma list → collect_parallel.py + collect_one_episode.py):
bash collect/collect_data.sh put_mouse_on_pad bench_demo_office_clean 0,1
```

A comma-separated GPU list dispatches seeds dynamically across those GPUs into **one**
run dir (`episode0..N-1`). `episode_num` comes from the task config.

### Policy rollouts

Use a **dedicated** `task_config` so policy episodes do not share a run dir with CuRobo:

```bash
bash policy/pi05/collect_rollout.sh <task> <task_config> <train_config> \
    <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]
```

The client is [`collect_rollout_client.py`](collect_rollout_client.py). Env vars:
`COLLECT_NUM` (default 100), `COLLECT_START_SEED`, `COLLECT_FIXED_SEED` (skip expert
solvability check), `ACTION_NOISE_VAR`.

### Replay extra modalities

Rebuilds the seeded scene, restores `_traj_data/episodeN_init.json`, replays the saved
joint path (`need_plan=False`), and records extra `data_type` keys from the overlay
config into `data/<task>/<config>_replay/`:

```bash
python collect/replay_trajectory.py <task> <collection_config> --replay-config replay_rich
```

### Grounding / inspection

Called automatically at collection time (`finalize_grounding`) when
`data_type.actor_bbox` is on. Writes `masking/episodeN.json` and bakes `grounding_mask`
into each camera group in the HDF5. Standalone re-run:

```bash
python collect/masking_resolve.py <run_dir> [ep] [--write] [--panel]
```

### Eval seeds

Writes `{BENCH_ROOT}/eval_seeds/<task>/<config>.txt` (no hdf5/mp4). Seeds start at
40000. Target: 20 seeds for `*_clean` configs, 2 otherwise.

```bash
python collect/precollect_eval_seeds.py <task> <config>
```

### LeRobot conversion

Details in [`lerobot_convert/README.md`](lerobot_convert/README.md). From the repo root
(env with `cv2`, `av`, `h5py`, `pandas`, `numpy`):

```bash
PYTHONPATH=collect python -m lerobot_convert.convert_scenes \
    --src /path/to/<task>_38scene_... \
    --out /path/to/lerobot_out
```

### Episode instructions

Regenerate language variants from `scene_info.json` (also run automatically at the end
of CuRobo collection):

```bash
python collect/generate_episode_instructions.py <task> <config> <max_num>
```

## Scripts

| File | Role |
|---|---|
| `collect_data.sh` | Entry wrapper. Single GPU → `collect_data.py`; comma GPU list → `collect_parallel.py`. Uses `$SIM_PYTHON` (repo `.venv`) if present. |
| `collect_data.py` | Sequential CuRobo collection: seed search, trajectory save, HDF5 merge, grounding, crash guards. |
| `collect_parallel.py` | Multi-GPU dispatcher. Workers pull seeds from one stream; episode index is claimed atomically (`slots.json`). |
| `collect_one_episode.py` | One seed: qualify (plan + success) then collect. Used by the parallel dispatcher. |
| `collect_rollout_client.py` | Policy-rollout client; same on-disk layout as CuRobo except `generator` / `success` attrs. |
| `masking_resolve.py` | Stage target/bin roles; write masking JSON; bake `grounding_mask` into HDF5. |
| `replay_trajectory.py` | Replay a collected run to add forgotten modalities. |
| `export_scene.py` | Per-episode t=0 scene geometry (`scene.npz` / `objects.json` / `scene_hash.txt`). Called from the collector. |
| `generate_episode_instructions.py` | Fill instruction templates from episode scene params. |
| `precollect_eval_seeds.py` | Expert-solvable eval seed lists only. |
| `verify_replay_determinism.py` | Bit-exact rebuild / cross-process / closed-loop replay checks. |
| `_env.py` | Path bootstrap (`WORKSPACE_ROOT`, `SIM_ROOT`, `BENCH_ROOT`, `DATA_ROOT`, …). |
| `replay_utils.py` | Shared scene rebuild / action replay helpers. |
| `lerobot_convert/` | HDF5 → LeRobot v2.1 (parquet + videos). |

## Output

One run dir per `(task, config)`:

```
data/<task>/<config>/
├── data/episodeN.hdf5            the episode (schema below)
├── video/episodeN.mp4            countertop cam, 30 fps, 1 video frame per dataset frame
├── scene/episodeN/
│   ├── scene.npz                 samples (M,3) f32 + normals (M,3) f32 + obj_id (M,) i32
│   │                             — surface samples of every scene object at t=0
│   │                             (0.5 cm spacing objects, 4 cm furniture/articulations)
│   ├── objects.json              per object: per_scene_id, name, tags
│   │                             (target/container/furniture/clutter/articulation),
│   │                             t=0 reference pose (pose_p, pose_q), n_samples
│   └── scene_hash.txt            sha1 over (name, id, rounded pose) — scene identity
├── masking/episodeN.json         grounding stage timeline (when actor_bbox is on)
├── _traj_data/
│   ├── episodeN.pkl              planned left/right joint paths (replay input)
│   └── episodeN_init.json        t=0 state: all actor poses, articulation root+qpos,
│                                 robot qpos/arm state, seed  (replay input)
├── scene_info.json               per episode: actor_id_map (id→name, incl. robot links),
│                                 role_names (target/destination + ids), object_pose_ids
│                                 (column order of /object_poses), seed, collision_metrics
│                                 (episode totals + hit-object names), language_perturbation,
│                                 cluttered_table_info, texture_info
├── seed.txt                      space-separated seeds; POSITION = episode index
├── slots.json · logs/ · instructions/   collection progress, per-seed logs, language variants
```

Policy-rollout runs use the same tree. `_traj_data/episodeN.pkl` then holds the policy
action sequence (`policy_actions_v1`) instead of a CuRobo joint path.

### `episodeN.hdf5` schema

T = saved frames, one per `save_freq` (default 15 physics substeps). Actor segmentation
pixels are raw `per_scene_id` (uint16 PNG); decode names via `scene_info.json`
`actor_id_map`. Role ids (`target_id` / `destination_id`) disambiguate same-model twins.

| Path | Shape / dtype | Content |
|---|---|---|
| `observation/<cam>/rgb` | (T,) JPEG bytes → 240×320×3 u8 | cams: head, left/right wrist, front, countertop, demo, demo_2 (all D435) |
| `observation/<cam>/depth` | (T,) PNG bytes → 240×320 u16 | depth in mm |
| `observation/<cam>/actor_segmentation` | (T,) PNG bytes → 240×320 u16 | per-object masks; pixel value = `per_scene_id` |
| `observation/<cam>/grounding_mask` | (T,) PNG bytes | baked target/bin mask (when `actor_bbox` is on) |
| `observation/<cam>/intrinsic_cv / extrinsic_cv / cam2world_gl` | (T,3,3)/(T,3,4)/(T,4,4) f32 | camera model per frame |
| `joint_action/{left_arm,left_gripper,right_arm,right_gripper,vector}` | vector: (T,14) f64 | **Drive target** (commanded joints). Written when `data_type.qpos` is on. |
| `joint_state/{left_arm,left_gripper,right_arm,right_gripper,vector}` | vector: (T,14) f64 | **Realized physics qpos** (`entity.get_qpos()`). Same layout as `joint_action`; not the same signal. |
| `endpose/…` | | task-space EE poses + normalized grippers |
| `object_poses` | (T, N, 7) f32 | pose of every non-robot actor `[x,y,z,qw,qx,qy,qz]`; column j = `object_pose_ids[j]` |
| `contact` | (T,) u8 | unintended robot↔world contact in the frame window (see below) |
| `contact_impulse` | (T,) f32 | max contact impulse (N·s) behind the contact flag that frame |
| `contact_pairs` | (T,) JSON bytes | `"a\|b"` body pairs that frame; non-robot bodies are `name#per_scene_id` |
| `collision` / `collision_impulse` / `collision_pairs` | as above | same triple for the collision flag (see below) |
| `pointcloud` | (T, 0) | empty placeholder — rebuild dense clouds from depth + K/E + masks |

Both `joint_action` and `joint_state` use `vector` = left arm (6) + left gripper (1) + right arm (6) + right gripper (1). Grippers are normalized to `[0, 1]`. `joint_action` is the controller target; `joint_state` is the actual joint angles after physics. t=0 robot qpos is also in `_traj_data/episodeN_init.json` (replay input), not as a per-frame HDF5 series.

**HDF5 attrs:** `generator` (`curobo_collision_aware` / `curobo_collision_unaware` /
policy name e.g. `pi05`), `seed`, `success` (task `check_success` at episode end),
`task_name`, `task_config`, `enable_collision_metrics`,
`planner_exclude_obstacles` (−1 omitted / 0 aware / 1 blind),
`planner_blind_to_obstacles` (resolved).

### Contact / collision flags

Both flags are OR-accumulated over each frame’s ~15 physics substeps and flushed per
saved frame. Thresholds in `_bench_base_task.py`: **1 cm / 0.1 rad** displacement,
**10 N** furniture impulse gate. Pair impulse must be > 1e-6 N·s (PhysX zero-impulse
margin contacts are ignored). With `enable_collision_metrics` off the arrays still
exist but are all zeros.

- **`contact[t]`** — the robot touched something it was not supposed to touch:
  a robot link, the **held target** knocking another object, or a robot-**actuated**
  body (e.g. a drawer being closed). Excludes robot self-contacts, wheel↔ground,
  the task **target**, all **destinations** (`des_obj*`), and intended grasps
  (anything passed to `grasp_actor`, including articulation handles).
- **`collision[t]`** — impulse-gated non-gripper robot↔furniture hit, **or** a static
  object that is (a) touched by the robot / held target, (b) displaced ≥ threshold
  from its baseline pose, and (c) **in motion**. An object shoved only by an actuated
  body (drawer close) is contact-only, never collision. Aftermath (object resting in
  its new pose, even while still touching) is contact-only. After 90 still substeps
  (0.36 s) the settled pose becomes the new baseline. No baseline snapshot ⇒ not a
  collision.
- **Episode counts** (`scene_info.collision_metrics`, categories
  `robot/target_to_static_object` and `robot_to_furniture`) count one event per
  displacement crossing. Knock → settle (90 still substeps) → knock again = 2.
  `intended_to_static_object` is unused (always 0). Hit-object names are included.

If thresholds change, collision can be recomputed offline from `/object_poses` +
`contact_pairs`/impulse without re-collecting. Link/sphere proximity: `scene.npz` +
`objects.json` + `joint_action` qpos + robot FK (`fk_basis.npz`). Shipped flags stay
valid for the thresholds they were recorded with.

## Notices

- **Resume.** Existing `episodeN.hdf5` is skipped and `seed.txt` continues. After
  changing what gets collected, delete old episodes or rename the config, or the run
  silently no-ops.
- **Matched pairs.** `use_seed: true` plus copying `seed.txt` from a finished run
  reproduces the same scenes/instructions.
- **Planner vs metrics.** Set both knobs explicitly. If `planner_exclude_obstacles` is
  omitted, planner clutter-blindness still follows `enable_collision_metrics`.
  `false` = planner sees clutter; `true` = planner is blind to clutter (furniture is
  always avoided). See the comments in `datagen_template.yml`.
- **Success filtering.** The CuRobo collector keeps only task-success episodes
  (`success` attr is always True on kept files). Policy rollouts keep failures too
  (`success=False`).
- **Cannot derive later.** `data_type.actor_bbox` (and `link_bbox`) are exact physics
  boxes including occluded extents — turn them on at collection time. Point clouds,
  2D boxes, visible-surface 3D boxes, and overlays can be rebuilt from depth + K/E +
  masks in a dataloader. Keep `data_type.pointcloud` off.
- **Grounding needs `actor_bbox`.** Without it, collection skips `finalize_grounding`.
- **aarch64 (GB10 / DGX Spark).** SAPIEN 3.0.0b1 has no aarch64 wheel — build from
  source: [`docs/setup_sapien_aarch64.md`](../docs/setup_sapien_aarch64.md).
- **Role gaps.** Some kitchen tasks store the target in task-specific attrs (grounding
  falls back to `_get_target_object_names()`). `put_can_next_to_basket` has no
  destination object — the goal is a pose next to an anchor.
- **Replay.** Replaying the same episode on a different machine can differ at
  millimeter scale and change the outcome. Restoring a mid-episode PhysX snapshot
  (`pack()` / `unpack()`) matches poses and joints but not the contact warm-start
  cache, so success labels can flip — rebuild from the scene seed and replay
  actions from step 0 instead.
- **Interpreter.** `collect_data.sh` prefers `$SIM_PYTHON`, then the repo `.venv`.
  `RT_DEBUG_STUCK=1` holds the viewer on the first seed-search failure.
- **Throughput (1× RTX 4080, d10, put_cup_on_coaster).** ≈ 82 s/kept episode including
  seed-search rejects; seed replay without search ≈ 37 s/episode. Episode ≈ 37 MB.
