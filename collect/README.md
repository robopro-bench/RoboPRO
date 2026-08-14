# Collect — data generation

How to collect RoboPRO episodes: per-object masks + depth + role annotations,
ready for large-scale runs. The collector lives in this directory and drives
the `sim/` runtime plus `benchmark/` tasks.

**Going forward:**

- ✅ **KEEP AS-IS (§2):** the perception/data plumbing — raw-id masks, ID→name/role maps,
  PNG compression, depth-based 3D lift, grounding. Verified end-to-end; please build on
  top rather than modify.
- 🔁 **YOURS TO BUILD (§3):** the labeling / collision-interpretation layer. We prototyped
  one (4-way outcome labels + filtering + planner-blindness for negative samples) and then
  **removed it from the collector** so you start from a clean slate — §3 summarizes what it
  was and where to find the full reference implementation in git history.
- One commented example config remains: `benchmark/bench_task_config/datagen_template.yml`.

Quick start: `source set_env.sh && bash collect/collect_data.sh <task> <config> 0`.
Grounding lives in `collect/masking_resolve.py` (called from the collector).

---

## 1. Change map (files the collector pipeline touches)

| File | What changed |
|---|---|
| `sim/envs/camera/camera.py` | segmentation returns **raw uint16 ids** (was palette-colorized RGB) |
| `sim/envs/_base_task.py` | + `get_actor_id_map()`, + `get_role_names()`; + optional `data_type.actor_bbox` recording (exact per-frame 3D boxes from physx) |
| `sim/envs/utils/pkl2hdf5.py` | + `seg_encoding()` — PNG-compresses uint16 masks into HDF5 |
| `collect/collect_data.py` | writes id-map/role sidecar into `scene_info.json`; organized per-episode output + end-of-run summary; per-episode crash guards (one bad episode no longer kills a run) |
| `collect/masking_resolve.py` | offline grounding: stage target/bin roles + bake masks into HDF5 |
| `collect/time_run.sh` | timing helper (wall time → sec/episode → dataset projection) |
| `benchmark/bench_task_config/datagen_template.yml` | single commented example config |

Nothing outside these files is modified — the `benchmark/bench_envs/` scene classes and
all stock `bench_*` configs are **untouched stock RoboPRO**.

---

## 2. ✅ KEEP AS-IS — the perception / data plumbing

These are verified end-to-end (masks↔ids↔names↔roles cross-checked visually and
programmatically across office/study/kitchen tasks; depth→3D lift metrically validated —
e.g. a reconstructed cup measures 4.3×4.4×6.9 cm at the true table height).

### 2.1 Raw-ID segmentation (`camera.py`)
Stock RoboPRO saved segmentation as **palette-colorized RGB images** — object identity
was irreversibly destroyed (you can't reliably map colors back to objects). Now
`_get_segmentation` returns the raw SAPIEN id planes as **uint16**:
mesh level = `seg_labels[..., 0]`, actor level = `seg_labels[..., 1]`.
Masks become usable training supervision. Configs enable it via
`data_type.actor_segmentation: true` (mesh level exists but stays off — sub-part ids
have no name map).

### 2.2 ID → name → role resolution (`_base_task.py` + sidecar)
- `get_actor_id_map()` → `{per_scene_id: name}` for every actor **including articulation
  links** (furniture decomposed into named links; unnamed robot articulations prefixed
  `robot/<link>`; ~70 entries typical).
- `get_role_names()` → the task's designated `target`/`destination` names **and their
  exact instance ids** (`target_id`/`destination_id` via `per_scene_id`). The ids
  disambiguate the target from same-model clutter (two `021_cup`s in one scene: id 7 =
  target, id 8 = clutter). Name keys retained as fallback for older data.
- Both are written into each episode's `scene_info.json` by `collect_data.py`.
- Obstacle convention (working definition): every known object that isn't
  target/destination/robot/scene-shell counts as an obstacle; spawned-clutter vs
  environment-object identity is preserved in the data so the split can be redefined
  later without re-collecting.

### 2.3 PNG-compressed masks (`pkl2hdf5.py`)
Raw uint16 masks at 7 cams × full trajectory = **370–450 MB/episode**. `seg_encoding()`
PNG-compresses each mask frame (lossless) before HDF5 write → **~37 MB/episode (~11×)**.
Decode on read with any PNG decoder (`cv2.imdecode` → uint16).

### 2.4 Cameras & 3D lift policy
- All cameras stay **D435 (320×240)** = identical to the benchmark's eval cameras — no
  train/test distribution mismatch. Extra viewpoints are inspection-only.
- `data_type.pointcloud` stays **off**: the stock cloud is head-cam-only, FPS-downsampled
  to 1024 points. Dense labeled clouds are instead **rebuilt from depth + stored
  intrinsics/extrinsics** (that's also how the method itself lifts masks to 3D) —
  on demand inside a dataloader (same math).
- Derived data policy: point clouds, 2D boxes, visible-surface 3D boxes, masks-as-images,
  and overlays are all **computable from the HDF5 after the fact** inside a dataloader,
  zero collection-time cost. The one thing that can't be
  derived later, so it's a collection knob: `data_type.actor_bbox` (exact full-extent
  3D boxes from the physics engine, incl. occluded parts; ~KBs/ep).

### 2.5 Collection robustness (`collect_data.py`)
Per-episode try/except (a CuRobo/mesh crash is logged, cleaned up, and the run
continues) + a no-frames guard (a plan that produces no executable motion yields no
HDF5 instead of crashing the pkl→HDF5 merge). Generic safety — worth keeping under any
labeling scheme. Also: per-episode banners + an end-of-run `COLLECTION SUMMARY`
(`collect/time_run.sh` parses those summary lines).

### 2.6 Grounding (`collect/masking_resolve.py`)
Called automatically at collection time (`finalize_grounding`) when
`data_type.actor_bbox` is on. Writes `masking/episodeN.json` and bakes
`grounding_mask` into each camera group in the HDF5.

Standalone re-run from the repo root:

```
python collect/masking_resolve.py <run_dir> [ep] [--write] [--panel]
```

Timing helper: `collect/time_run.sh`.

---

## 3. 🔁 THE LABELING LAYER — removed; yours to design

**What we had (brief):** a prototype for positive/negative contrast data — a config knob
that made the CuRobo planner *blind* to clutter (its smooth path then naturally collides
→ negative samples on the same seeds as the positives), a **measured** per-episode
outcome (task-success × collision → 4 labels: success / success-with-accident /
crashed-and-failed / failed-no-accident), a `keep_labels` config filter that kept only
the wanted labels per run, a collision-free seed-acceptance mode for exact positive
counts, and the outcome stamped into each HDF5's attrs.

**It was removed from the collector** (the 4-way taxonomy and the hard ± split don't
generalize across teams). The full working implementation is preserved in git history at
commit **`e248b2d`**:

```bash
git diff e248b2d^..e248b2d -- sim/script/collect_data.py benchmark/bench_envs/
```

**Recommendation from what worked:** make the new design a **config knob + filter
system** — the collector measures a per-episode outcome (possibly with more than 4
labels), and each team's config declares which labels to keep, so every team maps
outcomes to their own ±/curriculum without forking the collector. Two things proved
valuable regardless of taxonomy: measure labels from the live scene (never assume them
from the run type), and keep task-success vs collision as separate axes. Stamping the
outcome into HDF5 attrs made every file self-describing for dataloaders.

**Facts you'll need:** the collision signal lives in
`benchmark/bench_envs/_bench_base_task.py` (`get_collision_metrics()` → contacts,
per-object events, `is_collision`), gated by `enable_collision_metrics`. Thresholds in
code: **2 cm / 0.2 rad / 10 N**.
⚠️ Stock coupling: `enable_collision_metrics: true` **also makes the planner blind to
clutter** (`update_world(exclude_obstacles=True)` in the four scene base classes) — you
will likely want to decouple those two (our removed version did it with one extra config
key; see the reference commit). Bonus signal worth keeping in mind: for collision
episodes, the metrics blob names **which objects were actually hit** — free causal
"this obstacle mattered" labels.

---

## 4. Practical notes

- Routing: run from the repo root with `source set_env.sh`; configs resolve by *name* from `benchmark/bench_task_config/`.
- ARM / aarch64 hosts (GB10 / DGX Spark): SAPIEN 3.0.0b1 has no aarch64 wheel and must be
  built from source — see `docs/setup_sapien_aarch64.md`. This is purely an environment
  build step; it does not affect the data schema, grounding, or 3D boxes produced here.
- Collection is **resumable**: existing `episodeN.hdf5` skipped, `seed.txt` continues.
  Corollary: after changing what gets collected, delete old episodes (or rename the
  config) or collection silently no-ops.
- Same scenes across two runs: stock `use_seed: true` + copying `seed.txt` from a
  finished run reproduces identical scenes/instructions (useful for matched-pair designs).
- Known role-semantics gaps (open design points): some kitchen tasks hold the target in
  task-specific attrs (grounding falls back to `_get_target_object_names()`);
  `put_can_next_to_basket` has **no destination object** — its goal is a pose next to an
  anchor. Destination semantics for such tasks are still an open question.
- Throughput anchor (1× RTX 4080, d10 clutter, put_cup_on_coaster): ≈ **82 s/kept
  episode** including seed-search rejects; seed replay without search ≈ 37 s/episode.
  Episode ≈ 37 MB. `collect/time_run.sh` re-measures and projects.

---

## 5. Dataset contents & format (current pipeline — 2026-07-09)

Everything below is written by `bash collect/collect_data.sh <task> <config> <gpu|gpu,gpu>`
(single-GPU stock, or comma list = dynamic multi-GPU dispatch via
`collect/collect_parallel.py` + `collect/collect_one_episode.py`). One run dir per
(task, config):

```
<save_path>/<task>/<config>/
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

### 5.1 `episodeN.hdf5` schema (T = saved frames, one per `save_freq`=15 physics substeps)

| Path | Shape / dtype | Content |
|---|---|---|
| `observation/<cam>/rgb` | (T,) JPEG bytes → 240×320×3 u8 | cams: head, left/right wrist, front, countertop, demo, demo_2 (all D435) |
| `observation/<cam>/depth` | (T,) PNG bytes → 240×320 u16 | depth in mm |
| `observation/<cam>/actor_segmentation` | (T,) PNG bytes → 240×320 u16 | per-object masks; pixel value = `per_scene_id`, decode via `actor_id_map` |
| `observation/<cam>/intrinsic_cv / extrinsic_cv / cam2world_gl` | (T,3,3)/(T,3,4)/(T,4,4) f32 | camera model per frame |
| `joint_action/{left_arm,left_gripper,right_arm,right_gripper,vector}` | vector: (T,14) f64 | joint-space action (vector = both arms + grippers) |
| `endpose/…` | | task-space EE poses + normalized grippers |
| `object_poses` | (T, N, 7) f32 | pose of every non-robot actor per frame `[x,y,z,qw,qx,qy,qz]`; column j = object `object_pose_ids[j]` (scene_info) |
| `contact` | (T,) u8 | robot↔world contact in the frame window (see 5.2) |
| `contact_impulse` | (T,) f32 | max contact impulse (N·s) behind the contact flag that frame |
| `contact_pairs` | (T,) JSON bytes | list of `"a\|b"` body pairs touching that frame; non-robot bodies are `name#per_scene_id` (exact instance — twins of one model are distinct; robot links plain). Datasets before 2026-07-10 use plain names |
| `collision` / `collision_impulse` / `collision_pairs` | as above | same triple for the collision flag (see 5.2) |
| `pointcloud` | (T, 0) | EMPTY stock placeholder — ignore (pointcloud stays off; rebuild dense clouds from depth+K/E+masks) |

**HDF5 attrs (self-describing provenance):** `generator` (words:
`curobo_collision_aware` / `curobo_collision_unaware` / policy name e.g. `pi05`),
`seed`, `success` (task check_success at episode end — CuRobo datasets keep only
successes so this reads True; policy-rollout datasets keep failures too),
`task_name`, `task_config`, `enable_collision_metrics`,
`planner_exclude_obstacles` (−1 omitted / 0 aware / 1 blind),
`planner_blind_to_obstacles` (resolved).

### 5.2 Per-frame contact / collision semantics (recorded AT COLLECTION TIME)

Both flags are OR-accumulated over each frame's ~15 physics substeps inside
`check_collisions()` and flushed per saved frame — **the labels ship with the
data; no postprocessing needed.** Current thresholds: **1 cm / 0.1 rad
displacement, 10 N furniture impulse gate** (`_bench_base_task.py` constants).

- `contact[t]` = the robot touched something it was NOT supposed to touch —
  directly (a robot link), through the HELD TARGET object (a carried book
  knocking a bottle), or through a robot-ACTUATED body (the drawer being closed
  shoving an item) — **with actual force exchange** (pair impulse > 1e-6 N·s;
  PhysX zero-impulse margin "contacts" are not touches). Excludes robot
  self-contacts, base wheel↔ground support, the task TARGET itself, ALL
  DESTINATIONS (`des_obj*`), and INTENDED contacts (anything the task passed to
  `grasp_actor` — drawer/appliance handles + their articulation links).
- `collision[t]` = impulse-gated non-gripper robot↔furniture hit, OR a static
  object simultaneously (a) touched by the robot / held target (an object
  shoved by a robot-ACTUATED body — the drawer being closed — is CONTACT-only,
  never collision: grasp_actor never runs in pure policy eval, so excluding it
  keeps eval collision numbers exactly comparable to collection),
  (b) displaced ≥ threshold from its episode-start pose, and (c) **in
  motion** — where the touch must be FORCE-BEARING (impulse > 1e-6 N·s): a
  forceless graze on a moving object, or an object sliding back along the arm
  after an already-counted knock, earns no collision frames (per-substep motion ≥ 0.1 mm/0.001 rad, or ≥ 1 mm/0.01 rad across a
  rolling 30-substep window; flag persists ≤ ~1 window ≈ 0.12 s past the last
  observed motion). The shove is flagged; the aftermath — the object resting in
  its displaced pose, even while still in contact — is contact-only. A
  destination box is never the "toucher": an object knocked against a `des_obj`
  and resting there stops flagging once it stops moving. Once a moved object
  has stayed still for 90 uninterrupted substeps (0.36 s — same clock as the
  watch), its settled pose becomes its NEW displacement baseline — a
  later graze is not a collision unless it moves the object past the thresholds
  again. No baseline snapshot ⇒
  NOT a collision — no phantom flags.
- Episode-level counts (scene_info `collision_metrics`; categories
  robot/target_to_static_object + robot_to_furniture (intended_to_static_object
  retired 2026-07-14, kept in the schema as always-0)) use
  displacement-driven counting: a touched static object is counted once per
  DISPLACEMENT EVENT, at the moment its cumulative pose change from baseline
  crosses threshold; the settled re-baseline ends the event and re-arms the
  object, so knock → settle (90 still substeps) → knock again = 2 collisions. The watch starts
  at a FORCE-BEARING touch (impulse > 1e-6 N·s — same principle as the contact
  flag; forceless margin contacts earn no causal credit) and stays live while
  the object is touched-with-force OR still moving (a slow topple is followed
  all the way down); it expires after 90 substeps (0.36 s) with neither, and
  expired watches are not revived (settling creep and object→object chains
  stay unattributed). The reference (baseline) pose likewise updates only
  after 90 UNINTERRUPTED still substeps — motion inside the window restarts
  the stillness count. The counting moment also sets that
  frame's `collision[t]` once (pair = the object's last toucher), so a delayed
  crossing that happens after contact ended is still localized in the
  per-frame labels.

**Recompute-later guarantee:** if thresholds/definitions ever change, collision
can be recomputed offline from `/object_poses` + `contact_pairs`/impulse without
re-collecting; link/sphere proximity from `scene.npz` + `objects.json` +
`joint_action` qpos + robot FK (`fk_basis.npz`). The shipped flags remain valid
for the thresholds they were recorded with.

### 5.3 Replay — collect forgotten modalities later

Every episode is replayable (bit-exact joint trajectory, verified max |Δq| = 0):

```
python script/replay_trajectory.py <task> <collection_config> --replay-config replay_rich
```

rebuilds the seeded scene, restores `_traj_data/episodeN_init.json` (authoritative),
re-runs the saved joint paths with `need_plan=False`, and records the extra
`data_type` modalities from `replay_rich.yml` into
`data/replay_data/<task>/<config>_replay/`.

### 5.4 Inspection

Re-run grounding or render a masking panel from the repo root:

```
python collect/masking_resolve.py <run_dir> [ep] [--write] [--panel]
```
