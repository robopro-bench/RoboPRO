# visualization/ — inspect & export collected episodes

Standalone tools (need only the `robopro` env: numpy/h5py/opencv); they import no
engine code and never modify the dataset. Everything they produce is **derived** from
the episode HDF5 + `scene_info.json` sidecar — generate it when you need it, delete it
freely (`data/<save>/<task>/<config>/export/` and `/viz/` are throwaway).

`<run_dir>` below = one (task, config) output directory, e.g.
`data/dataset/put_cup_on_coaster/datagen_template`. Role colors everywhere:
**target = red, destination = green, obstacles = orange, robot = blue.**

Run from the repo root (or `sim/` with `../visualization/...`).

---

## The four tools

```bash
python visualization/inspect_hdf5.py  <run_dir>/data/episode0.hdf5  # HDF5 tree: keys, shapes, dtypes, attrs
python visualization/export.py        <run_dir> <ep> [flags]        # derive panels / clouds / boxes (see below)
python visualization/viz_episode.py   <run_dir> <ep> [flags]        # richer single-episode viz (+.ply, topdown, legend)
python visualization/flag_timeline.py <run_dir> [ep] [flags]        # per-frame contact/collision timeline (see below)
```

`export.py` is the everyday one. `viz_episode.py` is a heavier inspector (adds top-down
projections, a `legend.json`, stored-vs-rebuilt cloud comparison); same flags.

---

## Cameras — which is which (READ THIS before picking `--cam`)

Collection captures **all** of these per episode, so which cameras train the policy is
chosen later at the dataloader — the data doesn't lock you in.

| camera | what it is | training role |
|---|---|---|
| `countertop_camera` | overhead scene cam (all 4 scenes) | **policy input** — the `cam_high` view (see note) |
| `left_camera` | left-wrist, egocentric | **policy input** → `cam_left_wrist` |
| `right_camera` | right-wrist, egocentric | **policy input** → `cam_right_wrist` |
| `head_camera` | robot-head egocentric (D435, 320×240) | current code's `cam_high`, **being revised** |
| `demo_camera`, `demo_camera_2` | staged third-person overview (office/study only) | visualization only |
| `front_camera` | front scene cam | visualization only |

**Training set (per the RoboPRO team): `countertop_camera` + `left_camera` + `right_camera`.**
The repo *currently* wires `head_camera` as `cam_high` (`process_data.py`,
`collect_rollout_client.py`), but the head view is being fixed and `cam_high` is moving to
`countertop_camera`; `demo_camera` is visualization-only. Because collection saves every
camera, this needs **no re-collection** — the dataloader just selects the final set. Match
whatever the benchmark's eval ultimately provides.

**For visualization → any camera works** (role assignment comes from `scene_info.json` and
is identical in every view — only the viewpoint changes). Prefer **`countertop_camera`**: it
exists in all four scenes and is a real training camera, so you inspect the masks on a view
the policy will actually use. `--cam all` renders one panel per camera for a thorough check.

---

## export.py — flags

| flag | default | meaning |
|---|---|---|
| `<run_dir> <ep>` | — | positional: the run dir and episode index (e.g. `0`) |
| `--what` | all of them | comma list of `pcd,bbox2d,bbox3d,bbox3d_exact,panel`, or `all` |
| `--cam` | `head_camera` | one camera, a comma list, or `all` |
| `--frames` | `first,mid,last` | frames for **pcd / bbox\*** only: `all`, `every:N`, or a list like `0,40,80` (keywords `first,mid,last` allowed). **The panel ignores this** — it's always 6 rows. |
| `--stride` | `1` | pixel stride for point clouds (`2` = ¼ the points, faster/smaller `.ply`) |
| `--split-env` | off | keep environment objects (furniture/appliances) a separate class instead of merging them into `obstacle` |
| `--out` | `<run_dir>/export/episode<ep>` | custom output dir |

### Examples

```bash
# 1. Quick masking check — just the 6-row panel from an overview cam (fast, no heavy files)
python visualization/export.py data/dataset/put_cup_on_coaster/datagen_template 0 --what panel --cam countertop_camera

# 2. What the POLICY sees — panels for the three training cameras
python visualization/export.py <run_dir> 0 --what panel --cam countertop_camera,left_camera,right_camera

# 3. Everything, every camera (thorough inspection)
python visualization/export.py <run_dir> 0 --what all --cam all

# 4. Point clouds + exact 3D boxes, sampled across the trajectory
python visualization/export.py <run_dir> 0 --what pcd,bbox3d_exact --cam countertop_camera --frames every:20

# 5. Full-trajectory boxes as data (every frame), no images
python visualization/export.py <run_dir> 0 --what bbox3d_exact --frames all
```

### Outputs (under `<run_dir>/export/episode<ep>/`)

| flag | files | content |
|---|---|---|
| `panel` | `panel[_<cam>].png` | quick-look grid — **always 6 rows** (first, last, 4 evenly spaced between; independent of `--frames`), cols = RGB \| depth \| seg \| role overlay \| 2D boxes |
| `pcd` | `pcd_<cam>_f<k>.ply` | dense role-colored point cloud (+ exact-box wireframes when available), open in MeshLab. Honors `--frames`. |
| `bbox2d` | `bbox2d.json` | per frame, per object: pixel box (xyxy) + id + name + role |
| `bbox3d` | `bbox3d.json` | per frame, per object: world AABB of its **visible surface** (what the camera sees) |
| `bbox3d_exact` | `bbox3d_exact.json` | per frame, per object: **exact full-extent ORIENTED box** (center/half_size/quat/corners) aligned to the object's own pose (incl. occluded parts). Needs `actor_bbox` in the HDF5. |
| always | `meta.json` | id→name map, role→ids, units, `actor_bbox_available` flag |

Notes:
- **Masks** aren't dumped as PNGs — they live in the HDF5 (`actor_segmentation`) and the
  panel renders them. **`.npz` clouds** aren't written — the dataloader rebuilds clouds
  from depth on the fly, so a stored copy is redundant.
- **`bbox3d` vs `bbox3d_exact`:** visible-surface boxes hug only the pixels a camera sees
  (a half-occluded mug → box around the visible half); exact boxes are the physics
  engine's true full-size box for every object, occluded or not. Exact must be recorded at
  collection (`data_type.actor_bbox: true`) — it can't be reconstructed afterward.
- Per-episode `.mp4`s are written automatically during collection (next to the data) — no
  tool needed just to watch a run.

---

## flag_timeline.py — contact/collision timelines

Reads the per-frame `contact`/`collision` labels recorded at collection time
(`enable_collision_metrics: true`) and prints contact/collision frame ranges with
video timecodes, peak impulses, per-window object pairs, and each hit object's
t0→window-end pose movement:

```bash
python visualization/flag_timeline.py <run_dir> [ep] [--min-impulse 0.01]
```

Default shows ALL contacts (incl. zero-impulse resting); `--min-impulse` hides
light touches. Full label semantics: DATA_GEN.md §5.2.

---

## Timing a collection run

`sim/time_run.sh` (bash, lives next to `collect_data.sh`):

```bash
cd sim
bash time_run.sh <task> <config> [gpu]      # wall time, sec/episode, full-dataset projection
```
