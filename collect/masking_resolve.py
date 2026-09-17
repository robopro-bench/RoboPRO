#!/usr/bin/env python3
"""Offline masking resolver: derive per-stage target + bin roles from recorded data.

The training masks need exactly two role classes per stage: TARGET (the object being
manipulated) and BIN (where it goes). This module derives them *offline* from what the
simulator already recorded per frame -- no primitive instrumentation, no re-collection:

  actor_bbox/{id,pose,aabb_min,aabb_max}   (T, N, ..)   per-frame pose + world AABB
  scene_info.json  actor_id_map / role_names / cluttered_table_info

Pipeline (per episode):
  1. MOTION SEGMENTATION -> stages. The manipulated object of each stage is simply the
     rigid actor that MOVES; a chain yields several disjoint motion windows in time, one
     ordered stage each. (verified: chain_apple_bin_bowl_rack_spoon_sink -> 3 stages.)
  2. BIN per stage, geometry-first with a light task-name intent gate:
       pick_*                          -> no bin (pick-only)
       name has next/beside/in_front   -> REGION centred on the nearest object (anchor)
       name has an explicit table dest -> REGION = table free-space
       name has drawer/cabinet/fridge/microwave, no rigid container found
                                       -> DEFERRED to Phase 2 (articulation link box)
       else                            -> smallest rigid AABB containing the settle point
                                          (object / rigid-fixture bin); else table region
  3. Roles are per-stage, so chains flip correctly (book = target then bin). Only target
     and bin are ever emitted; a region's anchor is internal metadata, never a mask class.

Writes <run_dir>/masking.json and prints a summary. Articulation-only tasks (open/close,
no rigid mover) and 'deferred' bins are completed in Phase 2 (replay-backfilled link boxes).
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np

# --- self-contained io + palette -------------------------------------------
# This module is the grounding engine: the resolver + the region/OBB logic
# that bakes target/bin masks into the HDF5 at collection time (finalize_grounding).
# It depends only on numpy / h5py / cv2 / json.
SCENE_NAMES = {"ground", "wall", "table", "floor"}   # shell: never a target/bin/obstacle
ROBOT_COLOR = np.array([80, 130, 230], np.uint8)
OTHER_COLOR = np.array([130, 220, 220], np.uint8)
ROLE_COLORS = {  # RGB, for overlays
    "target": np.array([230, 40, 40], np.uint8),
    "destination": np.array([40, 200, 60], np.uint8),
    "obstacle": np.array([255, 160, 20], np.uint8),
}
TARGET_COL = np.array([230, 40, 40], np.uint8)   # training target mask = red
BIN_COL = np.array([40, 200, 60], np.uint8)      # training bin mask = green
ART_COL = np.array([200, 70, 210], np.uint8)     # articulation links (unresolved fixtures)


def decode_frames(ds, flags):
    return [cv2.imdecode(np.frombuffer(bytes(b), np.uint8), flags) for b in ds]


def load_seg(ds):
    """Segmentation dataset -> (T,H,W) uint16. Handles PNG-encoded and raw."""
    if ds.dtype.kind == "S":
        return np.stack(decode_frames(ds, cv2.IMREAD_UNCHANGED)).astype(np.uint16)
    arr = np.asarray(ds)
    if arr.ndim == 4:
        sys.exit("!! segmentation is COLORIZED (T,H,W,3) legacy data — "
                 "re-collect with the patched camera.py (raw uint16 ids).")
    return arr.astype(np.uint16)


def load_sidecar(run_dir, ep):
    """scene_info.json for one episode -> (ep_info, id_map, roles, robot_ids)."""
    with open(os.path.join(run_dir, "scene_info.json")) as f:
        ep_info = json.load(f).get(f"episode_{ep}", {})
    id_map = {}
    for k, v in ep_info.get("actor_id_map", {}).items():
        id_map[int(k)] = ("robot" + v) if v.startswith("/") else v  # old data: robot art unnamed
    roles = ep_info.get("role_names", {})
    robot_ids = {i for i, n in id_map.items() if n.startswith("robot/")}
    return ep_info, id_map, roles, robot_ids


_OBB_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _quat_to_R(q):
    """quaternion(s) (...,4) wxyz -> rotation matrix/matrices (...,3,3). Works for a
    single quat (4,) or a batched array."""
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), float)
    R[..., 0, 0] = 1 - 2 * (y*y + z*z); R[..., 0, 1] = 2 * (x*y - z*w); R[..., 0, 2] = 2 * (x*z + y*w)
    R[..., 1, 0] = 2 * (x*y + z*w); R[..., 1, 1] = 1 - 2 * (x*x + z*z); R[..., 1, 2] = 2 * (y*z - x*w)
    R[..., 2, 0] = 2 * (x*z - y*w); R[..., 2, 1] = 2 * (y*z + x*w); R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def _obb_to_aabb(center, half, quat):
    """Oriented box arrays (center/half (...,3), quat (...,4)) -> the axis-aligned bound
    (amin, amax). Used to derive an AABB for internal containment checks when only the
    OBB is stored."""
    R = _quat_to_R(quat)
    ext = np.matmul(np.abs(R), np.asarray(half, float)[..., None])[..., 0]
    center = np.asarray(center, float)
    return center - ext, center + ext


def _obb_dict(center, half, quat):
    """Single oriented box (center, half, quat) -> serializable dict with 8 world corners."""
    center, half = np.asarray(center, float), np.asarray(half, float)
    corners = (_OBB_SIGNS * half) @ _quat_to_R(quat).T + center
    return {"center": center.round(4).tolist(), "half_size": half.round(4).tolist(),
            "quat": np.asarray(quat, float).round(6).tolist(), "corners": corners.round(4).tolist()}


# ---- tunables (Phase 3 knobs) -------------------------------------------------
MOVE_MIN = 0.05      # a "mover" must displace >= 5 cm over the episode
ANG_MOVE_MIN = 0.35  # ...OR rotate >= ~20 deg (revolute doors barely translate)
SPEED_THR = 0.0015   # >= 1.5 mm/frame counts as "moving" for window extraction
ANG_THR = 0.02       # >= ~1.1 deg/frame of rotation counts as "moving"
GAP_MERGE = 25       # merge motion windows separated by < this many frames (regrasp)
MIN_WIN = 4          # drop motion windows shorter than this many frames
PAD = 0.02           # AABB containment padding (m)
ON_TOP_GAP = 0.12    # a target may sit up to this far ABOVE a support's top (on-top bins)
ELEVATED = 0.04      # settle this far above the table + no rigid bin + fixtures present
                     # => resting in an articulated fixture we can't box yet (Phase 2)
ANCHOR_MAX = 0.30    # anchor for a "next to" region must be within this of the settle (m)
REGION_MARGIN = 0.06 # disk radius floor = anchor xy half-extent + this (m)
TABLE_OBJ_PAD = 0.05 # table free-space bins keep this clearance (m) around every other
                     # object's footprint (a destination shouldn't hug an existing object).
                     # Tunable; also recorded per-bin in masking.json ("obj_pad").
VIS_FRAMES = 6       # frames sampled per camera for the visibility filter
MIN_PIXELS = 30      # an id must cover >= this many pixels somewhere to be maskable
GRIP_NEAR = 0.28     # a real target stays within this of a gripper while carried
                     # (endpose ref sits ~19 cm above the grasped object -> measured)
MIN_HELD_FRAC = 0.25 # >= this fraction of its motion window spent near a gripper

FIXTURE_TOKENS = ("drawer", "cabinet", "fridge", "microwave")
NEXT_TOKENS = ("next_to", "infront", "beside")              # relational -> anchor region
TABLE_TOKENS = ("onto_table", "on_table", "to_table")       # explicit table dest


def _norm(name):
    """actor name -> comparable token: lowercase letters only (drops NNN_ prefixes,
    separators, digits). '101_milk-tea'->'milktea', '042_wooden_box'->'woodenbox'."""
    return "".join(c for c in name.lower() if c.isalpha())


# ---- helpers ------------------------------------------------------------------

def task_from_run_dir(run_dir):
    # .../<scene>/<task>/<config>  -> <task>
    return os.path.basename(os.path.dirname(os.path.abspath(run_dir)))


def _read_bbox_group(g):
    """One bbox group -> (id, center(T,N,3), quat(T,N,4), amin(T,N,3), amax(T,N,3)).
    Handles the OBB schema (obb_center/half/quat, current) and the legacy pose+AABB
    schema. Center/quat feed motion segmentation; amin/amax feed containment/tables."""
    bid = g["id"][:]
    if "obb_center" in g:
        center, half, quat = g["obb_center"][:], g["obb_half"][:], g["obb_quat"][:]
        amin, amax = _obb_to_aabb(center, half, quat)
        return bid, center, quat, amin, amax
    pose = g["pose"][:]                          # legacy (T, N, 7)
    return bid, pose[:, :, :3], pose[:, :, 3:7], g["aabb_min"][:], g["aabb_max"][:]


def load_actor_bbox(h5path):
    """Per-frame id/center/orientation/AABB for rigid actors, with articulation links
    (link_bbox) appended as extra columns when present -- so drawer/fridge/cabinet/
    microwave links resolve by the SAME motion + containment rules as rigid actors.
    Absent link_bbox (older data) => rigid-only. Accepts both the OBB and legacy schema."""
    with h5py.File(h5path, "r") as f:
        if "actor_bbox" not in f:
            raise SystemExit(f"!! {h5path} has no actor_bbox group (need data_type.actor_bbox on)")
        bid, pose, quat, amin, amax = _read_bbox_group(f["actor_bbox"])
        if "link_bbox" in f:
            lbid, lpose, lquat, lamin, lamax = _read_bbox_group(f["link_bbox"])
            T = min(bid.shape[0], lbid.shape[0])
            bid = np.concatenate([bid[:T], lbid[:T]], axis=1)
            pose = np.concatenate([pose[:T], lpose[:T]], axis=1)
            quat = np.concatenate([quat[:T], lquat[:T]], axis=1)
            amin = np.concatenate([amin[:T], lamin[:T]], axis=1)
            amax = np.concatenate([amax[:T], lamax[:T]], axis=1)
    return (bid, pose.astype(np.float64), quat.astype(np.float64),
            amin.astype(np.float64), amax.astype(np.float64))


def _decode_seg_frame(ds, t):
    import cv2
    b = ds[t]
    if ds.dtype.kind == "S":
        return cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_UNCHANGED).astype(np.uint16)
    return np.asarray(b).astype(np.uint16)


def compute_visible_ids(h5path):
    """Set of seg ids that cover >= MIN_PIXELS somewhere, unioned over all cameras at
    VIS_FRAMES sampled frames. Excludes invisible synthetic markers (des_obj / anchor
    'box' with 0 pixels) which must never become a mask -- a bin you can't see is not a
    bin. A real object occluded in one view is caught in another / another frame."""
    counts = {}
    with h5py.File(h5path, "r") as f:
        obs = f["observation"]
        cams = list(obs.keys())
        T = f["actor_bbox/id"].shape[0]
        frames = np.unique(np.linspace(0, T - 1, VIS_FRAMES).astype(int))
        for cam in cams:
            ds = obs[cam].get("actor_segmentation")
            if ds is None:
                continue
            for t in frames:
                seg = _decode_seg_frame(ds, int(t))
                ids, c = np.unique(seg, return_counts=True)
                for i, n in zip(ids.tolist(), c.tolist()):
                    counts[int(i)] = counts.get(int(i), 0) + int(n)
    return {i for i, n in counts.items() if n >= MIN_PIXELS}


def is_scene(name):
    base = name.split("/")[-1]
    return base in SCENE_NAMES or name.startswith("robot/") or name.startswith("robot")


def contiguous_windows(mask):
    """bool array -> list of [start, end) windows where mask is True."""
    wins, s = [], None
    for t, m in enumerate(mask):
        if m and s is None:
            s = t
        elif not m and s is not None:
            wins.append([s, t]); s = None
    if s is not None:
        wins.append([s, len(mask)])
    return wins


def merge_windows(wins, gap):
    out = []
    for w in wins:
        if out and w[0] - out[-1][1] < gap:
            out[-1][1] = w[1]
        else:
            out.append(list(w))
    return out


# ---- stage 1: motion segmentation --------------------------------------------

def _held_fraction(pose_win, ee_l, ee_r, s, e):
    """Fraction of a motion window in which the object is within GRIP_NEAR of a gripper.
    A manipulated target is carried (near an ee); an incidentally-knocked object is not."""
    if ee_l is None or ee_r is None:
        return 1.0
    dl = np.linalg.norm(pose_win - ee_l[s:e], axis=1)
    dr = np.linalg.norm(pose_win - ee_r[s:e], axis=1)
    near = np.minimum(dl, dr) < GRIP_NEAR
    return float(near.mean()) if len(near) else 0.0


def _qang(q1, q2):
    """angle (rad) between quaternions along the last axis; sign-invariant."""
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(d)


def segment_stages(bid, pose, quat, id_map, ee_l=None, ee_r=None):
    """Return ordered stages: [{target_id, target_name, col, start, end, settle}].
    Motion = translation OR rotation (revolute doors barely translate). A stage is a
    mover's motion window; incidental (never-held) rigid movers are dropped."""
    T, N = bid.shape
    col_id = [int(bid[0, k]) for k in range(N)]           # columns are stable per episode
    speed = np.linalg.norm(np.diff(pose, axis=0), axis=2)  # (T-1, N) translation/frame
    aspeed = _qang(quat[:-1], quat[1:])                    # (T-1, N) rotation/frame
    total = np.linalg.norm(pose - pose[0:1], axis=2).max(axis=0)   # (N,)
    atotal = _qang(quat[0:1], quat).max(axis=0)                    # (N,)
    stages = []
    for k in range(N):
        i = col_id[k]
        name = id_map.get(i, f"id{i}")
        if is_scene(name) or (total[k] < MOVE_MIN and atotal[k] < ANG_MOVE_MIN):
            continue
        moving = (speed[:, k] > SPEED_THR) | (aspeed[:, k] > ANG_THR)
        wins = merge_windows(contiguous_windows(moving), GAP_MERGE)
        for s, e in wins:
            if e - s < MIN_WIN:
                continue
            ef = min(e, T - 1)
            disp = np.linalg.norm(pose[ef, k] - pose[s, k])
            rot = float(_qang(quat[s, k], quat[ef, k]))
            if disp < MOVE_MIN and rot < ANG_MOVE_MIN:
                continue
            # gripper-proximity filter drops incidentally-knocked rigid clutter. Skip it
            # for articulation links: a link's frame origin sits far from the handle the
            # gripper holds (~30 cm), and a link that moves >MOVE_MIN was opened/closed
            # on purpose (you don't knock a drawer 18 cm by accident).
            is_link = "/" in name and not name.startswith("robot")
            if not is_link and _held_fraction(pose[s:e, k], ee_l, ee_r, s, e) < MIN_HELD_FRAC:
                continue                                     # incidental knock, not a target
            settle = min(e, T - 1)                          # end of motion = at rest
            stages.append({"target_id": i, "target_name": name, "col": k,
                           "start": int(s), "end": int(e), "settle": int(settle)})
    stages.sort(key=lambda st: st["start"])
    for idx, st in enumerate(stages):
        st["stage"] = idx
    return stages, col_id


# ---- stage 2: bin resolution --------------------------------------------------

def containers_at(pt, frame, bid, amin, amax, id_map, exclude_col, visible):
    """Rigid actors that the target's settle point sits IN or ON, best first.
    xy must be inside the padded xy-AABB. Two tiers, tier-0 preferred, smallest volume
    within a tier: tier 0 = settle z truly inside the box's z-span (deep containers:
    box/sink/file-holder the object is down inside); tier 1 = settle z just ABOVE the
    top, within ON_TOP_GAP (flat supports: pad/coaster/book rested on top). Tiering
    stops a short box from stealing an object that is really down inside a tall holder.
    Invisible markers are excluded via `visible`."""
    N = bid.shape[1]
    hits = []
    for k in range(N):
        if k == exclude_col:
            continue
        i = int(bid[frame, k])
        name = id_map.get(i, f"id{i}")
        if is_scene(name) or i not in visible:
            continue
        lo, hi = amin[frame, k], amax[frame, k]
        if not (np.all(pt[:2] >= lo[:2] - PAD) and np.all(pt[:2] <= hi[:2] + PAD)):
            continue
        if (lo[2] - PAD) <= pt[2] <= (hi[2] + PAD):
            tier = 0                                   # truly inside the vertical span
        elif hi[2] + PAD < pt[2] <= hi[2] + ON_TOP_GAP:
            tier = 1                                   # resting on top
        else:
            continue
        vol = float(np.prod(np.clip(hi - lo, 1e-4, None)))
        hits.append((tier, vol, i, name))
    hits.sort()
    return [(v, i, n) for (_t, v, i, n) in hits]


def nearest_object(pt, frame, bid, pose, id_map, exclude_col, visible):
    """closest visible, non-scene, non-target actor to pt (fallback 'next to' anchor)."""
    N = bid.shape[1]
    best = None
    for k in range(N):
        if k == exclude_col:
            continue
        i = int(bid[frame, k])
        name = id_map.get(i, f"id{i}")
        if is_scene(name) or i not in visible:
            continue
        d = float(np.linalg.norm(pose[frame, k] - pt))
        if best is None or d < best[0]:
            best = (d, i, name, k)
    return best


def anchor_from_name(task, bid, id_map, visible, frame):
    """The object the task name says to place next to ('..._next_to_mouse' -> mouse),
    matched to a VISIBLE actor that has an actor_bbox column. Returns (col, id, name)
    or None. This beats nearest-object, which can latch onto an invisible goal marker."""
    t = task.lower().replace("in_front_of", "infront").replace("_of_", "_")
    noun = None
    for kw in ("next_to", "infront", "beside"):
        if kw in t:
            noun = _norm(t.split(kw, 1)[1])
            break
    if not noun:
        return None
    cols = {int(bid[frame, k]): k for k in range(bid.shape[1])}
    cands = []
    for i, k in cols.items():
        if i not in visible or is_scene(id_map.get(i, "")):
            continue
        nm = _norm(id_map.get(i, ""))
        if nm and (noun in nm or nm in noun):
            cands.append((k, i, id_map.get(i, "")))
    return cands[0] if cands else None


def table_surface(bid, amax, id_map, frame):
    """(table_id, top_z) from the actor named 'table'."""
    N = bid.shape[1]
    for k in range(N):
        i = int(bid[frame, k])
        if id_map.get(i, "").split("/")[-1] == "table":
            return i, float(amax[frame, k, 2])
    return None, None


def resolve_bin(st, task, bid, pose, amin, amax, id_map, visible, has_articulation):
    """Return a bin dict for one stage."""
    name_l = task.lower()
    settle = st["settle"]
    pt = pose[settle, st["col"]]

    # open/close: the mover is an ARTICULATION LINK (door/drawer) -> the link itself
    # is the target, no bin (the goal is a joint-state change, not a placement).
    tname = st.get("target_name", "")
    if "/" in tname and not tname.startswith("robot"):
        return {"bin_type": "none", "reason": "open/close: target is the articulation link (door/drawer)"}

    if name_l.startswith("pick"):
        return {"bin_type": "none", "reason": "pick-only"}

    # relational placement -> region centred on the named anchor
    if any(tok in name_l.replace("in_front_of", "infront") for tok in NEXT_TOKENS):
        anch = anchor_from_name(task, bid, id_map, visible, settle)
        if anch is None:
            # anchor named but not a rigid actor (e.g. 'in front of microwave') -> Phase 2
            if any(tok in name_l for tok in FIXTURE_TOKENS):
                return {"bin_type": "deferred_articulation",
                        "reason": "relational anchor is an articulated fixture (Phase 2)"}
            near = nearest_object(pt, settle, bid, pose, id_map, st["col"], visible)
            if near and near[0] <= ANCHOR_MAX:
                anch = (near[3], near[1], near[2])
        if anch is not None:
            ak, aid, aname = anch
            ac = pose[settle, ak, :2]
            half = float(np.max((amax[settle, ak, :2] - amin[settle, ak, :2]) / 2))
            reach = float(np.linalg.norm(ac - pt[:2]))          # region must reach the landing
            radius = round(max(half + REGION_MARGIN, reach + 0.03), 4)
            tz = table_surface(bid, amax, id_map, settle)[1]
            return {"bin_type": "region_anchor", "anchor_id": aid, "anchor_name": aname,
                    "center": [float(ac[0]), float(ac[1])], "radius": radius, "surface_z": tz,
                    "reason": f"next-to {aname} (landing {reach*100:.1f}cm off centre)"}
        tid, tz = table_surface(bid, amax, id_map, settle)
        return {"bin_type": "region_table", "table_id": tid, "surface_z": tz,
                "obj_pad": TABLE_OBJ_PAD, "reason": "relational, no anchor -> table"}

    if any(tok in name_l for tok in TABLE_TOKENS):
        tid, tz = table_surface(bid, amax, id_map, settle)
        return {"bin_type": "region_table", "table_id": tid, "surface_z": tz,
                "obj_pad": TABLE_OBJ_PAD, "reason": "explicit table destination"}

    # geometry-first: smallest rigid container the target sits in/on
    hits = containers_at(pt, settle, bid, amin, amax, id_map, st["col"], visible)
    if hits:
        vol, i, nm = hits[0]
        return {"bin_type": "object", "bin_id": i, "bin_name": nm,
                "reason": f"smallest container ({vol*1e3:.2f}L)"}

    # no rigid container. articulated fixture? -> Phase 2
    tid, tz = table_surface(bid, amax, id_map, settle)
    if any(tok in name_l for tok in FIXTURE_TOKENS):
        return {"bin_type": "deferred_articulation",
                "reason": f"fixture {[t for t in FIXTURE_TOKENS if t in name_l]} has no AABB (Phase 2)"}
    if has_articulation and tz is not None and pt[2] > tz + ELEVATED:
        return {"bin_type": "deferred_articulation",
                "reason": f"settle {(pt[2]-tz)*100:.0f}cm above table + fixtures present "
                          "-> likely in an articulated fixture (Phase 2)"}

    # placed in open space on the table
    return {"bin_type": "region_table", "table_id": tid, "surface_z": tz,
            "obj_pad": TABLE_OBJ_PAD, "reason": "open-space placement -> table free-space"}


# ---- top level ----------------------------------------------------------------

def role_target_ids(roles, id_map):
    """IDs the task SANCTIONS as manipulable, from scene_info's role_names (target /
    destination / intended-contact objects). Used to reject rigid clutter that merely
    got physically nudged -- e.g. a plant shoved by an opening drawer -- from being
    mislabeled a target. Returns None when the episode carries no usable role info, so
    the caller falls back to motion-only detection and nothing is filtered."""
    if not roles:
        return None
    names = set()
    for key in ("target_object_names", "destination_object_names", "intended_contact_names"):
        for n in (roles.get(key) or []):
            tok = _norm(n)
            if tok:
                names.add(tok)
    if not names:
        return None
    allowed = set()
    for key in ("target_id", "destination_id"):
        v = roles.get(key)
        if isinstance(v, (int, np.integer)):
            allowed.add(int(v))
    for i, nm in id_map.items():
        if _norm(nm.split("/")[-1]) in names or _norm(nm) in names:
            allowed.add(int(i))
    return allowed


def resolve_episode(run_dir, ep, task=None, verbose=True):
    task = task or task_from_run_dir(run_dir)
    h5 = os.path.join(run_dir, "data", f"episode{ep}.hdf5")
    ep_info, id_map, roles, robot_ids = load_sidecar(run_dir, ep)
    bid, pose, quat, amin, amax = load_actor_bbox(h5)
    T = bid.shape[0]

    visible = compute_visible_ids(h5)
    with h5py.File(h5, "r") as f:
        ee_l = f["endpose/left_endpose"][:, :3] if "endpose/left_endpose" in f else None
        ee_r = f["endpose/right_endpose"][:, :3] if "endpose/right_endpose" in f else None
    stages, col_id = segment_stages(bid, pose, quat, id_map, ee_l, ee_r)
    # role-aware target filter: keep articulation links (drawers/doors) and any object the
    # task sanctions in role_names; drop rigid clutter that only got nudged (e.g. a plant
    # shoved by an opening drawer) so it can never be mislabeled a target. Purely
    # subtractive -- bin resolution below is untouched -- and a no-op when roles are absent.
    allowed = role_target_ids(roles, id_map)
    if allowed is not None:
        kept = [st for st in stages
                if ("/" in st["target_name"] and not st["target_name"].startswith("robot"))
                or int(st["target_id"]) in allowed]
        if kept:                                    # never filter down to nothing
            stages = kept
            for idx, st in enumerate(stages):       # renumber after filtering
                st["stage"] = idx
    art_links = sorted({n for i, n in id_map.items()
                        if "/" in n and not n.startswith("robot") and "__jointframe__" not in n})
    has_articulation = len(art_links) > 0

    for st in stages:
        st["bin"] = resolve_bin(st, task, bid, pose, amin, amax, id_map, visible, has_articulation)

    # per-frame -> stage index (stage i owns [start_i, start_{i+1}))
    frame_stage = [-1] * T
    for idx, st in enumerate(stages):
        end = stages[idx + 1]["start"] if idx + 1 < len(stages) else T
        for t in range(st["start"], end):
            frame_stage[t] = idx
        for t in range(0, st["start"]):            # pre-roll shows the upcoming stage
            if frame_stage[t] == -1:
                frame_stage[t] = idx

    task_type = ("articulation_only" if not stages
                 else "chain" if len(stages) > 1 else "single")
    out = {
        "task": task, "task_type": task_type, "num_frames": int(T),
        "role_names_raw": roles, "articulation_links": art_links,
        "stages": [{k: v for k, v in st.items() if k != "col"} for st in stages],
        "frame_stage": frame_stage,
    }
    if verbose:
        _print_summary(out)
    return out


def _print_summary(out):
    print(f"\n=== {out['task']}  [{out['task_type']}, {out['num_frames']} frames] ===")
    if out["task_type"] == "articulation_only":
        print(f"  (no rigid mover; articulation-only) links: {out['articulation_links']}")
    for st in out["stages"]:
        b = st["bin"]
        bt = b["bin_type"]
        if bt == "object":
            binstr = f"bin=OBJECT {b['bin_id']}:{b['bin_name']}"
        elif bt == "region_anchor":
            binstr = f"bin=REGION~{b['anchor_name']} r={b['radius']}m"
        elif bt == "region_table":
            binstr = "bin=REGION table-free-space"
        elif bt == "deferred_articulation":
            binstr = "bin=DEFERRED(articulation, Phase 2)"
        else:
            binstr = "bin=none"
        print(f"  stage {st['stage']}: target {st['target_id']}:{st['target_name']:22s} "
              f"f[{st['start']}..{st['end']}] -> {binstr}  ({b['reason']})")


# ------------------------------------------------ per-frame roles (for export) --

def table_ids_of(id_map):
    return {i for i, n in id_map.items() if n.split("/")[-1] == "table"}


def stage_at(masking, k):
    """The stage that owns frame k (or None)."""
    fs = masking.get("frame_stage", [])
    if not (0 <= k < len(fs)):
        return None
    si = fs[k]
    if si is None or not (0 <= si < len(masking["stages"])):
        return None
    return masking["stages"][si]


def frame_role_sets(masking, k, id_map, visible, robot_ids):
    """Per-frame {target, destination(bin object), obstacle} id sets. The bin is an
    id only when it's an object/link; region bins (anchor/table) carry no id (rendered
    separately). Obstacle = every other visible non-scene object."""
    st = stage_at(masking, k)
    target, dest = set(), set()
    if st is not None:
        target = {int(st["target_id"])}
        b = st["bin"]
        if b["bin_type"] == "object":
            dest = {int(b["bin_id"])}
    obst = {i for i in visible
            if i in id_map and not is_scene(id_map[i])} - target - dest - set(robot_ids)
    return {"target": target, "destination": dest, "obstacle": obst}


# ------------------------------------------------------------- panel rendering --

def backproject_full(depth_mm, K, E):
    """(H,W) mm depth -> (H,W,3) world xyz + (H,W) valid mask (CV world->cam E)."""
    H, W = depth_mm.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth_mm.astype(np.float32) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pcam = np.stack([x, y, z], -1).reshape(-1, 3)
    R, t = E[:, :3], E[:, 3]
    return ((pcam - t) @ R).reshape(H, W, 3), (z > 0)


def table_obstacle_rects(actor_bbox, frame, id_map, robot_ids, target_id, pad, surface_z):
    """Padded xy footprints [(x0,y0,x1,y1), ...] of every rigid object resting on the
    table (excludes the scene shell, the robot, and the target being placed). Used to
    carve clearance out of a table free-space bin so the destination never hugs an
    existing object. Same computation a dataloader would run from actor_bbox."""
    rects = []
    if actor_bbox is None:
        return rects
    ids = actor_bbox["id"][frame]
    mns, mxs = actor_bbox["aabb_min"][frame], actor_bbox["aabb_max"][frame]
    rob = set(int(r) for r in (robot_ids or []))
    for j in range(len(ids)):
        i = int(ids[j])
        if i == target_id or i in rob:
            continue
        n = id_map.get(i, "") if id_map else ""
        if not n or is_scene(n):
            continue
        mn, mx = np.asarray(mns[j], float), np.asarray(mxs[j], float)
        if surface_z is not None and (mx[2] < surface_z - 0.03 or mn[2] > surface_z + 0.5):
            continue  # not resting on this table surface (below it, or floating high)
        rects.append((float(mn[0] - pad), float(mn[1] - pad),
                      float(mx[0] + pad), float(mx[1] + pad)))
    return rects


def region_bin_mask(binspec, seg, table_ids, world, valid, obstacle_rects=None):
    """Boolean (H,W) mask for a REGION bin (anchor disk or whole-table free-space).
    obstacle_rects (world-xy padded footprints) carve clearance out of a table bin."""
    bt = binspec["bin_type"]
    if bt == "region_table":
        m = np.isin(seg, list(table_ids))
        if valid is not None:
            m = m & valid
        if obstacle_rects and world is not None:
            wx, wy = world[:, :, 0], world[:, :, 1]
            excl = np.zeros(seg.shape, bool)
            for x0, y0, x1, y1 in obstacle_rects:
                excl |= (wx >= x0) & (wx <= x1) & (wy >= y0) & (wy <= y1)
            m = m & ~excl
        return m
    if bt == "region_anchor" and world is not None:
        on_table = np.isin(seg, list(table_ids)) & valid
        cx, cy = binspec["center"]
        d = np.linalg.norm(world[:, :, :2] - np.array([cx, cy]), axis=2)
        return on_table & (d <= binspec["radius"])
    return np.zeros(seg.shape, bool)


def masking_overlay(rgb, seg, target_mask, binmask, robot_ids):
    out = (rgb.astype(np.float32) * 0.5).astype(np.uint8)
    if robot_ids:
        m = np.isin(seg, list(robot_ids))
        out[m] = (0.6 * out[m] + 0.4 * ROBOT_COLOR).astype(np.uint8)
    if binmask is not None and binmask.any():
        out[binmask] = (0.35 * out[binmask] + 0.65 * BIN_COL).astype(np.uint8)
    if target_mask is not None and target_mask.any():
        out[target_mask] = (0.25 * out[target_mask] + 0.75 * TARGET_COL).astype(np.uint8)
    return out


def _label_bar(w, text, h=22):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text[:118], (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
    return bar


def bin_caption(b):
    bt = b["bin_type"]
    return {"object": f"bin={b.get('bin_name')}",
            "region_anchor": f"bin=REGION around {b.get('anchor_name')} (r={b.get('radius')}m)",
            "region_table": "bin=REGION table free-space",
            "deferred_articulation": "bin=DEFERRED (articulation)",
            "none": ("bin=none (open/close)" if "open/close" in b.get("reason", "")
                     else "bin=none (pick-only)")}.get(bt, bt)


def render_masking_panel(run_dir, ep, cam="countertop_camera", out_path=None, masking=None):
    """Stage-aware panel: one row per stage (grasp + settle frame) with target=red,
    bin=green (object mask / region). Written to <run_dir>/masking_viz/ by default."""
    if masking is None:
        masking = resolve_episode(run_dir, ep, verbose=False)
    ep_info, id_map, roles, robot_ids = load_sidecar(run_dir, ep)
    table_ids = table_ids_of(id_map)
    h5 = os.path.join(run_dir, "data", f"episode{ep}.hdf5")
    with h5py.File(h5, "r") as f:
        g = f[f"observation/{cam}"]
        rgb = np.stack(decode_frames(g["rgb"], cv2.IMREAD_COLOR))[..., ::-1]
        seg = load_seg(g["actor_segmentation"])
        depth = np.stack(decode_frames(g["depth"], cv2.IMREAD_UNCHANGED)).astype(np.float32)
        K = np.asarray(g["intrinsic_cv"]); E = np.asarray(g["extrinsic_cv"])
        # prefer the BAKED grounding masks (what the dataloader gets); fall back to
        # re-deriving them for pre-bake data.
        gmask = (np.stack(decode_frames(g["grounding_mask"], cv2.IMREAD_UNCHANGED))
                 if "grounding_mask" in g else None)
        abox = None
        if "actor_bbox" in f:                    # footprints for table padding (both schemas)
            bid, _, _, amin, amax = _read_bbox_group(f["actor_bbox"])
            abox = {"id": bid, "aabb_min": amin, "aabb_max": amax}
    src = "HDF5 grounding_mask" if gmask is not None else "re-derived"
    stages, rows, W = masking["stages"], [], rgb.shape[2]
    if not stages:
        art = {i for i, n in id_map.items()
               if "/" in n and not n.startswith("robot") and "__jointframe__" not in n}
        fr = len(rgb) // 2
        ov = (rgb[fr].astype(np.float32) * 0.5).astype(np.uint8)
        m = np.isin(seg[fr], list(art))
        ov[m] = (0.4 * ov[m] + 0.6 * ART_COL).astype(np.uint8)
        rows.append(_label_bar(W * 2, f"{masking['task']}: ARTICULATION-ONLY (magenta=fixture)"))
        rows.append(np.concatenate([rgb[fr], ov], 1))
    for st in stages:
        b = st["bin"]
        for tag, fr in [("grasp", st["start"]), ("settle", st["settle"])]:
            fr = int(np.clip(fr, 0, len(rgb) - 1))
            if gmask is not None:                            # read the BAKED masks
                tmask, bm = gmask[fr] == 1, gmask[fr] == 2
            else:                                            # re-derive (pre-bake data)
                tmask = seg[fr] == st["target_id"]
                if b["bin_type"] in ("region_anchor", "region_table"):
                    Kf = K[fr] if K.ndim == 3 else K
                    Ef = E[fr] if E.ndim == 3 else E
                    world, valid = backproject_full(depth[fr], Kf, Ef)
                    rects = (table_obstacle_rects(abox, fr, id_map, robot_ids, int(st["target_id"]),
                                                  b.get("obj_pad", TABLE_OBJ_PAD), b.get("surface_z"))
                             if b["bin_type"] == "region_table" else None)
                    bm = region_bin_mask(b, seg[fr], table_ids, world, valid, rects)
                elif b["bin_type"] == "object":
                    bm = seg[fr] == b["bin_id"]
                else:
                    bm = None
            ov = masking_overlay(rgb[fr], seg[fr], tmask, bm, robot_ids)
            rows.append(_label_bar(W * 2,
                f"stage {st['stage']} [{tag} f{fr}]: target={st['target_name']}  {bin_caption(b)}  ({src})"))
            rows.append(np.concatenate([rgb[fr], ov], 1))
    panel = np.concatenate(rows, 0)
    if out_path is None:
        d = os.path.join(run_dir, "masking_viz")
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, f"episode{ep}_{cam}.png")
    cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return out_path


# =============================================================================
# COLLECTION-TIME GROUNDING (bake into the HDF5)
# finalize_grounding() runs once per kept episode: it writes the grounding masks
# (target/bin, PNG-encoded like segmentation) into the HDF5 and a masking/episode
# {ep}.json stage-timeline sidecar. The dataloader then reads the masks + the OBB
# fields straight from the HDF5 -- it does NOT import this module.
# =============================================================================

def write_masking_json(run_dir, ep, verbose=False):
    """Resolve one episode and write <run_dir>/masking/episode{ep}.json. Called at
    collect time so masking is a first-class output beside the HDF5 / scene_info."""
    out = resolve_episode(run_dir, ep, verbose=verbose)
    d = os.path.join(run_dir, "masking")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"episode{ep}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    return p


def load_masking(run_dir, ep):
    """masking for one episode: read masking/episode{ep}.json if it exists, else
    resolve on the fly from the HDF5 (works whether or not collection wrote it)."""
    p = os.path.join(run_dir, "masking", f"episode{ep}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return resolve_episode(run_dir, ep, verbose=False)


def _grounding_label_map(seg, st, table_ids, id_map, robot_ids, abox, frame, world_valid):
    """(H,W) uint8 map: 0=background, 1=target, 2=bin, for one frame's stage. Target
    wins over bin where they'd overlap. This is what gets baked into the HDF5."""
    lab = np.zeros(seg.shape, np.uint8)
    if st is None:
        return lab
    b = st["bin"]
    bt = b["bin_type"]
    if bt == "object":
        lab[seg == int(b["bin_id"])] = 2
    elif bt.startswith("region"):
        world, valid = world_valid
        if world is not None:
            rects = (table_obstacle_rects(abox, frame, id_map, robot_ids, int(st["target_id"]),
                                          b.get("obj_pad", TABLE_OBJ_PAD), b.get("surface_z"))
                     if bt == "region_table" else None)
            lab[region_bin_mask(b, seg, table_ids, world, valid, rects)] = 2
    lab[seg == int(st["target_id"])] = 1
    return lab


def bake_grounding_masks(run_dir, ep, cams=None, masking=None):
    """Write a per-frame grounding label map (0=none,1=target,2=bin, PNG-encoded like the
    segmentation) into the HDF5 under observation/<cam>/grounding_mask, for every camera.
    The padding/region math is applied here ONCE so a dataloader reads finished masks and
    never calls back into this module. Idempotent (overwrites)."""
    if masking is None:
        masking = load_masking(run_dir, ep)
    _, id_map, _, robot_ids = load_sidecar(run_dir, ep)
    table_ids = table_ids_of(id_map)
    need_region = any(s["bin"]["bin_type"].startswith("region") for s in masking["stages"])
    with h5py.File(os.path.join(run_dir, "data", f"episode{ep}.hdf5"), "a") as f:
        abox = None
        if "actor_bbox" in f:
            bid, _, _, amin, amax = _read_bbox_group(f["actor_bbox"])
            abox = {"id": bid, "aabb_min": amin, "aabb_max": amax}
        obs = f["observation"]
        for cam in (cams or list(obs.keys())):
            g = obs.get(cam)
            if g is None or "actor_segmentation" not in g:
                continue
            seg = load_seg(g["actor_segmentation"])           # (T,H,W)
            depth = K = E = None
            if need_region and "depth" in g:
                depth = np.stack(decode_frames(g["depth"], cv2.IMREAD_UNCHANGED)).astype(np.float32)
                K, E = np.asarray(g["intrinsic_cv"]), np.asarray(g["extrinsic_cv"])
            enc = []
            for t in range(seg.shape[0]):
                st = stage_at(masking, t)
                wv = (None, None)
                if st is not None and st["bin"]["bin_type"].startswith("region") and depth is not None:
                    Kf = K[t] if K.ndim == 3 else K
                    Ef = E[t] if E.ndim == 3 else E
                    wv = backproject_full(depth[t], Kf, Ef)
                lab = _grounding_label_map(seg[t], st, table_ids, id_map, robot_ids, abox, t, wv)
                enc.append(cv2.imencode(".png", lab)[1].tobytes())
            maxlen = max(len(b) for b in enc)
            if "grounding_mask" in g:
                del g["grounding_mask"]
            g.create_dataset("grounding_mask", data=np.array(enc, dtype=f"S{maxlen}"))
    return masking


def finalize_grounding(run_dir, ep, cams=None, obj_pad=None):
    """Collection hook (called per kept episode): resolve the stages, write the metadata
    sidecar masking/episode{ep}.json, and bake the grounding masks into the HDF5 -- so the
    episode ships self-contained (HDF5 has masks + oriented boxes, no functions needed)."""
    global TABLE_OBJ_PAD
    if obj_pad is not None:
        TABLE_OBJ_PAD = float(obj_pad)
    masking = resolve_episode(run_dir, ep, verbose=False)
    d = os.path.join(run_dir, "masking")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"episode{ep}.json"), "w") as fp:
        json.dump(masking, fp, indent=2)
    bake_grounding_masks(run_dir, ep, cams=cams, masking=masking)
    return masking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("episode", type=int, nargs="?", default=0)
    ap.add_argument("--task", default=None)
    ap.add_argument("--write", action="store_true",
                    help="write masking/episode{ep}.json (the collection-time output)")
    ap.add_argument("--panel", action="store_true", help="also render the masking panel")
    ap.add_argument("--cam", default="countertop_camera")
    args = ap.parse_args()
    out = resolve_episode(args.run_dir, args.episode, task=args.task)
    if args.write:
        print("\nwrote", write_masking_json(args.run_dir, args.episode))
    if args.panel:
        print("wrote", render_masking_panel(args.run_dir, args.episode, args.cam, masking=out))


if __name__ == "__main__":
    main()
