"""
Deterministic single-occluder spawn and target visibility distribution.

Places the target (001_bottle id 9) in the upper third of the reachable region (in
front of the back furniture) and spawns one tall milk-box occluder (038_milk-box id 2,
scale 1.0) in a narrow band just in front (-y, robot/camera side) of the target, with no
clutter and no binary search. Then measures the t=0 countertop `visible_fraction`
distribution across seeds, reusing the measurement and plotting code of
analyze_natural_visibility.py.

visible_fraction = visible_target_px(with occluder) / full_target_px(no occluder),
same seed (target pose is fixed before the occluder is added).

USAGE (from the repo root):
    source set_env.sh
    cd sim
    python script/bench_script/analyze_occluder_visibility.py \
        --seed-start 0 --num-seeds 50 --occluding-object-distance 10,10 --bins 20 \
        --out-dir ../scripts/validation/results/phase2_occluder

    # narrow-region range (each seed draws a random offset within it) to see how
    # distance affects occlusion:
    python script/bench_script/analyze_occluder_visibility.py \
        --num-seeds 40 --occluding-object-distance 7,13 \
        --out-dir ../scripts/validation/results/phase2_occluder

    # re-plot only:
    python script/bench_script/analyze_occluder_visibility.py --plot-only \
        --out-dir ../scripts/validation/results/phase2_occluder
"""
import os
import json
import argparse
import collections
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien.core as sapien
import torch
import transforms3d as t3d
from curobo.types.state import JointState

from setup_paths import setup_paths
setup_paths()
sim_root = Path(os.environ["SIM_ROOT"])
os.chdir(sim_root)

from envs.utils import rand_pose, create_actor, create_box, ArmTag, cal_quat_dis
from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from visualize_task_scene import get_env_class
# measurement, rollout and plotting helpers shared with the natural-visibility sweep
from analyze_natural_visibility import (build_cfg, DR_CLEAN, save_overlay, analyze, _resolve_target,
                                        CAMERA, run_rollout, analyze_rollout, effective_out_dir)

# Target object: 001_bottle id 9 (8-way grasp ring, so the arm can side-reach around the
# occluder). Set TARGET_MODEL/TARGET_ID to 047_mouse id 0 for the stock top-down mouse task.
TARGET_MODEL = "001_bottle"
TARGET_ID = 9
# Target spawn region: back half of the table (back furniture removed). y <= 0.20 keeps it
# 0.15 m off the back edge (table y in [-0.35, 0.35]); |x| <= 0.15 keeps it 0.45 m off each
# side edge (table x in [-0.6, 0.6]), leaving room for the side waypoint around the box and
# keeping target and waypoint inside the arm's reach envelope.
TARGET_XLIM = (-0.15, 0.15)
TARGET_YLIM = (0.0, 0.20)
PAD_XY = (0.0, -0.28)          # destination pad parked at the front, out of the occluder zone
OCCLUDER_QPOS = [0.66, 0.66, -0.25, -0.25]   # same upright carton orientation as the stock milk-box

# Two-step (around-the-box) planning: CuRobo does not reliably route around a tall obstacle
# between start and goal, so before the grasp (and before the place) the gripper first moves
# to a waypoint beside the occluder on the arm's own side, then reaches in. The waypoint's
# x-offset from the box centre is OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP, i.e.
# SIDE_WAYPOINT_GAP is the clearance from the box edge to the gripper. A waypoint outside
# the reachable x-range is clamped to REACH_X_LIMIT on the same side (see _box_side_x).
OCC_HALF_FOOTPRINT = 0.08      # milk-box base half-diagonal (~0.11 x 0.122, incl. yaw)
SIDE_WAYPOINT_GAP = 0.24       # clearance from the box edge to the waypoint (gripper)
REACH_X_LIMIT = 0.5
GRASP_CANDIDATE_LIMIT = 4
# Post-grasp lift height (m). With a 0.1 m lift, the pre_beside_box move often started in
# collision with the re-enabled table (INVALID_START_STATE_WORLD_COLLISION) because the held
# object's voxelized collision volume (attach_object) had not cleared it. 0.15 gave the best
# success rate in validation sweeps; 0.2 was worse (a larger lift moves the arm into a
# different posture rather than adding clearance). Must match between dry run and execution.
GRASP_LIFT_HEIGHT = 0.15
# Minimum fraction of GRASP_LIFT_HEIGHT the target object itself (not just the gripper) must
# rise for the grasp to count. A missed grasp still reports plan_success=True for the lift
# while the object's z stays flat (observed rise 0-0.6% of the commanded lift), so 0.5
# leaves a wide margin.
GRASP_VERIFY_MIN_RISE_FRACTION = 0.5
# Max drift (m) of the object's position in the gripper's local frame, relative to the
# baseline captured right after attach_object, before a stage is treated as a lost grasp.
# plan_success only reflects the arm's motion plan, so a dropped object (10-60 cm drift
# observed) is otherwise not detected. 0.03 tolerates simulation jitter.
OBJECT_RETENTION_TOLERANCE = 0.03
# Max rotation drift (rad, ~11 deg) of the object's orientation in the gripper's local
# frame. Catches an object rotating loose without translating past
# OBJECT_RETENTION_TOLERANCE, which would leave CuRobo's attached collision model (a fixed
# rigid transform) inconsistent with the physical object.
OBJECT_RETENTION_ROTATION_TOLERANCE = 0.2
# How close (m) the held object must be to the intended final pose (self.des_obj_pose) for
# place_actor's descent to count as done (skip remaining moves, open the gripper) rather
# than a loss or an ambiguous mid-descent contact (see DESCENT_CONTACT_DRIFT_TOLERANCE).
# Mirrors put_mouse_on_pad.check_success() (independent x/y error each < 0.02) with a small
# margin; a looser combined radius would accept poses that check_success() still rejects.
OBJECT_PLACEMENT_XY_TOLERANCE = 0.018
OBJECT_PLACEMENT_Z_TOLERANCE = 0.03
# Contact handling uses the task's exact per-axis success threshold. Once the object has
# settled, a gripper-relative retention check is not meaningful, so release only when the
# task's own XY predicate passes.
CONTACT_RELEASE_XY_TOLERANCE = 0.02
# Gripper-relative drift (m) beyond OBJECT_RETENTION_TOLERANCE that place_actor's descent
# still treats as possible early contact rather than a loss: the object settling onto the
# pad/table slightly early changes the gripper-relative transform like a drop would.
# Observed early-contact cases drifted ~6-7 cm, genuine drops 30 cm-1.4 m. Below this the
# descent continues toward the target; above it the grasp counts as lost.
DESCENT_CONTACT_DRIFT_TOLERANCE = 0.15
# How close (m) the held object's world z must be to the resting height (des_obj_pose[2])
# for a drift within DESCENT_CONTACT_DRIFT_TOLERANCE to be attributed to surface contact.
# A real drop passes through the same drift band while still well above the surface, so
# drift magnitude alone cannot tell the two apart. Looser than OBJECT_PLACEMENT_Z_TOLERANCE
# since it only needs to confirm plausible contact, not the final resting pose.
DESCENT_CONTACT_HEIGHT_TOLERANCE = 0.06
# tstep_fraction schedule for create_grasp_approach_metric on place_actor's descent slices.
# tstep_fraction only enforces the straight-line/orientation-locked cost over the last
# (1 - tstep_fraction) of the timesteps; the rest is a soft-cost-only portion. CuRobo's
# default (0.8) let a 3 cm slice swing the gripper ~56 cm off the line before snapping back,
# while a fixed tight 0.2 removed the excursion but made some over-constrained slices
# infeasible. Attempts therefore start tight and relax toward the default across the retry
# budget; the path-safety filters below (DESCENT_MAX_PATH_DEVIATION etc.) still reject a
# circuitous solution at every fraction.
DESCENT_APPROACH_TSTEP_FRACTIONS = [0.2, 0.4, 0.6, 0.8]
# Max perpendicular distance (m) of a descent slice's planned EE path from the finite
# straight segment between its start and end pose. Well above the sub-2 mm wobble of normal
# slices, well below the ~56 cm detour above, and larger than DESCENT_SLICE_SIZE to allow
# small legitimate detours. Measured against the clamped segment: an unclamped line
# projection would miss a path that overshoots past the target along the same axis.
DESCENT_MAX_PATH_DEVIATION = 0.08
# Max ratio of a descent slice's Cartesian path length (sum of EE waypoint distances) to its
# straight-line distance; catches back-and-forth paths that stay close to the line.
# Conservative and not yet tuned on a large sample.
DESCENT_MAX_PATH_LENGTH_RATIO = 3.0
# Max ratio of a slice's joint-space path length to its direct start-to-end joint distance;
# catches joint-space winding that an EE-only check misses (a redundant arm can reach nearly
# the same EE pose through looping joint paths). Provisional, like the ratio above.
DESCENT_MAX_JOINT_TRAVEL_RATIO = 3.0
# Max range (rad, max - min) of any single joint over a slice; flags one joint spinning
# through an implausible range while the L2-aggregate travel ratio still looks fine.
# Provisional.
DESCENT_MAX_JOINT_RANGE = 1.0
# Max joint-space distance (rad, L2 over active joints) between a slice's start and end;
# catches a monotonic jump to a distant IK branch, which has a travel ratio near 1.0 and can
# keep every joint under DESCENT_MAX_JOINT_RANGE (six joints at ~0.9 rad is ~2.2 rad L2).
# Provisional.
DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT = 0.5
# Absolute cap (rad) on a slice's joint-space path length, the absolute companion to
# DESCENT_MAX_JOINT_TRAVEL_RATIO. Slightly above the endpoint cap since path length is never
# shorter than the endpoint distance. Provisional.
DESCENT_MAX_JOINT_PATH_LENGTH = 0.6
# Local landing search (place_actor's final approach). Commanding CuRobo to one exact final
# pose often fails with FINETUNE_TRAJOPT_FAIL at the surface, where the attached object's
# collision margin against the table is tightest. Instead, search a small grid of nearby
# landing poses, accept one that plans cleanly and passes the trajectory-safety filters, and
# let physics settle the final contact.
#
# Per-axis XY offsets (m) around the nominal target, combined into a grid. The max offset
# (1.5 cm) stays under OBJECT_PLACEMENT_XY_TOLERANCE (1.8 cm) and check_success's 2 cm.
LANDING_XY_OFFSETS = [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015]
# Candidate release heights (m above the resting height), lowest first: a lower release
# leaves less for physics to settle, a higher one gives CuRobo more collision margin.
LANDING_RELEASE_HEIGHTS = [0.01, 0.03, 0.05, 0.08]
# Max IK-feasible candidates (closest to the nominal target first) to fully plan per
# release height; IK checks are cheap, CuRobo plans are not.
LANDING_MAX_CANDIDATES_PER_HEIGHT = 12
# Stop planning further candidates at a height once this many have been accepted (plan
# success and path-safety filter).
LANDING_MIN_ACCEPTED_TO_STOP = 3
# Max XY distance (m) from the target for _local_landing_search_and_place to run (it also
# requires the object to be retained). Its candidate moves are short by design, so from
# farther away its filters would reject a legitimate longer approach. XY and Z are gated
# separately: XY bounds which of the +-1.5 cm offsets can land within check_success's
# tolerance, while the remaining vertical gap is bounded by LANDING_SEARCH_MAX_Z_DISTANCE.
LANDING_SEARCH_TRIGGER_DISTANCE = float(os.environ.get("LANDING_SEARCH_TRIGGER_DISTANCE", "0.10"))
# Max vertical distance (m) from the target for the same gate. Looser than the XY gate
# because the search itself picks a release height (LANDING_RELEASE_HEIGHTS, up to 8 cm).
LANDING_SEARCH_MAX_Z_DISTANCE = float(os.environ.get("LANDING_SEARCH_MAX_Z_DISTANCE", "0.15"))
# Max number of different contact points play_once tries when grasp_verify detects a
# missed grasp, reusing the ranking of _select_pick_place_candidate (via exclude_cp_ids).
GRASP_VERIFY_MAX_CANDIDATES = 3
# Replay attached-object trajectories more slowly without changing their
# planned geometric path. A factor of 2 inserts one interpolated control point
# between each pair and scales commanded velocity accordingly.
ATTACHED_TRAJECTORY_SLOWDOWN = max(
    1, int(os.environ.get("ATTACHED_TRAJECTORY_SLOWDOWN", "2")))
# Slice length (m) for place_actor's deterministic descent (see
# _plan_pose_with_descent_slices). place_actor's moves are short, mostly straight final
# approaches with a fixed target orientation (held via approach_axis), so fixed-size slices
# along the straight line suit them better than an error-informed shrink. 0.04 m is within
# the 3-5 cm range used for these slices.
DESCENT_SLICE_SIZE = 0.04
SIDE_WAYPOINT_GAPS = (0.20, 0.24, 0.28)
# Fallback gaps, tried only when none of the SIDE_WAYPOINT_GAPS beside_box targets passes
# check_ik_batch. For boxes far from the shoulder in x, an IK sweep at fixed y/z showed a
# reachability cutoff between gap 0.16 and 0.18, so all of SIDE_WAYPOINT_GAPS (0.20-0.28)
# can be unreachable while smaller gaps work. The preferred gaps leave more clearance, so
# these are only searched as a fallback.
SIDE_WAYPOINT_GAPS_FALLBACK = (0.16, 0.12, 0.08)
# Direct-plan retries per combo in the post-attach placement search. Right after lift and
# enable_table the live qpos sits close to the table, and whether CuRobo flags it as
# INVALID_START_STATE_WORLD_COLLISION depends on its internal trajopt seeding: the same
# post_grasp_escape/pre_beside_box move can fail every time in one run and succeed in a
# replay. A few retries (one extra trajopt call each) cost much less than bridge waypoints.
PLACEMENT_SEARCH_RETRIES = 3
# Waypoint z lifts above grasp height. In an IK-reachability sweep the waypoint was feasible
# about twice as often at +0.15 m (64%) as at +0.0/+0.06 m (29-36%); x/gap and higher lifts
# (+0.30 m, 54%) mattered much less.
SIDE_WAYPOINT_Z_LIFTS = (0.0, 0.06, 0.15)
PLACE_CLEARANCE_ZS = (1.05, 1.15, 1.25)
# Fallback clearance heights, tried when none of PLACE_CLEARANCE_ZS is reachable for
# lift_above_box (same idea as SIDE_WAYPOINT_GAPS_FALLBACK). A check_ik_batch height sweep
# (relax_orientation=True) at the beside_box x/y put the reachable window around
# z ~0.8-1.05: 1.15 and 1.25 are unreachable in any orientation, and 1.05 sometimes needs a
# relaxed orientation.
PLACE_CLEARANCE_ZS_FALLBACK = (0.95, 0.90, 0.85)
# Stages that plan with a fixed (non-relaxed) orientation even when the rest of the
# placement chain uses relax_orientation=True. center_over_pad is the last subgoal before
# place_actor's strict final descent, so it must hand off a known orientation. See
# _plan_pose_trajectory_sequence.
PLACEMENT_STRICT_ORIENTATION_STAGES = ("placement:center_over_pad",)
# Max iterations of the adaptive waypoint-shrink retry (see
# _plan_pose_with_shrinking_waypoint) used when a direct Cartesian plan fails; 0 keeps pure
# direct planning. Each failed attempt pulls the target toward the arm's current pose and
# retries, then chains a follow-up attempt at the real target once a shrunk intermediate
# succeeds. The POST_GRASP_ESCAPE_ATTEMPTS environment variable is accepted as an alias.
LOCAL_WAYPOINT_ATTEMPTS = int(os.environ.get(
    "LOCAL_WAYPOINT_ATTEMPTS", os.environ.get("POST_GRASP_ESCAPE_ATTEMPTS", "5")))
# Stop shrinking once the remaining distance (m) from the current pose to the (possibly
# shrunk) target drops below this.
WAYPOINT_SHRINK_MIN_DISTANCE = float(os.environ.get("WAYPOINT_SHRINK_MIN_DISTANCE", "0.05"))
# A second (top_down) waypoint orientation and a +-0.05 m y-offset search both failed at
# the same rate as the baseline (the bottleneck is kinematic unreachability of the
# position), so both are single-valued to keep the expensive CuRobo batch-plan grasp check
# below cheap.
WAYPOINT_ORIENTATIONS = ("grasp_aligned",)
WAYPOINT_Y_OFFSETS = (0.0,)
# Ordered chain stages (see _plan_candidate), used to rank how far a candidate got before
# failing, so the fallback (when no candidate fully plans) picks the one that progressed
# furthest instead of the first one generated, which may rank first geometrically but be
# unreachable.
STAGE_ORDER = ["waypoint", "pre_grasp", "grasp", "lift", "post_grasp_escape",
               "pre_beside_box", "beside_box", "pre_lift_above_box",
               "lift_above_box", "over_box_to_pad_y", "center_over_pad"]

# Reject a (seed, offset) build if the occluder would sit within OCC_PAD_CLEARANCE
# (edge-to-edge) of the destination pad, so the occluder can never block/overlap the
# target's final placement. Center-to-center threshold = pad_half + occluder_half + gap.
OCC_PAD_CLEARANCE = 0.05
_PAD_HALF = 0.06               # pad half-size (create_box half_size xy)
_OCC_HALF = 0.06               # milk-box base half-extent (~0.11 x 0.122 footprint)
OCC_PAD_MIN_DIST = _PAD_HALF + _OCC_HALF + OCC_PAD_CLEARANCE
# Placement retries (target/occluder pose resampling) are pure geometry with no actor or
# physics work, so retrying inside load_actors is cheap. In an offline simulation over
# TARGET_XLIM/TARGET_YLIM/PAD_XY (500 seeds per range), OCC_DISTANCE_CM ranges 15,20 /
# 10,15 / 30,40 converge within a few attempts, while 20,30 is harder (~2% single-draw
# success) and still left 3.8% failures at 200 attempts, hence the cap of 500. A seed that
# still fails is skipped in run() via occluder_pad_clearance_ok.
_MAX_PLACEMENT_ATTEMPTS = 500

# Yaw-aware edge-to-edge geometry. OCC_DISTANCE_CM is a true surface gap, not a
# center-to-center offset: these helpers turn each object's mesh AABB and yaw into a
# footprint rectangle and solve for the center-to-center placement that yields the gap.
_MODEL_DATA_CACHE = {}


def _yaw_from_quat(q):
    """World-frame yaw (about Z) of a pose quaternion. Every actor here is first stood
    upright by a fixed base qpos (local mesh Y -> world Z) and then optionally yawed, so
    the local mesh X axis stays within the world XY plane and the yaw can be read directly
    off the rotation matrix."""
    R = t3d.quaternions.quat2mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _footprint_half_extents(extents, scale):
    """(half-x, half-y) world footprint of an object's mesh AABB at yaw=0 (local mesh
    X/Z -> world X/Y once stood upright; local Y is height -> world Z)."""
    return float(extents[0] * scale[0] / 2.0), float(extents[2] * scale[2] / 2.0)


def _model_footprint_half_extents(modelname, model_id, scale_override=None):
    """Footprint half-extents for modelname/model_id, read from its model_data{id}.json.
    Mirrors create_actor's scale resolution (explicit scale_override, else the model's
    "scale", else [1,1,1]) without creating the actor, so target/occluder placement can be
    resolved by a retry loop before any expensive actor/mesh work."""
    key = (modelname, model_id)
    model_data = _MODEL_DATA_CACHE.get(key)
    if model_data is None:
        path = Path(os.environ["ASSETS_ROOT"]) / "objects" / modelname / f"model_data{model_id}.json"
        with open(path) as f:
            model_data = json.load(f)
        _MODEL_DATA_CACHE[key] = model_data
    scale = scale_override if scale_override is not None else model_data.get("scale", [1.0, 1.0, 1.0])
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]
    return _footprint_half_extents(model_data["extents"], scale)


def _occluder_footprint_half_extents():
    """Milk-box (model_id=2) footprint half-extents; scale is always [1,1,1]."""
    return _model_footprint_half_extents("038_milk-box", 2, [1.0, 1.0, 1.0])


def _rect_corners(center_xy, yaw, half_x, half_y):
    """World-space corners (in order, so consecutive pairs are edges) of a yaw-rotated
    rectangle footprint centered at center_xy."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[half_x, half_y], [half_x, -half_y], [-half_x, -half_y], [-half_x, half_y]])
    rot = local @ np.array([[c, -s], [s, c]]).T
    return rot + np.asarray(center_xy, dtype=float)


def _closest_pt_seg_seg(p1, q1, p2, q2):
    """Distance between closest points on 2D segments p1-q1 and p2-q2 (Ericson,
    Real-Time Collision Detection 5.1.9; handles degenerate and parallel cases)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _rects_overlap(poly_a, poly_b):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (poly_a, poly_b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis = axis / norm
            proj_a, proj_b = poly_a @ axis, poly_b @ axis
            if proj_a.max() < proj_b.min() - 1e-9 or proj_b.max() < proj_a.min() - 1e-9:
                return False
    return True


def _min_rect_distance(center_a, yaw_a, half_a, center_b, yaw_b, half_b):
    """True minimum distance between two yaw-rotated rectangle footprints (0 if they
    overlap). Exact for convex quadrilaterals, not an axis-aligned approximation."""
    poly_a = _rect_corners(center_a, yaw_a, *half_a)
    poly_b = _rect_corners(center_b, yaw_b, *half_b)
    if _rects_overlap(poly_a, poly_b):
        return 0.0
    best = float("inf")
    for i in range(4):
        for j in range(4):
            best = min(best, _closest_pt_seg_seg(poly_a[i], poly_a[(i + 1) % 4],
                                                 poly_b[j], poly_b[(j + 1) % 4]))
    return best


def _solve_center_offset_for_gap(target_center, target_yaw, target_half, occ_yaw, occ_half,
                                 desired_gap, direction=np.array([0.0, -1.0])):
    """Bisect the center-to-center distance D (along `direction` from target_center) at
    which the true edge-to-edge distance between the target and occluder footprints
    equals `desired_gap`. Distance between two disjoint convex shapes translated apart
    grows monotonically with the separation, so bisection converges regardless of yaw."""
    diag_a = float(np.hypot(*target_half))
    diag_b = float(np.hypot(*occ_half))
    lo, hi = 0.0, desired_gap + diag_a + diag_b + 0.05
    for _ in range(60):
        mid = (lo + hi) / 2.0
        occ_center = np.asarray(target_center, dtype=float) + mid * direction
        gap = _min_rect_distance(target_center, target_yaw, target_half, occ_center, occ_yaw, occ_half)
        if gap < desired_gap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def dr_measure(clutter_density):
    """Domain-randomization for the measurement build: optional default (non-curated)
    table clutter at the given density. Density 0 -> clean (occluder only)."""
    if clutter_density and clutter_density > 0:
        return {"cluttered_table": True, "obstacle_density": int(clutter_density),
                "clean_background_rate": 0}
    return dict(DR_CLEAN)


def make_occluder_task():
    Base = get_env_class("put_mouse_on_pad", bench_subdir="office")

    class OccluderTask(Base):
        # The back furniture (shelf/cabinet/file-holder) is not spawned, to free up gripper
        # workspace. Set True to restore the full office.
        SPAWN_BACK_FURNITURE = False
        # Drop the orientation-hold constraint on the final grasp approach so CuRobo does not
        # reject it with INVALID_PARTIAL_POSE_COST_METRIC (see grasp_actor_from_table below).
        # Set False to restore the stock straight-in, wrist-locked approach.
        DROP_GRASP_ORIENTATION_CONSTRAINT = True
        # When True, play_once stops right after the lift (pickup only); used by
        # pickup_reachability_map.py to probe IK from the picked-up state.
        PICKUP_ONLY = False
        spawn_occluder = False
        occluder_offset = 0.2         # desired edge-to-edge gap (metres, -y of the target);
                                       # load_actors solves the actual center-to-center
                                       # placement that yields this true surface distance
        target_model = TARGET_MODEL
        target_id = TARGET_ID
        target_xlim = TARGET_XLIM
        target_ylim = TARGET_YLIM
        fixed_pad_xy = PAD_XY

        def load_actors(self):
            # Target: random within the upper-third band (varies per seed). When an occluder is
            # spawned, target/occluder placement is resolved before any actor is created (pure
            # geometry) by redrawing up to _MAX_PLACEMENT_ATTEMPTS times until the occluder clears
            # the destination pad. The target position is redrawn too, since re-yawing the occluder
            # alone only shifts its footprint by a few cm.
            target_pose = rand_pose(xlim=list(self.target_xlim), ylim=list(self.target_ylim),
                                    qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])

            self.occluder_pad_clearance_ok = True
            occ_yaw = occ_half = target_half = dist = probe_pose = None
            if self.spawn_occluder:
                target_scale_override = self.item_info["scales"].get(self.target_model, {}).get(
                    f"{self.target_id}", None)
                target_half = _model_footprint_half_extents(
                    self.target_model, self.target_id, target_scale_override)
                occ_half = _occluder_footprint_half_extents()

                placement_ok = False
                for attempt in range(_MAX_PLACEMENT_ATTEMPTS):
                    tx, ty = float(target_pose.p[0]), float(target_pose.p[1])
                    target_yaw = _yaw_from_quat(target_pose.q)
                    # sample the occluder's yaw/z the normal way; x/y are placeholders,
                    # overwritten below once the yaw-aware offset is solved for.
                    probe_pose = rand_pose(xlim=[0.0], ylim=[0.0], qpos=OCCLUDER_QPOS,
                                           rotate_rand=True, rotate_lim=[0, 3.14, 0])
                    occ_yaw = _yaw_from_quat(probe_pose.q)
                    dist = _solve_center_offset_for_gap((tx, ty), target_yaw, target_half,
                                                        occ_yaw, occ_half, self.occluder_offset)
                    oy = ty - dist
                    pad_dist = float(np.hypot(tx - self.fixed_pad_xy[0], oy - self.fixed_pad_xy[1]))
                    if pad_dist >= OCC_PAD_MIN_DIST:
                        placement_ok = True
                        break
                    # Redraw only if another attempt will use the draw; redrawing on the last
                    # attempt would leave target_pose inconsistent with (dist, probe_pose).
                    if attempt < _MAX_PLACEMENT_ATTEMPTS - 1:
                        target_pose = rand_pose(xlim=list(self.target_xlim), ylim=list(self.target_ylim),
                                                qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
                self.occluder_pad_clearance_ok = placement_ok

            # scale from the task yaml if present, else None -> model's own scale
            self.target_obj = create_actor(
                scene=self, pose=target_pose, modelname=self.target_model, convex=True,
                model_id=self.target_id,
                scale=self.item_info["scales"].get(self.target_model, {}).get(f"{self.target_id}", None),
            )
            self.target_obj.set_mass(0.05)

            # destination pad (flat; parked at the front so it never conflicts with the occluder)
            px, py = self.fixed_pad_xy
            pad_pose = rand_pose(xlim=[px], ylim=[py], qpos=[1, 0, 0, 0], rotate_rand=False)
            self.color_name, self.color_value = "Gray", (0.5, 0.5, 0.5)
            self.des_obj = create_box(scene=self, pose=pad_pose, half_size=[0.06, 0.06, 0.0005],
                                      color=self.color_value, name="box", is_static=True)
            self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
            self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
            self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
            self.des_obj_pose[2] += 0.02

            # Occluder: one tall milk-box at the target's x, just in front (-y), at the placement
            # resolved above. self.occluder_offset is the desired edge-to-edge gap (m), not the
            # center-to-center distance; _solve_center_offset_for_gap converts it so the gap holds
            # regardless of either object's yaw or footprint.
            if self.spawn_occluder:
                tx, ty = float(target_pose.p[0]), float(target_pose.p[1])
                ox, oy = tx, ty - dist
                occ_pose = sapien.Pose([ox, oy, float(probe_pose.p[2])], probe_pose.q)

                self.occluder = create_actor(
                    scene=self, pose=occ_pose, modelname="038_milk-box", convex=True,
                    model_id=2, scale=[1.0, 1.0, 1.0],
                )
                self.occluder.set_mass(0.1)
                # Register the occluder in CuRobo's collision world so the planner avoids it. It is
                # not flagged is_obstacle, so it survives the exclude_obstacles=True pass used in
                # collision-metrics/eval mode, which only drops procedural clutter.
                self.collision_list.append({
                    "actor": self.occluder,
                    "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/038_milk-box/collision/base2.glb",
                })

        def play_once(self):
            # Expert plan. The forward (grasp) part below is fixed; the backward (placement) part
            # is the subgoal list from self._backward_subgoal_poses() (one CuRobo move per subgoal),
            # which is where placement subgoals are added, removed or reordered.
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"

            # Structured failure record: the first stage whose checkpoint sees plan_success go
            # False, plus the CuRobo fail_reason captured by the plan+replay helpers (None for
            # stages that use the standard self.move() path). Read by run() after run_rollout()
            # and written to records.jsonl. Reset per episode since the env is reused across seeds.
            self.rollout_failure_stage = None
            self.rollout_failure_reason = None
            self.rollout_candidate_info = None
            self._last_fail_reason = None
            self._grasp_baseline_transform = None
            self._slow_attached_replay = False
            # Per-episode diagnostics also written to records.jsonl: per-attempt grasp history
            # (contact points tried and why each failed), per-check retention drift (position and
            # rotation), and per-slice descent detail (CuRobo's position/rotation error on each
            # place_actor segment).
            self.rollout_grasp_attempts = []
            self.rollout_retention_checks = []
            self.rollout_descent_slices = []

            def checkpoint(name):
                # The dry-run candidate search only checks that the waypoint/pre_grasp/grasp/lift/
                # placement poses are reachable via a chained last_qpos rollout; it never calls
                # attach_object/enable_table (which change the collision world) and never tests
                # place_actor's final descent. Recording the first failing stage shows which
                # real-execution step broke.
                if not self.plan_success and self.rollout_failure_stage is None:
                    self.rollout_failure_stage = name
                    self.rollout_failure_reason = self._last_fail_reason
                if log_move:
                    print(f"[play_once] after {name}: plan_success={self.plan_success}")

            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")
            # On a missed grasp (grasp_verify below), retry with a different contact point from
            # _select_pick_place_candidate's ranking (via exclude_cp_ids). Failed attempts reset
            # plan_success/rollout_failure_* so a later success is not recorded alongside a stale
            # failure.
            tried_cp_ids = set()
            candidate_plan = None
            cp_id = None
            grasp_succeeded = False
            retry_reset_pose = list(self.get_arm_pose(arm_tag))
            for grasp_attempt in range(1, GRASP_VERIFY_MAX_CANDIDATES + 1):
                candidate_plan = self._select_pick_place_candidate(arm_tag, exclude_cp_ids=tried_cp_ids)
                cp_id = candidate_plan["contact_point_id"]
                if cp_id is None:
                    # No ranked contact point at all, so retrying cannot help. Executing the grasp
                    # anyway would fall back to an unconstrained default grasp
                    # (contact_point_id=None) whose failure hides the real cause, so fail fast.
                    self.plan_success = False
                    self._last_fail_reason = "no_ranked_contact_points"
                    checkpoint("grasp_candidate_selection")
                    self.rollout_grasp_attempts.append({
                        "attempt": grasp_attempt,
                        "contact_point_id": None,
                        "success": False,
                        "failed_stage": self.rollout_failure_stage,
                        "fail_reason": self.rollout_failure_reason,
                    })
                    break
                tried_cp_ids.add(cp_id)

                if candidate_plan["grasp_waypoint"] is not None:
                    wp_trajectory = candidate_plan.get("grasp_waypoint_trajectory")
                    if wp_trajectory is not None:
                        self._replay_planned_sequence(arm_tag, wp_trajectory)
                    else:
                        self._plan_and_replay_pose(arm_tag, candidate_plan["grasp_waypoint"])
                    checkpoint("waypoint_move")
                self._grasp_via_cached_trajectories(arm_tag, candidate_plan)
                checkpoint("grasp")

                pre_lift_object_z = float(self.target_obj.get_pose().p[2])
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=GRASP_LIFT_HEIGHT))
                checkpoint("lift")
                if self.PICKUP_ONLY:        # stop at the picked-up state (reachability probe)
                    return
                # Verify the grasp captured the object. plan_success only says CuRobo planned the
                # commanded gripper move; on a missed grasp the gripper lifts while the object stays
                # on the table. Without this check, attach_object below would bake the object's
                # on-table pose into CuRobo's collision model as if rigidly held, and the resulting
                # collision with the re-enabled table would be misattributed to the placement stages.
                if self.plan_success:
                    object_rise = float(self.target_obj.get_pose().p[2] - pre_lift_object_z)
                    if object_rise < GRASP_LIFT_HEIGHT * GRASP_VERIFY_MIN_RISE_FRACTION:
                        self.plan_success = False
                        self._last_fail_reason = f"grasp_missed(object_z_rise={object_rise:.3f}m)"
                checkpoint("grasp_verify")
                attempt_record = {
                    "attempt": grasp_attempt,
                    "contact_point_id": cp_id,
                    "success": self.plan_success,
                    "failed_stage": self.rollout_failure_stage,
                    "fail_reason": self.rollout_failure_reason,
                }
                self.rollout_grasp_attempts.append(attempt_record)
                if self.plan_success:
                    grasp_succeeded = True
                    break
                if grasp_attempt < GRASP_VERIFY_MAX_CANDIDATES and cp_id is not None:
                    if log_move:
                        print(f"[grasp_verify] attempt {grasp_attempt} missed (cp_id={cp_id}); "
                              f"retrying with next candidate")
                    self.plan_success = True
                    self.rollout_failure_stage = None
                    self.rollout_failure_reason = None
                    self._last_fail_reason = None
                    # The failed attempt closed the gripper; reopen it and re-enable the table
                    # before selecting another contact point. The waypoint_move/pre_grasp leg of
                    # _grasp_via_cached_trajectories plans with table collision active (it only
                    # disables the table right before the final approach), so leaving it disabled
                    # would let the next plan pass through the table.
                    self.move(self.open_gripper(arm_tag))
                    self.enable_table(enable=True)
                    # Candidate plans are generated from the live arm state, and a missed grasp
                    # leaves the arm at its lifted endpoint. Return to the common pre-attempt pose
                    # before selecting another contact point.
                    self._plan_and_replay_pose(arm_tag, retry_reset_pose)
                    attempt_record["reset_success"] = bool(self.plan_success)
                    if not self.plan_success:
                        reset_reason = self._last_fail_reason or "unknown"
                        attempt_record["reset_fail_reason"] = reset_reason
                        self._last_fail_reason = f"grasp_retry_reset_failed:{reset_reason}"
                        checkpoint("grasp_retry_reset")
                        break
                else:
                    break
            if not grasp_succeeded:
                # Retry cleanup can temporarily restore plan_success so the next
                # candidate can run. Exhausting the loop is still a failed grasp.
                self.plan_success = False
                if self._last_fail_reason is None:
                    self._last_fail_reason = "grasp_candidates_exhausted"
                checkpoint("grasp_verify")
                return
            self.attach_object(
                self.target_obj,
                f"{os.environ['ASSETS_ROOT']}/objects/{self.target_model}/collision/base{self.target_id}.glb",
                str(arm_tag),
            )
            checkpoint("attach_object")
            # Baseline for _object_retained: the object's pose relative to the gripper right
            # after grasp_verify passed. Later replay steps compare against it.
            self._grasp_baseline_transform = self._gripper_relative_object_transform(arm_tag)
            self._slow_attached_replay = True
            self.enable_table(enable=True)
            checkpoint("enable_table")

            # Re-select the placement geometry in the real attached-object world, ranking
            # candidates by whether the full post-attach motion chain plans, not just by endpoint
            # IK. The winning trajectories are cached and replayed exactly instead of re-planned.
            _beside_box_pose = candidate_plan["placement_subgoals"][0][1]
            _lift_z = _beside_box_pose[2]
            _quat = list(_beside_box_pose[3:])
            placement_plan = self._select_attached_placement_plan(
                arm_tag,
                tgt_y=self._place_target_y,
                lift_z=_lift_z,
                quat=_quat,
            )
            candidate_plan["placement_subgoals"] = placement_plan["placement_subgoals"]
            if log_move:
                print(f"[placement] verified x_side={placement_plan['x_side']:.3f} "
                      f"clearance_z={placement_plan['clearance_z']:.2f} "
                      f"escape={placement_plan.get('escape_idx', 0)} "
                      f"quat={placement_plan['quat']}")
            for stage_name, traj in placement_plan["trajectories"]:
                self._replay_planned_move(arm_tag, traj)
                if self.plan_success and not self._object_retained(arm_tag, context=stage_name):
                    self.plan_success = False
                    self._last_fail_reason = f"object_lost:{stage_name}"
                checkpoint(stage_name)
            # If _select_attached_placement_plan finds no combo that plans all the way through
            # (verified=False), it returns the deepest partial chain, whose trajectory list stops
            # at the failing stage. The loop above then completes without a failing checkpoint,
            # so the fallback is marked as an explicit failure here.
            if self.plan_success and not placement_plan.get("verified", False):
                self.plan_success = False
                self._last_fail_reason = placement_plan.get("fail_reason", "unverified_placement_fallback")
                checkpoint(placement_plan.get("failed_stage") or "placement:unverified_fallback")
            # place_actor lowers the object from the last subgoal onto the pad.
            # constrain="free": check_success is position-only, so alignment buys nothing and is a
            # likely IK_FAIL cause for a tall object held near its top.
            # dis=0.02: a 5 mm final gap leaves the held object's collision volume so close to the
            # table that trajopt finetuning fails (FINETUNE_TRAJOPT_FAIL); check_success() has no
            # height requirement, so a higher release costs nothing.
            # approach_axis=2 uses PoseCostMetric.create_grasp_approach_metric (as in CuRobo's
            # examples/isaac_sim/simple_stacking.py) for a straight-line z descent over the last
            # tstep_fraction of the trajectory. It sets offset_tstep_fraction >= 0, which skips
            # CuRobo's start-vs-goal orientation match check that rejects the plain constraint_pose
            # path (the reason DROP_GRASP_ORIENTATION_CONSTRAINT exists).
            place_arm_tag, place_actions = self.place_actor(
                self.target_obj, arm_tag=arm_tag, target_pose=self.des_obj_pose,
                constrain="free", pre_dis=0.05, dis=0.02,
            )
            if not place_actions:        # place_actor bails (plan_success already False)
                return                   # -> avoid place_actions[0] IndexError below
            for a in place_actions:
                if a.action == "move":
                    a.args["approach_axis"] = 2
            # place_pre_pose and place_pose are both reachable and collision-free with the object
            # attached, but center_over_pad -> place_pre_pose is one large move (vertical drop,
            # lateral shift and orientation change) that trajopt cannot smooth. Insert a verified
            # intermediate pose (position + orientation blend), as on the grasp side.
            mid_pose = self._verified_intermediate(arm_tag, candidate_plan["placement_subgoals"][-1][1], place_actions[0].target_pose)
            self._plan_and_replay_pose(arm_tag, mid_pose)
            if self.plan_success and not self._object_retained(arm_tag, context="placement:pre_place_descent"):
                self.plan_success = False
                self._last_fail_reason = "object_lost:placement:pre_place_descent"
            checkpoint("placement:pre_place_descent")
            if log_move:
                # Pure collision-aware IK check of the endpoints at the live qpos with the object
                # attached; separates "pose unreachable" from "no smooth path to the pose".
                check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
                move_poses = [a.target_pose for a in place_actions if a.action == "move"]
                ik_ok = check_ik(move_poses)
                print(f"[place_actor] pure-IK reachability (attached, live qpos): {list(ik_ok)}")
            # Commanding CuRobo down to place_actor's single exact final pose made place_actor
            # the dominant failure (FINETUNE_TRAJOPT_FAIL), since the attached object's margin
            # against the table is tightest there. Instead, search nearby landing poses within
            # check_success's tolerance and let physics settle the final contact.
            self._local_landing_search_and_place(place_arm_tag)
            checkpoint("place_actor")

        def _blend_pose(self, pose_a, pose_b, t=0.5):
            """Position lerp + quaternion nlerp (normalized linear interpolation, a cheap
            approximation of slerp that is good enough for a transit waypoint) between two flat
            [x,y,z,qw,qx,qy,qz] poses. Used to split a large move into smaller ones that trajopt
            solves more easily."""
            pa, pb = np.asarray(pose_a, dtype=float), np.asarray(pose_b, dtype=float)
            pos = (1 - t) * pa[:3] + t * pb[:3]
            qa, qb = pa[3:], pb[3:]
            if np.dot(qa, qb) < 0:      # quaternion double-cover: align signs before blending
                qb = -qb
            q = (1 - t) * qa + t * qb
            q = q / np.linalg.norm(q)
            return list(pos) + list(q)

        def _verified_intermediate(self, arm_tag, pose_a, pose_b, fractions=(0.5, 0.35, 0.65, 0.25, 0.75)):
            """Pick a transit waypoint between pose_a and pose_b that passes check_ik_batch.
            A plain midpoint blend can itself be unreachable, so several blend fractions are
            tried and the first reachable one is used (the plain midpoint if none is)."""
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            candidates = [self._blend_pose(pose_a, pose_b, t) for t in fractions]
            ok = check_ik(candidates)
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[intermediate] fractions={fractions} ok={list(ok)}")
            for candidate, reachable in zip(candidates, ok):
                if reachable:
                    return candidate
            return candidates[0]  # none verified reachable; fall back to the plain midpoint

        def _backward_subgoal_poses(self, arm_tag, x_side, clearance_z, lift_pose):
            """Ordered backward (placement) subgoals for a chosen candidate path."""
            quat = list(lift_pose[3:])
            tgt_y = self._place_target_y
            pad_x, pad_y = self.fixed_pad_xy
            lift_z = float(lift_pose[2])
            subgoals = []
            subgoals.append(("beside_box", [x_side, tgt_y, lift_z, *quat]))
            subgoals.append(("lift_above_box", [x_side, tgt_y, clearance_z, *quat]))
            subgoals.append(("over_box_to_pad_y", [x_side, pad_y, clearance_z, *quat]))
            subgoals.append(("center_over_pad", [pad_x, pad_y, clearance_z, *quat]))
            return subgoals

        def _post_grasp_escape_poses(self, arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS):
            """Local escape waypoint candidates from the current held-object pose.

            Candidate zero is `None`, preserving the direct route. The remaining
            candidates are the bounded local waypoint family for this specific
            attached-object transition out of the grasped state.
            """
            current = list(self.get_arm_pose(arm_tag))
            return [None] + self._local_waypoint_candidates(
                arm_tag, current, current, attempts=attempts)

        def _placement_execution_steps(self, arm_tag, placement_subgoals, escape_pose=None):
            """Build the exact post-attach placement chain to execute."""
            current_pose = list(self.get_arm_pose(arm_tag))
            steps = []
            if escape_pose is not None:
                steps.append(("placement:post_grasp_escape", escape_pose))
                current_pose = escape_pose
            if placement_subgoals:
                beside_box_pose = placement_subgoals[0][1]
                mid_to_beside_box = self._verified_intermediate(arm_tag, current_pose, beside_box_pose)
                steps.append(("placement:pre_beside_box", mid_to_beside_box))

            prev_pose = None
            for name, pose in placement_subgoals:
                if name == "lift_above_box" and prev_pose is not None:
                    mid_to_lift_above_box = self._verified_intermediate(arm_tag, prev_pose, pose)
                    steps.append(("placement:pre_lift_above_box", mid_to_lift_above_box))
                steps.append((f"placement:{name}", pose))
                prev_pose = pose
            return steps

        def _plan_pose_trajectory_sequence(self, arm_tag, stage_poses, start_qpos,
                                           local_attempts=LOCAL_WAYPOINT_ATTEMPTS, retries=1,
                                           relax_orientation=False):
            """Plan a labeled pose chain and keep the exact successful trajectories.

            relax_orientation is withheld for PLACEMENT_STRICT_ORIENTATION_STAGES regardless of
            the caller's request. Relaxing orientation on the last placement subgoal
            (center_over_pad, the handoff into place_actor's orientation-sensitive descent) let
            CuRobo converge to arbitrary held-object orientations that caused large placement
            misses even though every stage reported plan_success=True. Interior transit stages
            stay relaxed (that is where IK_FAIL was the bottleneck); any orientation drift there
            is corrected by the strict final stage."""
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            trajectories = []
            start_pose = list(self.get_arm_pose(arm_tag))
            for stage_name, pose in stage_poses:
                stage_relax_orientation = relax_orientation and stage_name not in PLACEMENT_STRICT_ORIENTATION_STAGES
                ok, fail_reason, qpos, stage_trajs = self._plan_pose_with_local_waypoint_retry(
                    arm_tag, pose, qpos, stage_name, start_pose=start_pose,
                    local_attempts=local_attempts, retries=retries,
                    relax_orientation=stage_relax_orientation)
                if not ok:
                    return False, stage_name, fail_reason, trajectories
                trajectories.extend(stage_trajs)
                start_pose = pose
            return True, None, None, trajectories

        def _select_attached_placement_plan(self, arm_tag, tgt_y, lift_z, quat,
                                            gaps=SIDE_WAYPOINT_GAPS, clearance_options=PLACE_CLEARANCE_ZS,
                                            fallback_gaps=SIDE_WAYPOINT_GAPS_FALLBACK,
                                            clearance_fallback=PLACE_CLEARANCE_ZS_FALLBACK):
            """Pick placement geometry by full attached-world motion-plan success.

            Endpoint IK is a useful prefilter, but most remaining failures come from the chained
            motion itself. This keeps the same geometry search space and scores candidates by
            whether the exact post-attach execution chain plans from the current live qpos. The
            winning trajectories are cached for immediate replay.

            Every stage searched here (post_grasp_escape, beside_box, lift_above_box,
            over_box_to_pad_y, center_over_pad and the blended pre_* transit waypoints) only
            carries the held object, and check_success() only looks at the final x/y. So the
            prefilters and trajopt calls use relax_orientation=True (position-only goals), which
            makes lift_above_box combos reachable that strict full-pose IK rejects."""
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            quat_options = (quat, GRASP_DIRECTION_DIC["top_down"])
            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            fallback = None
            fallback_depth = -1
            failure_breakdown = collections.Counter()
            escape_poses = self._post_grasp_escape_poses(arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS)

            # Try gaps first, and fallback_gaps only if nothing in gaps was reachable (a box far
            # from the shoulder can make every SIDE_WAYPOINT_GAPS beside_box target IK-infeasible
            # regardless of clearance/escape/quat).
            for gap_values in (gaps, fallback_gaps):
                any_reachable_x = False
                for quat_choice in quat_options:
                    # Cheap prefilter (no trajopt): beside_box's z (lift_z) and y (tgt_y) do not depend
                    # on clearance_z/escape_idx, so one batched IK call over all gaps skips the whole
                    # cz x escape_idx loop (up to 30 trajopt chains per gap) for gaps that cannot succeed.
                    gap_list = list(gap_values)
                    x_list = [self._box_side_x(arm_tag, gap=gap) for gap in gap_list]
                    beside_box_ok = check_ik([[x, tgt_y, lift_z, *quat_choice] for x in x_list],
                                              relax_orientation=True)
                    for gap, x, reachable in zip(gap_list, x_list, beside_box_ok):
                        if not bool(reachable):
                            failure_breakdown[("placement:beside_box", "IK_prefilter_unreachable")] += 1
                            if log_move:
                                print(f"[placement-plan] gap={gap:.2f} x_side={x:.3f} quat={quat_choice} "
                                      f"skipped: beside_box IK-unreachable (prefilter)")
                            continue
                        any_reachable_x = True
                        # Same prefilter one level deeper: lift_above_box shares (x, tgt_y) and only
                        # varies z (clearance), so batch-check every clearance value before the
                        # escape_idx loop of trajopt chains.
                        cz_list = list(clearance_options) + list(clearance_fallback)
                        lift_above_box_ok = check_ik([[x, tgt_y, cz, *quat_choice] for cz in cz_list],
                                                      relax_orientation=True)
                        for cz, cz_reachable in zip(cz_list, lift_above_box_ok):
                            if not bool(cz_reachable):
                                failure_breakdown[("placement:lift_above_box", "IK_prefilter_unreachable")] += 1
                                if log_move:
                                    print(f"[placement-plan] gap={gap:.2f} x_side={x:.3f} clearance_z={cz:.2f} "
                                          f"quat={quat_choice} skipped: lift_above_box IK-unreachable (prefilter)")
                                continue
                            placement_subgoals = self._backward_subgoal_poses(
                                arm_tag,
                                x_side=x,
                                clearance_z=cz,
                                lift_pose=[0, 0, lift_z, *quat_choice],
                            )
                            for escape_idx, escape_pose in enumerate(escape_poses):
                                stage_poses = self._placement_execution_steps(
                                    arm_tag, placement_subgoals, escape_pose=escape_pose)
                                ok, failed_stage, fail_reason, trajectories = self._plan_pose_trajectory_sequence(
                                    arm_tag, stage_poses, start_qpos, local_attempts=0,
                                    retries=PLACEMENT_SEARCH_RETRIES, relax_orientation=True)
                                if ok:
                                    return {
                                        "x_side": x,
                                        "clearance_z": cz,
                                        "quat": quat_choice,
                                        "escape_idx": escape_idx,
                                        "placement_subgoals": placement_subgoals,
                                        "trajectories": trajectories,
                                        "verified": True,
                                    }
                                depth = STAGE_ORDER.index(failed_stage.split("placement:")[-1]) if (
                                    failed_stage and failed_stage.startswith("placement:")
                                    and failed_stage.split("placement:")[-1] in STAGE_ORDER
                                ) else -1
                                if depth > fallback_depth:
                                    fallback = {
                                        "x_side": x,
                                        "clearance_z": cz,
                                        "quat": quat_choice,
                                        "escape_idx": escape_idx,
                                        "placement_subgoals": placement_subgoals,
                                        "trajectories": trajectories,
                                        "verified": False,
                                        "failed_stage": failed_stage,
                                        "fail_reason": fail_reason,
                                    }
                                    fallback_depth = depth
                                failure_breakdown[(failed_stage, fail_reason)] += 1
                                if log_move:
                                    print(f"[placement-plan] x_side={x:.3f} clearance_z={cz:.2f} "
                                          f"escape={escape_idx} quat={quat_choice} failed at "
                                          f"{failed_stage} ({fail_reason})")
                if any_reachable_x or gap_values is not gaps:
                    break
                if log_move:
                    print(f"[placement-plan] none of gaps={gap_values} verified beside_box-reachable "
                          f"for any quat; trying fallback gaps={fallback_gaps}")
            if log_move and fallback is not None:
                print(f"[placement-plan] no fully planned geometry; using deepest fallback "
                      f"(depth={fallback_depth}) with breakdown={failure_breakdown}")
            if fallback is not None:
                return fallback
            placement_subgoals = self._backward_subgoal_poses(
                arm_tag,
                x_side=self._box_side_x(arm_tag, gap=gaps[0]),
                clearance_z=clearance_options[0],
                lift_pose=[0, 0, lift_z, *quat],
            )
            return {
                "x_side": self._box_side_x(arm_tag, gap=gaps[0]),
                "clearance_z": clearance_options[0],
                "quat": quat,
                "placement_subgoals": placement_subgoals,
                "trajectories": [],
                "verified": False,
                "failed_stage": None,
                "fail_reason": "no_geometry_attempted",
            }

        def _box_side_x(self, arm_tag, gap=SIDE_WAYPOINT_GAPS[1]):
            """World x beside the occluder on the arm's OWN side (right -> +x, left -> -x),
            OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP from the box centre, clamped to the reach
            limit on that same side (never flipped across to the wrong side)."""
            if self.spawn_occluder and getattr(self, "occluder", None) is not None:
                box_x = float(self.occluder.get_pose().p[0])
            else:
                box_x = float(self.target_obj.get_pose().p[0])
            side = 1.0 if str(arm_tag) == "right" else -1.0
            x = box_x + side * (OCC_HALF_FOOTPRINT + gap)
            return side * REACH_X_LIMIT if abs(x) > REACH_X_LIMIT else x

        def _around_box_waypoint(self, arm_tag, ref_pose, gap=SIDE_WAYPOINT_GAPS[1], z_lift=0.0,
                                  orient="grasp_aligned", y_offset=0.0):
            """Grasp subgoal: a pose beside the occluder (via _box_side_x), near the box's
            horizontal (y) line. Position uses ref_pose's height plus the side-waypoint offset,
            with y = box y-line + y_offset (searched around 0, since exactly the box y-line is
            IK-unreachable for some seeds). Orientation is ref_pose's own (grasp_aligned, matching
            the reach-in angle) or a neutral top-down one (orient="top_down"). None when no
            occluder is present."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return None
            wp = list(ref_pose)
            wp[0] = self._box_side_x(arm_tag, gap=gap)
            wp[1] = float(self.occluder.get_pose().p[1]) + float(y_offset)   # box's y-line +/- offset
            wp[2] += float(z_lift)
            if orient == "top_down":
                wp[3:] = GRASP_DIRECTION_DIC["top_down"]
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[around_box] arm={arm_tag} orient={orient} y_offset={y_offset:.2f} -> waypoint "
                      f"x={wp[0]:.3f} y={wp[1]:.3f} z={wp[2]:.3f}")
            return wp

        def _rank_side_grasp_ids(self, actor, arm_tag, limit=GRASP_CANDIDATE_LIMIT):
            """Rank a few side grasps so the search can try several candidates instead of
            committing to a single local optimum."""
            side = 1.0 if str(arm_tag) == "right" else -1.0
            cx = float(actor.get_pose().p[0])
            conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            ranked = []
            for i, cm in actor.iter_contact_points("matrix"):
                if cm is None:
                    continue
                g = np.asarray(cm, dtype=float) @ conv
                R = g[:3, :3]
                approach = R @ np.array([1.0, 0.0, 0.0])      # gripper +x approach (world)
                horiz = abs(float(approach[2]))               # 0 = perfectly horizontal
                if horiz > 0.35:                              # skip clearly tilted/vertical
                    continue
                grasp_x = float(g[0, 3]) + float((R @ np.array([-0.12, 0.0, 0.0]))[0])
                arm_side = side * (grasp_x - cx)              # >0 = gripper on arm's side
                score = arm_side - horiz                      # arm-side + as horizontal as possible
                ranked.append((score, i))
            ranked.sort(reverse=True)
            ids = [i for _, i in ranked[:limit]]
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[grasp_id] arm={arm_tag} candidates={ids}")
            return ids

        def _evaluate_descent_metrics(self, metrics, max_joint_endpoint_displacement=DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT,
                                      max_joint_path_length=DESCENT_MAX_JOINT_PATH_LENGTH):
            """Gate a planned trajectory's path-safety metrics (from
            _trajectory_path_metrics) against the descent-slice thresholds shared by
            _plan_pose_with_descent_slices and _local_landing_search_and_place. The latter
            passes joint_scale-adjusted endpoint/path-length caps since its candidates cover a
            longer segment than a nominal descent slice.

            Returns (accepted, sort_key, reject_reason): sort_key is
            (path_length, joint_path_length, max_joint_range) for ranking accepted
            candidates by shortest-path-first, or (0.0, 0.0, 0.0) when there's
            nothing to measure (degenerate <2-waypoint plan, trusted as-is).
            reject_reason is None when accepted."""
            if metrics is None:
                return True, (0.0, 0.0, 0.0), None
            if (metrics["max_perp_deviation"] <= DESCENT_MAX_PATH_DEVIATION
                    and metrics["path_length_ratio"] <= DESCENT_MAX_PATH_LENGTH_RATIO
                    and metrics["joint_travel_ratio"] <= DESCENT_MAX_JOINT_TRAVEL_RATIO
                    and metrics["max_joint_range"] <= DESCENT_MAX_JOINT_RANGE
                    and metrics["joint_direct_dist"] <= max_joint_endpoint_displacement
                    and metrics["joint_path_length"] <= max_joint_path_length):
                return True, (metrics["path_length"], metrics["joint_path_length"], metrics["max_joint_range"]), None
            reject_reason = (
                f"path_filter_rejected(dev={metrics['max_perp_deviation']:.3f},"
                f"len_ratio={metrics['path_length_ratio']:.2f},"
                f"joint_ratio={metrics['joint_travel_ratio']:.2f},"
                f"max_joint_range={metrics['max_joint_range']:.3f},"
                f"joint_dist={metrics['joint_direct_dist']:.3f}(max={max_joint_endpoint_displacement:.3f}),"
                f"joint_path={metrics['joint_path_length']:.3f}(max={max_joint_path_length:.3f}))")
            return False, None, reject_reason

        def _pick_side_grasp_id(self, actor, arm_tag):
            ranked = self._rank_side_grasp_ids(actor, arm_tag, limit=1)
            return ranked[0] if ranked else None

        def _geometric_grasp_pose(self, actor, cp_id, pre_dis=0.0):
            """Grasp pose for a contact point, computed geometrically (same math as
            get_grasp_pose but WITHOUT choose_best_pose's planning gate). Used only to
            orient/place the side waypoint, so a box-blocked direct plan can't veto it."""
            cm = actor.get_contact_point(cp_id, "matrix")
            if cm is None:
                return None
            conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            g = np.asarray(cm, dtype=float) @ conv
            R = g[:3, :3]
            p = g[:3, 3] + R @ np.array([-0.12 - pre_dis, 0.0, 0.0])
            q = t3d.quaternions.mat2quat(R)
            return list(p) + list(q)

        def _time_stretch_trajectory(self, result, factor):
            """Densify a planned joint path while preserving its geometry."""
            factor = max(1, int(factor))
            positions = np.asarray(result.get("position"))
            if factor == 1 or positions.ndim < 2 or len(positions) < 2:
                return result

            stretched = deepcopy(result)
            sample = np.linspace(0.0, len(positions) - 1,
                                 (len(positions) - 1) * factor + 1)
            lo = np.floor(sample).astype(int)
            hi = np.minimum(lo + 1, len(positions) - 1)
            alpha = (sample - lo).reshape((-1,) + (1,) * (positions.ndim - 1))
            for key, derivative_order in (("position", 0), ("velocity", 1),
                                          ("acceleration", 2), ("jerk", 3)):
                values = result.get(key)
                if values is None:
                    continue
                values = np.asarray(values)
                if values.ndim == 0 or len(values) != len(positions):
                    continue
                blended = values[lo] * (1.0 - alpha) + values[hi] * alpha
                if derivative_order:
                    blended = blended / (factor ** derivative_order)
                stretched[key] = blended.astype(values.dtype, copy=False)
            return stretched

        def _replay_planned_move(self, arm_tag, cached_result):
            """Execute an already-planned trajectory (position/velocity arrays from an
            earlier plan_path call) directly via take_dense_action, without re-planning.
            Mirrors left_move_to_pose/right_move_to_pose's need_plan/joint_path bookkeeping.

            CuRobo's trajopt has no uniqueness guarantee for a Cartesian target: re-planning
            to the same pose can converge to a different, equally valid arm posture (up to
            1.47 rad qpos drift observed between a dry-run plan and a live re-plan), which
            flips downstream grasp reachability in about half of those cases. Replaying the
            verified trajectory keeps the live qpos identical to what the dry run assumed."""
            if not self.plan_success:
                return
            if cached_result is None or cached_result.get("status") != "Success":
                self.plan_success = False
                self._last_fail_reason = (cached_result or {}).get("fail_reason", "unknown")
                return
            if str(arm_tag) == "left":
                if self.need_plan:
                    self.left_joint_path.append(deepcopy(cached_result))
                else:
                    cached_result = deepcopy(self.left_joint_path[self.left_cnt])
                    self.left_cnt += 1
                replay_result = self._time_stretch_trajectory(
                    cached_result, ATTACHED_TRAJECTORY_SLOWDOWN
                    if self._slow_attached_replay else 1)
                control_seq = {"left_arm": replay_result, "left_gripper": None, "right_arm": None, "right_gripper": None}
            else:
                if self.need_plan:
                    self.right_joint_path.append(deepcopy(cached_result))
                else:
                    cached_result = deepcopy(self.right_joint_path[self.right_cnt])
                    self.right_cnt += 1
                replay_result = self._time_stretch_trajectory(
                    cached_result, ATTACHED_TRAJECTORY_SLOWDOWN
                    if self._slow_attached_replay else 1)
                control_seq = {"left_arm": None, "left_gripper": None, "right_arm": replay_result, "right_gripper": None}
            self.take_dense_action(control_seq)

        def _gripper_relative_object_transform(self, arm_tag):
            """The held object's full pose (position and orientation) in the gripper's local
            frame. Stable under gripper motion while the object is rigidly held, unlike a
            world-frame offset, which changes with gripper orientation even with zero slip.
            Returns (relative_position, relative_quaternion)."""
            gripper_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=float)
            gripper_rot = t3d.quaternions.quat2mat(gripper_pose[3:])
            object_pose = self.target_obj.get_pose()
            object_pos = np.asarray(object_pose.p, dtype=float)
            object_rot = t3d.quaternions.quat2mat(np.asarray(object_pose.q, dtype=float))
            relative_pos = gripper_rot.T @ (object_pos - gripper_pose[:3])
            relative_quat = t3d.quaternions.mat2quat(gripper_rot.T @ object_rot)
            return relative_pos, relative_quat

        def _object_near_placement_target(self, xy_tolerance=OBJECT_PLACEMENT_XY_TOLERANCE,
                                          z_tolerance=OBJECT_PLACEMENT_Z_TOLERANCE):
            """True if the held object already satisfies put_mouse_on_pad.check_success()'s
            placement criterion: independent x/y error each within xy_tolerance (mirroring the
            per-axis 0.02 check rather than a looser combined radius) and z within z_tolerance
            of the resting height. Used by place_actor's descent to stop and open the gripper
            once the placement is good. See DESCENT_CONTACT_DRIFT_TOLERANCE for drift that is
            neither this precise nor a genuine loss."""
            if not getattr(self, "des_obj_pose", None):
                return False
            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            target = np.asarray(self.des_obj_pose, dtype=float)
            dx = abs(float(obj_pos[0] - target[0]))
            dy = abs(float(obj_pos[1] - target[1]))
            z_dist = float(obj_pos[2] - target[2])  # positive: object sits above the target resting height
            return dx <= xy_tolerance and dy <= xy_tolerance and -z_tolerance <= z_dist <= z_tolerance

        def _object_near_support_height(self, height_tolerance=DESCENT_CONTACT_HEIGHT_TOLERANCE):
            """True if the held object's world z is plausibly at the placement surface
            (independent of x/y). Gates DESCENT_CONTACT_DRIFT_TOLERANCE's ambiguous-contact
            branch: a genuine drop shows the same gripper-relative drift while still high above
            the surface."""
            if not getattr(self, "des_obj_pose", None):
                return False
            obj_z = float(self.target_obj.get_pose().p[2])
            target_z = float(self.des_obj_pose[2])
            return abs(obj_z - target_z) <= height_tolerance

        def _placement_xy_error(self):
            """Signed target-minus-object XY error in world coordinates."""
            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            target = np.asarray(self.des_obj_pose, dtype=float)
            return target[:2] - obj_pos[:2]


        def _object_retained(self, arm_tag, tolerance=OBJECT_RETENTION_TOLERANCE,
                             rotation_tolerance=OBJECT_RETENTION_ROTATION_TOLERANCE, context=None,
                             return_drift=False):
            """True if the held object is still where it was (position and orientation)
            relative to the gripper when _grasp_baseline_transform was captured (right after
            attach_object in play_once), i.e. the grasp has not slipped, rotated loose or been
            dropped. Needed because plan_success only reflects the arm's motion plan, not the
            held object's state. Returns True if no baseline has been captured yet.

            Strict everywhere, including place_actor's descent; callers there check
            _object_near_placement_target first, since a legitimate early placement also changes
            the gripper-relative transform and would otherwise look like a drop."""
            if self._grasp_baseline_transform is None:
                return (True, 0.0, 0.0) if return_drift else True
            baseline_pos, baseline_quat = self._grasp_baseline_transform
            current_pos, current_quat = self._gripper_relative_object_transform(arm_tag)
            pos_drift = float(np.linalg.norm(current_pos - baseline_pos))
            rot_drift = float(cal_quat_dis(baseline_quat, current_quat) * np.pi)
            retained = pos_drift <= tolerance and rot_drift <= rotation_tolerance
            self.rollout_retention_checks.append({
                "context": context, "pos_drift": pos_drift, "rot_drift": rot_drift, "retained": retained,
            })
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[object-retained] context={context} pos_drift={pos_drift:.4f}m (tol={tolerance}) "
                      f"rot_drift={rot_drift:.4f}rad (tol={rotation_tolerance}) retained={retained}")
            if return_drift:
                return retained, pos_drift, rot_drift
            return retained

        def _trajectory_path_metrics(self, arm_tag, result):
            """Safety-filter metrics for a planned descent-slice trajectory, computed with the
            arm planner's forward kinematics. Used by _plan_pose_with_descent_slices to reject
            CuRobo trajopt solves that are valid but take an implausibly circuitous path for a
            short straight-line slice (see DESCENT_MAX_PATH_DEVIATION and the related
            thresholds). Returns None if the trajectory has fewer than 2 waypoints.

            - max_perp_deviation: max distance of any waypoint's EE position from the finite
              segment between the trajectory's start and end EE pose (projection clamped to
              [0, line_len], so an overshoot along the same axis is caught).
            - path_length_ratio: Cartesian path length (sum of consecutive EE waypoint
              distances) over the straight-line distance; catches back-and-forth paths.
            - joint_path_length / joint_direct_dist: absolute joint-space path length and
              direct start-to-end joint distance (rad, L2 norm), exposed separately so a
              monotonic jump to a distant IK branch (travel ratio near 1.0) is caught by its
              magnitude.
            - joint_travel_ratio: joint_path_length / joint_direct_dist; catches joint-space
              winding that an EE-only check misses. A near-zero denominator (here or in
              path_length_ratio) with a non-negligible numerator gives inf instead of 1.0, so a
              loop with almost identical start and end does not look like "nothing moved".
            - max_joint_range: largest single-joint (max - min) excursion (rad); flags one
              joint spinning through an implausible range."""
            positions = result.get("position")
            if positions is None or len(positions) < 2:
                return None
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            joint_state = JointState.from_position(
                torch.tensor(positions, dtype=torch.float32).cuda(),
                joint_names=planner.active_joints_name,
            )
            kin_state = planner.motion_gen.compute_kinematics(joint_state)
            ee_pos = np.array(kin_state.ee_pos_seq.to("cpu"), dtype=float)
            start_pos, end_pos = ee_pos[0], ee_pos[-1]
            line_vec = end_pos - start_pos
            line_len = float(np.linalg.norm(line_vec))
            if line_len < 1e-6:
                max_perp_deviation = float(np.max(np.linalg.norm(ee_pos - start_pos, axis=1)))
            else:
                line_unit = line_vec / line_len
                rel = ee_pos - start_pos
                t = np.clip(rel @ line_unit, 0.0, line_len)
                closest = start_pos + np.outer(t, line_unit)
                max_perp_deviation = float(np.max(np.linalg.norm(ee_pos - closest, axis=1)))
            path_length = float(np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)))
            if line_len > 1e-6:
                path_length_ratio = path_length / line_len
            elif path_length > 1e-6:
                path_length_ratio = float("inf")
            else:
                path_length_ratio = 1.0

            joint_pos = np.asarray(positions, dtype=float)
            joint_path_length = float(np.sum(np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)))
            joint_direct_dist = float(np.linalg.norm(joint_pos[-1] - joint_pos[0]))
            if joint_direct_dist > 1e-6:
                joint_travel_ratio = joint_path_length / joint_direct_dist
            elif joint_path_length > 1e-6:
                joint_travel_ratio = float("inf")
            else:
                joint_travel_ratio = 1.0
            max_joint_range = float(np.max(joint_pos.max(axis=0) - joint_pos.min(axis=0)))

            return {
                "max_perp_deviation": max_perp_deviation,
                "path_length": path_length,
                "path_length_ratio": path_length_ratio,
                "joint_path_length": joint_path_length,
                "joint_direct_dist": joint_direct_dist,
                "joint_travel_ratio": joint_travel_ratio,
                "max_joint_range": max_joint_range,
            }

        def _descent_tstep_fraction_for_attempt(self, attempt_idx, total_attempts,
                                               fractions=DESCENT_APPROACH_TSTEP_FRACTIONS):
            """Progressive relaxation schedule over a descent slice's retry budget: early
            attempts use the tightest fraction, later ones relax toward CuRobo's default (see
            DESCENT_APPROACH_TSTEP_FRACTIONS)."""
            tier = min(len(fractions) - 1, attempt_idx * len(fractions) // max(1, total_attempts))
            return fractions[tier]

        def _plan_pose_with_descent_slices(self, arm_tag, target_pose, qpos, stage_label, start_pose,
                                          retries=1, constraint_pose=None, approach_axis=None,
                                          slice_size=DESCENT_SLICE_SIZE, near_contact=False):
            """Deterministic incremental descent for place_actor's moves, used instead of the
            generic shrink retry (_plan_pose_with_shrinking_waypoint). place_actor's moves are
            short, mostly straight final approaches with a fixed target orientation (held via
            approach_axis), so fixed-size slices along the straight line fit them better than an
            adaptive shrink. Each slice is planned and replayed immediately, with
            _object_retained checked after every segment, so a mid-descent drop is caught at the
            slice where it happens and completed slices stay committed if a later one fails.

            Returns (ok, fail_reason, qpos, placed). placed=True means the object already
            satisfies check_success; the caller (_execute_actions_via_plan_and_replay) then skips
            the remaining 'move' actions and opens the gripper."""
            plan_func = self._arm_plan_func(arm_tag)
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            start = np.asarray(start_pose, dtype=float)
            target = np.asarray(target_pose, dtype=float)
            total_dist = float(np.linalg.norm(target[:3] - start[:3]))
            num_slices = max(1, int(np.ceil(total_dist / slice_size)))
            segment_length = total_dist / num_slices
            # create_grasp_approach_metric's offset_position must be smaller than the segment it
            # is applied to. CuRobo's default (0.05) exceeds a 4 cm slice, and the unscaled offset
            # caused FINETUNE_TRAJOPT_FAIL, so it is scaled with the segment length.
            approach_offset = min(0.05, segment_length * 0.8) if approach_axis is not None else 0.05
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            last_reason = "unknown"
            for i in range(1, num_slices + 1):
                # Interpolate both position and orientation (_blend_pose); copying the target
                # quaternion onto every slice made the first slice perform the whole orientation
                # change in one small position step.
                waypoint = np.asarray(self._blend_pose(start, target, i / num_slices), dtype=float)
                total_attempts = max(1, int(retries))
                # Collect every attempt that plans and passes the path-safety filter across the
                # relaxation schedule, then replay the shortest accepted one, so a looser fraction
                # that finds a shorter path can win. Bounded by retries.
                accepted = []  # list of (path_length, joint_path_length, max_joint_range, result, attempt_idx, attempt_record)
                for attempt in range(total_attempts):
                    fraction = self._descent_tstep_fraction_for_attempt(attempt, total_attempts)
                    candidate = plan_func(list(waypoint), last_qpos=qpos0,
                                         constraint_pose=constraint_pose, approach_axis=approach_axis,
                                         approach_offset=approach_offset,
                                         tstep_fraction=fraction, near_contact=near_contact)
                    candidate["tstep_fraction"] = fraction
                    attempt_record = {
                        "stage": stage_label, "slice": i, "num_slices": num_slices,
                        "attempt": attempt, "tstep_fraction": fraction,
                        "near_contact": near_contact,
                        "status": candidate.get("status"),
                        "position_error": candidate.get("position_error"),
                        "rotation_error": candidate.get("rotation_error"),
                    }
                    if candidate.get("status") != "Success":
                        last_reason = candidate.get("fail_reason", "unknown")
                        attempt_record["fail_reason"] = last_reason
                        self.rollout_descent_slices.append(attempt_record)
                        continue
                    metrics = self._trajectory_path_metrics(arm_tag, candidate)
                    attempt_record["path_metrics"] = metrics
                    is_accepted, sort_key, last_reason = self._evaluate_descent_metrics(metrics)
                    attempt_record["accepted"] = is_accepted
                    if is_accepted:
                        accepted.append(sort_key + (candidate, attempt, attempt_record))
                    else:
                        attempt_record["reject_reason"] = last_reason
                        if log_move:
                            print(f"[descent-slice] {stage_label} slice {i}/{num_slices} attempt "
                                  f"{attempt} (tstep_fraction={fraction}): rejected: {last_reason}")
                    self.rollout_descent_slices.append(attempt_record)
                if not accepted:
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices} failed "
                              f"({total_attempts} attempts, none accepted; last={last_reason})")
                    return False, last_reason, qpos0, False
                # Lexicographic: shortest Cartesian path, then shortest joint-space path, then
                # smallest single-joint excursion (similar Cartesian paths can have very different
                # joint motion).
                accepted.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
                best_path_length, best_joint_path_length, best_max_joint_range, result, best_attempt, best_record = accepted[0]
                best_record["selected"] = True
                if log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded "
                          f"(attempt {best_attempt}, tstep_fraction={result['tstep_fraction']}, "
                          f"path_length={best_path_length:.3f}m, "
                          f"joint_path_length={best_joint_path_length:.3f}rad, "
                          f"{len(accepted)}/{total_attempts} attempts accepted)")
                qpos0 = self._roll_qpos_forward(arm_tag, qpos0, result)
                self._replay_planned_move(arm_tag, result)
                if not self.plan_success:
                    return False, self._last_fail_reason, qpos0, False
                # Check arrival before retention: the object settling onto the target surface early
                # changes the gripper-relative transform like a drop, but it is the intended outcome.
                # Stop descending (further descent could push the object into the surface) and
                # report placed=True so the caller skips the remaining moves.
                if self._object_near_placement_target():
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: object "
                              f"already at a placement that would pass check_success; done")
                    return True, None, qpos0, True
                retained, pos_drift, rot_drift = self._object_retained(
                    arm_tag, context=f"{stage_label}:slice{i}", return_drift=True)
                if not retained:
                    xy_error = self._placement_xy_error()
                    supported_contact = (
                        pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                        and self._object_near_support_height())
                    if (supported_contact
                            and np.all(np.abs(xy_error) < CONTACT_RELEASE_XY_TOLERANCE)):
                        self.rollout_descent_slices.append({
                            "stage": f"{stage_label}:contact-release",
                            "dx": float(xy_error[0]), "dy": float(xy_error[1]),
                            "correction_needed": False,
                        })
                        return True, None, qpos0, True
                    # Ambiguous contact requires bounded position drift, bounded rotation drift
                    # (otherwise an object that tipped loose without translating would pass on
                    # position alone) and the object near the support surface (a genuine drop shows
                    # the same 4-15 cm drift while still high above it).
                    if (pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                            and rot_drift <= OBJECT_RETENTION_ROTATION_TOLERANCE
                            and self._object_near_support_height()):
                        # Ambiguous: not a passing placement and outside strict retention, but the
                        # drift is far below any observed drop (30 cm-1.4 m) and consistent with early,
                        # imprecise contact. Keep descending so the remaining slices can still converge
                        # on the target.
                        if log_move:
                            print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: drift "
                                  f"{pos_drift:.3f}m looks like early contact, not a loss; continuing")
                    else:
                        self.plan_success = False
                        self._last_fail_reason = f"object_lost:{stage_label}:slice{i}"
                        return False, self._last_fail_reason, qpos0, False
                elif log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded")
            return True, None, qpos0, False

        def _local_landing_search_and_place(self, arm_tag):
            """Land the held object near the target instead of forcing place_actor's descent to
            one exact final pose, which often fails with FINETUNE_TRAJOPT_FAIL at the surface
            where the attached object's collision margin against the table is tightest.

            Searches a grid of XY offsets (within check_success's tolerance) at the staging
            release height max(LANDING_RELEASE_HEIGHTS). IK-feasible candidates are planned
            closest to the nominal target first, filtered with the same trajectory-safety checks
            as _plan_pose_with_descent_slices, and the shortest accepted path is replayed. From
            that verified pose the object is lowered through the remaining release heights with
            short descent slices, then the gripper opens and physics settles the final contact.
            Requires the object to be retained and within LANDING_SEARCH_TRIGGER_DISTANCE (XY)
            and LANDING_SEARCH_MAX_Z_DISTANCE (Z) of the target, since the candidate moves are
            short by design.

            Candidate object poses are converted to gripper poses with get_place_pose, which uses
            the actual grasp transform (actor pose vs. end-effector pose), so the result is
            correct for whichever grasp the episode used.

            Sets plan_success=False with a descriptive fail_reason if no candidate passes; the
            caller's checkpoint() records it like any other stage failure."""
            plan_func = self._arm_plan_func(arm_tag)
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            target = np.asarray(self.des_obj_pose, dtype=float)

            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            xy_distance = float(np.hypot(obj_pos[0] - target[0], obj_pos[1] - target[1]))
            z_distance = float(abs(obj_pos[2] - target[2]))
            retained = self._object_retained(arm_tag, context="landing-search:precondition")
            if not retained or xy_distance > LANDING_SEARCH_TRIGGER_DISTANCE or z_distance > LANDING_SEARCH_MAX_Z_DISTANCE:
                self.plan_success = False
                self._last_fail_reason = (f"landing_search_preconditions_not_met(retained={retained},"
                                          f"xy_distance={xy_distance:.3f}m,z_distance={z_distance:.3f}m)")
                if log_move:
                    print(f"[landing-search] preconditions not met: retained={retained} "
                          f"xy_distance={xy_distance:.3f}m (max {LANDING_SEARCH_TRIGGER_DISTANCE}m) "
                          f"z_distance={z_distance:.3f}m (max {LANDING_SEARCH_MAX_Z_DISTANCE}m)")
                return

            # Plan only to the highest release height (8 cm) as a staging pose, then descend
            # sequentially with the near-contact profile; direct 1-5 cm landing grids produced
            # hundreds of solver failures in validation sweeps.
            height_tiers = [[max(LANDING_RELEASE_HEIGHTS)]]
            for tier in height_tiers:
                tier_accepted = []  # (path_length, joint_path_length, max_joint_range, result, height, record)
                for height in tier:
                    candidates = []  # (dx, dy, obj_pose)
                    for dx in LANDING_XY_OFFSETS:
                        for dy in LANDING_XY_OFFSETS:
                            obj_pose = target.copy()
                            obj_pose[0] += dx
                            obj_pose[1] += dy
                            obj_pose[2] = target[2] + height
                            candidates.append((dx, dy, obj_pose))
                    # Sort by radius (closest to the nominal target first), then by angle, so ties
                    # at the same radius do not systematically favor negative dx (LANDING_XY_OFFSETS
                    # lists negatives first and the sort is stable).
                    candidates.sort(key=lambda c: (np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])))

                    gripper_poses = [
                        self.get_place_pose(self.target_obj, arm_tag, obj_pose, constrain="free", pre_dis=0.0)
                        for _, _, obj_pose in candidates
                    ]
                    ik_ok = list(check_ik(gripper_poses))
                    current_ee_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=float)
                    qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
                    qpos0 = np.array(qpos, dtype=np.float64, copy=True)

                    planned = 0
                    height_accepted = 0
                    for (dx, dy, _), gripper_pose, feasible in zip(candidates, gripper_poses, ik_ok):
                        record = {"stage": "landing-search", "height": height, "dx": dx, "dy": dy,
                                  "ik_feasible": bool(feasible)}
                        if not feasible:
                            self.rollout_descent_slices.append(record)
                            continue
                        if planned >= LANDING_MAX_CANDIDATES_PER_HEIGHT:
                            continue  # not attempted (not a rejection), so not recorded
                        planned += 1
                        segment_length = float(np.linalg.norm(np.asarray(gripper_pose[:3]) - current_ee_pose[:3]))
                        approach_offset = min(0.05, segment_length * 0.8) if segment_length > 1e-6 else 0.02
                        if height == max(LANDING_RELEASE_HEIGHTS):
                            # The staging target is collision-clear and is a transit, not a descent; the
                            # approach metric caused repeatable 2-2.5x path-length loops here.
                            candidate_result = plan_func(list(gripper_pose), last_qpos=qpos0)
                        else:
                            candidate_result = plan_func(
                                list(gripper_pose), last_qpos=qpos0, approach_axis=2,
                                approach_offset=approach_offset, tstep_fraction=0.4)
                        record["status"] = candidate_result.get("status")
                        if candidate_result.get("status") != "Success":
                            record["fail_reason"] = candidate_result.get("fail_reason", "unknown")
                            record["accepted"] = False
                            self.rollout_descent_slices.append(record)
                            continue
                        metrics = self._trajectory_path_metrics(arm_tag, candidate_result)
                        record["path_metrics"] = metrics
                        # The absolute joint caps were calibrated for ~4 cm descent slices
                        # (DESCENT_SLICE_SIZE); a landing candidate can be ~10 cm away and needs more
                        # joint travel even for a clean plan (unscaled caps accepted almost nothing).
                        # Scale them by segment length, as approach_offset is.
                        joint_scale = max(1.0, segment_length / DESCENT_SLICE_SIZE)
                        max_joint_endpoint_displacement = DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT * joint_scale
                        max_joint_path_length = DESCENT_MAX_JOINT_PATH_LENGTH * joint_scale
                        record["joint_scale"] = joint_scale
                        is_accepted, sort_key, reject_reason = self._evaluate_descent_metrics(
                            metrics, max_joint_endpoint_displacement, max_joint_path_length)
                        record["accepted"] = is_accepted
                        if is_accepted:
                            tier_accepted.append(sort_key + (candidate_result, height, dx, dy, record))
                            height_accepted += 1
                        else:
                            record["reject_reason"] = reject_reason
                        self.rollout_descent_slices.append(record)
                        if height_accepted >= LANDING_MIN_ACCEPTED_TO_STOP:
                            break
                    if log_move:
                        print(f"[landing-search] height={height:.3f}m: {height_accepted} accepted "
                              f"({sum(ik_ok)}/{len(candidates)} IK-feasible, {planned} planned)")

                if not tier_accepted:
                    continue

                tier_accepted.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
                best_path_length, best_joint_path_length, _, result, best_height, best_dx, best_dy, best_record = tier_accepted[0]
                best_record["selected"] = True
                if log_move:
                    print(f"[landing-search] tier {tier} succeeded: {len(tier_accepted)} candidates "
                          f"accepted across the tier, chose height={best_height:.3f}m "
                          f"path_length={best_path_length:.3f}m joint_path_length={best_joint_path_length:.3f}rad")
                self._replay_planned_move(arm_tag, result)
                if not self.plan_success:
                    return
                if not self._object_retained(arm_tag, context="landing-search"):
                    self.plan_success = False
                    self._last_fail_reason = "object_lost:landing-search"
                    return

                release_height = best_height
                descent_failure = None
                if best_height == max(LANDING_RELEASE_HEIGHTS):
                    # The collision-clear solve is a staging pose. From this
                    # verified state, make only short constrained downward moves.
                    for lower_height in sorted(
                            (h for h in LANDING_RELEASE_HEIGHTS if h < best_height), reverse=True):
                        lower_obj_pose = target.copy()
                        lower_obj_pose[:3] += np.array([best_dx, best_dy, lower_height])
                        lower_gripper_pose = self.get_place_pose(
                            self.target_obj, arm_tag, lower_obj_pose,
                            constrain="free", pre_dis=0.0)
                        live_qpos = (self.robot.left_entity.get_qpos() if str(arm_tag) == "left"
                                     else self.robot.right_entity.get_qpos())
                        ok, reason, _, placed = self._plan_pose_with_descent_slices(
                            arm_tag, lower_gripper_pose,
                            np.asarray(live_qpos, dtype=np.float64),
                            f"landing-descent:{lower_height:.3f}",
                            start_pose=self.get_arm_pose(arm_tag),
                            retries=PLACEMENT_SEARCH_RETRIES, approach_axis=2,
                            near_contact=True)
                        if not ok:
                            descent_failure = reason
                            if not self.plan_success:
                                return
                            break
                        release_height = lower_height
                        if placed:
                            break

                self.rollout_descent_slices.append({
                    "stage": "landing-release", "height": release_height,
                    "dx": best_dx, "dy": best_dy,
                    "descent_failure": descent_failure,
                })
                self.move(self.open_gripper(arm_tag))
                return

            self.plan_success = False
            self._last_fail_reason = "no_valid_landing_candidate"
            if log_move:
                print(f"[landing-search] failed: no candidate at any height "
                      f"{LANDING_RELEASE_HEIGHTS} planned and passed the filter")

        def _execute_actions_via_plan_and_replay(self, arm_tag, actions, retries=1):
            """Execute each 'move' Action via deterministic descent slicing (see
            _plan_pose_with_descent_slices), which plans and replays incrementally and checks
            grasp retention after every slice; non-move actions (gripper open/close) go through
            self.move.

            retries: place_actor's failures are FINETUNE_TRAJOPT_FAIL, a marginal trajopt
            feasibility problem rather than unreachability, and CuRobo's trajopt seeding is
            randomized, so the same setup can succeed on a retry. Each retry re-attempts only the
            failed slice, not the whole chain."""
            if not self.plan_success:
                return
            start_pose = list(self.get_arm_pose(arm_tag))
            move_idx = 0
            already_placed = False
            for a in actions:
                if a.action != "move":
                    self.move((arm_tag, [a]))
                    if not self.plan_success:
                        return
                    continue
                if already_placed:
                    # An earlier move already reached a placement that passes check_success
                    # (placed=True): skip the remaining descent moves but still run the non-move
                    # actions (gripper open) that follow.
                    continue
                move_idx += 1
                qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
                ok, fail_reason, _, placed = self._plan_pose_with_descent_slices(
                    arm_tag, a.target_pose, np.array(qpos, dtype=np.float64, copy=True),
                    f"place_actor:move{move_idx}", start_pose=start_pose, retries=retries,
                    constraint_pose=a.args.get("constraint_pose"),
                    approach_axis=a.args.get("approach_axis"))
                if not ok:
                    self.plan_success = False
                    self._last_fail_reason = fail_reason
                    return
                start_pose = list(self.get_arm_pose(arm_tag))
                already_placed = placed

        def _plan_and_replay_pose(self, arm_tag, pose):
            """Plan a single pose from the current live qpos and immediately replay it via
            _replay_planned_move, instead of self.move(self.move_to_pose(...)). This exposes the
            CuRobo fail_reason via self._last_fail_reason for the failure recorder and uses the
            same execution primitive as the rest of this file."""
            if not self.plan_success:
                return
            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            ok, fail_reason, _, trajectories = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pose, np.array(start_qpos, dtype=np.float64, copy=True), "pose")
            if not ok:
                self.plan_success = False
                self._last_fail_reason = fail_reason
                return
            self._replay_planned_sequence(arm_tag, trajectories)

        def _grasp_via_cached_trajectories(self, arm_tag, candidate_plan, gripper_pos=0.0):
            """Replay the dry-run-verified pre-grasp and grasp trajectories.

            As in grasp_actor_from_table, the table is disabled after reaching pre-grasp and
            before replaying the final cached approach. Falls back to live planning only when
            the candidate has no cached trajectories.
            """
            pre_traj = candidate_plan.get("pre_grasp_trajectory")
            grasp_traj = candidate_plan.get("grasp_trajectory")
            if pre_traj is None or grasp_traj is None:
                return self.grasp_actor_from_table(
                    self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1,
                    contact_point_id=candidate_plan["contact_point_id"])
            self._replay_planned_sequence(arm_tag, pre_traj)
            self.enable_table(enable=False)
            self._replay_planned_sequence(arm_tag, grasp_traj)
            self.move(self.close_gripper(arm_tag, pos=gripper_pos))

        def grasp_actor_from_table(self, actor, arm_tag, pre_grasp_dis=0.1, grasp_dis=0,
                                   gripper_pos=0.0, contact_point_id=None):
            # Same as Office_base_task.grasp_actor_from_table, but strips the orientation-hold
            # constraint_pose from the approach actions (DROP_GRASP_ORIENTATION_CONSTRAINT).
            if not self.DROP_GRASP_ORIENTATION_CONSTRAINT:
                return super().grasp_actor_from_table(
                    actor, arm_tag, pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis,
                    gripper_pos=gripper_pos, contact_point_id=contact_point_id)
            _, actions = self.grasp_actor(
                actor=actor, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis, gripper_pos=gripper_pos, contact_point_id=contact_point_id)
            if not actions:            # grasp_actor bails (e.g. an earlier move set
                return                 # plan_success=False) -> avoid actions[0] IndexError
            for a in actions:
                if getattr(a, "args", None):
                    a.args.pop("constraint_pose", None)  # -> unconstrained plan in move()
            self.move((arm_tag, [actions[0]]))
            self.enable_table(enable=False)
            self.move((arm_tag, actions[1:]))

        def _arm_plan_func(self, arm_tag):
            return self.robot.left_plan_path if str(arm_tag) == "left" else self.robot.right_plan_path

        def _arm_active_joint_indices(self, arm_tag):
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            return [planner.all_joints.index(name) for name in planner.active_joints_name if name in planner.all_joints]

        def _roll_qpos_forward(self, arm_tag, qpos, plan_result):
            qpos_next = np.array(qpos, dtype=np.float64, copy=True)
            active_indices = self._arm_active_joint_indices(arm_tag)
            qpos_next[active_indices] = np.asarray(plan_result["position"][-1], dtype=np.float64)
            return qpos_next

        def _local_waypoint_candidates(self, arm_tag, start_pose, target_pose, attempts=LOCAL_WAYPOINT_ATTEMPTS):
            """Bridge candidates near the current/start pose for any Cartesian move."""
            try:
                attempts = max(0, int(attempts))
            except (TypeError, ValueError):
                attempts = LOCAL_WAYPOINT_ATTEMPTS
            if attempts == 0:
                return []
            start = np.asarray(start_pose, dtype=float)
            target = np.asarray(target_pose, dtype=float)
            side = 1.0 if str(arm_tag) == "right" else -1.0
            specs = [
                ("offset", 0.00, 0.00, 0.04, 0.00),
                ("offset", 0.04, 0.00, 0.04, 0.00),
                ("offset", 0.08, 0.00, 0.06, 0.00),
                ("offset", 0.00, -0.04, 0.06, 0.00),
                ("offset", 0.04, -0.04, 0.08, 0.00),
                ("offset", 0.08, -0.04, 0.08, 0.00),
                ("offset", 0.00, 0.04, 0.06, 0.00),
                ("blend", 0.00, 0.00, 0.04, 0.35),
                ("blend", 0.04, 0.00, 0.06, 0.35),
                ("blend", 0.00, -0.04, 0.08, 0.35),
                ("offset", 0.10, 0.00, 0.10, 0.00),
                ("offset", 0.10, -0.06, 0.10, 0.00),
                # A larger pure-vertical lift (no lateral offset; the direction that resolved
                # the escape/pre_beside_box INVALID_START_STATE_WORLD_COLLISION failures) and a
                # later blend fraction. Cheap because infeasible bridge candidates are filtered
                # by IK before any trajopt call (see _plan_pose_with_local_waypoint_retry).
                ("offset", 0.00, 0.00, 0.10, 0.00),
                ("blend", 0.00, 0.00, 0.00, 0.50),
            ]
            candidates = []
            for mode, dx_side, dy, dz, blend_t in specs[:attempts]:
                pose = self._blend_pose(start, target, blend_t) if mode == "blend" else list(start)
                pose[0] = float(np.clip(pose[0] + side * dx_side, -REACH_X_LIMIT, REACH_X_LIMIT))
                pose[1] = float(np.clip(pose[1] + dy, -0.33, 0.30))
                pose[2] = float(np.clip(pose[2] + dz, 0.78, 1.30))
                candidates.append(pose)
            return candidates

        def _plan_pose_with_shrinking_waypoint(self, arm_tag, target_pose, qpos, stage_label,
                                              start_pose, constraint_pose=None, approach_axis=None,
                                              relax_orientation=False,
                                              max_iterations=LOCAL_WAYPOINT_ATTEMPTS,
                                              min_distance=WAYPOINT_SHRINK_MIN_DISTANCE):
            """Adaptive waypoint-shrink retry, used when a direct plan to target_pose has
            already failed.

            CuRobo's MotionGenResult sets position_error/rotation_error even on
            FINETUNE_TRAJOPT_FAIL (see motion_gen.py's _plan_from_solve_state), so a failed
            attempt shows how far off the optimizer landed. Each iteration either (a) retries
            the original target from wherever the previous iteration landed, or (b) if that
            fails, shrinks the target toward the current pose and retries the smaller hop. The
            shrink is sized from position_error when available (back off by about the residual
            gap plus a margin, so a near miss shrinks a little and a wild miss a lot), else a
            plain 50% halving (e.g. IK_FAIL, where trajopt never ran). Shrinking toward the
            current pose, which is reachable, converges toward something plannable. Stops on
            success at the real target, after max_iterations, or once the remaining hop is
            shorter than min_distance. CuRobo's failed trajectory is never replayed (its path
            was not verified collision-free); only its error magnitude sizes the next plan."""
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            plan_func = self._arm_plan_func(arm_tag)
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            current_pose = np.asarray(start_pose, dtype=float)
            original_target = np.asarray(target_pose, dtype=float)
            goal = original_target.copy()
            trajectories = []
            last_reason = "unknown"
            for iteration in range(1, max(1, int(max_iterations)) + 1):
                result = plan_func(list(goal), last_qpos=qpos0, constraint_pose=constraint_pose,
                                   approach_axis=approach_axis, relax_orientation=relax_orientation)
                reached_original = bool(np.allclose(goal[:3], original_target[:3], atol=1e-6))
                if result.get("status") == "Success":
                    qpos0 = self._roll_qpos_forward(arm_tag, qpos0, result)
                    label = stage_label if reached_original else f"{stage_label}:shrink{iteration}"
                    trajectories.append((label, result))
                    if reached_original:
                        return True, None, qpos0, trajectories
                    if log_move:
                        print(f"[waypoint-shrink] {stage_label} iter {iteration}: reached shrunk "
                              f"intermediate, retrying real target from there")
                    current_pose = goal.copy()
                    goal = original_target.copy()
                    continue
                last_reason = result.get("fail_reason", "unknown")
                remaining_dist = float(np.linalg.norm(goal[:3] - current_pose[:3]))
                if remaining_dist < min_distance:
                    if log_move:
                        print(f"[waypoint-shrink] {stage_label} iter {iteration} failed ({last_reason}); "
                              f"remaining distance {remaining_dist:.3f}m < {min_distance}m, giving up")
                    break
                position_error = result.get("position_error")
                if position_error is not None and 0 < position_error < remaining_dist:
                    # Back off by about how far short CuRobo's best attempt landed (2x margin, to
                    # avoid retrying at the same edge of infeasibility).
                    target_remaining = max(min_distance, remaining_dist - 2.0 * position_error)
                    shrink_fraction = target_remaining / remaining_dist
                else:
                    shrink_fraction = 0.5
                goal[:3] = current_pose[:3] + (goal[:3] - current_pose[:3]) * shrink_fraction
                if log_move:
                    print(f"[waypoint-shrink] {stage_label} iter {iteration} failed ({last_reason}, "
                          f"position_error={position_error}); shrinking target toward current pose "
                          f"(fraction={shrink_fraction:.2f}) -> new distance "
                          f"{float(np.linalg.norm(goal[:3] - current_pose[:3])):.3f}m")
            return False, last_reason, qpos0, trajectories

        def _plan_pose_with_local_waypoint_retry(self, arm_tag, target_pose, qpos, stage_label,
                                                start_pose=None, retries=1, constraint_pose=None,
                                                approach_axis=None, local_attempts=LOCAL_WAYPOINT_ATTEMPTS,
                                                relax_orientation=False):
            """Plan one target; if direct planning fails, try the adaptive
            waypoint-shrink retry (see _plan_pose_with_shrinking_waypoint).

            relax_orientation=True: position-only goal (PoseCostMetric.reach_partial_pose, see
            planner.py's plan_path) for transit moves where only the target position matters. A
            check_ik_batch(relax_orientation=True) sweep showed this makes otherwise-infeasible
            combos reachable (e.g. lift_above_box at low clearance_z)."""
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            start_pose = list(self.get_arm_pose(arm_tag)) if start_pose is None else list(start_pose)
            last_reason = "unknown"
            plan_func = self._arm_plan_func(arm_tag)
            for _ in range(max(1, int(retries))):
                result = plan_func(target_pose, last_qpos=qpos0,
                                   constraint_pose=constraint_pose, approach_axis=approach_axis,
                                   relax_orientation=relax_orientation)
                if result.get("status") == "Success":
                    return True, None, self._roll_qpos_forward(arm_tag, qpos0, result), [(stage_label, result)]
                last_reason = result.get("fail_reason", "unknown")
            if local_attempts <= 0:
                return False, last_reason, qpos0, []
            return self._plan_pose_with_shrinking_waypoint(
                arm_tag, target_pose, qpos0, stage_label, start_pose,
                constraint_pose=constraint_pose, approach_axis=approach_axis,
                relax_orientation=relax_orientation, max_iterations=local_attempts)

        def _replay_planned_sequence(self, arm_tag, trajectories):
            """Replay one planned trajectory or a list of labeled trajectories."""
            if trajectories is None:
                return
            if isinstance(trajectories, dict):
                trajectories = [trajectories]
            for item in trajectories:
                traj = item[1] if isinstance(item, tuple) else item
                self._replay_planned_move(arm_tag, traj)
                if not self.plan_success:
                    return

        def _plan_pose_sequence(self, arm_tag, poses, start_qpos, stage_labels=None):
            """Returns (success, failed_stage_label, fail_reason, final_qpos).
            failed_stage_label/fail_reason are None on success; failed_stage_label is
            the label (from stage_labels, positionally aligned with poses) of the
            first pose in the chain that failed to plan, and fail_reason is CuRobo's
            own MotionGenStatus string for that failure (e.g. IK_FAIL vs a
            collision-specific status). final_qpos is the qpos reached by the last
            successfully-planned pose (== start_qpos if the very first pose failed),
            so a caller can resume planning a follow-on sub-chain from where this one
            left off (e.g. after toggling world state between segments)."""
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            start_pose = list(self.get_arm_pose(arm_tag))
            for idx, pose in enumerate(poses):
                label = stage_labels[idx] if stage_labels else f"pose{idx}"
                ok, fail_reason, qpos, _ = self._plan_pose_with_local_waypoint_retry(
                    arm_tag, pose, qpos, label, start_pose=start_pose)
                if not ok:
                    return False, label, fail_reason, qpos
                start_pose = pose
            return True, None, None, qpos

        def _candidate_specs(self, arm_tag, cp_ids):
            for cp_id in cp_ids:
                grasp_pre_pose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.1)
                grasp_pose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.0)
                if grasp_pre_pose is None or grasp_pose is None:
                    continue
                lift_pose = list(grasp_pose)
                lift_pose[2] += GRASP_LIFT_HEIGHT
                occluder_present = self.spawn_occluder and getattr(self, "occluder", None) is not None
                gaps = SIDE_WAYPOINT_GAPS if occluder_present else (SIDE_WAYPOINT_GAPS[1],)
                z_lifts = SIDE_WAYPOINT_Z_LIFTS if occluder_present else (0.0,)
                orients = WAYPOINT_ORIENTATIONS if occluder_present else ("grasp_aligned",)
                y_offsets = WAYPOINT_Y_OFFSETS if occluder_present else (0.0,)
                for gap in gaps:
                    x_side = self._box_side_x(arm_tag, gap=gap)
                    for z_lift in z_lifts:
                        for orient in orients:
                            for y_offset in y_offsets:
                                grasp_waypoint = self._around_box_waypoint(
                                    arm_tag, grasp_pre_pose, gap=gap, z_lift=z_lift,
                                    orient=orient, y_offset=y_offset)
                                for clearance_z in PLACE_CLEARANCE_ZS:
                                    placement_subgoals = self._backward_subgoal_poses(
                                        arm_tag,
                                        x_side=x_side,
                                        clearance_z=clearance_z,
                                        lift_pose=lift_pose,
                                    )
                                    yield {
                                        "contact_point_id": cp_id,
                                        "grasp_waypoint": grasp_waypoint,
                                        "lift_pose": lift_pose,
                                        "placement_subgoals": placement_subgoals,
                                        "gap": gap,
                                        "y_offset": y_offset,
                                        "z_lift": z_lift,
                                        "orient": orient,
                                        "clearance_z": clearance_z,
                                    }

        def _plan_grasp_side(self, arm_tag, candidate, cache):
            """Waypoint -> pre_grasp -> grasp -> lift, cached per (cp_id, gap, z_lift, orient,
            y_offset). The result does not depend on clearance_z, but _candidate_specs yields one
            candidate per clearance_z with the same waypoint/grasp, and get_grasp_pose runs a
            10-rotation CuRobo batch plan, so caching avoids redoing that work 3x.
            Returns (ok, failed_stage, fail_reason, qpos, lift_pose, trajectories), where
            trajectories is {"waypoint"/"pre_grasp"/"grasp": <raw plan_path result>}, so real
            execution can replay the exact verified trajectories instead of re-planning (a
            re-plan to the same Cartesian pose can converge to a different joint configuration;
            see _replay_planned_move).
            pre_grasp/grasp trajectories are planned chained (pre_grasp's landed qpos is grasp's
            start), matching physical execution, unlike the get_grasp_pose feasibility checks
            below, which mirror choose_grasp_pose's unchained check."""
            key = (candidate["contact_point_id"], candidate["gap"], candidate["z_lift"],
                  candidate["orient"], candidate["y_offset"])
            if key in cache:
                return cache[key]

            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            trajectories = {}
            if candidate["grasp_waypoint"] is not None:
                ok, fail_reason, qpos, waypoint_trajectory = self._plan_pose_with_local_waypoint_retry(
                    arm_tag, candidate["grasp_waypoint"], qpos, "waypoint")
                if not ok:
                    result = (False, "waypoint", fail_reason, None, None, None)
                    cache[key] = result
                    return result
                trajectories["waypoint"] = waypoint_trajectory
            grasp_side_start_qpos = qpos  # pre_grasp/grasp poses are both determined from here (unchained)

            # Validate with the same grasp-pose selection real execution uses (get_grasp_pose ->
            # choose_best_pose's rotation search and batch-plan check) instead of the geometric
            # approximation, so a passing candidate matches what grasp_actor_from_table()
            # executes. choose_grasp_pose tests pre_grasp/grasp independently from the same
            # starting qpos, which is mirrored here.
            cp_id = candidate["contact_point_id"]
            pre_grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                                 pre_dis=0.1, last_qpos=grasp_side_start_qpos)
            if pre_grasp_pose is None:
                result = (False, "pre_grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result
            grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                             pre_dis=0.0, last_qpos=grasp_side_start_qpos)
            if grasp_pose is None:
                result = (False, "grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result

            # Plan and capture the trajectories to replay, chained (grasp starts where pre_grasp
            # lands), matching physical execution.
            ok, fail_reason, qpos, pre_grasp_trajectory = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pre_grasp_pose, qpos, "pre_grasp", start_pose=candidate.get("grasp_waypoint"))
            if not ok:
                result = (False, "pre_grasp", fail_reason, None, None, None)
                cache[key] = result
                return result
            trajectories["pre_grasp"] = pre_grasp_trajectory

            ok, fail_reason, qpos, grasp_trajectory = self._plan_pose_with_local_waypoint_retry(
                arm_tag, grasp_pose, qpos, "grasp", start_pose=pre_grasp_pose)
            if not ok:
                result = (False, "grasp", fail_reason, None, None, None)
                cache[key] = result
                return result
            trajectories["grasp"] = grasp_trajectory

            lift_pose = list(grasp_pose)
            lift_pose[2] += GRASP_LIFT_HEIGHT
            ok, failed_stage, fail_reason, qpos = self._plan_pose_sequence(
                arm_tag, [lift_pose], qpos, stage_labels=["lift"])
            if not ok:
                result = (ok, failed_stage, fail_reason, None, None, None)
                cache[key] = result
                return result

            result = (True, None, None, qpos, lift_pose, trajectories)
            cache[key] = result
            return result

        def _plan_candidate(self, arm_tag, candidate, grasp_side_cache):
            ok, failed_stage, fail_reason, qpos, lift_pose, trajectories = self._plan_grasp_side(
                arm_tag, candidate, grasp_side_cache)
            if not ok:
                return ok, failed_stage, fail_reason
            # Carried to play_once so real execution can replay these exact verified
            # trajectories for the waypoint/pre_grasp/grasp moves instead of
            # independently re-planning to the same poses (see _plan_grasp_side's
            # docstring for why that matters).
            candidate["grasp_waypoint_trajectory"] = trajectories.get("waypoint")
            candidate["pre_grasp_trajectory"] = trajectories.get("pre_grasp")
            candidate["grasp_trajectory"] = trajectories.get("grasp")

            # Placement subgoal orientation derives from the grasp orientation. Rebuild it from
            # the real lift_pose above (candidate["placement_subgoals"] came from the geometric
            # approximation in _candidate_specs) and write it back so play_once uses it too.
            x_side = self._box_side_x(arm_tag, gap=candidate["gap"])
            placement_subgoals = self._backward_subgoal_poses(
                arm_tag, x_side=x_side, clearance_z=candidate["clearance_z"], lift_pose=lift_pose)
            candidate["placement_subgoals"] = placement_subgoals
            candidate["lift_pose"] = lift_pose

            # Real execution calls attach_object before planning the placement subgoals; the
            # check above does not, so it can certify a pose blocked by the held object's own
            # collision volume (beside_box was observed to flip from feasible to infeasible once
            # attached). The held object's pose is approximated as lift_pose.
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            object_dict = {
                "name": self.target_obj.get_name(),
                "pose": list(lift_pose),
                "file_path": f"{os.environ['ASSETS_ROOT']}/objects/{self.target_model}/collision/base{self.target_id}.glb",
                "scale": self.target_obj.scale,
            }
            placement_poses = [pose for _, pose in placement_subgoals]
            placement_labels = [name for name, _ in placement_subgoals]
            try:
                planner.attach_object(object_dict, qpos, arms_tag=str(arm_tag))
                ok2, failed_stage2, fail_reason2, _ = self._plan_pose_sequence(
                    arm_tag, placement_poses, qpos, stage_labels=placement_labels)
            finally:
                planner.detach_object()
            return ok2, failed_stage2, fail_reason2

        def _select_pick_place_candidate(self, arm_tag, exclude_cp_ids=None):
            # Selected-candidate metadata for the failure recorder (see play_once): whether the
            # executed candidate was fully dry-run-verified or a fallback, and for a fallback how
            # far it got and the failure breakdown, so records.jsonl separates "verified plan
            # failed in execution" from "no workable plan found".
            #
            # exclude_cp_ids: contact points that already failed a real grasp_verify check (see
            # play_once), skipped so a retry picks a different contact point.
            cp_ids = self._rank_side_grasp_ids(self.target_obj, arm_tag)
            if exclude_cp_ids:
                cp_ids = [cp for cp in cp_ids if cp not in exclude_cp_ids]
            if not cp_ids:
                self.rollout_candidate_info = {"verified": False, "reason": "no_ranked_contact_points"}
                return {
                    "contact_point_id": None,
                    "grasp_waypoint": None,
                    "placement_subgoals": self._backward_subgoal_poses(
                        arm_tag,
                        x_side=self._box_side_x(arm_tag),
                        clearance_z=PLACE_CLEARANCE_ZS[1],
                        lift_pose=list(self.get_arm_pose(arm_tag)),
                    ),
                }

            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            fallback = None
            fallback_depth = -1
            fallback_stage = None
            tried = 0
            failure_breakdown = {}
            grasp_side_cache = {}
            for candidate in self._candidate_specs(arm_tag, cp_ids):
                tried += 1
                ok, failed_stage, fail_reason = self._plan_candidate(arm_tag, candidate, grasp_side_cache)
                if ok:
                    if log_move:
                        print(
                            "[candidate] selected "
                            f"cp_id={candidate['contact_point_id']} gap={candidate['gap']:.2f} "
                            f"z_lift={candidate['z_lift']:.2f} orient={candidate['orient']} "
                            f"y_offset={candidate['y_offset']:.2f} "
                            f"clearance_z={candidate['clearance_z']:.2f} (tried {tried})"
                        )
                    self.rollout_candidate_info = {
                        "verified": True, "contact_point_id": candidate["contact_point_id"], "tried": tried,
                    }
                    return candidate
                # Fallback = the candidate that progressed furthest through the chain before
                # failing, not the first one generated (a later cp_id can be more reachable than
                # the top-ranked one).
                depth = STAGE_ORDER.index(failed_stage) if failed_stage in STAGE_ORDER else -1
                if depth > fallback_depth:
                    fallback, fallback_depth, fallback_stage = candidate, depth, failed_stage
                reasons = failure_breakdown.setdefault(failed_stage, {})
                reasons[fail_reason] = reasons.get(fail_reason, 0) + 1

            if fallback is not None and log_move:
                print(
                    f"[candidate] no fully planned candidate found ({tried} tried); "
                    f"using deepest-progress fallback cp_id={fallback['contact_point_id']} "
                    f"(died at '{fallback_stage}', depth {fallback_depth}); "
                    f"failure breakdown (stage -> {{reason: count}}): {failure_breakdown}"
                )
            self.rollout_candidate_info = {
                "verified": False,
                "contact_point_id": fallback["contact_point_id"] if fallback else None,
                "tried": tried,
                "dry_run_fallback_stage": fallback_stage,
                "dry_run_failure_breakdown": failure_breakdown,
            }
            return fallback

    return OccluderTask


def run(args):
    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    dist_lo_cm, dist_hi_cm = (float(x) for x in args.occluding_object_distance.split(","))
    if dist_lo_cm > dist_hi_cm:
        dist_lo_cm, dist_hi_cm = dist_hi_cm, dist_lo_cm
    dist_range_label = args.occluding_object_distance
    clutter_densities = [int(d) for d in args.clutter_densities.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> {success,fail}/video/episode{N}.mp4)

    env = make_occluder_task()()
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  "
          f"occluding_object_distance={dist_lo_cm}-{dist_hi_cm}cm (per-seed random draw)  "
          f"clutter_densities={clutter_densities}  rollout={rollout}  camera={CAMERA}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: same scene, NO occluder ---
            env.spawn_occluder = False
            try:
                env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed ({type(e).__name__}: {e}); skipping seed")
                safe_close()
                continue
            safe_close()
            if full_px <= 0:
                print(f"[seed {seed}] full_target_px=0 on clean scene; skipping seed.")
                continue

            # One occluder distance per seed, drawn uniformly from the configured cm range and
            # converted to meters. Seeded by the seed alone so the draw is reproducible across
            # reruns of the same seed range.
            off = float(np.random.default_rng(int(seed)).uniform(dist_lo_cm, dist_hi_cm)) / 100.0
            # per-build coin flip: drop the occluder with prob no_occluder_prob
            show = bool(np.random.default_rng(int(seed) * 1000 + int(round(off * 100))).random()
                        >= args.no_occluder_prob)
            # Pad clearance depends on both objects' yaw (off is an edge-to-edge gap), which is
            # only known once load_actors builds the scene, so it is checked via
            # env.occluder_pad_clearance_ok right after the first build below and the whole
            # seed is rejected there.
            env.spawn_occluder = show
            env.occluder_offset = off
            pad_reject = False
            for cd in clutter_densities:
                try:
                    env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed,
                                               dr_measure(cd)))
                    if show and not env.occluder_pad_clearance_ok:
                        print(f"[seed {seed}] occluder@off={off} lands too close to the pad "
                              f"(< {OCC_PAD_MIN_DIST:.3f}m); rejecting this build.")
                        pad_reject = True
                        safe_close()
                        break
                    target = _resolve_target(env)
                    pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                    res = env.measure_target_visibility(target, camera_name=CAMERA, denominator=full_px)
                    if args.save_images:
                        tag = "occ" if show else "noocc"
                        save_overlay(
                            env, res["mask"],
                            img_dir / f"seed{seed:04d}_off{off:.2f}_cd{cd:02d}_{tag}.png",
                            f"seed {seed} off={off:.2f} {tag} clut={cd}  "
                            f"vis_px={res['visible_pixel_count']} full={full_px} "
                            f"frac={res['visible_fraction']:.3f} {res['bucket']} pose_match={pose_ok}",
                        )
                except Exception as e:
                    print(f"[seed {seed} off{off:.2f} cd{cd}] build failed ({type(e).__name__}: {e}); skipping")
                    safe_close()
                    continue
                safe_close()

                rec = {"seed": int(seed), "offset": float(off),
                       "occ_distance_range_cm": dist_range_label, "full_px": int(full_px),
                       "visible_px": int(res["visible_pixel_count"]),
                       "visible_fraction": float(res["visible_fraction"]),
                       "bucket": res["bucket"], "in_fov": bool(res["in_fov"]),
                       "pose_match": pose_ok, "occluder_shown": show,
                       "clutter_density": int(cd)}
                # --- expert curobo rollout on the same scene (video + success) ---
                # env.spawn_occluder / occluder_offset persist, so the rollout
                # build reproduces the same occluder placement as the measurement.
                if rollout:
                    rollout_result = run_rollout(env, "put_mouse_on_pad", args.base_config, seed,
                                                 dr_measure(cd), out_dir, ep_counter)
                    success = bool(rollout_result["success"])
                    rec["rollout_success"] = success
                    rec["attached_trajectory_slowdown"] = ATTACHED_TRAJECTORY_SLOWDOWN
                    rec["rollout_ep"] = ep_counter
                    artifact_info = rollout_result.get("artifact_info") or {}
                    rec["rollout_bucket"] = artifact_info.get("bucket", "success" if success else "fail")
                    rec["rollout_video"] = artifact_info.get(
                        "video_relpath", f"{rec['rollout_bucket']}/video/episode{ep_counter}.mp4"
                    )
                    rec["rollout_data"] = artifact_info.get(
                        "hdf5_relpath", f"{rec['rollout_bucket']}/data/episode{ep_counter}.hdf5"
                    )
                    # Structured failure/candidate metadata set by play_once (see checkpoint() and
                    # _select_pick_place_candidate); plain attributes on the reused env, so they
                    # survive run_rollout's close_env(). None on success.
                    rec["rollout_failure_stage"] = getattr(env, "rollout_failure_stage", None)
                    rec["rollout_failure_reason"] = getattr(env, "rollout_failure_reason", None)
                    rec["rollout_candidate_info"] = getattr(env, "rollout_candidate_info", None)
                    rec["rollout_grasp_attempts"] = getattr(env, "rollout_grasp_attempts", None)
                    rec["rollout_retention_checks"] = getattr(env, "rollout_retention_checks", None)
                    rec["rollout_descent_slices"] = getattr(env, "rollout_descent_slices", None)
                    print(f"    seed {seed} off={off:.2f} cd={cd} {res['bucket']}: "
                          f"rollout {'SUCCESS' if success else 'FAIL'} -> "
                          f"{rec['rollout_bucket']}/episode{ep_counter}")
                    ep_counter += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            if pad_reject:
                continue
            print(f"[{si+1}/{len(seeds)}] seed {seed}: off={off:.3f}m full_px={full_px} done")

    safe_close()
    print(f"\nsweep complete -> {jsonl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--occluding-object-distance", default="20,20",
                    help="occluder edge-to-edge gap range 'lo,hi' in cm in front of the "
                         "target (true closest-surface distance between the two objects' "
                         "yaw-rotated footprints, not center-to-center); each seed "
                         "independently draws a random gap uniformly from [lo, hi] "
                         "(pass e.g. 20,20 for a single fixed distance)")
    ap.add_argument("--clutter-densities", default="0",
                    help="table clutter density/densities to sweep in the measurement scene, "
                         "comma-separated (0=off). e.g. 0,8,15")
    ap.add_argument("--group-by", default="occ_distance_range_cm",
                    choices=["occ_distance_range_cm", "offset", "clutter_density"],
                    help="analysis grouping variable for the bucket/histogram figures. "
                         "occ_distance_range_cm (default) groups by the configured "
                         "--occluding-object-distance range as a whole; since each seed "
                         "draws its own offset from that range, grouping by exact "
                         "'offset' gives very small groups.")
    ap.add_argument("--no-occluder-prob", type=float, default=0.2,
                    help="probability the milk-box occluder is NOT spawned for a build (default 0.2)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/phase2_occluder")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (writes to <out-dir>_rollout, "
                         "saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        run(args)
    group_label = {"occ_distance_range_cm": "occluder edge-to-edge gap range (cm)",
                   "offset": "occluder edge-to-edge gap (m)",
                   "clutter_density": "table clutter density"}[args.group_by]
    out_dir = effective_out_dir(args)
    analyze_kwargs = dict(group_key=args.group_by, group_label=group_label,
                          suptitle="Visibility with one milk-box occluder (countertop)",
                          bar_title=f"Bucket proportions vs {group_label}")
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)


if __name__ == "__main__":
    main()
