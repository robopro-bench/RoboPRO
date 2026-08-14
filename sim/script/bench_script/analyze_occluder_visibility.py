"""
Phase 2 part 1 (#35): deterministic single-occluder spawn + visibility distribution.

Places the target (001_bottle id 9) in the upper third of the reachable region (in
front of the back furniture) and spawns ONE tall milk-box occluder (038_milk-box id 2,
scale 1.0) in a narrow band just in front (-y, robot/camera side) of the target -- no
clutter, no binary search. Then measures the t=0 countertop `visible_fraction`
distribution across seeds, reusing the Phase 1 harness (measurement + plotting).

visible_fraction = visible_target_px(with occluder) / full_target_px(no occluder),
same seed (target pose is fixed before the occluder is added).

USAGE (from the benchmark folder):
    cd benchmark
    source set_env.sh
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
# reuse the Phase 1 harness verbatim
from analyze_natural_visibility import (build_cfg, DR_CLEAN, save_overlay, analyze, _resolve_target,
                                        CAMERA, run_rollout, analyze_rollout, effective_out_dir)

# target object: 001_bottle id 9 (8-way grasp ring -> side-reaches around the occluder).
# Swap TARGET_MODEL/TARGET_ID back to 047_mouse id 0 for the stock top-down mouse.
TARGET_MODEL = "001_bottle"
TARGET_ID = 9
# target spawn: back half of the table (back furniture removed). Capped at y=0.20 so it
# stays 0.15 m off the back edge (table y in [-0.35, 0.35]); x kept to [-0.15, 0.15] so it
# stays 0.45 m off each side edge (table x in [-0.6, 0.6]) -> leaves room for the around-
# the-box side waypoint and keeps target + waypoint inside the arm's reach envelope.
TARGET_XLIM = (-0.15, 0.15)
TARGET_YLIM = (0.0, 0.20)
PAD_XY = (0.0, -0.28)          # destination pad parked at the front, out of the occluder zone
OCCLUDER_QPOS = [0.66, 0.66, -0.25, -0.25]   # same upright carton orientation as the stock milk-box

# Two-step (around-the-box) planning: curobo won't reliably route around a tall obstacle
# between start and goal, so before the grasp (and before the place) we send the gripper
# to a waypoint beside the occluder on the arm's own side, then reach in. The waypoint's
# x-offset from the box CENTRE = OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP, i.e. SIDE_WAYPOINT
# _GAP is the clearance from the box EDGE to the gripper (not from centre). If the waypoint
# would leave the reachable x-range we flip to the other (table-centre) side. All tunable.
OCC_HALF_FOOTPRINT = 0.08      # milk-box base half-diagonal (~0.11 x 0.122, incl. yaw)
SIDE_WAYPOINT_GAP = 0.24       # clearance from the box EDGE to the waypoint (gripper)
REACH_X_LIMIT = 0.5
GRASP_CANDIDATE_LIMIT = 4
# The pre_beside_box move (right after lift+attach_object+enable_table) was found
# empirically to fail with MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION in
# 100% of its failures (5/5 in a 24-seed sweep) -- meaning the STARTING qpos for
# that plan is already colliding with the just-re-enabled table, before any target
# pose even matters. A 0.1m lift isn't always enough clearance for the held
# object's true collision volume (voxelized via attach_object) to clear the table
# once its collision is turned back on. Bumping the post-grasp lift height directly
# targets this; must stay consistent across dry-run and real execution.
# 0.15 (first attempt) reduced this failure 5/5 -> 4/24 seeds and raised the
# occluder-present success rate to 5/16. Pushing further to 0.2 was WORSE on both
# counts (4/16 successes, 6 pre_beside_box collisions) -- not monotonic, likely
# because more lift shifts the arm into a different (sometimes worse) posture
# rather than simply buying more clearance. Reverted to the better-validated 0.15;
# some of the 0.2 regression is plausibly sampling noise (documented CuRobo
# run-to-run nondeterminism) rather than a true causal effect, but 0.15 is the
# only value with two consistent, positive data points.
GRASP_LIFT_HEIGHT = 0.15
# Minimum fraction of GRASP_LIFT_HEIGHT the TARGET OBJECT itself (not just the
# gripper) must actually rise for the grasp to be considered real. Confirmed
# empirically (seeds 11/19/26): a fully missed grasp still reports the lift
# MOVE as plan_success=True (CuRobo found a valid plan for the commanded
# gripper motion) while the object's z stays flat (0.0-0.6% of the commanded
# rise) -- a wide margin below any plausible "successful but marginal" grasp,
# so 0.5 has a lot of room without risking false positives on a real grasp.
GRASP_VERIFY_MIN_RISE_FRACTION = 0.5
# Max drift (meters) of the object's position, expressed in the gripper's LOCAL
# frame, from the baseline captured right after attach_object, before a stage
# is considered to have lost the grasp. Confirmed empirically (seed 27): every
# placement stage reported plan_success=True while the object physically fell
# mid-transit (~10-60cm drift) -- plan_success only reflects the arm's motion
# plan, never the held object's actual state. 0.03 gives room for simulation
# jitter while still catching any real slip/drop, which in every observed case
# was at least an order of magnitude larger.
OBJECT_RETENTION_TOLERANCE = 0.03
# Max rotation drift (radians) of the object's orientation, expressed in the
# gripper's local frame, from the baseline -- translation alone misses a
# held object rotating loose without translating past OBJECT_RETENTION_
# TOLERANCE, leaving CuRobo's attached collision model (which assumes a
# fixed rigid transform) inconsistent with the physical object. ~11 degrees.
OBJECT_RETENTION_ROTATION_TOLERANCE = 0.2
# How close (meters) the held object's actual pose must be to the intended
# final placement pose (self.des_obj_pose) to treat place_actor's descent as
# genuinely DONE (skip remaining moves, open the gripper) rather than a loss
# or an ambiguous mid-descent contact (see DESCENT_CONTACT_DRIFT_TOLERANCE).
# Deliberately mirrors put_mouse_on_pad.check_success()'s own criterion
# (independent x/y error each < 0.02) with a small safety margin, INSTEAD of
# a looser combined radius: an earlier 8cm combined-radius version accepted
# seed 9 (dx=3.24cm) and seed 12 (dx=6.11cm, dy=2.85cm) as "arrived" even
# though check_success() would still reject both on at least one axis --
# stopping the descent there only guaranteed a failure the remaining slices
# might otherwise have corrected.
OBJECT_PLACEMENT_XY_TOLERANCE = 0.018
OBJECT_PLACEMENT_Z_TOLERANCE = 0.03
# Contact handling uses the task's exact per-axis success threshold. Once the
# object has settled, a gripper-relative retention check is no longer meaningful;
# release only when the task's own XY predicate already passes.
CONTACT_RELEASE_XY_TOLERANCE = 0.02
# Gripper-relative drift (meters) beyond OBJECT_RETENTION_TOLERANCE that's
# still treated as ambiguous-possible-contact rather than an outright loss,
# for place_actor's descent specifically: the object settling onto the pad/
# table a bit early (before reaching the precise final target) changes the
# gripper-relative transform similarly to a drop, but isn't one. Confirmed
# empirically: benign early-contact cases (seeds 9/12) showed ~6-7cm drift,
# while genuine catastrophic drops (seeds 20/25/27) showed 30cm-1.4m -- a
# wide, unambiguous gap. Below this, the descent continues toward the real
# target instead of failing early; above it, it's an unambiguous loss.
DESCENT_CONTACT_DRIFT_TOLERANCE = 0.15
# How close (meters) the held object's world-frame z must be to the intended
# resting height (self.des_obj_pose[2]) for a DESCENT_CONTACT_DRIFT_TOLERANCE-
# sized drift to be plausibly explained by contact with the placement surface,
# rather than the early stage of a genuine drop -- a real drop passes through
# the same 4-15cm drift band while still well above the surface (falling
# under gravity, unconstrained by the descent's planned trajectory), so
# position drift magnitude ALONE can't distinguish the two. Looser than
# OBJECT_PLACEMENT_Z_TOLERANCE (which gates a full "already placed" stop)
# since this only needs to confirm plausible surface contact, not a precise
# final resting pose.
DESCENT_CONTACT_HEIGHT_TOLERANCE = 0.06
# create_grasp_approach_metric's tstep_fraction for place_actor's descent
# slices specifically -- lower than CuRobo's own default (0.8, appropriate for
# a single large grasp-approach jump) because these slices are already tiny
# (DESCENT_SLICE_SIZE). tstep_fraction only holds the straight-line/
# orientation-locked cost for the trajectory's LAST (1-tstep_fraction)
# fraction of timesteps -- the rest is a soft-cost-only "free" portion.
# Confirmed empirically (seed 8, place_actor:move1 slice 4): with a FIXED
# 0.8, a 3cm slice's trajopt solve ballooned to 980 timesteps and swung the
# gripper 56cm away (up to z=0.70m) before snapping back onto the line for
# the final 20% -- the free portion has no incentive to stay short once
# trajopt escalates to a much longer time horizon (which it does more
# readily right at the final approach, where collision margins against the
# placement surface are tightest).
#
# A single fixed tight value isn't enough either: re-testing with a FIXED
# 0.2 fixed the excursion (deviation dropped from 56cm to ~2cm) but made one
# slice fail FINETUNE_TRAJOPT_FAIL on all 10 retries -- an overconstrained
# geometry where the exact straight line genuinely conflicts with collision
# margins needs at least some slack, and retrying the identical
# overconstrained problem 10 times can't discover that. Progressive
# relaxation instead: start tight (least detour room) for a few attempts,
# then relax toward CuRobo's own default across the remaining retry budget
# -- giving a genuinely-infeasible-under-tight-constraint slice a real
# chance to solve, while a hard path-safety filter (DESCENT_MAX_PATH_
# DEVIATION and friends below) still rejects any candidate at ANY fraction
# that takes an implausibly circuitous path, so relaxing never re-admits
# the original 56cm-loop failure mode.
DESCENT_APPROACH_TSTEP_FRACTIONS = [0.2, 0.4, 0.6, 0.8]
# Max perpendicular distance (meters) a descent slice's planned Cartesian EE
# path may stray from the FINITE straight-line SEGMENT between its own start
# and end pose before being rejected -- a backstop against the pathological-
# detour failure mode above, independent of which tstep_fraction produced
# it. Well above the sub-2mm wobble seen on well-behaved slices and well
# below the 56cm pathological case, and larger than a single slice's own
# straight-line distance (DESCENT_SLICE_SIZE) to leave room for legitimate
# small detours. Must be measured against the CLAMPED segment, not the
# infinite line through it -- an unclamped projection would report near-zero
# deviation for a trajectory that overshoots straight past the target along
# the same axis, which is exactly the kind of detour this needs to catch.
DESCENT_MAX_PATH_DEVIATION = 0.08
# Max ratio of a descent slice's actual Cartesian path length (sum of
# consecutive EE waypoint distances) to its straight-line distance --
# catches a trajectory that stays geometrically close to the line (small
# DESCENT_MAX_PATH_DEVIATION) but travels back and forth along it far more
# than the direct distance requires. A conservative multiple of 1.0 (a
# perfectly straight path); not tuned against a large empirical sample yet,
# so treat as provisional pending broader validation.
DESCENT_MAX_PATH_LENGTH_RATIO = 3.0
# Max ratio of a descent slice's total joint-space path length (sum of
# consecutive joint-position deltas) to the direct start->end joint-space
# distance -- catches joint-space winding that an EE-only Cartesian check
# can miss entirely (a redundant/near-singular arm can reach nearly the same
# EE pose through very different, looping joint configurations). Same
# provisional-pending-validation caveat as DESCENT_MAX_PATH_LENGTH_RATIO.
DESCENT_MAX_JOINT_TRAVEL_RATIO = 3.0
# Max excursion (radians) of any SINGLE joint's range (max-min position)
# across a descent slice's trajectory -- flags one joint spinning through an
# implausible range for what should be a tiny slice even when the aggregate
# DESCENT_MAX_JOINT_TRAVEL_RATIO looks acceptable (that ratio can stay small
# if only one joint winds while the others move normally, since it's an
# L2-norm aggregate). Same provisional-pending-validation caveat.
DESCENT_MAX_JOINT_RANGE = 1.0
# Max absolute joint-space distance (radians, L2 norm across all active
# joints) between a descent slice's own trajectory start and end -- catches
# a monotonic (non-winding) move to a DISTANT IK branch that the ratio-only
# checks above can miss entirely: a straight-line jump has joint_travel_ratio
# near 1.0 regardless of magnitude, and several joints each moving under
# DESCENT_MAX_JOINT_RANGE individually can still sum to a large aggregate
# displacement (e.g. six joints at ~0.9 rad each is an L2 norm of ~2.2 rad,
# passing max_joint_range=1.0 easily). A tiny Cartesian slice should need
# only a small joint-space move; not tuned against a large empirical sample
# yet, so treat as provisional pending broader validation.
DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT = 0.5
# Max absolute joint-space path length (radians) traveled during a descent
# slice -- an absolute companion to DESCENT_MAX_JOINT_TRAVEL_RATIO (which
# only bounds the path as a MULTIPLE of the endpoint distance, so it can't
# catch a monotonic jump to a distant branch on its own, same gap as
# DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT above). Slightly above that
# endpoint-displacement cap since path length is always >= endpoint
# distance. Same provisional-pending-validation caveat.
DESCENT_MAX_JOINT_PATH_LENGTH = 0.6
# Local landing-region search (place_actor's final approach): instead of
# forcing CuRobo to reach one exact final pose -- which kept failing
# FINETUNE_TRAJOPT_FAIL right at the surface, where the attached object's
# collision margin against the table is tightest -- search a small grid of
# nearby landing poses and accept the first one (at the lowest feasible
# release height) that plans cleanly and passes the same trajectory-safety
# filters used elsewhere, then let physics settle the final contact instead
# of commanding CuRobo all the way down to it.
#
# XY offsets (meters) tried per axis around the nominal target, combined
# into a grid -- matches check_success's own per-axis 2cm tolerance with a
# safety margin (max offset 1.5cm, comfortably under OBJECT_PLACEMENT_XY_
# TOLERANCE's 1.8cm).
LANDING_XY_OFFSETS = [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015]
# Candidate release heights (meters above the target's resting height),
# tried lowest first -- a lower release means less for physics to settle
# and stays closest to the nominal target, but a higher one gives CuRobo
# more collision-margin room to actually plan to when the exact surface
# height is too tight to solve.
LANDING_RELEASE_HEIGHTS = [0.01, 0.03, 0.05, 0.08]
# Max number of IK-feasible candidates (closest-to-nominal-target first) to
# actually PLAN per release height -- IK checks are cheap, full CuRobo plans
# aren't, so this bounds cost instead of exhaustively planning every
# candidate in the grid.
LANDING_MAX_CANDIDATES_PER_HEIGHT = 12
# Stop planning further candidates at a given height once this many have
# been accepted (passed both plan success and the path-safety filter) --
# picks among a reasonable sample instead of paying for the full budget
# above every time.
LANDING_MIN_ACCEPTED_TO_STOP = 3
# Max XY distance (meters) from the target, and required object retention,
# for _local_landing_search_and_place to even attempt its search -- its
# candidate moves are short by design, so if it were ever invoked from
# farther away its filters would reject a legitimate longer approach
# instead of a genuinely bad one. The caller currently only reaches this
# function once already close (via the placement chain + pre_place_descent),
# but this makes that assumption an explicit, checked precondition rather
# than an unstated one.
#
# Deliberately XY-only, not a combined 3D radius: XY offset directly bounds
# which of the +-1.5cm candidate offsets could plausibly land within
# check_success's own tolerance, while the remaining VERTICAL gap is a
# different concern (how far the arm still has to descend) governed by
# LANDING_SEARCH_MAX_Z_DISTANCE below instead -- a combined 3D check would
# treat "5cm lateral, no vertical drop left" the same as "no lateral offset,
# 5cm still to descend", which aren't comparable for this search.
LANDING_SEARCH_TRIGGER_DISTANCE = float(os.environ.get("LANDING_SEARCH_TRIGGER_DISTANCE", "0.10"))
# Max vertical (Z) distance (meters) from the target for the same gate --
# looser than the XY condition since the whole point of this search is
# picking a release height (LANDING_RELEASE_HEIGHTS, up to 8cm) rather than
# requiring the arm to already be at the exact final height.
LANDING_SEARCH_MAX_Z_DISTANCE = float(os.environ.get("LANDING_SEARCH_MAX_Z_DISTANCE", "0.15"))
# Max number of DIFFERENT contact-point candidates to actually try grasping in
# play_once when grasp_verify detects a missed grasp, before giving up on the
# episode entirely. _select_pick_place_candidate already ranks/generates
# several candidates for the dry-run search -- this reuses that same ranking
# to pick a genuinely different contact point on retry (via exclude_cp_ids)
# instead of ending the episode after the first missed grasp.
GRASP_VERIFY_MAX_CANDIDATES = 3
# Replay attached-object trajectories more slowly without changing their
# planned geometric path. A factor of 2 inserts one interpolated control point
# between each pair and scales commanded velocity accordingly.
ATTACHED_TRAJECTORY_SLOWDOWN = max(
    1, int(os.environ.get("ATTACHED_TRAJECTORY_SLOWDOWN", "2")))
# Slice length (meters) for place_actor's deterministic descent (see
# _plan_pose_with_descent_slices). place_actor's moves are short, mostly-
# straight-line final approaches with KNOWN structure (fixed target
# orientation, held via approach_axis) -- a scalar-error-informed shrink
# doesn't exploit that the way a fixed-size slice along the straight line to
# the target does. 0.04 sits in the requested 3-5cm range.
DESCENT_SLICE_SIZE = 0.04
SIDE_WAYPOINT_GAPS = (0.20, 0.24, 0.28)
# Fallback gaps tried ONLY when none of SIDE_WAYPOINT_GAPS's beside_box target verifies
# reachable (check_ik_batch, no trajopt) -- confirmed empirically (seed 9, right arm,
# box far from the shoulder in x): a check_ik_batch sweep at fixed y/z showed a clean
# reachable/unreachable cutoff at gap<=0.16 vs gap>=0.18, i.e. ALL of SIDE_WAYPOINT_GAPS
# (0.20-0.28) were unreachable for this box position while smaller gaps were fine. The
# fixed 3-value set silently assumed the arm can always reach OCC_HALF_FOOTPRINT+gap
# from the box on its own side, which breaks down for boxes positioned far from the
# shoulder. Only consulted as a fallback (extra search cost) when the preferred, more
# clearance-safe gaps all fail to verify.
SIDE_WAYPOINT_GAPS_FALLBACK = (0.16, 0.12, 0.08)
# Direct-plan retries for the post-attach placement search specifically (seed 8 class):
# a check_ik_batch/live-replay probe showed the SAME post_grasp_escape/pre_beside_box
# move -- confirmed independently reachable via pure IK -- failed 180/180 times with
# INVALID_START_STATE_WORLD_COLLISION in one background run, then succeeded on EVERY
# attempt in a fresh replay of the identical seed: the live qpos right after lift+
# enable_table sits close enough to the table that whether it's flagged in collision is
# sensitive to CuRobo's own internal trajopt seeding (the same marginal-feasibility
# nondeterminism already documented for place_actor, just showing up one stage earlier
# now that the search finds this stage instead of silently skipping it). Retrying the
# direct plan a few times before falling back to bridge waypoints gives that seeding
# more chances to land on a working solution, at far lower cost than enabling bridges
# (each retry is 1 extra trajopt call per combo; each bridge is ~local_attempts more).
PLACEMENT_SEARCH_RETRIES = 3
# A direct IK-reachability sweep (check_ik_batch diagnostic) showed the waypoint's
# feasibility roughly DOUBLES at +0.15m above grasp height (64%) vs the +0.0/+0.06m
# band this used to test alone (29-36%) -- x/gap and going even higher (+0.30, 54%)
# both matter far less. +0.15 is the empirically-found sweet spot, not a guess.
SIDE_WAYPOINT_Z_LIFTS = (0.0, 0.06, 0.15)
PLACE_CLEARANCE_ZS = (1.05, 1.15, 1.25)
# Fallback clearance heights tried when NONE of PLACE_CLEARANCE_ZS verifies reachable
# for lift_above_box (mirrors SIDE_WAYPOINT_GAPS_FALLBACK's reasoning for gaps): a
# check_ik_batch(relax_orientation=True) height sweep at each seed's beside_box x/y
# showed the REAL reachable window sits around z~0.8-1.05, i.e. PLACE_CLEARANCE_ZS's
# two higher values (1.15, 1.25) are unreachable outright regardless of orientation,
# and even 1.05 needs orientation relaxed to be reachable for some gaps. These lower
# values give the search somewhere to fall back to if 1.05 isn't enough either.
PLACE_CLEARANCE_ZS_FALLBACK = (0.95, 0.90, 0.85)
# Stages that must plan with a fixed (non-relaxed) orientation even when the rest
# of the placement chain uses relax_orientation=True: center_over_pad is the last
# placement subgoal before place_actor's own strict final descent, so it needs to
# hand off a KNOWN, expected orientation, not whatever CuRobo happened to converge
# to under a free-orientation goal. See _plan_pose_trajectory_sequence.
PLACEMENT_STRICT_ORIENTATION_STAGES = ("placement:center_over_pad",)
# Max iterations for the adaptive waypoint-shrink retry (see
# _plan_pose_with_shrinking_waypoint), used whenever a direct Cartesian plan
# fails. 0 preserves pure direct planning. Replaces the previous fixed-offset
# bridge-pool fallback: instead of blindly trying hand-picked offset directions,
# a failed attempt's target is iteratively pulled halfway toward the arm's
# current pose and retried, chaining a follow-up attempt at the REAL target
# once a shrunk intermediate succeeds. Default lowered from the old pool's 9 to
# 5 since this converges toward an always-eventually-reachable point (the
# current pose itself), needing fewer attempts than searching fixed offsets
# blindly. POST_GRASP_ESCAPE_ATTEMPTS is accepted as a backward-compatible
# alias for runs launched before this became universal.
LOCAL_WAYPOINT_ATTEMPTS = int(os.environ.get(
    "LOCAL_WAYPOINT_ATTEMPTS", os.environ.get("POST_GRASP_ESCAPE_ATTEMPTS", "5")))
# Give up shrinking once the remaining distance from the current pose to the
# (possibly already-shrunk) target drops below this -- not worth chasing an
# even-tinier hop that makes no real progress toward the actual destination.
WAYPOINT_SHRINK_MIN_DISTANCE = float(os.environ.get("WAYPOINT_SHRINK_MIN_DISTANCE", "0.05"))
# Both a second ("top_down") waypoint orientation and a +/-0.05m y-offset search were
# tried here and measured empirically to fail at the SAME rate as the baseline (both
# orientations: ~equal IK_FAIL proportion; y-offset: no change in failure proportion
# either) -- i.e. neither is the actual bottleneck (that's pure kinematic unreachability
# at the position itself). Collapsed back to single values so this search space doesn't
# multiply the cost of the (expensive, real CuRobo batch-plan) grasp-pose check below
# for no measured benefit. Re-widen only with new evidence, not speculatively.
WAYPOINT_ORIENTATIONS = ("grasp_aligned",)
WAYPOINT_Y_OFFSETS = (0.0,)
# Ordered chain stages (see _plan_candidate) -- used to rank how far a candidate got
# before failing, so the fallback (used when NO candidate fully plans) can pick the
# one that progressed furthest instead of whichever was generated first. Confirmed via
# a check_ik_batch reach-envelope probe: a fully-reachable contact point (both pre_grasp
# and grasp) can exist among the ones tried, but the old "first-generated" fallback
# picked an unreachable one instead because it happened to rank #1 geometrically.
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
# Placement retries (target/occluder pose resampling) are pure geometry -- no actor or
# physics work -- so retrying hundreds of times inside load_actors before ever creating
# an actor is essentially free next to the mesh/physics/curobo cost load_actors already
# pays regardless. Simulated against TARGET_XLIM/TARGET_YLIM/PAD_XY (500 seeds/range,
# cap=200): OCC_DISTANCE_CM=15,20 / 10,15 / 30,40 all converge in a handful of attempts
# with zero residual failures; 20,30 is a genuinely harder band (~2% single-draw success
# rate -- that gap size collides with the fixed pad across much of the target's spawn
# area) and still had 3.8% residual failures at cap=200. Bumped to 500 as cheap insurance
# (~(1-0.021)**500 ~= 0.002% projected residual, not re-simulated). Any seed that still
# fails after the cap falls through to the existing occluder_pad_clearance_ok check in
# run(), which rejects/skips it exactly as before -- just far more rarely.
_MAX_PLACEMENT_ATTEMPTS = 500

# --- yaw-aware edge-to-edge geometry (OCC_DISTANCE_CM is a true surface gap, not a
# center-to-center offset -- these turn each object's mesh AABB + yaw into a footprint
# rectangle and solve for the center-to-center placement that yields the requested gap).
_MODEL_DATA_CACHE = {}


def _yaw_from_quat(q):
    """World-frame yaw (about Z) of a pose quaternion. Every actor here is first stood
    upright by a fixed base qpos (local mesh Y -> world Z) and then optionally yawed, so
    the local mesh X axis always ends up rotated purely within the world XY plane --
    reading it off the rotation matrix sidesteps having to invert that qpos algebra."""
    R = t3d.quaternions.quat2mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _footprint_half_extents(extents, scale):
    """(half-x, half-y) world footprint of an object's mesh AABB at yaw=0 (local mesh
    X/Z -> world X/Y once stood upright; local Y is height -> world Z)."""
    return float(extents[0] * scale[0] / 2.0), float(extents[2] * scale[2] / 2.0)


def _model_footprint_half_extents(modelname, model_id, scale_override=None):
    """Footprint half-extents for modelname/model_id, read straight from its
    model_data{id}.json -- mirrors create_actor's own scale resolution (explicit
    scale_override wins, else the model's own "scale", else [1,1,1]) without needing
    the actor to already exist, so target/occluder placement can be resolved by a
    pre-creation retry loop before any (expensive) actor/mesh work happens."""
    key = (modelname, model_id)
    model_data = _MODEL_DATA_CACHE.get(key)
    if model_data is None:
        path = Path(os.environ["BENCH_ROOT"]) / "assets" / "objects" / modelname / f"model_data{model_id}.json"
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
    Real-Time Collision Detection 5.1.9 -- handles degenerate/parallel cases)."""
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
    overlap) -- exact for convex quadrilaterals, not an axis-aligned approximation."""
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
        # Remove the back furniture (shelf/cabinet/file-holder) for now to free up
        # gripper workspace. Flip back to True (or delete) to restore the full office.
        SPAWN_BACK_FURNITURE = False
        # Lever A: drop the orientation-hold constraint on the final grasp approach so
        # curobo won't reject it with INVALID_PARTIAL_POSE_COST_METRIC. Set False (or
        # delete this flag + the grasp_actor_from_table override below) to restore the
        # stock straight-in, wrist-locked constrained approach.
        DROP_GRASP_ORIENTATION_CONSTRAINT = True
        # When True, play_once stops right after the lift (pickup only) -- used by
        # pickup_reachability_map.py to reach the picked-up state, then probe IK.
        PICKUP_ONLY = False
        spawn_occluder = False
        occluder_offset = 0.2         # desired EDGE-TO-EDGE gap (metres, -y of the target);
                                       # load_actors solves the actual center-to-center
                                       # placement that yields this true surface distance
        target_model = TARGET_MODEL
        target_id = TARGET_ID
        target_xlim = TARGET_XLIM
        target_ylim = TARGET_YLIM
        fixed_pad_xy = PAD_XY

        def load_actors(self):
            # target: random within the upper-third band (varies per seed -> distribution).
            # When an occluder is being spawned, target/occluder placement is resolved
            # BEFORE any actor is created (pure geometry -- no mesh/physics cost) by
            # retrying the random draw up to _MAX_PLACEMENT_ATTEMPTS times until the
            # occluder clears the destination pad, instead of building the whole scene
            # and only then discovering (and discarding) a pad conflict. Re-yawing the
            # occluder alone isn't enough to escape most conflicts (it only shifts the
            # projected footprint by a few cm); target position dominates whether the
            # occluder lands near the pad, so it's resampled too on each retry.
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
                    # Only resample if another attempt will actually consume the new
                    # draw -- on the last attempt, redrawing target_pose here would
                    # leave it inconsistent with the (dist, probe_pose) computed above
                    # for the failed check, since nothing re-validates this final draw.
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

            # occluder: one tall milk-box, deterministically at mouse-x, just in front (-y),
            # at the (target/occluder pose, gap-solved) placement resolved above.
            # self.occluder_offset is the desired EDGE-TO-EDGE gap (meters); it is NOT the
            # center-to-center distance -- that's solved for below (via _solve_center_
            # offset_for_gap) so the requested gap holds regardless of either object's
            # yaw or footprint size.
            if self.spawn_occluder:
                tx, ty = float(target_pose.p[0]), float(target_pose.p[1])
                ox, oy = tx, ty - dist
                occ_pose = sapien.Pose([ox, oy, float(probe_pose.p[2])], probe_pose.q)

                self.occluder = create_actor(
                    scene=self, pose=occ_pose, modelname="038_milk-box", convex=True,
                    model_id=2, scale=[1.0, 1.0, 1.0],
                )
                self.occluder.set_mass(0.1)
                # Register the occluder in curobo's collision world so the planner
                # actually avoids it. NOT flagged is_obstacle -> it survives the
                # exclude_obstacles=True pass used in collision-metrics/eval mode
                # ("... curobo planner skips clutter obstacles"), which only drops
                # is_obstacle=True procedural clutter.
                self.collision_list.append({
                    "actor": self.occluder,
                    "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/038_milk-box/collision/base2.glb",
                })

        def play_once(self):
            # Expert plan. FORWARD (grasp) below is fixed. BACKWARD (placement) is a list
            # of subgoals returned by self._backward_subgoal_poses() -- EDIT THAT METHOD to
            # add / remove / reorder placement subgoals; each subgoal is one curobo move.
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"

            # Structured failure recorder (replaces having to grep '[play_once] after'
            # lines out of ROBOTWIN_LOG_MOVE logs by hand): the FIRST stage whose
            # checkpoint sees plan_success go False, plus whatever specific CuRobo
            # fail_reason our own plan+replay helpers captured for it (None for
            # stages still going through the standard self.move() path, which
            # doesn't expose a reason). Read by run() after run_rollout() and written
            # into records.jsonl. Reset per-episode since env is reused across seeds.
            self.rollout_failure_stage = None
            self.rollout_failure_reason = None
            self.rollout_candidate_info = None
            self._last_fail_reason = None
            self._grasp_baseline_transform = None
            self._slow_attached_replay = False
            # Richer diagnostics read by run() into records.jsonl alongside the
            # single-value failure/candidate fields above: per-attempt grasp
            # history (which contact points were tried and why each failed, not
            # just the last one), per-check retention drift (position + rotation,
            # not just the pass/fail it produced), and per-slice descent detail
            # (CuRobo's own position/rotation error on each place_actor segment).
            self.rollout_grasp_attempts = []
            self.rollout_retention_checks = []
            self.rollout_descent_slices = []

            def checkpoint(name):
                # Diagnostic: the dry-run candidate search only verifies the
                # waypoint/pre_grasp/grasp/lift/placement POSES are independently
                # reachable via a chained last_qpos rollout -- it never calls
                # attach_object/enable_table (which change the live collision world)
                # and never tests place_actor's final descent at all. A candidate can
                # pass the dry run and still die at any of these real-execution-only
                # steps, or at a step the dry run "verified" but which replans from a
                # genuinely different qpos in real execution. This pinpoints which.
                if not self.plan_success and self.rollout_failure_stage is None:
                    self.rollout_failure_stage = name
                    self.rollout_failure_reason = self._last_fail_reason
                if log_move:
                    print(f"[play_once] after {name}: plan_success={self.plan_success}")

            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")
            # Retry with a genuinely different contact-point candidate when
            # grasp_verify (below) detects a missed grasp, instead of ending the
            # episode after the first miss. _select_pick_place_candidate already
            # ranks/generates several candidates for the dry-run search --
            # exclude_cp_ids reuses that same ranking rather than re-deriving it.
            # Intermediate failed attempts reset plan_success/rollout_failure_*
            # so a LATER successful attempt isn't left alongside a stale failure
            # recording from an earlier, since-recovered-from miss.
            tried_cp_ids = set()
            candidate_plan = None
            cp_id = None
            grasp_succeeded = False
            retry_reset_pose = list(self.get_arm_pose(arm_tag))
            for grasp_attempt in range(1, GRASP_VERIFY_MAX_CANDIDATES + 1):
                candidate_plan = self._select_pick_place_candidate(arm_tag, exclude_cp_ids=tried_cp_ids)
                cp_id = candidate_plan["contact_point_id"]
                if cp_id is None:
                    # No viable contact-point candidate at all (e.g. the object's
                    # current geometry/pose gives _rank_side_grasp_ids nothing to
                    # rank) -- retrying can't help since nothing about the object
                    # or arm has changed. Executing the grasp anyway would fall
                    # through to an unconstrained default grasp
                    # (grasp_actor_from_table with contact_point_id=None) whose
                    # failure then overwrites the real cause with a generic
                    # grasp/lift error and reason=None. Fail fast with the real
                    # reason instead.
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
                # Verify the grasp actually captured the object before trusting it. plan_success
                # only reflects whether CuRobo found a valid motion plan for the COMMANDED
                # gripper move -- it says nothing about whether the object moved with it.
                # Confirmed empirically (seeds 11/19/26): the gripper lifted the full commanded
                # GRASP_LIFT_HEIGHT while the object stayed frozen at its original table pose
                # (z-rise 0.0-0.6% of the commanded height) -- a fully missed grasp, not a
                # marginal one. Left unchecked, attach_object below bakes the object's real
                # (still-on-table) pose into CuRobo's collision model as if it were rigidly
                # held by the now-lifted gripper, producing a nonsensical attached-mesh
                # position that immediately collides with the table the instant
                # enable_table(True) runs a few lines down -- misattributed as a placement/
                # pre_beside_box planning bug rather than what it actually is: a failed grasp.
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
                    # The failed attempt closed the gripper; restore both the
                    # gripper and table state before selecting another contact point. The table must go back to
                    # ENABLED here, not stay disabled: _grasp_via_cached_
                    # trajectories' own waypoint_move/pre_grasp leg expects to
                    # plan WITH table collision active (that's why it only
                    # disables the table partway through, right before the
                    # close final approach) -- leaving it disabled would let
                    # the next candidate's waypoint/pre_grasp plan straight
                    # through the table.
                    self.move(self.open_gripper(arm_tag))
                    self.enable_table(enable=True)
                    # Candidate plans are generated from the live arm state. A
                    # missed grasp leaves the arm at its lifted endpoint, so
                    # selecting the next candidate there made every retry solve
                    # a different, frequently colliding problem. Return to the
                    # common pre-attempt pose before selecting another contact.
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
                f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
                str(arm_tag),
            )
            checkpoint("attach_object")
            # Baseline for _object_retained: the object's full pose relative to the
            # gripper right now, while we still trust the grasp (grasp_verify just
            # passed). Every subsequent replay step compares against this.
            self._grasp_baseline_transform = self._gripper_relative_object_transform(arm_tag)
            self._slow_attached_replay = True
            self.enable_table(enable=True)
            checkpoint("enable_table")

            # Re-select the placement geometry under the REAL attached-object world,
            # but rank candidates by whether the FULL post-attach motion chain plans
            # successfully, not just by endpoint IK. Cache the winning trajectories so
            # these stages are replayed exactly instead of being planned again live.
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
            # _select_attached_placement_plan's search can exhaust every geometry/
            # escape combo without ever finding one that plans ALL the way through
            # (verified=False): it then falls back to the DEEPEST partial chain it
            # found, whose "trajectories" list is truncated at the stage that kept
            # failing (confirmed empirically: seed 8/9 checkpoint traces jumped
            # straight from an early placement subgoal to place_actor's
            # pre_place_descent, skipping lift_above_box/over_box_to_pad_y/
            # center_over_pad entirely, yet still reported rollout_success=True).
            # The loop above only iterates what's actually IN that truncated list,
            # so it silently "completes" without ever calling checkpoint() for the
            # missing subgoals -- plan_success never goes False. Make the fallback
            # an explicit, correctly-attributed failure instead of a silent skip.
            if self.plan_success and not placement_plan.get("verified", False):
                self.plan_success = False
                self._last_fail_reason = placement_plan.get("fail_reason", "unverified_placement_fallback")
                checkpoint(placement_plan.get("failed_stage") or "placement:unverified_fallback")
            # place_actor lowers the object from the last subgoal onto the pad.
            # constrain="free": check_success is position-only, so alignment buys nothing
            # and was a likely IK_FAIL cause for a tall object held near its top.
            # dis=0.02 (was 0.005): the wide-sweep checkpoints showed 3/3 episodes that
            # reached this step failed here with MotionGenStatus.FINETUNE_TRAJOPT_FAIL --
            # a 5mm final gap puts the held object's collision volume so close to the
            # table that trajopt's finetune can't converge on a smooth, collision-margin
            # -respecting descent. check_success() (put_mouse_on_pad.py) only checks x/y
            # position + gripper state, no z/height requirement, so a looser release
            # height costs nothing and gives finetune room to converge.
            # That alone didn't fix it: [place_actor] logging showed place_pre_pose ->
            # place_pose isn't a pure vertical drop, it also shifts diagonally in x/y,
            # and still hit FINETUNE_TRAJOPT_FAIL. CuRobo's own stacking example
            # (examples/isaac_sim/simple_stacking.py) uses PoseCostMetric.
            # create_grasp_approach_metric for exactly this: a straight-line approach
            # blended into only the last tstep_fraction of the trajectory. Unlike the
            # plain constraint_pose path (why DROP_GRASP_ORIENTATION_CONSTRAINT exists),
            # it sets offset_tstep_fraction >= 0, which skips CuRobo's start-vs-goal
            # orientation match check, so it won't reject a differing start/goal
            # orientation the way constraint_pose did. approach_axis=2 asks for a
            # straight-line descent (z) into both the pre-pose and final pose.
            place_arm_tag, place_actions = self.place_actor(
                self.target_obj, arm_tag=arm_tag, target_pose=self.des_obj_pose,
                constrain="free", pre_dis=0.05, dis=0.02,
            )
            if not place_actions:        # place_actor bails (plan_success already False)
                return                   # -> avoid place_actions[0] IndexError below
            for a in place_actions:
                if a.action == "move":
                    a.args["approach_axis"] = 2
            # None of the above fixed it: a pure-IK check (object attached, live qpos)
            # showed place_pre_pose/place_pose are BOTH independently reachable and
            # collision-free -- the endpoint is fine. So FINETUNE_TRAJOPT_FAIL here is a
            # path-smoothness problem, not a reachability/collision one: center_over_pad
            # -> place_pre_pose is one big trajopt problem (large vertical drop + lateral
            # shift + orientation change all at once) that the optimizer can't
            # interpolate smoothly, even though both ends are individually valid. Same
            # fix pattern as the grasp side: break it into a smaller intermediate step
            # (position+orientation blend) instead of one large jump.
            mid_pose = self._verified_intermediate(arm_tag, candidate_plan["placement_subgoals"][-1][1], place_actions[0].target_pose)
            self._plan_and_replay_pose(arm_tag, mid_pose)
            if self.plan_success and not self._object_retained(arm_tag, context="placement:pre_place_descent"):
                self.plan_success = False
                self._last_fail_reason = "object_lost:placement:pre_place_descent"
            checkpoint("placement:pre_place_descent")
            if log_move:
                # Diagnostic: is the endpoint itself reachable (pure collision-aware IK,
                # no trajectory optimization/smoothness), at the LIVE qpos with the
                # object actually attached right now? Distinguishes "the pose itself is
                # unreachable/colliding" from "the pose is fine but trajopt can't find a
                # smooth path there" -- four different trajopt/collision-margin fixes
                # all failed identically, so this tells us whether to keep tuning the
                # solver or reconsider the target pose itself.
                check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
                move_poses = [a.target_pose for a in place_actions if a.action == "move"]
                ik_ok = check_ik(move_poses)
                print(f"[place_actor] pure-IK reachability (attached, live qpos): {list(ik_ok)}")
            # Forcing CuRobo all the way down to place_actor's one exact final
            # pose (via _execute_actions_via_plan_and_replay, retries 4 -> 10)
            # was the prior approach here; the broad-sweep validation (seeds
            # 8-31) confirmed it made place_actor the dominant remaining
            # failure (100% MotionGenStatus.FINETUNE_TRAJOPT_FAIL), since the
            # attached object's collision margin against the table is
            # tightest exactly at that one pose. Searching a small grid of
            # nearby landing poses instead -- accepting any that plans
            # cleanly within check_success's own tolerance, then letting
            # physics settle the final contact -- avoids ever asking CuRobo
            # to solve that one hardest point.
            self._local_landing_search_and_place(place_arm_tag)
            checkpoint("place_actor")

        def _blend_pose(self, pose_a, pose_b, t=0.5):
            """Position lerp + quaternion nlerp (normalized linear interpolation --
            a cheap, good-enough slerp approximation for a transit waypoint that
            doesn't need to be exact) between two flat [x,y,z,qw,qx,qy,qz] poses.
            Used to insert an intermediate step in a large single move so trajopt
            gets a smaller, easier motion to solve instead of one big jump."""
            pa, pb = np.asarray(pose_a, dtype=float), np.asarray(pose_b, dtype=float)
            pos = (1 - t) * pa[:3] + t * pb[:3]
            qa, qb = pa[3:], pb[3:]
            if np.dot(qa, qb) < 0:      # quaternion double-cover: align signs before blending
                qb = -qb
            q = (1 - t) * qa + t * qb
            q = q / np.linalg.norm(q)
            return list(pos) + list(q)

        def _verified_intermediate(self, arm_tag, pose_a, pose_b, fractions=(0.5, 0.35, 0.65, 0.25, 0.75)):
            """Same verify-before-committing principle as
            _select_attached_placement_plan, applied to a transit waypoint: a naive
            midpoint blend can itself be unreachable (confirmed empirically -- seed
            11's pre_beside_box waypoint failed at the plain t=0.5 blend). Try a small
            spread of blend fractions and use whichever verifies reachable via
            check_ik_batch, falling back to the plain midpoint if none do."""
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
            """Ordered BACKWARD (placement) subgoals for a chosen candidate path."""
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
            """Build the exact post-attach placement chain we want to execute."""
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

            relax_orientation is withheld for PLACEMENT_STRICT_ORIENTATION_STAGES
            regardless of the caller's request: relaxing orientation on the LAST
            placement subgoal (center_over_pad, the handoff into place_actor's own
            strict, orientation-sensitive final descent) let CuRobo converge to an
            arbitrary held-object orientation there -- confirmed empirically to
            cause catastrophic placement misses (one seed landed the object ~51m
            from the pad, almost certainly a real collision/physics ejection from
            an orientation the attached-object collision approximation never
            validated) despite every stage reporting plan_success=True. Interior
            transit stages stay relaxed (that's where the real IK_FAIL bottleneck
            was, and they're always followed by at least one more stage -- ending
            at this forced-strict one -- so any orientation drift there gets
            corrected before the place_actor handoff."""
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
            """Pick placement geometry by FULL attached-world motion-plan success.

            Endpoint IK is a useful prefilter, but the remaining failures here have
            increasingly been about the chained motion itself. This helper keeps the
            same geometry search space, then scores candidates by whether the exact
            post-attach execution chain plans successfully from the current live qpos.
            The winning trajectories are cached for immediate replay.

            Every stage this searches over (post_grasp_escape, beside_box,
            lift_above_box, over_box_to_pad_y, center_over_pad, and the blended
            pre_* transit waypoints between them) is a pure carry-the-held-object
            move -- check_success() never looks at orientation en route, only the
            final x/y placement. So both the prefilters and the real trajopt calls
            below use relax_orientation=True (position-only goals): a
            check_ik_batch(relax_orientation=True) sweep confirmed this opens up
            real, currently-dead combos at lift_above_box that strict full-pose IK
            rejects outright."""
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            quat_options = (quat, GRASP_DIRECTION_DIC["top_down"])
            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            fallback = None
            fallback_depth = -1
            failure_breakdown = collections.Counter()
            escape_poses = self._post_grasp_escape_poses(arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS)

            # gaps first, fallback_gaps ONLY if NOTHING in gaps ever verified reachable
            # (the seed-9 case: box far enough from the shoulder that ALL of SIDE_
            # WAYPOINT_GAPS's beside_box targets are IK-infeasible, so the full trajopt
            # chain never even had a chance regardless of clearance/escape/quat).
            for gap_values in (gaps, fallback_gaps):
                any_reachable_x = False
                for quat_choice in quat_options:
                    # Cheap prefilter (no trajopt): beside_box's height (lift_z) and y
                    # (tgt_y) don't depend on clearance_z/escape_idx, so checking every
                    # gap's reachability in one batched IK call here skips the ENTIRE cz
                    # x escape_idx loop (up to 30 full trajopt chains per gap) for a gap
                    # whose target could never succeed regardless of how it's reached.
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
                        # Same idea as the beside_box prefilter above, one level deeper:
                        # lift_above_box shares this (x, tgt_y) and only varies in z
                        # (clearance) -- batch-check every clearance value here rather
                        # than discovering it's dead only after the full escape_idx loop
                        # of trajopt chains.
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
            horizontal (y) line. Position comes from ref_pose's height + the side-waypoint
            offset, with y = box's y-line + y_offset (searched around 0 -- pinning exactly
            to the box's y-line was found to be IK-unreachable at some seeds regardless of
            x/orientation). Orientation is either ref_pose's own (grasp_aligned -- matches
            the eventual reach-in angle) or a neutral top-down orientation (orient=
            "top_down"). None when no occluder is present."""
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
            """Rank a few side grasps so we can search across candidates instead of
            committing to exactly one local optimum."""
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
            _plan_pose_with_descent_slices and _local_landing_search_and_place --
            the latter passes joint_scale-adjusted endpoint/path-length caps since
            its candidates cover a longer segment than a nominal descent slice.

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
            """Execute an ALREADY-PLANNED trajectory (position/velocity arrays from a
            prior plan_path call) directly via take_dense_action, skipping a fresh
            plan_path call entirely. Mirrors left_move_to_pose/right_move_to_pose's
            need_plan/joint_path bookkeeping but never re-plans.

            Why this exists: CuRobo's trajopt has no uniqueness guarantee for a given
            Cartesian target -- re-planning to the SAME pose independently (as
            left_move_to_pose/right_move_to_pose normally do) can converge to a
            genuinely different, equally-valid arm posture. Confirmed empirically
            (qpos drift up to 1.47 rad between a dry-run-verified plan and a fresh
            live re-plan to the identical waypoint pose), and that drift flips
            downstream grasp reachability in ~50% of cases -- explaining most of the
            grasp-stage failures seen despite the dry run having just verified a
            working grasp. Replaying the exact verified trajectory removes the
            divergence: the live qpos after this call is guaranteed to match what the
            dry run assumed, since it's the same trajectory, not a new one."""
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
            """The held object's FULL pose (position AND orientation) expressed in
            the gripper's LOCAL frame -- stable under gripper translation/rotation
            as long as the object is still rigidly held, unlike a raw world-frame
            offset (which changes with gripper orientation even with zero slip).
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
            """Has the held object already reached a pose that WOULD PASS
            put_mouse_on_pad.check_success() -- independent x/y error each under
            xy_tolerance (mirroring check_success's own per-axis 0.02 criterion,
            not a looser combined radius)? Used by place_actor's descent to stop
            and open the gripper immediately once this is genuinely true, rather
            than continuing to descend past a placement that's already good.
            Deliberately conservative: an earlier combined-radius version
            accepted seed 9 (dx=3.24cm) and seed 12 (dx=6.11cm) as "arrived" even
            though check_success() would still reject both on at least one
            axis -- see DESCENT_CONTACT_DRIFT_TOLERANCE for how a drift that
            ISN'T yet this precise, but also isn't a genuine loss, is handled."""
            if not getattr(self, "des_obj_pose", None):
                return False
            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            target = np.asarray(self.des_obj_pose, dtype=float)
            dx = abs(float(obj_pos[0] - target[0]))
            dy = abs(float(obj_pos[1] - target[1]))
            z_dist = float(obj_pos[2] - target[2])  # positive: object sits above the target resting height
            return dx <= xy_tolerance and dy <= xy_tolerance and -z_tolerance <= z_dist <= z_tolerance

        def _object_near_support_height(self, height_tolerance=DESCENT_CONTACT_HEIGHT_TOLERANCE):
            """Is the held object's world-frame z plausibly at the placement
            surface right now (independent of x/y)? Used to gate
            DESCENT_CONTACT_DRIFT_TOLERANCE's ambiguous-contact branch: a
            gripper-relative drift in that band is only explainable as early
            contact if the object is actually near the resting height it would
            be contacting -- a genuine drop shows the same drift magnitude
            while still high above the surface, since it's falling under
            gravity rather than following the descent's planned trajectory."""
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
            """True if the held object is still where it was (position AND
            orientation) relative to the gripper when _grasp_baseline_transform
            was captured (right after attach_object in play_once) -- i.e. the
            grasp hasn't slipped, rotated loose, or been dropped since. Confirmed
            empirically (seed 27): every placement stage reported
            plan_success=True while the object physically fell mid-transit, since
            plan_success only reflects the ARM's motion plan, never the HELD
            OBJECT's actual state. Returns True (nothing to check) if no baseline
            has been captured yet.

            Strict everywhere, including place_actor's descent -- callers there
            should check _object_near_placement_target FIRST and short-circuit
            before ever calling this, since a legitimate early placement also
            changes the gripper-relative transform and would otherwise look
            identical to a drop."""
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
            """Safety-filter metrics for a planned descent-slice trajectory,
            via the arm's planner's own forward kinematics. Used by
            _plan_pose_with_descent_slices to catch CuRobo trajopt solves
            that are technically valid but take an implausibly circuitous
            path for what should be a tiny straight-line slice -- see
            DESCENT_MAX_PATH_DEVIATION and friends for the empirical case
            that motivated this (a 3cm slice's solve swinging the gripper
            56cm away before snapping back). Returns None if the trajectory
            has fewer than 2 waypoints (nothing to check).

            - max_perp_deviation: max distance any waypoint's EE position
              strays from the closest point on the FINITE segment between
              the trajectory's own start/end EE pose (projection clamped to
              [0, line_len] -- an unclamped projection would report
              near-zero deviation for a trajectory that overshoots straight
              past the target along the same axis).
            - path_length_ratio: actual Cartesian path length (sum of
              consecutive EE waypoint distances) over the straight-line
              distance -- catches back-and-forth paths that stay close to
              the line but travel much farther than the direct distance.
            - joint_path_length / joint_direct_dist: absolute joint-space
              path length and direct start->end joint-space distance
              (radians, L2 norm) -- exposed directly (not just as a ratio)
              so a monotonic jump to a distant IK branch can be caught by
              its own absolute magnitude, since such a jump has
              joint_travel_ratio near 1.0 regardless of how far it goes.
            - joint_travel_ratio: joint_path_length / joint_direct_dist --
              catches joint-space WINDING an EE-only check can miss (a
              redundant/near-singular arm can reach nearly the same EE pose
              via very different, looping joint paths). A near-zero
              joint_direct_dist (or line_len for path_length_ratio) with a
              non-negligible numerator returns inf rather than silently
              collapsing to a "perfect" 1.0 -- a real loop with almost
              identical start/end shouldn't look the same as "nothing
              moved."
            - max_joint_range: the largest single joint's (max-min)
              excursion across the trajectory, in radians -- flags one
              joint spinning through an implausible range even when the
              aggregate joint_travel_ratio looks acceptable."""
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
            """Progressive relaxation schedule across a descent slice's retry
            budget: front-load attempts on the tightest (least detour room)
            fraction, then relax toward CuRobo's own default across the
            remaining attempts -- see DESCENT_APPROACH_TSTEP_FRACTIONS for why
            a single fixed value (either tight or loose) isn't enough on its
            own."""
            tier = min(len(fractions) - 1, attempt_idx * len(fractions) // max(1, total_attempts))
            return fractions[tier]

        def _plan_pose_with_descent_slices(self, arm_tag, target_pose, qpos, stage_label, start_pose,
                                          retries=1, constraint_pose=None, approach_axis=None,
                                          slice_size=DESCENT_SLICE_SIZE, near_contact=False):
            """Deterministic incremental descent, used for place_actor's moves
            instead of the generic scalar-error-informed shrink retry
            (_plan_pose_with_shrinking_waypoint). place_actor's moves are short,
            mostly-straight-line final approaches with KNOWN structure (fixed
            target orientation, held via approach_axis) -- exploiting that
            structure with fixed-size slices along the straight line to the
            target is more appropriate than an adaptive shrink that doesn't know
            it. Plans AND REPLAYS each slice immediately (not a virtual dry-run
            chain), checking _object_retained after every segment -- catches a
            mid-descent drop at the EXACT slice it happens (confirmed
            empirically: seed 27 dropped the object mid-transit while every
            stage still reported plan_success=True) instead of only at the end,
            and naturally preserves whatever slices already succeeded if a
            later one fails, without needing a separate replay-before-return
            step (each slice is already committed by the time the next runs).

            Returns (ok, fail_reason, qpos, placed). placed=True means the
            object has already reached a pose that would pass check_success --
            the caller (_execute_actions_via_plan_and_replay) must skip any
            remaining 'move' actions and go straight to opening the gripper,
            since continuing to descend past an already-good placement could
            only make it worse."""
            plan_func = self._arm_plan_func(arm_tag)
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            start = np.asarray(start_pose, dtype=float)
            target = np.asarray(target_pose, dtype=float)
            total_dist = float(np.linalg.norm(target[:3] - start[:3]))
            num_slices = max(1, int(np.ceil(total_dist / slice_size)))
            segment_length = total_dist / num_slices
            # create_grasp_approach_metric's offset_position must stay smaller
            # than the segment it's applied to -- CuRobo's own default (0.05) is
            # LARGER than a 4cm slice, asking it to set up a straight-line
            # pre-approach point farther away than the entire move itself.
            # Confirmed to matter: 6 episodes still failed FINETUNE_TRAJOPT_FAIL
            # with the unscaled 5cm offset on every slice.
            approach_offset = min(0.05, segment_length * 0.8) if approach_axis is not None else 0.05
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            last_reason = "unknown"
            for i in range(1, num_slices + 1):
                # Interpolate BOTH position and orientation (_blend_pose does
                # position lerp + quaternion nlerp) -- copying the target's
                # quaternion onto every slice made the FIRST slice perform the
                # entire orientation change within just one small position step.
                waypoint = np.asarray(self._blend_pose(start, target, i / num_slices), dtype=float)
                total_attempts = max(1, int(retries))
                # Collect every attempt that plans successfully AND passes the
                # path-safety filter, across the whole progressive-relaxation
                # schedule, then replay the SHORTEST accepted one -- not just
                # the first success. Trying every attempt regardless of an
                # early pass (rather than stopping at the first accepted
                # candidate) is what lets a tight-fraction attempt lose to a
                # looser one that happens to find a genuinely shorter path,
                # and is bounded by retries (already small, e.g. 10).
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
                                  f"{attempt} (tstep_fraction={fraction}): rejected -- {last_reason}")
                    self.rollout_descent_slices.append(attempt_record)
                if not accepted:
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices} failed "
                              f"({total_attempts} attempts, none accepted; last={last_reason})")
                    return False, last_reason, qpos0, False
                # Lexicographic: shortest Cartesian path first, then shortest
                # joint-space path, then smallest single-joint excursion --
                # two candidates can have near-identical Cartesian paths but
                # very different joint motion, so Cartesian length alone
                # isn't enough to prefer the more sensible one.
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
                # Check arrival BEFORE retention, not as a fallback after a
                # failed retention check: the object settling onto the target
                # surface early (before the gripper's own nominal endpoint)
                # changes the gripper-relative transform exactly like a drop
                # would, but it's the intended outcome -- stop descending
                # further (which could otherwise ram the held object into the
                # surface it already reached) and report this move done, via
                # placed=True (the caller must skip any remaining moves).
                if self._object_near_placement_target():
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: object "
                              f"already at a placement that would pass check_success -- done")
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
                    # Ambiguous-contact requires ALL of: bounded position
                    # drift, bounded ROTATION drift (a rotation-only loss --
                    # object tipped/rotated loose without much translating --
                    # would otherwise sail through on position alone, quietly
                    # defeating the retention check's own rotation term), and
                    # the object plausibly near the support surface it would
                    # be contacting (a genuine drop shows the same 4-15cm
                    # position drift while still high above the surface,
                    # since it's falling under gravity rather than following
                    # the descent's planned trajectory).
                    if (pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                            and rot_drift <= OBJECT_RETENTION_ROTATION_TOLERANCE
                            and self._object_near_support_height()):
                        # Ambiguous: not yet at a passing placement (the check
                        # above already ruled that out) and not within strict
                        # retention tolerance either, but the drift magnitude is
                        # well below anything a genuine drop has shown (30cm-
                        # 1.4m) and consistent with early, imprecise contact.
                        # Don't fail -- let the remaining slices keep trying to
                        # converge on the real target instead of giving up on
                        # a placement that's already known to be too far off.
                        if log_move:
                            print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: drift "
                                  f"{pos_drift:.3f}m looks like early contact, not a loss -- continuing")
                    else:
                        self.plan_success = False
                        self._last_fail_reason = f"object_lost:{stage_label}:slice{i}"
                        return False, self._last_fail_reason, qpos0, False
                elif log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded")
            return True, None, qpos0, False

        def _local_landing_search_and_place(self, arm_tag):
            """Replaces place_actor's forced descent to one exact final pose
            (move1/move2 via _execute_actions_via_plan_and_replay), which kept
            failing MotionGenStatus.FINETUNE_TRAJOPT_FAIL right at the surface
            -- the attached object's collision margin against the table is
            tightest exactly there, and CuRobo's trajopt struggles to converge
            on ANY smooth path into it, let alone the one exact pose asked for.

            Instead searches a small grid of nearby landing poses -- XY
            offsets within check_success's own tolerance, release heights
            from low to high -- and requires only ONE of them to actually
            plan and pass the same trajectory-safety filters used by
            _plan_pose_with_descent_slices, then replays ONLY that one and
            opens the gripper, letting physics settle the final contact
            instead of commanding CuRobo all the way down to it. Explicitly
            gated on the object being retained and within
            LANDING_SEARCH_TRIGGER_DISTANCE (XY) / LANDING_SEARCH_MAX_Z_
            DISTANCE (vertical) of the target -- this function's
            candidate moves are short by design, so calling it from farther
            away would apply the same short-path filters to a legitimately
            longer approach and reject it.

            Heights are grouped into two tiers ([1cm,3cm] then [5cm,8cm]):
            ALL accepted candidates across an entire tier are compared
            before picking a winner, not just the first height with any
            success -- otherwise a marginal 1cm solve could beat a cleanly
            planned 3cm one that was never even tried, since candidates
            aren't compared across different heights independently.

            Candidate object poses are converted to gripper poses via
            get_place_pose, NOT a fixed offset above the pad -- get_place_pose
            derives the required gripper pose from the ACTUAL current grasp
            transform (actor pose vs. end-effector pose), so it's correct
            regardless of which grasp this particular episode happened to
            use, unlike a fixed "gripper N cm above the pad" assumption which
            would put the object at the wrong height for a different grasp.

            Sets plan_success=False with a descriptive fail_reason if no
            candidate at any height passes; the caller's checkpoint() records
            it as an ordinary failure, same as any other stage."""
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
                    print(f"[landing-search] preconditions not met -- retained={retained} "
                          f"xy_distance={xy_distance:.3f}m (max {LANDING_SEARCH_TRIGGER_DISTANCE}m) "
                          f"z_distance={z_distance:.3f}m (max {LANDING_SEARCH_MAX_Z_DISTANCE}m)")
                return

            # Skip direct 1--5 cm grids: the sweep produced 509 solver failures.
            # Stage at 8 cm, then descend sequentially with the near-contact profile.
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
                    # Sort by radius (closest to nominal target first), then
                    # by angle -- groups by distance WITHOUT letting ties at
                    # the same radius systematically favor one sign over
                    # another (plain grid iteration order puts every
                    # negative-dx candidate before any positive-dx one at
                    # the same radius, since LANDING_XY_OFFSETS lists
                    # negatives first and Python's sort is stable).
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
                            continue  # never attempted -- not a rejection, don't record
                        planned += 1
                        segment_length = float(np.linalg.norm(np.asarray(gripper_pose[:3]) - current_ee_pose[:3]))
                        approach_offset = min(0.05, segment_length * 0.8) if segment_length > 1e-6 else 0.02
                        if height == max(LANDING_RELEASE_HEIGHTS):
                            # This collision-clear target is a staging transit,
                            # not a descent. The approach metric caused the
                            # repeatable 2--2.5x loops seen in landing_v2.
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
                        # DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT/PATH_LENGTH
                        # were calibrated for descent slices (~DESCENT_SLICE_
                        # SIZE=4cm each) -- a landing-search candidate can be
                        # up to ~10cm away, which naturally needs more joint
                        # travel even for a perfectly clean plan. Reusing the
                        # unscaled absolute caps rejected almost everything
                        # (confirmed empirically: 1/288 candidates accepted
                        # across a 6-seed sanity run). Scale by how much
                        # longer this candidate's segment is than a nominal
                        # slice, same way approach_offset already scales.
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
            _plan_pose_with_descent_slices), which plans AND replays incrementally
            and checks grasp retention after every slice; non-move actions
            (gripper open/close) execute normally through self.move.

            retries: place_actor's failures are 100% MotionGenStatus.FINETUNE_TRAJOPT_FAIL
            (confirmed empirically) -- a marginal-feasibility trajopt-difficulty problem,
            not a reachability one. We've directly observed the SAME setup succeed on
            one run and fail on another (CuRobo's trajopt has internal randomized
            seeding), so retrying a failed plan_func call with fresh internal seeding
            has a real chance of finding a solution the first attempt missed. Each
            retry re-attempts ONLY the failed slice, not the whole chain."""
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
                    # A previous move already reached a placement that would
                    # pass check_success (placed=True) -- skip any remaining
                    # descent moves (continuing could only push an already-good
                    # placement out of tolerance or into the surface) but still
                    # fall through to whatever non-move actions (gripper open)
                    # come after this one.
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
            """Plan a single pose from the CURRENT live qpos and immediately replay
            it via _replay_planned_move, instead of self.move(self.move_to_pose(...)).
            For a lone plan-then-immediately-execute call like this there's no
            separate earlier verification pass to diverge from (unlike waypoint/
            grasp/place_actor), so this isn't fixing a re-plan-divergence bug here --
            its value is exposing the specific CuRobo fail_reason via
            self._last_fail_reason for the structured failure recorder, and using the
            same execution primitive as the rest of this file for consistency."""
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

            Matching the legacy execution path, disable the table after reaching
            pre-grasp and before replaying the final cached approach. Fall back to
            live planning only when the candidate has no cached trajectories.
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
            # Same as Office_base_task.grasp_actor_from_table, but (Lever A) strips the
            # orientation-hold constraint_pose from the approach actions before executing.
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
                # Added when the bridge search gained an IK prefilter (see
                # _plan_pose_with_local_waypoint_retry): a bigger pure-vertical lift
                # (no lateral offset -- the direction that resolved the escape/
                # pre_beside_box INVALID_START_STATE_WORLD_COLLISION failures) and a
                # later blend fraction, both cheap to add now that infeasible
                # candidates are filtered before spending a trajopt call on them.
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
            """Adaptive waypoint-shrink retry, used when a direct plan to target_pose
            has already failed.

            CuRobo's MotionGenResult sets position_error/rotation_error UNCONDITIONALLY
            even on FINETUNE_TRAJOPT_FAIL (confirmed via source read of motion_gen.py's
            _plan_from_solve_state) -- a failed attempt isn't pure noise, it tells us
            how far off the optimizer's best attempt landed. Instead of only trying
            hand-picked fixed offset directions (the old _local_waypoint_candidates
            bridge pool), each iteration here either (a) retries the ORIGINAL target
            from wherever the previous iteration actually landed, or (b) if that
            fails, shrinks the target toward the current pose and retries that
            smaller hop instead. The shrink amount is informed by position_error when
            CuRobo reports one: back the target off by roughly the residual gap
            (plus a safety margin) instead of blindly halving every time, so a
            near-miss shrinks only a little (another attempt might close a small
            gap) while a wild miss shrinks a lot. Falls back to a plain 50% halving
            when no usable position_error is available (e.g. IK_FAIL, where
            trajopt never ran). Shrinking toward the current pose (by definition
            already reachable) converges toward something plannable even for
            targets that are hard/far, unlike fixed offsets which can themselves be
            unreachable. Stops on success at the real target, on max_iterations, or
            once the remaining hop is already < min_distance from the current pose
            (not worth shrinking further). Note: this does NOT replay CuRobo's own
            failed near-miss trajectory (its full path was never verified
            collision-safe, only its final error magnitude is known) -- it only
            uses the error MAGNITUDE to size a fresh, independently-planned hop."""
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
                    # Back off by roughly how far short CuRobo's own best attempt
                    # landed (2x margin so we don't retry right at the same edge
                    # of infeasibility), instead of blindly halving every time.
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

            relax_orientation=True: position-only goal (PoseCostMetric.
            reach_partial_pose, see planner.py's plan_path) -- for transit-only moves
            where the target's orientation doesn't matter, only its position. A
            check_ik_batch(relax_orientation=True) sweep confirmed this opens up
            real, currently-dead combos (e.g. lift_above_box at low clearance_z)."""
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
            """Waypoint -> pre_grasp -> grasp -> lift, cached per (cp_id, gap, z_lift,
            orient, y_offset) -- this doesn't depend on clearance_z, but
            _candidate_specs yields one candidate per clearance_z sharing the same
            waypoint/grasp, so without caching this (expensive: get_grasp_pose does a
            real 10-rotation CuRobo batch-plan) work would be redone 3x for nothing.
            Returns (ok, failed_stage, fail_reason, qpos, lift_pose, trajectories).
            trajectories is {"waypoint"/"pre_grasp"/"grasp": <raw plan_path result>},
            captured so real execution can REPLAY these EXACT verified trajectories
            instead of independently re-planning to the same poses: confirmed
            empirically that an independent re-plan to the same Cartesian pose can
            converge to a genuinely different (though equally valid) joint
            configuration -- CuRobo's trajopt has no uniqueness guarantee -- with
            qpos drift up to 1.47 rad between the two, which flips downstream grasp
            reachability in ~50% of cases with large drift. This already fixed the
            waypoint; pre_grasp/grasp trajectories are captured the same way here
            because grasp_actor_from_table's own independent re-plan to the SAME
            pose choose_best_pose already verified is exactly the same failure mode,
            one step later in the chain.
            pre_grasp/grasp trajectories are planned CHAINED (pre_grasp's landed
            qpos feeds grasp's start), matching how they're physically executed one
            after another -- unlike the get_grasp_pose feasibility checks below,
            which mirror choose_grasp_pose's own unchained pose determination."""
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

            # Validate the SAME grasp-pose selection real execution uses
            # (get_grasp_pose -> choose_best_pose's rotation search + batch-plan
            # check, now that choose_best_pose's shortest-plan logic is fixed)
            # instead of the geometric approximation -- a candidate that "passes"
            # here can no longer diverge from what grasp_actor_from_table() actually
            # executes. choose_grasp_pose's own check_pose tests pre_grasp/grasp
            # independently from the SAME starting qpos (it doesn't chain through
            # pre_grasp's landed state) -- mirrored here for consistency.
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

            # Now plan+capture the ACTUAL trajectories to replay, CHAINED (pre_grasp
            # then grasp from wherever pre_grasp lands), matching physical execution.
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
            # Carried to play_once so real execution can REPLAY these exact verified
            # trajectories for the waypoint/pre_grasp/grasp moves instead of
            # independently re-planning to the same poses (see _plan_grasp_side's
            # docstring for why that matters).
            candidate["grasp_waypoint_trajectory"] = trajectories.get("waypoint")
            candidate["pre_grasp_trajectory"] = trajectories.get("pre_grasp")
            candidate["grasp_trajectory"] = trajectories.get("grasp")

            # Placement subgoal orientation is derived from the grasp orientation --
            # rebuild from the REAL lift_pose above (candidate["placement_subgoals"]
            # was built in _candidate_specs from the geometric approximation) and
            # write the corrected version back onto the candidate so real execution
            # (play_once) uses it too, not just this validation pass.
            x_side = self._box_side_x(arm_tag, gap=candidate["gap"])
            placement_subgoals = self._backward_subgoal_poses(
                arm_tag, x_side=x_side, clearance_z=candidate["clearance_z"], lift_pose=lift_pose)
            candidate["placement_subgoals"] = placement_subgoals
            candidate["lift_pose"] = lift_pose

            # Real execution calls attach_object (holding the target) before planning
            # the placement subgoals; the check above doesn't, so it can wrongly
            # certify a placement pose that's actually blocked by the held object's
            # own collision volume. Confirmed empirically: "beside_box" flips from
            # IK-feasible to infeasible once attached with this same chained qpos.
            # Approximate the held object's pose as lift_pose (the gripper is
            # essentially at the object once it's holding it).
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            object_dict = {
                "name": self.target_obj.get_name(),
                "pose": list(lift_pose),
                "file_path": f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
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
            # Selected-candidate metadata for the structured failure recorder (see
            # play_once): whether the executed candidate was fully dry-run-verified
            # or a fallback, and if a fallback, how far it got in the dry run and
            # the full failure breakdown across everything tried. Lets records.jsonl
            # distinguish "the dry run found a working plan and it still failed for
            # real" from "the dry run never found anything workable to begin with"
            # without re-deriving it from ROBOTWIN_LOG_MOVE logs.
            #
            # exclude_cp_ids: contact points already tried and found to fail a REAL
            # grasp_verify check (see play_once) -- skipped so a retry after a
            # missed grasp picks a genuinely different contact point instead of
            # re-selecting the same one that just failed.
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
                # Fallback = the candidate that progressed FURTHEST through the chain
                # before failing, not just the first one generated -- a later cp_id can
                # be far more reachable than the top-ranked one even though it's tried
                # later in the loop.
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
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

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

            # One occluder distance per seed, drawn uniformly from the configured cm
            # range and converted to meters -- keyed on seed alone (not seed+offset,
            # since offset is no longer an externally chosen sweep value) so the draw
            # is still reproducible across reruns of the same seed range.
            off = float(np.random.default_rng(int(seed)).uniform(dist_lo_cm, dist_hi_cm)) / 100.0
            # per-build coin flip: drop the occluder with prob no_occluder_prob
            show = bool(np.random.default_rng(int(seed) * 1000 + int(round(off * 100))).random()
                        >= args.no_occluder_prob)
            # Pad-clearance rejection can't be pre-estimated from `off` any more -- `off`
            # is now an edge-to-edge gap, and the actual center-to-center placement (and
            # thus the occluder's real xy) depends on both objects' yaw, only known once
            # load_actors actually builds the scene. So this is checked for real (via
            # env.occluder_pad_clearance_ok, set in load_actors) right after the first
            # build below, and the whole seed is rejected there instead of pre-build.
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
                    # Structured failure/candidate metadata set by play_once (see
                    # its checkpoint()/_select_pick_place_candidate) -- survives
                    # run_rollout's internal close_env() since it's just plain
                    # attributes on the (reused) env object. None on success.
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
                    help="occluder EDGE-TO-EDGE gap range 'lo,hi' in CM in front of the "
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
                         "--occluding-object-distance range as a whole -- appropriate "
                         "since each seed now draws its own offset from within that "
                         "range, so per-exact-offset grouping ('offset') no longer "
                         "yields meaningful per-group sample sizes.")
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
