#!/usr/bin/env python
"""Measure the determinism the rebuild-and-replay protocol depends on.

Phases (run from sim/, RoboPRO conda env):

  rebuild     N fresh rebuilds replaying identical stored actions. Every
              fingerprint and label must match. This is the protocol's core
              claim. Needs no policy and no model server.

  crossproc   Same, but writes its fingerprint to --out so two separate
              processes can be compared with --compare A.npz B.npz. Proves the
              guarantee survives a process boundary (and therefore a machine).

  closedloop  N rebuilds with an identical noise key, running the POLICY in the
              loop. Exercises ray-traced rendering feeding live inference --
              the one path the other phases never touch. Requires --port.

  restore     Control: the WRONG way. Snapshot/restore instead of rebuilding.
              Requires the `lookahead` package; skipped with a message if absent.
              Included because the failure is silent -- restore reproduces every
              observable (poses, velocities, joint angles) bit-exactly, so
              nothing looks wrong until labels start flipping.

Example:
    python script/verify_replay_determinism.py --phase rebuild \\
        --task_name put_milktea_next_to_laptop --task_config bench_demo_office_d14 \\
        --seed 0 --actions <episode>.npz --T 100 --repeats 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.append("./")
sys.path.append("./description/utils")
from script.bench_script.setup_paths import setup_paths  # noqa: E402

setup_paths()
sys.path.insert(0, "script")
from replay_utils import (build_args, rebuild, replay_actions,  # noqa: E402
                          state_fingerprint)


def load_actions(path):
    d = np.load(path)
    for k in ("actions", "branch_actions", "arr_0"):
        if k in d:
            a = d[k]
            return a.reshape(-1, a.shape[-1])
    raise SystemExit(f"{path} has no recognised action array (keys: {list(d)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["rebuild", "crossproc", "closedloop", "restore"])
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    ap.add_argument("--task_name", default="put_milktea_next_to_laptop")
    ap.add_argument("--task_config", default="bench_demo_office_d14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actions", help="npz with recorded action rows")
    ap.add_argument("--T", type=int, default=100, help="prefix length before the tail")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--step_lim", type=int, default=300)
    ap.add_argument("--port", type=int, help="model server port (closedloop only)")
    ap.add_argument("--instruction", default="Put the milk tea next to the laptop.")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.compare:
        A, B = np.load(a.compare[0]), np.load(a.compare[1])
        d = np.abs(A["fp"] - B["fp"])
        print(f"cross-process: dim={A['fp'].size}  max|diff| = {d.max():.3e}  "
              f"exact = {bool((d == 0).all())}   labels {bool(A['label'])} vs {bool(B['label'])}")
        return

    args = build_args(a.task_name, a.task_config)
    acts = load_actions(a.actions) if a.actions else None
    task = None

    if a.phase in ("rebuild", "crossproc"):
        print(f"=== {a.phase}: {a.repeats} rebuild(s), identical actions ===")
        fps, labs = [], []
        for r in range(a.repeats):
            task = rebuild(task, a.task_name, args, a.seed,
                           step_lim=a.step_lim, instruction=a.instruction)
            replay_actions(task, acts)
            fps.append(state_fingerprint(task))
            labs.append(bool(task.eval_success or task.check_success()))
            d = float(np.abs(fps[-1] - fps[0]).max())
            print(f"  rebuild {r}: success={labs[-1]}  max|diff vs 0| = {d:.3e}")
        ok = all(np.array_equal(f, fps[0]) for f in fps) and len(set(labs)) <= 1
        print(f"  -> BIT-EXACT ACROSS REBUILDS: {ok}")
        if a.out:
            np.savez(a.out, fp=fps[0], label=labs[0])
            print(f"  [out] {a.out}")

    elif a.phase == "closedloop":
        if not a.port:
            raise SystemExit("closedloop needs --port (model server)")
        import eval_policy_client as E
        eval_func = E.eval_function_decorator("pi05", "eval")
        model = E.ModelClient(port=a.port, pi0_step=50)
        print(f"=== closedloop: {a.repeats} rebuild(s), identical noise key, policy in the loop ===")
        runs = []
        for r in range(a.repeats):
            task = rebuild(task, a.task_name, args, a.seed,
                           step_lim=a.step_lim, instruction=a.instruction)
            model.reset_model()
            model.set_noise_episode("verify_closedloop")
            rows, orig = [], task.take_action

            def rec(action, *aa, _o=orig, **kw):
                rows.append(np.asarray(action, dtype=np.float64).copy())
                return _o(action, *aa, **kw)

            task.take_action = rec
            try:
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    eval_func(task, model, task.get_obs())
            finally:
                task.take_action = orig
            runs.append((np.stack(rows) if rows else np.zeros((0, 14)),
                         state_fingerprint(task),
                         bool(task.eval_success or task.check_success())))
            print(f"  run {r}: steps={len(rows)} success={runs[-1][2]}")
        ok = True
        for r in range(1, len(runs)):
            same = runs[r][0].shape == runs[0][0].shape
            am = float(np.abs(runs[r][0] - runs[0][0]).max()) if same else float("nan")
            fm = float(np.abs(runs[r][1] - runs[0][1]).max())
            ok &= bool(same and am == 0.0 and fm == 0.0 and runs[r][2] == runs[0][2])
            print(f"  run {r} vs 0: action max|diff| = {am}   fp max|diff| = {fm:.3e}   "
                  f"label match = {runs[r][2] == runs[0][2]}")
        print(f"  -> CLOSED-LOOP BIT-EXACT: {ok}")

    elif a.phase == "restore":
        try:
            from lookahead import rollback
        except Exception as e:
            print(f"[skip] restore control needs the `lookahead` package ({e}).")
            print("       Its point is only to show why this protocol rebuilds instead.")
            return
        print(f"=== restore (CONTROL -- the wrong way): {a.repeats} restores ===")
        task = rebuild(task, a.task_name, args, a.seed,
                       step_lim=a.step_lim, instruction=a.instruction)
        replay_actions(task, acts[:a.T])
        snap = rollback.snapshot(task)
        fp_T = state_fingerprint(task)
        labs, fps = [], []
        for r in range(a.repeats):
            rollback.restore(task, snap)
            d0 = float(np.abs(state_fingerprint(task) - fp_T).max())
            replay_actions(task, acts[a.T:])
            fps.append(state_fingerprint(task))
            labs.append(bool(task.eval_success or task.check_success()))
            print(f"  restore {r}: state restored exactly (max|diff| {d0:.3e}) "
                  f"-> success={labs[-1]}  tail max|diff vs 0| "
                  f"{float(np.abs(fps[-1]-fps[0]).max()):.3e}")
        print(f"  -> labels {[int(x) for x in labs]}   "
              f"STABLE: {len(set(labs)) == 1}")
        print("  note: the restored STATE is bit-exact; what differs is the PhysX")
        print("        warm-start cache, which pack() does not capture.")

    if task is not None:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
