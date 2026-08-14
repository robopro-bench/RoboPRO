"""
Batch runner: runs test_collision_metrics.py for every office/study/kitchenl/kitchens task,
saves video + collision data under <output_dir>/<subdir>/tasks/<task_name>/<instance_N>/.

USAGE:
    cd sim
    source set_env.sh
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py [options]

EXAMPLES:
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py --output-dir ./col_results --seed 0
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py --tasks mouse_on_pad move_cup
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py --subdir office --output-dir ./collision_test
    # Kitchen tasks with randomized config, 5 instances each:
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py --subdir kitchenl --num-instances 5
    python $BENCH_ROOT/bench_script/run_collision_metrics_all.py --subdir kitchenl --num-instances 3 --task-config bench_demo_randomized
"""
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path

# ── task registry ──────────────────────────────────────────────────────────────
KITCHEN_TASKS = [
    "move_bottle_from_fridge_next_to_can",
    "move_bottle",
    "move_can_from_cabinet_to_basket",
    "move_milk_close_fridge",
    "pick_bottle_from_fridge",
    "pick_boxdrink_from_basket",
    "pick_can_from_basket",
    "pick_can_from_cabinet",
    "pick_milk_box_from_fridge",
    "pick_sauce_can_from_cabinet",
    "put_bottle_in_basket",
    "put_bottle_in_fridge",
    "put_can_close_cabinet",
    "put_can_in_cabinet",
    "put_can_infront_of_microwave",
    "put_can_next_to_basket",
    "put_milk_box_in_fridge",
    "put_sauce_can_in_basket",
    "put_sauce_can_in_cabinet",
    "switch_can_with_bottle_in_basket",
]

OFFICE_TASKS = [
    "close_drawer",
    "move_items_around",
    "open_drawer",
    "organize_table",
    "put_book_in_fileholder",
    "put_book_on_book",
    "put_milktea_next_to_laptop",
    "put_milktea_on_shelf",
    "put_mouse_next_to_stapler",
    "put_mouse_on_pad",
    "put_phone_next_to_cube",
    "put_phone_on_holder",
    "put_rubikscube_in_drawer",
    "put_rubikscube_next_to_milktea",
    "put_stapler_in_drawer",
    "put_stapler_next_to_mouse",
    "put_stapler_on_book",
    "set_up_table",
    "store_rubikscube_on_shelf",
    "store_stapler_in_drawer",
]

STUDY_TASKS = [
    "empty_box",
    "move_book_onto_table",
    "move_cup_next_to_book",
    "move_cup_onto_table",
    "move_cup_put_pen_in_cup",
    "move_cup",
    "move_cups_into_box",
    "move_pen_to_box",
    "move_seal_cup_next_to_box",
    "move_seal_next_to_box",
    "move_seal_next_to_pencup",
    "move_seal_onto_book",
    "move_seal_onto_table",
    "put_cup_in_box",
    "put_cup_on_coaster",
    "put_cup_on_table",
    "put_glue_in_box",
    "put_pen_in_box",
    "put_pen_in_pencup",
    "put_seal_in_box",
]

KITCHENS_TASKS = [
    "chain_apple_bin_bowl_rack_spoon_sink_ks",
    "chain_apple_sink_plate_bread_board_ks",
    "chain_bowl_rack_apple_sink_ks",
    "chain_heat_hamburger_ks",
    "chain_serve_hamburger_ks",
    "close_microwave_ks",
    "drop_apple_in_bin_ks",
    "move_hamburger_onto_plate_ks",
    "pick_apple_from_bowl_ks",
    "pick_apple_from_sink_ks",
    "pick_fork_from_sink_ks",
    "pick_hamburger_from_microwave_ks",
    "place_bowl_in_dishrack_ks",
    "put_bowl_in_sink_ks",
    "put_bread_on_board_ks",
    "put_hamburger_in_microwave_ks",
    "put_plate_in_sink_ks",
    "put_spoon_in_dishrack_ks",
    "put_spoon_in_sink_ks",
    "put_spoon_on_plate_ks",
]

# task_name -> bench_subdir
TASK_SUBDIR: dict[str, str] = {t: "office" for t in OFFICE_TASKS}
TASK_SUBDIR.update({t: "study" for t in STUDY_TASKS})
TASK_SUBDIR.update({t: "kitchenl" for t in KITCHEN_TASKS})
TASK_SUBDIR.update({t: "kitchens" for t in KITCHENS_TASKS})


def _col(code: str, text: str) -> str:
    codes = {"GREEN": "32", "RED": "31", "YELLOW": "33", "CYAN": "36", "BOLD": "1"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"


def run_task(task_name: str, subdir: str, task_config: str, seed: int,
             base_output: Path, capture_every: int, contact_image_every: int,
             timeout: int, instance_idx: int = 0) -> dict:
    """Run test_collision_metrics.py for one task instance. Returns a result summary dict."""
    task_output = (base_output / subdir / task_name / f"instance_{instance_idx}").resolve()
    task_output.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parent / "test_collision_metrics.py"
    cmd = [
        sys.executable, str(script),
        task_name, task_config,
        "--seed", str(seed),
        "--output-dir", str(task_output),
        "--bench-subdir", subdir,
        "--capture-every", str(capture_every),
        "--contact-image-every", str(contact_image_every),
        "--render-freq", "0",
    ]

    print(_col("CYAN", f"\n{'='*60}"))
    print(_col("BOLD", f"  [{subdir}] {task_name}"))
    print(_col("CYAN", f"{'='*60}"))
    print(f"  output → {task_output}")
    print(f"  cmd: {' '.join(cmd[2:])}\n")

    run_prefix = f"{task_name}_{task_config}"
    log_file = task_output / "run.log"
    result = {"task": task_name, "subdir": subdir, "instance": instance_idx,
              "seed": seed, "status": "unknown", "output_dir": str(task_output)}

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        # stream output to console AND write to log
        print(proc.stdout)
        log_file.write_text(proc.stdout)

        if proc.returncode == 0:
            result["status"] = "ok"
            print(_col("GREEN", f"  ✓ {task_name} finished"))
        else:
            result["status"] = "error"
            result["returncode"] = proc.returncode
            print(_col("RED", f"  ✗ {task_name} exited with code {proc.returncode}"))

        # parse metrics json if present (named {task}_{config}_collision_metrics.json by test script)
        metrics_path = task_output / f"{run_prefix}_collision_metrics.json"
        if metrics_path.exists():
            result["metrics"] = json.loads(metrics_path.read_text())
            result["metrics_path"] = str(metrics_path)

        collision_log_path = task_output / f"{run_prefix}_collision_log.json"
        if collision_log_path.exists():
            result["collision_log_path"] = str(collision_log_path)

        video_path = task_output / f"{run_prefix}.mp4"
        if video_path.exists():
            result["video_path"] = str(video_path)
            print(f"  video  → {video_path}")
        if metrics_path.exists():
            print(f"  metrics → {metrics_path}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print(_col("RED", f"  ✗ {task_name} TIMED OUT after {timeout}s"))
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        print(_col("RED", f"  ✗ {task_name} exception: {e}"))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run test_collision_metrics.py for all (or selected) tasks"
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Subset of task names to run (default: all)")
    parser.add_argument("--subdir", choices=["office", "study", "kitchenl", "kitchens"], default=None,
                        help="Only run tasks from this subdir")
    parser.add_argument("--task-config", default="bench_demo_clean",
                        help="Task config name (default: bench_demo_clean)")
    parser.add_argument("--num-instances", type=int, default=1,
                        help="Number of instances (randomized runs) per task (default: 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./collision_metrics_results")
    parser.add_argument("--capture-every", type=int, default=5,
                        help="Capture video frame every N physics steps")
    parser.add_argument("--contact-image-every", type=int, default=50,
                        help="Save contact debug image every N steps")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-task timeout in seconds (default: 600)")
    args = parser.parse_args()

    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    # Build task list
    all_tasks = [(t, s) for t, s in TASK_SUBDIR.items()]
    if args.subdir:
        all_tasks = [(t, s) for t, s in all_tasks if s == args.subdir]
    if args.tasks:
        requested = set(args.tasks)
        unknown = requested - TASK_SUBDIR.keys()
        if unknown:
            print(_col("RED", f"Unknown tasks: {unknown}"))
            sys.exit(1)
        all_tasks = [(t, s) for t, s in all_tasks if t in requested]

    num_instances = args.num_instances
    total_runs = len(all_tasks) * num_instances
    print(_col("BOLD", f"\nRunning collision metrics for {len(all_tasks)} tasks × {num_instances} instance(s) = {total_runs} runs"))
    print(f"Output root: {base_output}")
    print(f"Config: {args.task_config}  base seed: {args.seed}")

    results = []
    for task_name, subdir in all_tasks:
        for i in range(num_instances):
            instance_seed = args.seed + i
            r = run_task(
                task_name=task_name,
                subdir=subdir,
                task_config=args.task_config,
                seed=instance_seed,
                base_output=base_output,
                capture_every=args.capture_every,
                contact_image_every=args.contact_image_every,
                timeout=args.timeout,
                instance_idx=i,
            )
            results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(_col("CYAN", f"\n{'='*60}"))
    print(_col("BOLD", "  SUMMARY"))
    print(_col("CYAN", f"{'='*60}"))

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    for r in ok:
        metrics_str = ""
        if "metrics" in r:
            m = r["metrics"]
            success = m.get("task_success")
            success_str = ("YES" if success else "NO") if success is not None else "-"
            metrics_str = (
                f"  success={success_str}"
                f"  robot↔furniture={m.get('robot_to_furniture','-')}"
                f"  robot↔static={m.get('robot_to_static_object','-')}"
                f"  target↔static={m.get('target_to_static_object','-')}"
            )
        inst = f"[i={r.get('instance', 0)} s={r.get('seed', '-')}] " if num_instances > 1 else ""
        print(_col("GREEN", f"  ✓ [{r['subdir']}] {r['task']} {inst}") + metrics_str)

    for r in failed:
        inst = f"[i={r.get('instance', 0)} s={r.get('seed', '-')}] " if num_instances > 1 else ""
        print(_col("RED", f"  ✗ [{r['subdir']}] {r['task']} {inst} ({r['status']})"))

    # Save summary json
    summary_path = base_output / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved: {summary_path}")
    print(f"Done. {len(ok)}/{len(results)} tasks succeeded.")


if __name__ == "__main__":
    main()
