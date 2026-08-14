#!/bin/bash
# Integration test for OOD obstacle expansion (#19)
#
# Runs expert rollouts with unseen_obstacles at density 10 for one task
# per scene family, across multiple seeds. Saves videos for review.
#
# Usage:
#   source set_env.sh  # repo root
#   cd sim
#   bash ../scripts/validation/integration_test_ood.sh
#
# Options (env vars):
#   SEEDS="0 1 2 3 4"      # seeds to test (default: 0 1 2)
#   CONFIG=bench_object_ood_unseen_obstacles_d10  # config to use
#   GPU_ID=0                # GPU to use

SEEDS="${SEEDS:-0 1 2}"
CONFIG="${CONFIG:-bench_object_ood_unseen_obstacles_d10}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

TASKS=(
    "put_mouse_on_pad:office"
    "put_cup_in_box:study"
    "put_bowl_in_sink_ks:kitchens"
    "put_bottle_in_fridge:kitchenl"
)

RESULTS_DIR="../scripts/validation/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RESULTS_DIR/integration_test_${TIMESTAMP}.log"

echo "Integration test: OOD obstacle expansion" | tee "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Seeds: $SEEDS" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

pass=0
fail=0
error=0

for entry in "${TASKS[@]}"; do
    task="${entry%%:*}"
    scene="${entry##*:}"

    echo "=== $scene / $task ===" | tee -a "$LOG"

    for seed in $SEEDS; do
        printf "  seed %-4s " "$seed" | tee -a "$LOG"

        # Use a per-task-seed save path so videos don't overwrite each other
        SAVE_SUBDIR="data/integration_test_${TIMESTAMP}/${scene}_${task}"
        mkdir -p "$SAVE_SUBDIR"

        output=$(python script/bench_script/visualize_task_scene.py \
            "$task" "$CONFIG" \
            --bench-subdir "$scene" \
            --rollout --no-render \
            --seed "$seed" --save_data 2>&1) || true

        success_line=$(echo "$output" | grep -i "^Success:" | tail -1)

        # Move the saved video to a unique name
        SRC_VIDEO="data/bench_data/video/episode_${task}_0.mp4"
        DST_VIDEO="$SAVE_SUBDIR/seed_${seed}.mp4"
        if [ -f "$SRC_VIDEO" ]; then
            mv "$SRC_VIDEO" "$DST_VIDEO"
            video_note="video: $DST_VIDEO"
        else
            # Try alternate naming patterns
            FOUND_VIDEO=$(find data/bench_data/video/ -name "*${task}*" -newer "$LOG" -type f 2>/dev/null | head -1)
            if [ -n "$FOUND_VIDEO" ]; then
                mv "$FOUND_VIDEO" "$DST_VIDEO"
                video_note="video: $DST_VIDEO"
            else
                video_note="(no video)"
            fi
        fi

        if echo "$success_line" | grep -qi "True"; then
            echo "PASS  $video_note" | tee -a "$LOG"
            pass=$((pass + 1))
        elif echo "$success_line" | grep -qi "False"; then
            echo "FAIL  $video_note" | tee -a "$LOG"
            fail=$((fail + 1))
        else
            echo "ERROR $video_note" | tee -a "$LOG"
            echo "$output" | tail -10 >> "$LOG"
            error=$((error + 1))
        fi
    done
    echo "" | tee -a "$LOG"
done

total=$((pass + fail + error))
echo "========================================" | tee -a "$LOG"
echo "RESULTS: $pass/$total passed, $fail failed, $error errors" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Videos saved to: data/integration_test_${TIMESTAMP}/"
echo "Full log: $LOG"

if [ "$fail" -gt 0 ] || [ "$error" -gt 0 ]; then
    echo ""
    echo "Note: some expert failures at density 10 are expected —"
    echo "heavy clutter can block CuRobo paths. Check if the failure"
    echo "rate is similar to before the OOD expansion."
fi
