#!/bin/bash
# Baseline comparison for OOD obstacle expansion (#19)
#
# Runs the same tasks and seeds as integration_test_ood.sh but with
# seen-obstacle configs (no OOD), so you can compare pass rates.
#
# Usage:
#   source set_env.sh  # repo root
#   cd sim
#   bash ../scripts/validation/baseline_test_ood.sh

SEEDS="${SEEDS:-0 1 2}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Each entry: task:scene:config
TASKS=(
    "put_mouse_on_pad:office:bench_demo_office_d10"
    "put_cup_in_box:study:bench_demo_study_d10"
    "put_bowl_in_sink_ks:kitchens:bench_demo_kitchens_d10"
    "put_bottle_in_fridge:kitchenl:bench_demo_kitchenl_d10"
)

RESULTS_DIR="../scripts/validation/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RESULTS_DIR/baseline_test_${TIMESTAMP}.log"

echo "Baseline test: seen obstacles (no OOD)" | tee "$LOG"
echo "Seeds: $SEEDS" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

pass=0
fail=0
error=0

for entry in "${TASKS[@]}"; do
    IFS=':' read -r task scene config <<< "$entry"

    echo "=== $scene / $task  (config: $config) ===" | tee -a "$LOG"

    for seed in $SEEDS; do
        printf "  seed %-4s " "$seed" | tee -a "$LOG"

        SAVE_SUBDIR="data/baseline_test_${TIMESTAMP}/${scene}_${task}"
        mkdir -p "$SAVE_SUBDIR"

        output=$(python script/bench_script/visualize_task_scene.py \
            "$task" "$config" \
            --bench-subdir "$scene" \
            --rollout --no-render \
            --seed "$seed" --save_data 2>&1) || true

        success_line=$(echo "$output" | grep -i "^Success:" | tail -1)

        SRC_VIDEO="data/bench_data/video/episode_${task}_0.mp4"
        DST_VIDEO="$SAVE_SUBDIR/seed_${seed}.mp4"
        if [ -f "$SRC_VIDEO" ]; then
            mv "$SRC_VIDEO" "$DST_VIDEO"
            video_note="video: $DST_VIDEO"
        else
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
echo "Videos saved to: data/baseline_test_${TIMESTAMP}/"
echo "Full log: $LOG"
echo ""
echo "Compare these results against the OOD integration test."
echo "If pass rates are similar, the OOD expansion didn't degrade anything."
