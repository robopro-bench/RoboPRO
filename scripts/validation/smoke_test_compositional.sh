#!/bin/bash
# Smoke test for compositional eval configs (#22)
#
# Runs one task per scene on each compositional config to confirm
# no loading errors or crashes.
#
# Usage:
#   cd sim
#   source set_env.sh
#   bash ../scripts/validation/smoke_test_compositional.sh

SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONFIGS=(
    bench_compositional_object_d6
    bench_compositional_object_d10
    bench_compositional_object_d15
    bench_compositional_object_vision_d6
    bench_compositional_object_vision_d10
    bench_compositional_object_vision_d15
    bench_compositional_full_d6
    bench_compositional_full_d10
    bench_compositional_full_d15
)

# One task per scene
TASKS=(
    "put_mouse_on_pad:office"
    "put_cup_in_box:study"
    "put_bowl_in_sink_ks:kitchens"
    "move_bottle:kitchenl"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="../scripts/validation/results"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/smoke_compositional_${TIMESTAMP}.log"

echo "Compositional config smoke test" | tee "$LOG"
echo "Seed: $SEED" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

pass=0
fail=0
error=0

for config in "${CONFIGS[@]}"; do
    echo "=== $config ===" | tee -a "$LOG"

    # Pick one task to test (rotate through scenes)
    num_tasks=${#TASKS[@]}
    idx=$(( (pass + fail + error) % num_tasks ))
    entry="${TASKS[$idx]}"
    task="${entry%%:*}"
    scene="${entry##*:}"

    printf "  %s/%s  seed %s  " "$scene" "$task" "$SEED" | tee -a "$LOG"

    output=$(python script/bench_script/visualize_task_scene.py \
        "$task" "$config" \
        --bench-subdir "$scene" \
        --rollout --no-render \
        --seed "$SEED" 2>&1) || true

    success_line=$(echo "$output" | grep -i "^Success:" | tail -1)

    if echo "$success_line" | grep -qi "True"; then
        echo "PASS" | tee -a "$LOG"
        pass=$((pass + 1))
    elif echo "$success_line" | grep -qi "False"; then
        echo "FAIL (expert)" | tee -a "$LOG"
        fail=$((fail + 1))
    else
        echo "ERROR" | tee -a "$LOG"
        echo "$output" | tail -5 >> "$LOG"
        error=$((error + 1))
    fi
done

total=$((pass + fail + error))
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "RESULTS: $pass/$total passed, $fail failed, $error errors" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Log: $LOG"
echo ""
echo "Any result other than ERROR means the config loads and runs correctly."
echo "FAIL just means the expert couldn't solve that seed (expected at high density)."
