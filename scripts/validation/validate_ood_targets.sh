#!/bin/bash
# Validate OOD target variants for Layer 2 (#21)
#
# For each candidate target variant, forces it as the only OOD target ID
# and runs an expert rollout to confirm grasp + place + check_success().
#
# Usage:
#   source set_env.sh  # repo root
#   cd sim
#   bash ../scripts/validation/validate_ood_targets.sh
#
# Options:
#   SEEDS="0 1 2"    # seeds per candidate (default: 0 1)
#   SCENE=office     # validate one scene only

SEEDS="${SEEDS:-0 1}"
TASK_OBJECTS="../benchmark/bench_task_config/task_objects.yml"
BACKUP="${TASK_OBJECTS}.bak"
CONFIG="bench_object_ood_unseen_targets_clean"

# Check config exists — need unseen_targets with density 0
CONFIG_PATH="../benchmark/bench_task_config/${CONFIG}.yml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Creating clean OOD targets config: $CONFIG_PATH"
    cat > "$CONFIG_PATH" << 'YAMLEOF'
# Minimal config for validating OOD target variants.
# unseen_targets=true, no clutter, no vision perturbation.
render_freq: 0
episode_num: 1
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
  obstacle_density: 0
  language_perturbation:
    enabled: true
    instruction_bank: benchmark/bench_task_config/instruction_bank.json
  object_perturbation:
    unseen_obstacles: false
    unseen_targets: true
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  demo_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  third_view: false
  depth: true
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./data/bench_data
clear_cache_freq: 1
collect_data: false
eval_video_log: true
enable_collision_metrics: false
YAMLEOF
fi

# Candidate targets: object:scene:task:candidate_ids
# One representative task per object per scene.
CANDIDATES=(
    # Office
    "043_book:office:put_book_in_fileholder:1"
    "046_alarm-clock:office:put_phone_next_to_cube:4"
    "048_stapler:office:put_stapler_in_drawer:5,6"
    "101_milk-tea:office:put_milktea_on_shelf:4,6"
    "061_battery:office:move_items_around:4"
    "073_rubikscube:office:put_rubikscube_in_drawer:1"
    "059_pencup:office:put_mouse_on_pad:4,6"
    # Study
    "021_cup:study:put_cup_in_box:4,6,7,8,9"
    "058_markpen:study:move_cup:2,4"
    "095_glue:study:put_glue_in_box:6"
    "100_seal:study:put_seal_in_box:1,4,6"
    # Kitchens
    "006_hamburg:kitchens:move_hamburger_onto_plate_ks:4,5"
    "035_apple:kitchens:drop_apple_in_bin_ks:0"
    "075_bread:kitchens:put_bread_on_board_ks:2,5,6"
    # Kitchenl
    "001_bottle:kitchenl:move_bottle:0,2,3,5,7,8,9,10"
    "038_milk-box:kitchenl:put_milk_box_in_fridge:1"
    "071_can:kitchenl:put_can_in_cabinet:3,6"
    "105_sauce-can:kitchenl:put_sauce_can_in_basket:0,2"
    "114_bottle:kitchenl:move_bottle:1,2,3"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="../scripts/validation/results"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/target_validation_${TIMESTAMP}.log"
VIDEO_DIR="data/target_validation_${TIMESTAMP}"

echo "OOD Target Variant Validation" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Backup task_objects.yml
cp "$TASK_OBJECTS" "$BACKUP"
trap 'cp "$BACKUP" "$TASK_OBJECTS"; rm -f "$BACKUP"' EXIT

pass=0
fail=0
error=0
skip=0

for entry in "${CANDIDATES[@]}"; do
    IFS=':' read -r obj scene task ids_str <<< "$entry"

    # Filter by scene if specified
    if [ -n "${SCENE:-}" ] && [ "$scene" != "$SCENE" ]; then
        continue
    fi

    IFS=',' read -ra ids <<< "$ids_str"

    for id in "${ids[@]}"; do
        label="${obj}/${id} on ${scene}/${task}"
        echo "--- $label ---" | tee -a "$LOG"

        # Patch task_objects.yml: set OOD targets for this scene+object to only this ID
        # Use python to safely modify YAML
        python3 -c "
import yaml, sys

with open('$TASK_OBJECTS') as f:
    d = yaml.safe_load(f)

ood = d.setdefault('object_ood', {})
scene_cfg = ood.setdefault('$scene', {})
targets = scene_cfg.setdefault('targets', {})
targets['$obj'] = [$id]

with open('$TASK_OBJECTS', 'w') as f:
    yaml.dump(d, f, default_flow_style=False, sort_keys=False)
" 2>&1

        if [ $? -ne 0 ]; then
            echo "  ERROR patching YAML" | tee -a "$LOG"
            error=$((error + 1))
            cp "$BACKUP" "$TASK_OBJECTS"
            continue
        fi

        for seed in $SEEDS; do
            printf "  seed %-4s " "$seed" | tee -a "$LOG"

            SAVE_SUBDIR="${VIDEO_DIR}/${scene}_${task}"
            mkdir -p "$SAVE_SUBDIR"

            output=$(python script/bench_script/visualize_task_scene.py \
                "$task" "$CONFIG" \
                --bench-subdir "$scene" \
                --rollout --no-render \
                --seed "$seed" --save_data 2>&1) || true

            success_line=$(echo "$output" | grep -i "^Success:" | tail -1)

            # Move video
            SRC_VIDEO="data/bench_data/video/episode_${task}_0.mp4"
            DST_VIDEO="${SAVE_SUBDIR}/${obj}_id${id}_seed${seed}.mp4"
            if [ -f "$SRC_VIDEO" ]; then
                mv "$SRC_VIDEO" "$DST_VIDEO"
                video_note="video: $DST_VIDEO"
            else
                FOUND=$(find data/bench_data/video/ -name "*${task}*" -newer "$LOG" -type f 2>/dev/null | head -1)
                if [ -n "$FOUND" ]; then
                    mv "$FOUND" "$DST_VIDEO"
                    video_note="video: $DST_VIDEO"
                else
                    video_note="(no video)"
                fi
            fi

            if echo "$success_line" | grep -qi "True"; then
                echo "PASS  ${obj}/${id}  $video_note" | tee -a "$LOG"
                pass=$((pass + 1))
            elif echo "$success_line" | grep -qi "False"; then
                echo "FAIL  ${obj}/${id}  $video_note" | tee -a "$LOG"
                fail=$((fail + 1))
            else
                echo "ERROR ${obj}/${id}  $video_note" | tee -a "$LOG"
                echo "$output" | tail -5 >> "$LOG"
                error=$((error + 1))
            fi
        done

        # Restore original YAML after each candidate
        cp "$BACKUP" "$TASK_OBJECTS"
    done
done

total=$((pass + fail + error))
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "RESULTS: $pass/$total passed, $fail failed, $error errors" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Videos: $VIDEO_DIR/"
echo "Log: $LOG"
echo ""
echo "Variants that PASS on at least one seed are safe to register."
echo "Review FAIL videos to check if the failure is grasp-related"
echo "(reject the variant) or seed-related (try more seeds)."
