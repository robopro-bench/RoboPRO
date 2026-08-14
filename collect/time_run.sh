#!/usr/bin/env bash
# time_run.sh — time one collect_data.sh run and project full-dataset generation time.
#
# Usage (from repo root):
#   bash collect/time_run.sh <task> <config> [gpu]
# Example:
#   bash collect/time_run.sh put_cup_on_coaster datagen_template 1
#
# Prints: wall time, episodes attempted vs kept, seconds per attempt / per kept,
# and a projection for TARGET_EPISODES total kept episodes on 1/4/20 GPUs.
set -u

COLLECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SIM_ROOT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${COLLECT_DIR}/../set_env.sh"
fi

TASK="${1:?usage: time_run.sh <task> <config> [gpu]}"
CONFIG="${2:?usage: time_run.sh <task> <config> [gpu]}"
GPU="${3:-0}"

# full-dataset target (kept episodes); override per call:
#   TARGET_EPISODES=24000 bash collect/time_run.sh ...
TARGET_EPISODES="${TARGET_EPISODES:-16000}"

LOG="$(mktemp /tmp/collect_time_XXXXXX.log)"
echo "▶ timing: bash collect/collect_data.sh $TASK $CONFIG $GPU   (log: $LOG)"
echo "────────────────────────────────────────────────────────────────────"
start=$(date +%s)
bash "${COLLECT_DIR}/collect_data.sh" "$TASK" "$CONFIG" "$GPU" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
end=$(date +%s)
wall=$(( end - start ))

# parse the COLLECTION SUMMARY the collector prints
att=$(grep -aoiE 'episodes attempted[[:space:]]+[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')
kept=$(grep -aoiE 'kept in dataset[[:space:]]+[0-9]+'   "$LOG" | tail -1 | grep -oE '[0-9]+$')
att=${att:-0}; kept=${kept:-0}

echo "════════════════════════════════════════════════════════════════════"
echo "TIMING RESULT · $TASK / $CONFIG  (exit $rc)"
echo "  wall clock            ${wall}s  (~$(awk "BEGIN{printf \"%.1f\", $wall/60}") min)"
echo "  episodes attempted    ${att}"
echo "  episodes kept         ${kept}"
if [ "$att" -gt 0 ]; then
  echo "  sec / attempted ep    $(awk "BEGIN{printf \"%.1f\", $wall/$att}")"
fi
if [ "$kept" -gt 0 ]; then
  echo "  sec / kept ep         $(awk "BEGIN{printf \"%.1f\", $wall/$kept}")"
fi

if [ "$att" -gt 0 ] && [ "$kept" -gt 0 ]; then
  echo "────────────────────────────────────────────────────────────────────"
  echo "PROJECTION → ${TARGET_EPISODES} kept episodes at THIS run's sec/kept rate"
  awk -v wall="$wall" -v kept="$kept" -v tgt="$TARGET_EPISODES" 'BEGIN{
    total = tgt * wall/kept;              # seconds on 1 GPU
    printf "  %-8s %10s %10s\n","GPUs","hours","days";
    split("1 4 20", g, " ");
    for(i=1;i<=3;i++){ n=g[i]; h=total/n/3600.0;
      printf "  %-8s %10.1f %10.2f\n", n, h, h/24.0; }
  }'
  echo "  NB: warmup (CuRobo init per launch) + seed-search rejects inflate real"
  echo "      time; time a low- AND a high-density config to bracket the range."
else
  echo "  (could not parse attempted/kept from the run — check $LOG)"
fi
