#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export BENCH_ROOT="$SCRIPT_DIR"
export SIM_ROOT="$WORKSPACE_ROOT/sim"

echo "BENCH_ROOT=$BENCH_ROOT"
echo "SIM_ROOT=$SIM_ROOT"
