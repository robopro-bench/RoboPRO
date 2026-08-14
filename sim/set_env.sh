#!/usr/bin/env bash

# absolute path to the RoboPRO simulation runtime (this directory)
export SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WORKSPACE_ROOT="$(cd "$SIM_ROOT/.." && pwd)"
export BENCH_ROOT="$WORKSPACE_ROOT/benchmark"

echo "BENCH_ROOT=$BENCH_ROOT"
echo "SIM_ROOT=$SIM_ROOT"
