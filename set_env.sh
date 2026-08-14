#!/usr/bin/env bash

# Repo-root env for collection and anything else that should not cd into sim/.
export WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIM_ROOT="$WORKSPACE_ROOT/sim"
export BENCH_ROOT="$WORKSPACE_ROOT/benchmark"
export DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/data}"
export POLICY_ROOT="$WORKSPACE_ROOT/policy"

echo "WORKSPACE_ROOT=$WORKSPACE_ROOT"
echo "SIM_ROOT=$SIM_ROOT"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "POLICY_ROOT=$POLICY_ROOT"
