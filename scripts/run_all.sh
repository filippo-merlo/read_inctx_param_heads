#!/usr/bin/env bash
# The whole official pipeline, in dependency order, on four GPUs (nvidia-smi indices).
#   scripts/run_all.sh <smoke|full> [gpu_a gpu_b gpu_c gpu_d]
set -euo pipefail

S=$(dirname "$0")/run_atlas.sh
MODE=$1
A=${2:-0} B=${3:-2} C=${4:-3} D=${5:-5}

$S "$MODE" configs
# Counterfactual and deduplicated closed-book feed head identification; oracle and
# closed-book are needed for Table 4 only. Launches are staggered: this node enforces a
# strict memory-commit limit, and four simultaneous model loads exceed it.
STAGGER=${STAGGER:-90}
( $S "$MODE" infer cf "$A" && $S "$MODE" bem cf "$A" && $S "$MODE" relevance cf "$A" \
    && $S "$MODE" composition "$A" ) & pids=($!)
sleep "$STAGGER"
( $S "$MODE" infer cb_dedupe "$B" && $S "$MODE" bem cb_dedupe "$B" \
    && $S "$MODE" relevance cb_dedupe "$B" ) & pids+=($!)
sleep "$STAGGER"
$S "$MODE" infer oracle "$C" & pids+=($!)
sleep "$STAGGER"
$S "$MODE" infer cb "$D" & pids+=($!)
for pid in "${pids[@]}"; do wait "$pid"; done  # set -e stops here if any chain failed
$S "$MODE" rank
$S "$MODE" save_ids
$S "$MODE" export
# One at a time: the BEM metric forks 8 DataLoader workers, which exceeds this node's
# strict memory-commit limit when several run at once.
for run in cf oracle cb; do $S "$MODE" evals "$run" "$A"; done
