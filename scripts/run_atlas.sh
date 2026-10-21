#!/usr/bin/env bash
# Run one stage of the official Atlas-of-ICL pipeline, in its README order, with our configs.
#
#   scripts/run_atlas.sh <smoke|full> configs                 render configs/atlas into the clone
#   scripts/run_atlas.sh <smoke|full> infer       <run> <gpu>  ablation_infer/main.py
#   scripts/run_atlas.sh <smoke|full> evals       <run> <gpu>  main.py with evals_only (Recall, EM, BEM, K-Prec)
#   scripts/run_atlas.sh <smoke|full> bem         <run> <gpu>  preprocessing/score_bem.py
#   scripts/run_atlas.sh <smoke|full> relevance   <run> <gpu>  preprocessing/get_relevance.py
#   scripts/run_atlas.sh <smoke|full> rank                     preprocessing/rank_mean_retrieval_histogram.py
#   scripts/run_atlas.sh <smoke|full> composition <gpu>        preprocessing/filter_heads_composition.py
#   scripts/run_atlas.sh <smoke|full> save_ids                 preprocessing/save_head_ids.py
#   scripts/run_atlas.sh <smoke|full> export                   scripts/export_heads.py -> results/heads.json (full)
#
# runs: cf (counterfactual), oracle, cb (closed-book), cb_dedupe (closed-book, deduplicated
# questions; the official code uses it only for relevance extraction).
# `configs` rewrites the clone's global_path.yml, so smoke and full must not run at once.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ATLAS=$ROOT/third_party/in-context-atlas
PY=$ATLAS/.venv/bin/python
MODE=$1
STAGE=$2
DATA=$ROOT/data/$MODE
LOGS=$ROOT/logs/$MODE
mkdir -p "$LOGS"

export JAVA_HOME=$HOME/.local/jdk-11
export PATH=$JAVA_HOME/bin:$PATH
export WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # heterogeneous node: make indices match nvidia-smi

render() {  # template -> destination, with @ROOT@ set to this mode's data dir
  sed "s#@ROOT@#$DATA#g" "$1" > "$2"
}

run() {  # dir log cmd...
  local dir=$1 log=$2; shift 2
  echo "[$(date -u +%FT%TZ)] $MODE $STAGE: $*" | tee -a "$LOGS/$log"
  (cd "$ATLAS/$dir" && "$PY" "$@") >> "$LOGS/$log" 2>&1
  echo "[$(date -u +%FT%TZ)] done" | tee -a "$LOGS/$log"
}

case $STAGE in
  configs)
    git -C "$ATLAS" clean -fq ablation_infer/exp_configs preprocessing/preprocess_configs
    render "$ROOT/configs/atlas/global_path.yml" "$ATLAS/configs/global_path.yml"
    for f in "$ROOT"/configs/atlas/exp_configs/*.yml; do
      render "$f" "$ATLAS/ablation_infer/exp_configs/$(basename "$f")"
    done
    for f in "$ROOT"/configs/atlas/preprocess_configs/*.yml; do
      render "$f" "$ATLAS/preprocessing/preprocess_configs/$(basename "$f")"
    done
    # Created up front: the official AblationConfig creates it with exists-then-makedirs,
    # which races when the inference runs start in parallel.
    mkdir -p "$DATA/results/ablation"
    echo "configs rendered for $MODE ($DATA)" ;;
  infer)
    export CUDA_VISIBLE_DEVICES=$4
    run ablation_infer "infer_$3.log" main.py --config_file "$3.yml" ;;
  evals)
    export CUDA_VISIBLE_DEVICES=$4
    run ablation_infer "evals_$3.log" main.py --config_file "$3_eval.yml" ;;
  bem)
    export CUDA_VISIBLE_DEVICES=$4
    # spaCy's transformer runs on CPU; its default of one thread per core oversubscribes
    # the node next to the GPU jobs (3.8 s/item measured, vs 0.14 s/item at 8 threads).
    export OMP_NUM_THREADS=8
    run preprocessing "bem_$3.log" score_bem.py --config_file "$3.yml" ;;
  relevance)
    export CUDA_VISIBLE_DEVICES=$4
    run preprocessing "relevance_$3.log" get_relevance.py --config_file "$3.yml" ;;
  rank)
    export CUDA_VISIBLE_DEVICES=""
    run preprocessing rank.log rank_mean_retrieval_histogram.py --config_file cf.yml cf cb_dedupe ;;
  composition)
    export CUDA_VISIBLE_DEVICES=$3
    run preprocessing composition.log filter_heads_composition.py --config_file cf.yml ;;
  save_ids)
    export CUDA_VISIBLE_DEVICES=""
    run preprocessing save_ids.log save_head_ids.py --config_file cf.yml ;;
  export)
    export CUDA_VISIBLE_DEVICES=""
    out=$ROOT/results/heads.json
    [ "$MODE" = full ] || out=$DATA/heads.json
    run . export.log "$ROOT/scripts/export_heads.py" "$out" ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
