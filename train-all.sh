#!/usr/bin/env bash
# Train all configurations that appear in the paper's results table.
# Each run writes to ./cache/results-{ablation}-{model_type}-{group_col}.pkl
# and is picked up automatically by 04-xai-analysis.py.
set -euo pipefail

export PYTHONHASHSEED=42

BASE="--group-col tree_id"

echo "=== GBDT ==="
uv run train.py --model-type gbdt  --ablation all             $BASE; echo ""
uv run train.py --model-type gbdt  --ablation no-defoliation  $BASE; echo ""
uv run train.py --model-type gbdt  --ablation plot-level-only $BASE; echo ""
uv run train.py --model-type gbdt  --ablation tree-level-only $BASE; echo ""

echo "=== ElasticNet ==="
uv run train.py --model-type elasticnet --ablation all             $BASE; echo ""
uv run train.py --model-type elasticnet --ablation no-defoliation  $BASE; echo ""
uv run train.py --model-type elasticnet --ablation plot-level-only $BASE; echo ""
uv run train.py --model-type elasticnet --ablation tree-level-only $BASE; echo ""

echo "=== LMM ==="
uv run train.py --model-type lmm --ablation all             $BASE; echo ""
uv run train.py --model-type lmm --ablation no-defoliation  $BASE; echo ""
uv run train.py --model-type lmm --ablation plot-level-only $BASE; echo ""
uv run train.py --model-type lmm --ablation tree-level-only $BASE; echo ""

BASE_TEMP_BLOCK="--group-col tree_id --temporal-cv"

echo "=== GBDT (no temporal CV) ==="
uv run train.py --model-type gbdt  --ablation all $BASE_TEMP_BLOCK; echo ""
uv run train.py --model-type gbdt  --ablation tree-level-only $BASE_TEMP_BLOCK; echo ""

echo "=== ElasticNet (no temporal CV) ==="
uv run train.py --model-type elasticnet --ablation all $BASE_TEMP_BLOCK; echo ""
uv run train.py --model-type elasticnet --ablation tree-level-only $BASE_TEMP_BLOCK; echo ""

echo "=== LMM (no temporal CV) ==="
uv run train.py --model-type lmm --ablation all $BASE_TEMP_BLOCK; echo ""
uv run train.py --model-type lmm --ablation tree-level-only $BASE_TEMP_BLOCK; echo ""

BASE_PLOT="--group-col plot_id"

echo "=== GBDT (plot_id) ==="
uv run train.py --model-type gbdt  --ablation all $BASE_PLOT; echo ""
uv run train.py --model-type gbdt  --ablation tree-level-only $BASE_PLOT; echo ""

echo "=== ElasticNet (plot_id) ==="
uv run train.py --model-type elasticnet --ablation all $BASE_PLOT; echo ""
uv run train.py --model-type elasticnet --ablation tree-level-only $BASE_PLOT; echo ""

echo "=== All configurations done ==="
