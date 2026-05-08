#!/usr/bin/env bash
# Train all configurations that appear in the paper's results table.
# Each run writes to ./cache/results-{ablation}-{model_type}-{group_col}.pkl
# and is picked up automatically by 04-xai-analysis.py.
set -euo pipefail

BASE="--group-col tree_id --temporal-cv"

echo "=== GBDT ==="
uv run train.py --model-type gbdt  --ablation all             $BASE; echo ""
uv run train.py --model-type gbdt  --ablation no-defoliation  $BASE; echo ""
uv run train.py --model-type gbdt  --ablation plot-level-only $BASE; echo ""
uv run train.py --model-type gbdt  --ablation tree-level-only $BASE; echo ""

echo "=== Lasso ==="
uv run train.py --model-type lasso --ablation all             $BASE; echo ""
uv run train.py --model-type lasso --ablation no-defoliation  $BASE; echo ""
uv run train.py --model-type lasso --ablation plot-level-only $BASE; echo ""
uv run train.py --model-type lasso --ablation tree-level-only $BASE; echo ""

echo "=== All configurations done ==="
