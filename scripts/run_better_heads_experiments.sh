#!/bin/bash
# Run ElasticNet and LightGBM experiments in parallel tmux sessions

# Kill existing sessions if they exist
tmux kill-session -t elasticnet 2>/dev/null
tmux kill-session -t lightgbm 2>/dev/null

# Create output directory
mkdir -p experiment_results

echo "=================================="
echo "🚀 LAUNCHING BETTER HEADS EXPERIMENTS"
echo "=================================="
echo ""
echo "Running two experiments in parallel:"
echo "  1. ElasticNet (30 min - diagnostic)"
echo "  2. LightGBM (2 hours - most likely win)"
echo ""

# Session 1: ElasticNet
tmux new-session -d -s elasticnet
tmux send-keys -t elasticnet "cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM" C-m
tmux send-keys -t elasticnet "uv run python scripts/generate_elasticnet_submission.py 2>&1 | tee experiment_results/elasticnet_log.txt" C-m

echo "✅ ElasticNet running in tmux session: elasticnet"

# Session 2: LightGBM
tmux new-session -d -s lightgbm
tmux send-keys -t lightgbm "cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM" C-m
tmux send-keys -t lightgbm "uv run python scripts/generate_lightgbm_submission.py 2>&1 | tee experiment_results/lightgbm_log.txt" C-m

echo "✅ LightGBM running in tmux session: lightgbm"

echo ""
echo "=================================="
echo "Monitor experiments:"
echo "=================================="
echo ""
echo "ElasticNet:"
echo "  tmux attach -t elasticnet"
echo "  tail -f experiment_results/elasticnet_log.txt"
echo ""
echo "LightGBM:"
echo "  tmux attach -t lightgbm"
echo "  tail -f experiment_results/lightgbm_log.txt"
echo ""
echo "Detach: Ctrl+B, D"
echo "Kill: tmux kill-session -t <session>"
echo ""
echo "=================================="
echo "Expected timeline:"
echo "=================================="
echo "  ElasticNet: ~30 min"
echo "  LightGBM:   ~2 hours"
echo ""
echo "Both sessions will keep running in background!"
echo "=================================="
