#!/bin/bash
# Run Boughter transfer learning experiment in tmux

SESSION_NAME="ginkgo_transfer"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create output directory
mkdir -p experiment_results

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Run the experiment with uv (to use correct dependencies)
tmux send-keys -t $SESSION_NAME "cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM" C-m
tmux send-keys -t $SESSION_NAME "uv run python scripts/ginkgo_boughter_transfer.py 2>&1 | tee experiment_results/boughter_transfer_esm1v_log.txt" C-m

echo "=================================="
echo "✅ Boughter transfer experiment running in tmux!"
echo "=================================="
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    # Attach to session"
echo "  tmux detach (Ctrl+B, D)          # Detach from session"
echo "  tmux kill-session -t $SESSION_NAME  # Kill session"
echo ""
echo "Monitor progress:"
echo "  tail -f experiment_results/boughter_transfer_esm1v_log.txt"
echo ""
echo "Session will keep running in background even if you detach."
echo "=================================="
