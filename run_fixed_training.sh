#!/bin/bash
# Run corrected training pipeline with all P0/P1 fixes applied
# Expected: CV ~70-71%, Jain test ~65-69%

SESSION_NAME="fixed_training"
LOG_FILE="logs/fixed_training_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p logs

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session and run training
tmux new-session -d -s $SESSION_NAME "python3 train.py config_boughter.yaml 2>&1 | tee $LOG_FILE"

echo "✅ Training started in tmux session: $SESSION_NAME"
echo "📋 Log file: $LOG_FILE"
echo ""
echo "Commands:"
echo "  • Attach: tmux attach -t $SESSION_NAME"
echo "  • Monitor: tail -f $LOG_FILE"
echo "  • Check status: tmux ls"
echo ""
echo "Expected timeline:"
echo "  • Embedding extraction: ~5-10 min"
echo "  • 10-fold CV: ~15-20 min"
echo "  • Training + Jain test: ~5-10 min"
echo "  • Total: ~30-45 min"
echo ""
echo "Expected results (WITH FIXES):"
echo "  • Boughter 10-CV: ~70-71% (was 67.5%)"
echo "  • Jain test: ~65-69% (was 55.3%)"
