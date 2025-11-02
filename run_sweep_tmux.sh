#!/bin/bash
# Launch hyperparameter sweep in tmux session

SESSION_NAME="hyperparam_sweep"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/hyperparam_sweep_${TIMESTAMP}.log"

echo "================================================================"
echo "LAUNCHING HYPERPARAMETER SWEEP IN TMUX"
echo "================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo ""
echo "To attach to session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from session (while inside):"
echo "  Ctrl+B, then D"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "================================================================"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p hyperparameter_sweep_results

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session and run sweep
tmux new-session -d -s $SESSION_NAME

# Send commands to tmux session
tmux send-keys -t $SESSION_NAME "echo '================================================================='" C-m
tmux send-keys -t $SESSION_NAME "echo 'HYPERPARAMETER SWEEP STARTING'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Timestamp: $TIMESTAMP'" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================================='" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "python3 train_hyperparameter_sweep.py 2>&1 | tee $LOG_FILE" C-m

echo "✓ Tmux session created and sweep started!"
echo ""
echo "Run 'tmux attach -t $SESSION_NAME' to view progress"
echo "Or run 'tail -f $LOG_FILE' to follow logs"
