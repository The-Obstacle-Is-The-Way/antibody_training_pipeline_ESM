#!/bin/bash
#
# Run Jain Novo Parity Inference in tmux
# Tests Boughter-trained model on corrected Jain 86-antibody dataset
#

SESSION_NAME="jain_inference"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./test_results/jain_novo_parity_86_${TIMESTAMP}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          Jain Novo Parity 86 Inference - Running in tmux                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset:     test_datasets/jain/VH_only_jain_novo_parity_86.csv"
echo "Model:       models/boughter_vh_esm1v_logreg.pkl"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Session:     ${SESSION_NAME}"
echo ""

# Kill existing session if it exists
tmux kill-session -t ${SESSION_NAME} 2>/dev/null

# Create new tmux session and run inference
tmux new-session -d -s ${SESSION_NAME} "
    echo '╔════════════════════════════════════════════════════════════════════════════╗';
    echo '║                    JAIN NOVO PARITY 86 INFERENCE                           ║';
    echo '╚════════════════════════════════════════════════════════════════════════════╝';
    echo '';
    echo 'Target Benchmark: Novo Nordisk 66.28% accuracy (68.6% in Sakhnini 2025)';
    echo 'Test Set: 86 antibodies (54 specific + 32 non-specific)';
    echo 'Corrected methodology: 0-10 flags, >=4 threshold, BVP included';
    echo '';
    echo 'Starting inference...';
    echo '';

    python3 test.py \
        --model models/boughter_vh_esm1v_logreg.pkl \
        --data test_datasets/jain/VH_only_jain_novo_parity_86.csv \
        --device mps \
        --batch-size 8 \
        --output-dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}_inference.log;

    echo '';
    echo '╔════════════════════════════════════════════════════════════════════════════╗';
    echo '║                         INFERENCE COMPLETE!                                ║';
    echo '╚════════════════════════════════════════════════════════════════════════════╝';
    echo '';
    echo 'Results saved to: ${OUTPUT_DIR}';
    echo 'Log file: ${OUTPUT_DIR}_inference.log';
    echo '';
    echo 'Press ENTER to exit tmux session...';
    read;
"

echo -e "${GREEN}✓ tmux session '${SESSION_NAME}' started!${NC}"
echo ""
echo "To attach to the session:"
echo -e "  ${BLUE}tmux attach -t ${SESSION_NAME}${NC}"
echo ""
echo "To view session list:"
echo -e "  ${BLUE}tmux ls${NC}"
echo ""
echo "To monitor progress:"
echo -e "  ${BLUE}tail -f ${OUTPUT_DIR}_inference.log${NC}"
echo ""
echo "Auto-attaching in 3 seconds..."
sleep 3

# Auto-attach to session
tmux attach -t ${SESSION_NAME}
