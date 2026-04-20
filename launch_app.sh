#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_app.sh — Start the Streamlit IT-ticket classifier
#
# Usage:
#   bash launch_app.sh              # default port 8501
#   bash launch_app.sh 8080         # custom port
#
# The VM must be running with GPU access.
# Access the app at: http://<VM-public-IP>:<PORT>
# ──────────────────────────────────────────────────────────────────────────────
set -e

PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the project's virtual environment if it exists
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✓ Activated .venv"
elif command -v poetry &>/dev/null; then
    echo "✓ Using poetry run"
    RUNNER="poetry run"
fi

# Load .env if present (picks up HF_TOKEN, AZURE keys, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo "✓ Loaded .env"
fi

echo ""
echo "  Starting IT Ticket Classifier on port $PORT"
echo "  → http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""

cd "$SCRIPT_DIR"

$RUNNER streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
