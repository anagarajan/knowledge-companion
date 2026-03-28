#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ingest.sh — Add a folder of PDFs to the knowledge base
#
# Usage:
#   ./ingest.sh <folder-name>            — ingest documents/<folder-name>
#   ./ingest.sh <folder-name> --force    — re-ingest even if already stored
#   ./ingest.sh <folder-name> --remove   — remove from knowledge base
#
# Examples:
#   ./ingest.sh HR
#   ./ingest.sh Legal --force
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/backend/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
die() { echo -e "${RED}✘  $*${NC}"; exit 1; }
ok()  { echo -e "${GREEN}✔${NC}  $*"; }

[[ -z "${1:-}" ]] && die "Usage: ./ingest.sh <folder-name> [--force] [--remove]"

FOLDER="$1"
shift
EXTRA_ARGS="$*"

FOLDER_PATH="$ROOT/documents/$FOLDER"

[[ -d "$FOLDER_PATH" ]] || die "Folder not found: $FOLDER_PATH"
[[ -f "$VENV/bin/activate" ]] || die "Backend not set up. Run ./start.sh first."

# Require the backend to be running (it means Postgres is up too)
curl -s http://localhost:8001/api/health &>/dev/null \
  || die "Backend is not running. Start the app first with: ./start.sh"

echo ""
echo "  Ingesting: $FOLDER_PATH"
echo "  ──────────────────────────────────────"

cd "$ROOT/backend"
"$VENV/bin/python" ingest.py --folder "$FOLDER_PATH" $EXTRA_ARGS

ok "Done. Open the app and select the '$FOLDER' folder to search it."
echo ""
