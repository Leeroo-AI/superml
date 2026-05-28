#!/usr/bin/env bash
# superml-cli installer
# Usage: bash install.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${HOME}/.local/bin"

mkdir -p "$BIN_DIR"
ln -sf "${REPO_DIR}/superml" "${BIN_DIR}/superml"
echo "Installed: ${BIN_DIR}/superml -> ${REPO_DIR}/superml"

# Check if ~/.local/bin is on PATH
if ! echo "$PATH" | grep -q "${BIN_DIR}"; then
    echo ""
    echo "Add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "Then run: source ~/.bashrc"
fi
