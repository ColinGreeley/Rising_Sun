#!/usr/bin/env bash
set -euo pipefail

# Build standalone Rising Sun executable for the current platform.
# Usage: bash scripts/build_standalone.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "=== Step 1: Build frontend ==="
if command -v node >/dev/null 2>&1; then
    NODE=node
elif [[ -x "$HOME/.nvm/versions/node/v24.11.0/bin/node" ]]; then
    export PATH="$HOME/.nvm/versions/node/v24.11.0/bin:$PATH"
    NODE=node
else
    echo "ERROR: Node.js 20+ required. Install via nvm or system package." >&2
    exit 1
fi
echo "  Using Node $(node --version)"
cd web/frontend
VITE_API_BASE_URL="" npm run build
cd "$ROOT"

echo ""
echo "=== Step 2: PyInstaller build ==="
# Use CPU-only torch to keep size down
export RISING_SUN_RAPIDOCR_USE_CUDA=0
pyinstaller --noconfirm --clean rising_sun.spec

echo ""
echo "=== Step 3: Package ==="
DIST_DIR="dist/RisingSun"
if [[ -d "$DIST_DIR" ]]; then
    PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    ARCHIVE="dist/RisingSun-${PLATFORM}-${ARCH}.tar.gz"
    tar -czf "$ARCHIVE" -C dist RisingSun
    SIZE=$(du -sh "$ARCHIVE" | cut -f1)
    echo "  Built: $ARCHIVE ($SIZE)"
else
    echo "ERROR: dist/RisingSun not found — build may have failed." >&2
    exit 1
fi

echo ""
echo "=== Done ==="
echo "  To run: ./dist/RisingSun/RisingSun"
echo "  To distribute: share $ARCHIVE"
