#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  BOOTSTRAP_PYTHON="$ROOT_DIR/venv/bin/python"
elif [[ -n "${MUSUBI_PYTHON:-}" ]]; then
  BOOTSTRAP_PYTHON="$MUSUBI_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.12)"
elif command -v python3.11 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.10)"
elif command -v python3 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3)"
else
  echo "Python 3 was not found." >&2
  echo "Create a virtual environment in ./venv or set MUSUBI_PYTHON to its Python executable." >&2
  exit 1
fi

"$BOOTSTRAP_PYTHON" "$ROOT_DIR/tools/bootstrap_environment.py"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"

EXTRA_ARGS=()
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  EXTRA_ARGS+=(--no-browser)
fi

echo "Starting Musubi Studio..."
exec "$PYTHON_BIN" -m modern_gui.server "${EXTRA_ARGS[@]}" "$@"
