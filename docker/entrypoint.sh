#!/usr/bin/env bash
set -e

export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}

PROJECT_DIR=/workspace/project
TARGET_DIR=/tmp/python-deps

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "[entrypoint] Installing project requirements (target)..."
  mkdir -p "$TARGET_DIR"
  uv pip install \
    --target "$TARGET_DIR" \
    -r "$PROJECT_DIR/requirements.txt"
fi

export PYTHONPATH="$TARGET_DIR:${PYTHONPATH}"

# mkdir -p /workspace/generated_tests
# mkdir -p /workspace/pynguin-report

echo "[entrypoint] Running pynguin..."
exec pynguin "$@"
