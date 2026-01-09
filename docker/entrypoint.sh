#!/usr/bin/env bash
set -euxo pipefail

export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}

PROJECT_DIR=/workspace/project
PROJECT_NAME=${PROJECT_NAME:?PROJECT_NAME is not set}

CACHE_ROOT=/workspace/.project-deps
TARGET_DIR=$CACHE_ROOT/$PROJECT_NAME/site-packages
FLAG_FILE=$CACHE_ROOT/$PROJECT_NAME/installed.flag

mkdir -p "$TARGET_DIR"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  if [ ! -f "$FLAG_FILE" ]; then
    echo "[entrypoint] Installing project deps for $PROJECT_NAME..."
    uv pip install \
      --target "$TARGET_DIR" \
      -r "$PROJECT_DIR/requirements.txt"
    touch "$FLAG_FILE"
  else
    echo "[entrypoint] Reusing cached project deps for $PROJECT_NAME"
  fi
fi

export PYTHONPATH="$TARGET_DIR:${PYTHONPATH:-}"

echo "[entrypoint] Running pynguin..."
exec pynguin "$@"
