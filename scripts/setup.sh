#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "Installing Python dependencies..."
.venv/bin/python -m pip install -q -e .
echo "Python dependencies installed successfully."

if ! command -v zstd >/dev/null 2>&1; then
  apt-get update --quiet
  DEBIAN_FRONTEND=noninteractive apt-get install -y zstd
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

echo "Setup complete."
