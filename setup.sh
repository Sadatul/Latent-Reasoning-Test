#!/usr/bin/env bash
# Create a uv environment in this folder and build the datasets.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh

uv venv .venv
uv pip install --python .venv huggingface_hub
.venv/bin/python prepare_datasets.py
