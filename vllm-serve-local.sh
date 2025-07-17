#!/usr/bin/env bash

set -euo pipefail

# Define model info and target directory
HF_MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_MODEL_DIR="./models/TinyLlama-1.1B-Chat-v1.0"

echo "Checking for local model at: $LOCAL_MODEL_DIR"

# If the local model directory doesn't exist or is empty, download it
if [ ! -d "$LOCAL_MODEL_DIR" ] || [ -z "$(ls -A "$LOCAL_MODEL_DIR")" ]; then
  echo "Downloading model $HF_MODEL_ID into $LOCAL_MODEL_DIR..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_MODEL_ID', local_dir='$LOCAL_MODEL_DIR', local_dir_use_symlinks=False)
"
else
  echo "Model already exists locally. Skipping download."
fi

# Start the vLLM OpenAI-compatible server from local model path
echo "Starting vLLM server from local path: $LOCAL_MODEL_DIR"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$LOCAL_MODEL_DIR" \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8888
