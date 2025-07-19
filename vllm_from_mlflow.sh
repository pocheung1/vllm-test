#!/bin/bash

# Purpose:
#   - Download a registered LLM from MLflow Model Registry
#   - Reconstruct the model directory in Hugging Face format required by vLLM
#   - Launch the vLLM OpenAI-compatible HTTP server to serve the model

set -e

# Define the registered model name and version
MODEL_NAME="TinyLlama-1.1B-Chat-v1.0"
MODEL_VERSION="1"

echo "Checking if model '$MODEL_NAME' version '$MODEL_VERSION' exists in MLflow..."

# Check if the specified model version exists in MLflow
python3 - <<EOF
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "${MODEL_NAME}"
model_version = "${MODEL_VERSION}"

try:
    mv = client.get_model_version(model_name, model_version)
    print(f"Model version {model_version} of '{model_name}' exists.")
except Exception:
    import sys
    print(f"Model version {model_version} of '{model_name}' not found.", file=sys.stderr)
    sys.exit(1)
EOF

echo "Downloading model from MLflow and preparing vLLM directory..."

# Download and assemble model directory in Hugging Face format for vLLM
VLLM_MODEL_DIR=$(python3 - <<EOF
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from pathlib import Path
import shutil
import os

# Construct MLflow model URI and download artifact
model_uri = f"models:/${MODEL_NAME}/${MODEL_VERSION}"
download_path = Path(ModelsArtifactRepository(model_uri).download_artifacts(""))

# Output directory to reconstruct the Hugging Face model for vLLM
output_path = Path("/tmp/vllm_model")
output_path.mkdir(parents=True, exist_ok=True)

# Expected MLflow subdirectories
model_dir = download_path / "model"
tokenizer_dir = download_path / "components" / "tokenizer"

# Check for critical files
if not (model_dir / "config.json").exists():
    raise RuntimeError("config.json not found under model/")
if not (tokenizer_dir / "tokenizer.model").exists():
    raise RuntimeError("tokenizer.model not found under components/tokenizer/")

# Copy model weights and tokenizer files into the output directory
shutil.copytree(model_dir, output_path, dirs_exist_ok=True)
shutil.copytree(tokenizer_dir, output_path, dirs_exist_ok=True)

# Return path to shell
print(output_path)
EOF
)

echo "Starting vLLM server..."

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8888
