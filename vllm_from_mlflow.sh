#!/bin/bash

set -e

# Define the registered model name and version/stage
MODEL_NAME="TinyLlama-1.1B-Chat-v1.0"
MODEL_VERSION="1"

echo "Checking if model '$MODEL_NAME' version '$MODEL_VERSION' exists in MLflow..."

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

echo "Downloading model from MLflow..."

LOCAL_MODEL_DIR=$(python3 - <<EOF
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

model_uri = f"models:/${MODEL_NAME}/${MODEL_VERSION}"
download_dir = ModelsArtifactRepository(model_uri).download_artifacts("")

# Look for the actual model subdirectory
model_dir = os.path.join(download_dir, "model")
if os.path.exists(os.path.join(model_dir, "config.json")):
    print(model_dir)
else:
    raise RuntimeError("Hugging Face config.json not found under model subdirectory")
EOF
)

echo "Model downloaded to $LOCAL_MODEL_DIR"

echo "Starting vLLM server..."

python3 -m vllm.entrypoints.openai.api_server \
  --model "$LOCAL_MODEL_DIR" \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8888
