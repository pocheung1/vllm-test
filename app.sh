#!/usr/bin/env bash

# Launch vLLM OpenAI-compatible API server

MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

echo "Starting vLLM server with model: $MODEL_NAME"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8888 \
