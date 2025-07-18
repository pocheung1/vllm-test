#!/usr/bin/env bash
#
# Script: vllm_app_test.sh
#
# This script tests a deployed LLM that exposes an OpenAI-compatible chat API
# by sending a prompt and printing the response.
#
# Usage:
#   ./vllm_app_test.sh
#   ./vllm_app_test.sh https://...
#   LLM_APP_URL=https://... ./vllm_app_test.sh
#   PROMPT="Translate this to Japanese: Hello" ./vllm_app_test.sh

set -eou pipefail

# Allow overriding the URL via argument or environment variable
LLM_APP_URL="${1:-${LLM_APP_URL:-https://pocheung90200.dmo-team-sandbox.domino.tech/apps/vllm}}"

# Allow prompt override via PROMPT env var
PROMPT=${PROMPT:-"What's the capital of Japan?"}

# Identify and validate the running app pod
if [[ -n "${LLM_APP_POD:-}" ]]; then
  # Validate user-specified pod
  POD=$(kubectl -n domino-compute get po --no-headers -o custom-columns=NAME:.metadata.name \
    -l dominodatalab.com/workload-type=App --field-selector=status.phase=Running | grep $LLM_APP_POD || true)

  if [[ -z "$POD" ]]; then
    echo "No running app pod found matching $LLM_APP_POD."
    exit 1
  fi
else
  # Find a running App pod
  POD=$(kubectl -n domino-compute get po --no-headers -o custom-columns=NAME:.metadata.name \
    -l dominodatalab.com/workload-type=App --field-selector=status.phase=Running | head -1)

  if [[ -z "$POD" ]]; then
    echo "No running app pod found."
    exit 1
  fi
fi
echo "App pod: $POD"

# Validate that the LLM_APP_URL is reachable and responds with HTTP 401
status_code=$(curl -s -o /dev/null -w "%{http_code}" "$LLM_APP_URL")
if [[ "$status_code" != 401 ]]; then
  echo "Error: $LLM_APP_URL should have returned 401 Unauthorized"
  exit 1
fi
echo "App url: $LLM_APP_URL"

# Get the access token from inside the pod
BEARER_TOKEN=$(kubectl exec -it $POD -c run -- curl http://localhost:8899/access-token)

if [[ -z "$BEARER_TOKEN" ]]; then
  echo "Failed to retrieve token from pod."
  exit 1
fi

# Export environment variables so the Python script can pick them up
export LLM_APP_URL
export BEARER_TOKEN

python vllm_app_test.py --prompt "$PROMPT"

