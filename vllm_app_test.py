#!/usr/bin/env python3
"""
Script: vllm_app_test.py

Purpose:
    An external test client for an OpenAI-compatible chat app that:
    - retrieves the app url and bearer token from command-line arguments or environment variables
    - creates an OpenAI client configured with:
      - a base url for an OpenAI-compatible endpoint backed by a vLLM
      - a bearer token to authenticate the client with the endpoint
    - sends a chat completion request to the chat app and prints the response

Usage:
    python vllm_app_test.py --url <LLM_APP_URL> --token <BEARER_TOKEN>
    or set environment variables:
        export LLM_APP_URL=<LLM_APP_URL>
        export BEARER_TOKEN=<BEARER_TOKEN>
        python vllm_app_test.py

    Optional:
        --prompt "<your input prompt>"
"""

import argparse
import os
import openai

# Argument parsing
parser = argparse.ArgumentParser(description="Test LLM app")
parser.add_argument("--url", help="URL of the deployed LLM app (or set LLM_APP_URL)")
parser.add_argument("--token", help="Bearer token (or set BEARER_TOKEN)")
parser.add_argument("--prompt", default="What's the capital of Japan?", help="Prompt to send to the model")
args = parser.parse_args()

# Retrieve app_url and bearer_token from the arguments or environment variables
# Argument takes precedence; fallback to environment variable
app_url = (args.url or os.getenv("LLM_APP_URL", "")).rstrip("/")
bearer_token = args.token or os.getenv("BEARER_TOKEN")

# Validate required inputs
if not app_url:
    raise ValueError("Missing app URL. Provide --url or set LLM_APP_URL environment variable.")
if not bearer_token:
    raise ValueError("Missing bearer token. Provide --token or set BEARER_TOKEN environment variable.")

# Initialize OpenAI-compatible client
client = openai.OpenAI(
    base_url=app_url + "/v1",
    api_key=bearer_token,
)

# Send chat completion request
response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[{"role": "user", "content": args.prompt}],
    temperature=0.7,
    max_tokens=50,
)

print(response.choices[0].message.content.strip())
