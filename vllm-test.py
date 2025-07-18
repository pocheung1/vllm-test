from openai import OpenAI
import argparse
import requests

# Run this test client inside any execution
# Example:
# python vllm-test.py --app-url https://pocheung90200.dmo-team-sandbox.domino.tech/apps/vllm-test/

# Argument parser
parser = argparse.ArgumentParser(description="Call vLLM chat API.")
parser.add_argument("--app-url", required=True, help="Base URL of the deployed vLLM app")
args = parser.parse_args()

# Construct base URL for OpenAI-compatible endpoint
APP_URL = args.app_url.rstrip("/")  # remove trailing slash if present
BASE_URL = APP_URL + "/v1"

# Get access token and create OpenAI client
client = OpenAI(
    base_url=BASE_URL,
    api_key=requests.get("http://localhost:8899/access-token").text,
)

# Send chat request
response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[
        {"role": "user", "content": "What's the capital of Japan?"},
    ],
    temperature=0.7,
    max_tokens=50,
)

# Print the assistant's reply
print(response.choices[0].message.content.strip())
