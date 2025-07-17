import openai

# Run this test client inside the run container of the vLLM app

client = openai.OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="EMPTY",  # Required placeholder for vLLM
)

response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print("Assistant:", response.choices[0].message.content.strip())
