from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
revision = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
model_type = "LLM"
model_dir = "/mnt/data/TinyLlama_TinyLlama-1_1B-Chat-v1_0-fe8a4e"

# Download model
if model_type == "LLM":
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, cache_dir=model_dir)
else:
    model = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=model_dir)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=model_dir)

print(f"Downloaded {model_type} model '{model_id}' at revision '{revision}' to '{model_dir}'")