import mlflow
from datasets import load_dataset
from mlflow import MlflowClient
from mlflow.exceptions import RestException
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline,
)

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
MERGED_MODEL_NAME = BASE_MODEL_NAME + "-finetuned"
ADAPTER_OUTPUT_DIR = "./adapter_weights"
LOG_MERGED_MODEL = False

client = MlflowClient()

mlflow.set_experiment("TinyLlama-fine-tuning")


def is_model_registered(model_name: str) -> bool:
    try:
        client.get_registered_model(model_name)
        return True
    except RestException:
        return False


def tokenize(example):
    tokens = tokenizer(example["quote"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# Step 1: Retrieve the base model and tokenizer

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
print("Downloaded base model and tokenizer")

if not is_model_registered(BASE_MODEL_NAME):
    print(f"Registering {BASE_MODEL_ID}...")

    with mlflow.start_run(run_name="log-base-model") as base_run:
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline("text-generation", model=base_model, tokenizer=tokenizer),
            tokenizer=tokenizer,
            artifact_path="base_model",
            input_example="What's the capital of France?"
        )
        mlflow.register_model(model_info.model_uri, BASE_MODEL_NAME)
        print(f"Registered base model: {BASE_MODEL_NAME}")


# Step 2: Apply LoRA adapter

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, lora_config)

print("LoRA adapter applied")


# Step 3: Prepare dataset and trainer

dataset = load_dataset("Abirate/english_quotes")['train'].train_test_split(test_size=0.1)
tokenized = dataset.map(tokenize)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Prepared dataset and trainer")


# Step 4: Fine-tuning

with mlflow.start_run(run_name="adapter-finetune") as run:
    mlflow.log_params({
        "registered_base_model": BASE_MODEL_ID,
        "adapter_type": "LoRA",
        "learning_rate": training_args.learning_rate,
        "epochs": training_args.num_train_epochs
    })

    trainer.train()

    # Log adapter weights only
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    mlflow.log_artifacts(ADAPTER_OUTPUT_DIR, artifact_path="adapters")
    print("Logged adapter weights")

    # Optionally merge and register final model
    if LOG_MERGED_MODEL:
        merged_model = model.merge_and_unload()
        merged_info = mlflow.transformers.log_model(
            transformers_model=pipeline("text-generation", model=merged_model, tokenizer=tokenizer),
            tokenizer=tokenizer,
            artifact_path="merged_model",
            input_example="What's the capital of France?"
        )
        mlflow.register_model(merged_info.model_uri, MERGED_MODEL_NAME)
        print(f"Registered merged model: {MERGED_MODEL_NAME}")
