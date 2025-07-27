from mlflow import MlflowClient
from mlflow.models import create_external_model

# Input arguments
model_type = "llm"
model_source = "hf"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_revision = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
model_task = "text-generation"

# Domino dataset info (mounted path)
dataset_name = "test-fe8a4e"
dataset_snapshot_id = "6884347fd3097042943decce"
dataset_mount_path = f"/mnt/data/{dataset_name}"
model_sub_path = f"models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/{model_revision}"
external_model_path = f"{dataset_mount_path}/{model_sub_path}"

# MLflow registered model name
# This sanitization is not needed after support for URL encoding the model id is added
registered_model_name = model_id.replace("/", "_") + "-ext"

model_version = create_external_model(
    name=registered_model_name,
    source=external_model_path,
    flavor="transformers",
    metadata={
        "domino.model.type": model_type,
        "domino.model.source": model_source,
        "domino.model.source.hf.model_id": model_id,
        "domino.model.source.hf.model_revision": model_revision,
        "domino.model.task": model_task,
        "domino.model.dataset_name": dataset_name,
        "domino.model.dataset_snapshot_id": dataset_snapshot_id,
    },
)

print(f"✅ Registered external model '{registered_model_name}' version {model_version.version}")

client = MlflowClient()

# Set registered model description
client.update_registered_model(
    name=model_version.name,
    description=f"{model_id} registered as an external model from Hugging Face."
)
