from mlflow.tracking import MlflowClient
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import mlflow.transformers

# Input arguments
model_dir = "/mnt/data/test-fe8a4e/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
revision = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
model_type = "LLM"

# Load the model and tokenizer from model_dir based on model_type
if model_type == "LLM":
    model = AutoModelForCausalLM.from_pretrained(model_dir)
else:
    model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

with mlflow.start_run() as run:
    # Log the model
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        input_example="Hello, how are you?",
    )

    # TODO Check for metadata conflicts if the registered model already exists.
    # For example, the model_id must match the existing tag.

    # Register the model
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name=model_id,
        description=f"{model_id} registered from Hugging Face",
    )

client = MlflowClient()

# Tag the registered model
if result.version == 1:
    client.set_registered_model_tag(name=model_id, key="mlflow.domino.model_id", value=model_id)
    client.set_registered_model_tag(name=model_id, key="mlflow.domino.model_type", value=model_type)

# Tag the registered model version
client.set_model_version_tag(
    name=model_id,
    version=result.version,
    key="mlflow.domino.model_version",
    value=revision,
)
