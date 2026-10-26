
import json
import logging
import os

import dagshub
import mlflow
from mlflow import MlflowClient


logger = logging.getLogger("model_registration")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)


def main():
    # Configure MLflow tracking with your DagsHub repository
    dagshub.init(
        repo_owner="Cheeku29",
        repo_name="emotion-detection-mlops",
        mlflow=True
    )

    model_info_path = "reports/experiment_info.json"

    if not os.path.exists(model_info_path):
        raise FileNotFoundError(
            f"Model information not found: {model_info_path}"
        )

    with open(model_info_path, "r", encoding="utf-8") as file:
        model_info = json.load(file)

    run_id = model_info["run_id"]
    model_path = model_info["model_path"]

    model_name = "emotion-detection-model"
    model_uri = f"runs:/{run_id}/{model_path}"

    client = MlflowClient()

    # Register the model version
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    logger.info(
        "Registered model %s, version %s",
        model_name,
        registered_model.version
    )

    # Assign a staging alias to the new version
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=registered_model.version
    )

    logger.info(
        "Model version %s assigned alias 'staging'",
        registered_model.version
    )


if __name__ == "__main__":
    main()
