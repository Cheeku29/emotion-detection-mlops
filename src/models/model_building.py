import os
import logging
import pickle
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import json

from sklearn.linear_model import LogisticRegression
from src.mlflow_setup import configure_dagshub
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_building_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except Exception:
        logger.exception("Failed to load parameters")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df

    except Exception:
        logger.exception("Failed to load data from %s", file_path)
        raise


def train_model(
    X_train,
    y_train,
    params: dict
) -> LogisticRegression:
    """Train Logistic Regression using the selected hyperparameters."""
    try:
        model = LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver=params["solver"],
            max_iter=params.get("max_iter", 1000),
            random_state=params.get("random_state", 42)
        )

        model.fit(X_train, y_train)

        logger.debug("Logistic Regression training completed")
        return model

    except Exception:
        logger.exception("Error during model training")
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model using pickle."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.debug("Model saved to %s", file_path)

    except Exception:
        logger.exception("Error occurred while saving the model")
        raise



def main() -> None:
    try:
        config = load_params("params.yaml")["model_building"]

        configure_dagshub()

        mlflow.set_experiment("Emotion Detection - Model Building")

        with mlflow.start_run(
            run_name="Logistic Regression - BoW Tuned"
        ):
            train_data = load_data(config["train_input"])

            # Separate features and target
            X_train = train_data.drop(columns=["label"]).values
            y_train = train_data["label"].values

            # Best hyperparameters from GridSearchCV
            model_params = {
                "C": config.get("C", 1),
                "penalty": config.get("penalty", "l1"),
                "solver": config.get("solver", "liblinear"),
                "max_iter": config.get("max_iter", 1000),
                "random_state": config.get("random_state", 42)
            }

            mlflow.log_params(model_params)
            mlflow.log_params({
                "model_type": "LogisticRegression",
                "feature_representation": "BoW"
            })

            # Train model
            model = train_model(
                X_train,
                y_train,
                model_params
            )

            # Save local model for DVC
            save_model(model, config["model_output"])

            # Log model artifact to MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle"
            )

            # Save information needed for model registration
            experiment_info = {
                "run_id": mlflow.active_run().info.run_id,
                "model_uri": model_info.model_uri
            }

            os.makedirs("reports", exist_ok=True)

            with open("reports/experiment_info.json", "w", encoding="utf-8") as file:
                json.dump(experiment_info, file, indent=4)

            mlflow.log_artifact("reports/experiment_info.json")

            logger.info("Model and experiment information saved successfully")

    except Exception:
        logger.exception("Failed to complete model building")
        raise


if __name__ == "__main__":
    main()
