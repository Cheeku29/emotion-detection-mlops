import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import yaml
import mlflow
from src.mlflow_setup import configure_dagshub

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_evaluation_errors.log"
    )
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


def load_model(file_path: str):
    """Load the trained model from a local pickle file."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.debug("Model loaded from %s", file_path)
        return model

    except Exception:
        logger.exception("Failed to load model from %s", file_path)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load the test dataset."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df

    except Exception:
        logger.exception("Failed to load data from %s", file_path)
        raise


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the model and return metrics."""
    try:
        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, y_pred, zero_division=0)
            )
        }

        # Calculate ROC-AUC when both classes are present
        if len(np.unique(y_test)) == 2:
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
                metrics["auc"] = float(
                    roc_auc_score(y_test, y_score)
                )
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_test)
                metrics["auc"] = float(
                    roc_auc_score(y_test, y_score)
                )
        else:
            logger.warning(
                "ROC-AUC skipped because test labels contain one class"
            )

        logger.debug("Evaluation metrics calculated: %s", metrics)
        return metrics

    except Exception:
        logger.exception("Error during model evaluation")
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to a JSON file."""
    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics saved to %s", file_path)

    except Exception:
        logger.exception("Failed to save metrics")
        raise


def main() -> None:
    try:
        params = load_params("params.yaml")["model_evaluation"]

        configure_dagshub()

        mlflow.set_experiment(
            "Emotion Detection - Model Evaluation"
        )

        with mlflow.start_run(
            run_name="Evaluate Tuned Logistic Regression"
        ):
            clf = load_model(params["model_input"])
            test_data = load_data(params["test_input"])

            X_test = test_data.drop(columns=["label"]).values
            y_test = test_data["label"].values

            # Evaluate
            metrics = evaluate_model(clf, X_test, y_test)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("feature_representation", "BoW")
            mlflow.log_param("evaluation_dataset", params["test_input"])

            # Save JSON metrics for the existing DVC pipeline
            save_metrics(metrics, params["metrics_output"])

            # Log JSON metrics as an MLflow artifact
            mlflow.log_artifact(
                params["metrics_output"],
                artifact_path="evaluation"
            )

            logger.info("Final evaluation metrics: %s", metrics)
            logger.debug("Model evaluation completed successfully")

    except Exception:
        logger.exception("Failed to complete model evaluation")
        raise


if __name__ == "__main__":
    main()
