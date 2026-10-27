import os
import logging
import pandas as pd
import yaml
import mlflow

from sklearn.model_selection import train_test_split
from src.mlflow_setup import configure_dagshub

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("errors.log")
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df

    except Exception:
        logger.exception("Failed to load data")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep happiness and sadness, and encode the labels."""
    try:
        df = df.drop(columns=["tweet_id"])

        final_df = df[
            df["sentiment"].isin(["happiness", "sadness"])
        ].copy()

        final_df["sentiment"] = final_df["sentiment"].map({
            "happiness": 1,
            "sadness": 0
        })

        logger.debug("Data preprocessing completed")
        return final_df

    except Exception:
        logger.exception("Failed to preprocess data")
        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str
) -> None:
    """Save train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(raw_data_path, "train.csv"),
            index=False
        )

        test_data.to_csv(
            os.path.join(raw_data_path, "test.csv"),
            index=False
        )

        logger.debug("Train and test data saved to %s", raw_data_path)

    except Exception:
        logger.exception("Failed to save train/test data")
        raise


def main() -> None:
    try:
        params = load_params("params.yaml")
        config = params["data_ingestion"]

        test_size = config["test_size"]
        random_state = config["random_state"]
        data_url = config["data_url"]

        configure_dagshub()

        mlflow.set_experiment("Emotion Detection - Data Ingestion")

        with mlflow.start_run(run_name="Data Ingestion"):

            # Log ingestion parameters
            mlflow.log_params({
                "test_size": test_size,
                "random_state": random_state,
                "target_classes": "happiness,sadness",
                "stratify": True
            })

            # Load and preprocess
            df = load_data(data_url)
            final_df = preprocess_data(df)

            # Log dataset statistics
            mlflow.log_param("source", data_url)
            mlflow.log_metric("filtered_dataset_rows", len(final_df))

            class_counts = final_df["sentiment"].value_counts()

            for label, count in class_counts.items():
                mlflow.log_metric(f"class_{label}_count", int(count))

            # Stratified split
            train_data, test_data = train_test_split(
                final_df,
                test_size=test_size,
                random_state=random_state,
                stratify=final_df["sentiment"]
            )

            # Save data
            save_data(
                train_data,
                test_data,
                data_path="./data"
            )

            mlflow.log_metrics({
                "train_rows": len(train_data),
                "test_rows": len(test_data)
            })

            logger.debug("Data ingestion completed successfully")

    except Exception:
        logger.exception("Data ingestion failed")
        raise


if __name__ == "__main__":
    main()
