import os
import logging
import pandas as pd
import yaml
import mlflow
import dagshub

from sklearn.feature_extraction.text import CountVectorizer


# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "feature_engineering_errors.log"
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


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)

        logger.debug("Data loaded from %s", file_path)
        return df

    except Exception:
        logger.exception("Failed to load data from %s", file_path)
        raise


def apply_bow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> tuple[pd.DataFrame, pd.DataFrame, CountVectorizer]:
    """Apply BoW, fitting the vectorizer only on training text."""
    try:
        required_columns = ["content", "sentiment"]

        for column in required_columns:
            if (
                column not in train_data.columns
                or column not in test_data.columns
            ):
                raise ValueError(f"Missing required column: {column}")

        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data["content"].astype(str).values
        y_train = train_data["sentiment"].values

        X_test = test_data["content"].astype(str).values
        y_test = test_data["sentiment"].values

        # Fit on training text only to avoid data leakage
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Keep CSV output format for compatibility with the existing DVC pipeline
        feature_names = vectorizer.get_feature_names_out()

        train_df = pd.DataFrame(
            X_train_bow.toarray(),
            columns=feature_names
        )
        train_df["label"] = y_train

        test_df = pd.DataFrame(
            X_test_bow.toarray(),
            columns=feature_names
        )
        test_df["label"] = y_test

        logger.debug(
            "BoW transformation completed with %d features",
            len(feature_names)
        )

        return train_df, test_df, vectorizer

    except Exception:
        logger.exception("Error during BoW transformation")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the transformed dataframe to CSV."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)

    except Exception:
        logger.exception("Failed to save data to %s", file_path)
        raise


def main() -> None:
    try:
        params = load_params("params.yaml")
        config = params["feature_engineering"]

        # Initialize DagsHub MLflow tracking
        dagshub.init(
            repo_owner="Cheeku29",
            repo_name="emotion-detection-mlops",
            mlflow=True
        )

        mlflow.set_experiment(
            "Emotion Detection - Feature Engineering"
        )

        with mlflow.start_run(
            run_name="BoW Feature Engineering"
        ):
            train_data = load_data(config["train_input"])
            test_data = load_data(config["test_input"])

            # Log parameters
            mlflow.log_params({
                "vectorizer": "CountVectorizer",
                "max_features": config["max_features"],
                "train_input": config["train_input"],
                "test_input": config["test_input"],
            })

            mlflow.log_metrics({
                "train_rows_input": len(train_data),
                "test_rows_input": len(test_data),
            })

            train_df, test_df, vectorizer = apply_bow(
                train_data,
                test_data,
                config["max_features"]
            )

            # Save the feature matrices
            save_data(train_df, config["train_output"])
            save_data(test_df, config["test_output"])

            # Log output statistics
            mlflow.log_metrics({
                "number_of_features": len(
                    vectorizer.get_feature_names_out()
                ),
                "train_rows_output": len(train_df),
                "test_rows_output": len(test_df),
                "train_output_columns": len(train_df.columns),
                "test_output_columns": len(test_df.columns),
            })

            # Log generated CSV files as MLflow artifacts
            mlflow.log_artifact(
                config["train_output"],
                artifact_path="features"
            )
            mlflow.log_artifact(
                config["test_output"],
                artifact_path="features"
            )

            logger.debug(
                "Feature engineering completed successfully"
            )

    except Exception:
        logger.exception("Failed to complete feature engineering")
        raise


if __name__ == "__main__":
    main()
