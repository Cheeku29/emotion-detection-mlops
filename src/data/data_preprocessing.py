import os
import re
import string
import logging
import pandas as pd
import yaml
import mlflow

from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.mlflow_setup import configure_dagshub


# Logging configuration
logger = logging.getLogger("data_transformation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("transformation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Initialize reusable preprocessing resources
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_params(params_path: str) -> dict[str, Any]:
    """Load preprocessing parameters from params.yaml."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters loaded from %s", params_path)
        return params["data_preprocessing"]

    except Exception:
        logger.exception("Failed to load parameters")
        raise


def lemmatization(text: str) -> str:
    """Lemmatize the text."""
    words = text.split()
    words = [LEMMATIZER.lemmatize(word) for word in words]
    return " ".join(words)


def remove_stop_words(text: str) -> str:
    """Remove English stop words."""
    words = [
        word for word in str(text).split()
        if word not in STOP_WORDS
    ]
    return " ".join(words)


def removing_numbers(text: str) -> str:
    """Remove digits from text."""
    return re.sub(r"\d+", "", text)


def lower_case(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def removing_punctuations(text: str) -> str:
    """Remove punctuation."""
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation),
        " ",
        text
    )
    text = text.replace("؛", "")
    return re.sub(r"\s+", " ", text).strip()


def removing_urls(text: str) -> str:
    """Remove URLs."""
    url_pattern = re.compile(
        r"https?://\S+|www\.\S+",
        flags=re.IGNORECASE
    )
    return url_pattern.sub("", text)


def remove_small_sentences(
    df: pd.DataFrame,
    text_column: str,
    min_words: int
) -> pd.DataFrame:
    """Remove rows containing fewer than min_words words."""
    df = df.copy()

    short_rows = df[text_column].fillna("").apply(
        lambda text: len(str(text).split()) < min_words
    )

    removed_rows = int(short_rows.sum())
    df = df.loc[~short_rows].copy()

    logger.debug("Removed %d short rows", removed_rows)
    return df


def normalize_text(
    df: pd.DataFrame,
    text_column: str,
    min_words: int
) -> pd.DataFrame:
    """Normalize text and remove short rows."""
    try:
        df = df.copy()

        if text_column not in df.columns:
            raise KeyError(f"Missing text column: {text_column}")

        rows_before = len(df)
        df[text_column] = df[text_column].fillna("").astype(str)

        df[text_column] = df[text_column].apply(removing_urls)
        df[text_column] = df[text_column].apply(lower_case)
        df[text_column] = df[text_column].apply(remove_stop_words)
        df[text_column] = df[text_column].apply(removing_numbers)
        df[text_column] = df[text_column].apply(removing_punctuations)
        df[text_column] = df[text_column].apply(lemmatization)

        df = remove_small_sentences(
            df,
            text_column,
            min_words
        )

        logger.debug(
            "Text normalization completed: %d -> %d rows",
            rows_before,
            len(df)
        )

        return df

    except Exception:
        logger.exception("Error during text normalization")
        raise


def main() -> None:
    try:
        params = load_params("params.yaml")

        input_dir = params["input_dir"]
        output_dir = params["output_dir"]
        text_column = params["text_column"]
        min_words = params["min_words"]

        configure_dagshub()

        mlflow.set_experiment(
            "Emotion Detection - Data Preprocessing"
        )

        with mlflow.start_run(
            run_name="Text Preprocessing"
        ):
            # Log preprocessing parameters
            mlflow.log_params({
                "text_column": text_column,
                "min_words": min_words,
                "input_dir": input_dir,
                "output_dir": output_dir
            })

            train_path = os.path.join(
                input_dir,
                params["train_filename"]
            )
            test_path = os.path.join(
                input_dir,
                params["test_filename"]
            )

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logger.debug("Training and testing data loaded")

            # Track input sizes
            mlflow.log_metrics({
                "train_rows_before": len(train_data),
                "test_rows_before": len(test_data)
            })

            # Transform the data
            train_processed_data = normalize_text(
                train_data,
                text_column,
                min_words
            )

            test_processed_data = normalize_text(
                test_data,
                text_column,
                min_words
            )

            # Track output sizes
            mlflow.log_metrics({
                "train_rows_after": len(train_processed_data),
                "test_rows_after": len(test_processed_data),
                "train_rows_removed": (
                    len(train_data) - len(train_processed_data)
                ),
                "test_rows_removed": (
                    len(test_data) - len(test_processed_data)
                )
            })

            # Save processed data
            os.makedirs(output_dir, exist_ok=True)

            train_output = os.path.join(
                output_dir,
                params["train_processed_filename"]
            )
            test_output = os.path.join(
                output_dir,
                params["test_processed_filename"]
            )

            train_processed_data.to_csv(
                train_output,
                index=False
            )
            test_processed_data.to_csv(
                test_output,
                index=False
            )

            # Log processed CSVs as MLflow artifacts
            mlflow.log_artifact(
                train_output,
                artifact_path="processed_data"
            )
            mlflow.log_artifact(
                test_output,
                artifact_path="processed_data"
            )

            logger.debug(
                "Processed data saved to %s",
                output_dir
            )

    except Exception:
        logger.exception("Failed to complete data preprocessing")
        raise


if __name__ == "__main__":
    main()
