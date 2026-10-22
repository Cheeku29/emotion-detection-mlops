import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import yaml
import logging

from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# logging configuration
logger = logging.getLogger('data_transformation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)

        logger.debug('Parameters loaded from %s', params_path)
        return params['data_preprocessing']

    except Exception:
        logger.exception('Failed to load parameters')
        raise


def lemmatization(text: str) -> str:
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text: str) -> str:
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def lower_case(text: str) -> str:
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text: str) -> str:
    """Remove punctuations from the text."""
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(
        r'https?://\S+|www\.\S+',
        flags=re.IGNORECASE
    )
    return url_pattern.sub('', text)


def remove_small_sentences(
    df: pd.DataFrame,
    text_column: str,
    min_words: int
) -> pd.DataFrame:
    """Remove rows with fewer than the minimum number of words."""
    df = df.copy()

    short_rows = df[text_column].fillna('').apply(
        lambda text: len(str(text).split()) < min_words
    )

    removed_rows = int(short_rows.sum())
    df = df.loc[~short_rows].copy()

    logger.debug('Removed %d short rows', removed_rows)
    return df


def normalize_text(
    df: pd.DataFrame,
    text_column: str,
    min_words: int
) -> pd.DataFrame:
    """Normalize the text data."""
    try:
        df = df.copy()

        if text_column not in df.columns:
            raise KeyError(f'Missing text column: {text_column}')

        df[text_column] = df[text_column].fillna('').astype(str)

        df[text_column] = df[text_column].apply(removing_urls)
        logger.debug('URLs removed')

        df[text_column] = df[text_column].apply(lower_case)
        logger.debug('Converted to lower case')

        df[text_column] = df[text_column].apply(remove_stop_words)
        logger.debug('Stop words removed')

        df[text_column] = df[text_column].apply(removing_numbers)
        logger.debug('Numbers removed')

        df[text_column] = df[text_column].apply(removing_punctuations)
        logger.debug('Punctuation removed')

        df[text_column] = df[text_column].apply(lemmatization)
        logger.debug('Lemmatization performed')

        df = remove_small_sentences(
            df, text_column, min_words
        )

        logger.debug('Text normalization completed')
        return df

    except Exception:
        logger.exception('Error during text normalization')
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')

        input_dir = params['input_dir']
        output_dir = params['output_dir']
        text_column = params['text_column']
        min_words = params['min_words']

        # Fetch the data from data/raw
        train_data = pd.read_csv(
            os.path.join(input_dir, params['train_filename'])
        )
        test_data = pd.read_csv(
            os.path.join(input_dir, params['test_filename'])
        )
        logger.debug('Training and testing data loaded')

        # Transform the data
        train_processed_data = normalize_text(
            train_data, text_column, min_words
        )
        test_processed_data = normalize_text(
            test_data, text_column, min_words
        )

        # Store the data inside data/interim
        os.makedirs(output_dir, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(
                output_dir, params['train_processed_filename']
            ),
            index=False
        )
        test_processed_data.to_csv(
            os.path.join(
                output_dir, params['test_processed_filename']
            ),
            index=False
        )

        logger.debug('Processed data saved to %s', output_dir)

    except Exception:
        logger.exception('Failed to complete data transformation')
        raise


if __name__ == '__main__':
    main()
