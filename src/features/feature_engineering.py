import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging


# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception:
        logger.exception('Failed to load parameters')
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception:
        logger.exception('Failed to load data from %s', file_path)
        raise


def apply_bow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply CountVectorizer to the data."""
    try:
        required_columns = ['content', 'sentiment']

        for column in required_columns:
            if column not in train_data.columns or column not in test_data.columns:
                raise ValueError(f'Missing required column: {column}')

        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['content'].astype(str).values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].astype(str).values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('Bag-of-Words applied and data transformed')
        return train_df, test_df

    except Exception:
        logger.exception('Error during BOW transformation')
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception:
        logger.exception('Failed to save data to %s', file_path)
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')
        config = params['feature_engineering']

        train_data = load_data(config['train_input'])
        test_data = load_data(config['test_input'])

        train_df, test_df = apply_bow(
            train_data,
            test_data,
            config['max_features']
        )

        save_data(train_df, config['train_output'])
        save_data(test_df, config['test_output'])

        logger.debug('Feature engineering completed successfully')

    except Exception:
        logger.exception('Failed to complete feature engineering')
        raise


if __name__ == '__main__':
    main()
