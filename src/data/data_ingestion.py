import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
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

    except FileNotFoundError:
        logger.exception('File not found: %s', params_path)
        raise
    except yaml.YAMLError:
        logger.exception('YAML error while loading parameters')
        raise
    except Exception:
        logger.exception('Unexpected error while loading parameters')
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df

    except Exception:
        logger.exception('Unexpected error occurred while loading data')
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=['tweet_id'], inplace=True)

        final_df = df[
            df['sentiment'].isin(['happiness', 'sadness'])
        ].copy()

        final_df['sentiment'] = final_df['sentiment'].replace(
            {'happiness': 1, 'sadness': 0}
        )

        logger.debug('Data preprocessing completed')
        return final_df

    except KeyError:
        logger.exception('Missing column in the dataframe')
        raise
    except Exception:
        logger.exception('Unexpected error during preprocessing')
        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str
) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(raw_data_path, 'train.csv'), index=False
        )
        test_data.to_csv(
            os.path.join(raw_data_path, 'test.csv'), index=False
        )

        logger.debug('Train and test data saved to %s', raw_data_path)

    except Exception:
        logger.exception('Unexpected error occurred while saving data')
        raise


def main() -> None:
    try:
        params = load_params(params_path='params.yaml')
        config = params['data_ingestion']

        test_size = config['test_size']
        random_state = config['random_state']
        data_url = config['data_url']

        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state
        )

        save_data(train_data, test_data, data_path='./data')

        logger.debug('Data ingestion completed successfully')

    except Exception:
        logger.exception('Failed to complete data ingestion')
        raise


if __name__ == '__main__':
    main()
