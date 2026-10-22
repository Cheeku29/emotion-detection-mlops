import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging


# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
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
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception:
        logger.exception('Failed to load data from %s', file_path)
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict
) -> GradientBoostingClassifier:
    """Train the Gradient Boosting model."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            random_state=params['random_state']
        )

        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf

    except Exception:
        logger.exception('Error during model training')
        raise


def save_model(model: GradientBoostingClassifier, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        logger.debug('Model saved to %s', file_path)

    except Exception:
        logger.exception('Error occurred while saving the model')
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')['model_building']

        train_data = load_data(params['train_input'])

        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        clf = train_model(X_train, y_train, params)

        save_model(clf, params['model_output'])

        logger.debug('Model building completed successfully')

    except Exception:
        logger.exception('Failed to complete model building')
        raise


if __name__ == '__main__':
    main()
