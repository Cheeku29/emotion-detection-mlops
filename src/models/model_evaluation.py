
# model evaluation

import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import yaml
import logging


# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
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


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception:
        logger.exception('Failed to load model from %s', file_path)
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


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        if len(np.unique(y_test)) < 2:
            raise ValueError(
                'ROC-AUC requires both classes in the test labels.'
            )

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc)
        }

        logger.debug('Model evaluation metrics calculated')
        return metrics_dict

    except Exception:
        logger.exception('Error during model evaluation')
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(metrics, file, indent=4)

        logger.debug('Metrics saved to %s', file_path)

    except Exception:
        logger.exception('Error occurred while saving metrics')
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')['model_evaluation']

        clf = load_model(params['model_input'])
        test_data = load_data(params['test_input'])

        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics, params['metrics_output'])

        logger.debug('Model evaluation completed successfully')

    except Exception:
        logger.exception('Failed to complete model evaluation')
        raise


if __name__ == '__main__':
    main()
