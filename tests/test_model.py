import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.pkl"
TEST_DATA_PATH = ROOT / "data" / "processed" / "test_bow.csv"


@pytest.fixture(scope="module")
def model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="module")
def vectorizer():
    with VECTORIZER_PATH.open("rb") as file:
        return pickle.load(file)


@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv(TEST_DATA_PATH)


def test_model_and_vectorizer_have_matching_feature_counts(model, vectorizer):
    assert model.n_features_in_ == len(
        vectorizer.get_feature_names_out()
    )


def test_model_predicts_expected_output_shape(model, test_data):
    features = test_data.drop(columns=["label"])
    predictions = model.predict(features.values)

    assert len(predictions) == len(test_data)
    assert set(predictions).issubset({0, 1})


def test_model_meets_minimum_holdout_performance(model, test_data):
    features = test_data.drop(columns=["label"])
    labels = test_data["label"]
    predictions = model.predict(features.values)

    assert accuracy_score(labels, predictions) >= 0.70
    assert f1_score(labels, predictions, zero_division=0) >= 0.70
