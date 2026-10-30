import importlib
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import mlflow.sklearn
import pytest


class FakeModel:
    def predict(self, features: pd.DataFrame):
        assert features.shape[1] == 2
        return [1]


class FakeVectorizer:
    def transform(self, texts):
        assert texts == ["great day"]
        return type(
            "Features",
            (),
            {"toarray": lambda self: np.array([[1, 0]])}
        )()

    def get_feature_names_out(self):
        return ["great", "day"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DAGSHUB_PAT", "test-token")

    with patch("src.mlflow_setup.dagshub.auth.add_app_token"), \
            patch("src.mlflow_setup.mlflow.set_tracking_uri"), \
            patch.object(
                mlflow.sklearn,
                "load_model",
                return_value=FakeModel()
            ):
        sys.modules.pop("flask_app.app", None)
        app_module = importlib.import_module("flask_app.app")
        app_module.vectorizer = FakeVectorizer()
        app_module.app.config.update(TESTING=True)
        with app_module.app.test_client() as test_client:
            yield test_client


def test_home_page(client):
    response = client.get("/")

    assert response.status_code == 200


def test_predict_returns_happy_for_positive_prediction(client):
    response = client.post("/predict", data={"text": "Great day!"})

    assert response.status_code == 200
    assert b"Happy" in response.data


def test_predict_requires_text(client):
    response = client.post("/predict", data={"text": ""})

    assert response.status_code == 200
    assert b"Please enter some text." in response.data
