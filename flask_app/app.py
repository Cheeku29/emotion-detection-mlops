
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from flask import Flask, render_template, request

from flask_app.preprocessing_utility import normalize_text
from src.mlflow_setup import configure_dagshub


configure_dagshub()

app = Flask(__name__)

MODEL_NAME = "emotion-detection-model"
MODEL_ALIAS = "production"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

# Load the fitted vectorizer
vectorizer_path = "models/vectorizer.pkl"

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(
        "models/vectorizer.pkl not found. "
        "Save the fitted vectorizer during feature engineering."
    )

import pickle

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html", result=None, text="")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()

    if not text:
        return render_template(
            "index.html",
            result=None,
            text=text,
            error="Please enter some text."
        )

    cleaned_text = normalize_text(text)
    features = vectorizer.transform([cleaned_text])

    # Pass the same feature names and order used during training
    features_df = pd.DataFrame(
        features.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Predict emotion
    prediction = int(model.predict(features_df)[0])

    result = "Happy" if prediction == 1 else "Sad"

    return render_template(
        "index.html",
        result=result,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)
