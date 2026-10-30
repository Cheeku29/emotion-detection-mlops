
import os
import pickle

import mlflow
import mlflow.sklearn

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

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

model_feature_count = getattr(model, "n_features_in_", None)
vectorizer_feature_count = len(vectorizer.get_feature_names_out())

if (
    model_feature_count is not None
    and model_feature_count != vectorizer_feature_count
):
    raise RuntimeError(
        "Production model and vectorizer are incompatible: "
        f"model expects {model_feature_count} features, "
        f"but vectorizer produces {vectorizer_feature_count}."
    )


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


    prediction = int(model.predict(features.toarray())[0])

    result = "Happy" if prediction == 1 else "Sad"

    return render_template(
        "index.html",
        result=result,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
