import os

import dagshub


REPO_OWNER = "Cheeku29"
REPO_NAME = "emotion-detection-mlops"


def configure_dagshub() -> None:
    """Configure authenticated MLflow tracking through DagsHub."""
    dagshub_token = os.getenv("DAGSHUB_PAT")

    if not dagshub_token:
        raise EnvironmentError(
            "DAGSHUB_PAT environment variable is not set"
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub.auth.add_app_token(dagshub_token)

    dagshub.init(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        mlflow=True
    )
