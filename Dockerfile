FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir gunicorn \
    && python -m nltk.downloader -d /usr/local/share/nltk_data \
        stopwords wordnet omw-1.4

ENV NLTK_DATA=/usr/local/share/nltk_data
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY flask_app ./flask_app
COPY src/mlflow_setup.py ./src/mlflow_setup.py
COPY models/vectorizer.pkl ./models/vectorizer.pkl

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]
