FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=${MODEL_VERSION}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY models/model_v${MODEL_VERSION}.joblib ./models/model_v${MODEL_VERSION}.joblib
COPY models/feature_list.json ./models/feature_list.json

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
