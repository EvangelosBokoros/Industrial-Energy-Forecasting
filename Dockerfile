FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-serving.txt ./requirements-serving.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --requirement requirements-serving.txt

RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY src ./src

COPY models/ensemble_2_0_model.joblib \
    ./models/ensemble_2_0_model.joblib

COPY config/serving/model_metadata.json \
    ./config/serving/model_metadata.json

COPY config/serving/input_reference.json \
    ./config/serving/input_reference.json

RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]