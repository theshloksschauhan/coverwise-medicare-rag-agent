# The vector index and embedding model are built into the image, so the container needs no external DB.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/models PORT=8080 \n    COVERWISE_RAW_DIR=/app/data/raw COVERWISE_INDEX_DIR=/app/data/index
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

COPY data/raw ./data/raw
RUN python -m coverwise.ingest

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8080
CMD ["sh", "-c", "uvicorn coverwise.api:app --host 0.0.0.0 --port ${PORT}"]
