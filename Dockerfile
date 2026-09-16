# syntax=docker/dockerfile:1

# ---- Builder: resolve and install Python dependencies into a venv ----
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# ---- Runtime: copy the venv and the app, nothing else ----
FROM python:3.12-slim AS runtime

# xgboost/scikit-learn load libgomp at import time; python:3.12-slim doesn't
# ship it, so importing xgboost fails without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ ./src/
COPY data/ ./data/

# Pipelines read raw CSVs from data/ and write trained-model/preprocessing
# artifacts to artifacts/ and run logs to logs/ (src/core/logger.py creates
# logs/ itself, but the others are created here so the dirs exist up front
# and can be bind-mounted for persistence across container runs).
RUN mkdir -p artifacts logs data/processed \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "src/main.py"]
CMD ["--training"]
