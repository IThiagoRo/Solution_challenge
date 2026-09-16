"""
FastAPI service that loads the trained fraud-detection
model from artifacts/ and scores incoming transactions
in real time.

Run from the project root:
    uvicorn api.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictionResponse, TransactionRequest
from src.core.exception import CustomException
from src.pipelines import inference

logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
PREPROCESSING_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing_params.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.pkl")

artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup and kept in memory for the process lifetime
    logger.info("Loading model artifacts from %s", ARTIFACTS_DIR)
    preprocessing_artifact, model_artifact = inference.load_artifacts(
        PREPROCESSING_PATH, MODEL_PATH
    )
    artifacts["preprocessing"] = preprocessing_artifact
    artifacts["model"] = model_artifact
    logger.info(
        "Model loaded: %s (threshold=%.3f)",
        model_artifact["model_name"],
        model_artifact["threshold"],
    )
    yield
    artifacts.clear()


app = FastAPI(
    title="MELI Fraud Detection API",
    description="Scores transactions for fraud risk.",
    lifespan=lifespan,
)


def _score(transactions: list[TransactionRequest]) -> list[PredictionResponse]:
    """Shared scoring path for both endpoints below.

    src.pipelines.inference.predict is batch-oriented (it expects a DataFrame),
    so a single transaction is just scored as a one-row batch -- this keeps
    /predict and /predict/batch on the exact same code path as the offline
    inference pipeline, instead of re-implementing the scoring logic here.
    """
    raw_df = pd.DataFrame([t.model_dump() for t in transactions])
    try:
        results = inference.predict(
            raw_df, artifacts["preprocessing"], artifacts["model"]
        )
    except CustomException as e:
        # A CustomException here means the input passed schema validation but
        # the pipeline itself couldn't process it (e.g. an unrecoverable
        # feature-engineering error) -- that's a client input problem, not a
        # server crash, hence 422 rather than a 500.
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    return [
        PredictionResponse(
            transaction_id=row.k,
            fraud_probability=round(float(row.fraud_proba), 6),
            # "approved" (P(fraud) < threshold) is the pipeline's business
            # decision; is_fraud is just its negation, spelled out for callers
            # who only care about the fraud/not-fraud verdict.
            is_fraud=not bool(row.approved),
            threshold=float(row.threshold),
            model_name=row.model_name,
        )
        for row in results.itertuples()
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe: reports which model is loaded and its decision
    threshold, or 503 if the service came up without a model (e.g. lifespan
    failed to find the artifact files).
    """
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=artifacts["model"]["model_name"],
        threshold=artifacts["model"]["threshold"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    """Score a single raw transaction and return its fraud verdict."""
    return _score([transaction])[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    transactions: list[TransactionRequest],
) -> list[PredictionResponse]:
    """Score several raw transactions in one call (one preprocessing/inference
    pass instead of one per transaction).
    """
    if not transactions:
        raise HTTPException(status_code=400, detail="Empty transaction list")
    return _score(transactions)
