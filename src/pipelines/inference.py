import logging
import sys

import pandas as pd

from src.core.exception import CustomException
from src.pipelines import preprocessing, training

logger = logging.getLogger(__name__)


def load_artifacts(preprocessing_path: str, model_path: str) -> tuple[dict, dict]:
    """Load the fitted preprocessing artifact and the trained model artifact from disk."""
    try:
        preprocessing_artifact = preprocessing.load_params(preprocessing_path)
        model_artifact = training.load_model(model_path)

        if preprocessing_artifact["feature_columns"] != model_artifact["feature_columns"]:
            logger.warning(
                "Feature schema mismatch between preprocessing artifact (%s) and model "
                "artifact (%s): using the model's schema for column alignment.",
                preprocessing_path, model_path,
            )
        return preprocessing_artifact, model_artifact
    except CustomException:
        raise
    except Exception as e:
        logger.error("Failed to load inference artifacts: %s", e)
        raise CustomException(str(e), sys) from e


def _extract_transaction_ids(raw_df: pd.DataFrame) -> pd.Series:
    """Extract column `k`, the per-transaction unique identifier.

    `k` is not a business feature and is dropped during feature engineering (see
    preprocessing.transform and notebooks/1_EDA.ipynb, Section 2), but it is exactly
    what is needed here to trace each prediction back to the transaction it came from
    -- so it is read directly from the raw input instead of from the engineered features.
    """
    try:
        transaction_ids = raw_df["k"]
    except KeyError as e:
        logger.error("Raw input is missing column 'k' (transaction identifier): %s", e)
        raise CustomException(str(e), sys) from e

    if transaction_ids.isna().any():
        logger.warning(
            "%d rows have a null transaction id ('k'); their predictions will not be "
            "traceable back to a unique transaction.",
            int(transaction_ids.isna().sum()),
        )
    if transaction_ids.duplicated().any():
        logger.warning(
            "%d duplicated values found in transaction id ('k'); it is expected to be "
            "unique per row.",
            int(transaction_ids.duplicated().sum()),
        )
    return transaction_ids


def predict(raw_df: pd.DataFrame, preprocessing_artifact: dict, model_artifact: dict) -> pd.DataFrame:
    """Score a batch of new, raw (not yet preprocessed) transactions.

    Applies the already-fitted feature engineering (never re-fitted on new/production
    data), aligns the result to the schema the model was trained on, and applies the
    business decision rule (approve if P(fraud) < threshold) with the threshold that
    was calibrated on validation in the training pipeline.
    """
    try:
        logger.info("Running inference on %d new rows", len(raw_df))

        transaction_ids = _extract_transaction_ids(raw_df)

        features = preprocessing.transform(raw_df, preprocessing_artifact["params"])
        features = preprocessing.align_columns(features, model_artifact["feature_columns"])

        model = model_artifact["model"]
        threshold = model_artifact["threshold"]
        fraud_proba = model.predict_proba(features)[:, 1]
        approved = fraud_proba < threshold

        results = pd.DataFrame({
            "k": transaction_ids.values,
            "fecha": raw_df["fecha"].values,
            "monto": raw_df["monto"].values,
            "fraud_proba": fraud_proba,
            "approved": approved,
            "threshold": threshold,
            "model_name": model_artifact["model_name"],
        })

        logger.info(
            "Inference complete: %d rows scored, %d approved, %d rejected",
            len(results), int(approved.sum()), int((~approved).sum()),
        )
        return results
    except CustomException:
        raise
    except Exception as e:
        logger.error("Failed to run inference: %s", e)
        raise CustomException(str(e), sys) from e


def inference_pipeline(raw_df: pd.DataFrame, preprocessing_path: str, model_path: str) -> pd.DataFrame:
    """End-to-end inference stage: load artifacts and score a batch of new raw transactions."""
    try:
        preprocessing_artifact, model_artifact = load_artifacts(preprocessing_path, model_path)
        return predict(raw_df, preprocessing_artifact, model_artifact)
    except CustomException:
        raise
    except Exception as e:
        logger.error("inference_pipeline failed: %s", e)
        raise CustomException(str(e), sys) from e
