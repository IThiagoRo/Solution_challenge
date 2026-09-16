import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from xgboost import XGBClassifier

from src.core.exception import CustomException

logger = logging.getLogger(__name__)

RANDOM_STATE_DEFAULT = 42
DEFAULT_THRESHOLDS = np.linspace(0, 1, 201)

MODEL_NAME = "XGBoost"

GAIN_RATE = 0.25
LOSS_RATE = 1.0
THEORETICAL_OPTIMAL_THRESHOLD = GAIN_RATE / (GAIN_RATE + LOSS_RATE)  # 0.20


def transaction_profit(is_fraud: int, approved: bool, amount: float) -> float:
    """Profit of a single transaction given its true label and the approve/reject decision."""
    if not approved:
        return 0.0
    return -LOSS_RATE * amount if is_fraud else GAIN_RATE * amount


def portfolio_profit(y_true, approved, amount) -> float:
    """Total profit of a set of transactions (vectorized)."""
    y_true = np.asarray(y_true)
    approved = np.asarray(approved)
    amount = np.asarray(amount)
    gains = np.where(
        ~approved,
        0.0,
        np.where(y_true == 1, -LOSS_RATE * amount, GAIN_RATE * amount),
    )
    return float(gains.sum())


def get_class_imbalance_ratio(y_train: pd.Series) -> float:
    """Compute the negative/positive class ratio, used as scale_pos_weight for XGBoost."""
    try:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        ratio = neg / pos
        logger.info(
            "Class imbalance in training set: %d legit / %d fraud -> ratio %.2f:1",
            neg,
            pos,
            ratio,
        )
        return ratio
    except Exception as e:
        logger.error("Failed to compute class imbalance ratio: %s", e)
        raise CustomException(str(e), sys) from e


def build_model(
    scale_pos_weight: float, random_state: int = RANDOM_STATE_DEFAULT
) -> XGBClassifier:
    """Build the (unfitted) XGBoost model, using the same hyperparameters chosen in
    notebooks/3_modeling.ipynb. scale_pos_weight handles class imbalance directly,
    without resampling.
    """
    try:
        # n_jobs=1 trades off training speed for full run-to-run determinism: XGBoost's
        # multithreaded histogram building is not guaranteed bit-for-bit reproducible,
        # which previously shifted the optimal threshold by one grid step between runs.
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
        logger.info("Built model: %s", MODEL_NAME)
        return model
    except Exception as e:
        logger.error("Failed to build model: %s", e)
        raise CustomException(str(e), sys) from e


def train_model(
    model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series
) -> XGBClassifier:
    """Fit the model on the training set."""
    try:
        logger.info("Training model: %s", MODEL_NAME)
        model.fit(X_train, y_train)
        logger.info("Finished training: %s", MODEL_NAME)
        return model
    except Exception as e:
        logger.error("Failed to train model: %s", e)
        raise CustomException(str(e), sys) from e


def evaluate_classification_metrics(
    y_true, y_proba, threshold: float = 0.5
) -> dict:
    """Compute threshold-agnostic (ROC-AUC, PR-AUC) and threshold-dependent (F1, F6) metrics."""
    try:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        return {
            "ROC-AUC": roc_auc_score(y_true, y_proba),
            "PR-AUC": average_precision_score(y_true, y_proba),
            f"F1 @{threshold}": fbeta_score(y_true, y_pred, beta=1),
            f"F6 @{threshold}": fbeta_score(y_true, y_pred, beta=6),
        }
    except Exception as e:
        logger.error("Failed to compute classification metrics: %s", e)
        raise CustomException(str(e), sys) from e


def optimize_threshold(
    y_true, y_proba, amount, thresholds: Optional[np.ndarray] = None
) -> tuple[float, float, np.ndarray]:
    """Sweep decision thresholds and return the one maximizing total business profit.

    A transaction is approved when `P(fraud) < threshold`. Returns
    (best_threshold, best_profit, full_profit_curve) so the curve can be reused
    for plotting or stability checks without recomputing it.
    """
    try:
        thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
        curve = np.array(
            [
                portfolio_profit(y_true, np.asarray(y_proba) < t, amount)
                for t in thresholds
            ]
        )
        best_idx = int(curve.argmax())
        best_threshold = float(thresholds[best_idx])
        best_profit = float(curve[best_idx])
        logger.info(
            "Optimal threshold: %.3f (profit=%.2f, theoretical=%.2f)",
            best_threshold,
            best_profit,
            THEORETICAL_OPTIMAL_THRESHOLD,
        )
        return best_threshold, best_profit, curve
    except Exception as e:
        logger.error("Failed to optimize decision threshold: %s", e)
        raise CustomException(str(e), sys) from e


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    monto_val: pd.Series,
    thresholds: Optional[np.ndarray] = None,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> dict:
    """End-to-end training stage: train the winning model type and calibrate its
    business decision threshold.

    Mirrors notebooks/3_modeling.ipynb's outcome directly: XGBoost was already chosen
    there by comparing it against Logistic Regression, Random Forest and LightGBM, so
    this pipeline does not repeat that comparison — it only (re)trains XGBoost on
    whatever training data it is given and re-optimizes the threshold on validation.
    """
    try:
        logger.info(
            "Starting training pipeline on %d training rows", len(X_train)
        )
        scale_pos_weight = get_class_imbalance_ratio(y_train)
        model = build_model(scale_pos_weight, random_state=random_state)
        model = train_model(model, X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        metrics_val = evaluate_classification_metrics(y_val, val_proba)
        best_threshold, best_profit, _ = optimize_threshold(
            y_val, val_proba, monto_val, thresholds
        )

        logger.info("Training pipeline complete: profit_val=%.2f", best_profit)
        return {
            "model_name": MODEL_NAME,
            "model": model,
            "metrics_val": metrics_val,
            "threshold": best_threshold,
            "profit_val": best_profit,
        }
    except CustomException:
        raise
    except Exception as e:
        logger.error("train_pipeline failed: %s", e)
        raise CustomException(str(e), sys) from e


def save_model(
    model_name: str, model, threshold: float, feature_columns: list, path: str
) -> None:
    """Persist the trained model, its business threshold and the expected feature schema."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": model_name,
                "model": model,
                "threshold": float(threshold),
                "feature_columns": feature_columns,
            },
            path,
        )
        logger.info("Saved final model artifact to %s", path)
    except Exception as e:
        logger.error("Failed to save final model artifact to %s: %s", path, e)
        raise CustomException(str(e), sys) from e


def load_model(path: str) -> dict:
    """Load a final model artifact previously saved with save_model()."""
    try:
        artifact = joblib.load(path)
        logger.info("Loaded final model artifact from %s", path)
        return artifact
    except Exception as e:
        logger.error(
            "Failed to load final model artifact from %s: %s", path, e
        )
        raise CustomException(str(e), sys) from e
