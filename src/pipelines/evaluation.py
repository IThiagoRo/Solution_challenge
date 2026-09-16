import logging
import sys

import numpy as np
import pandas as pd

from src.core.exception import CustomException
from src.pipelines import training

logger = logging.getLogger(__name__)

TOP_N_FEATURES_DEFAULT = 15


def evaluate_on_test(
    model,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    monto_test: pd.Series,
) -> dict:
    """Final, one-shot evaluation on a held-out test set: classification metrics at
    both the default (0.5) and business threshold, plus profit against the two
    trivial baselines (approve everything / reject everything).

    The threshold is expected to already be calibrated on validation (see
    training.train_pipeline) -- it is never re-optimized here, to avoid overfitting
    the decision policy to the very data used to report final performance.
    """
    try:
        logger.info(
            "Evaluating model on %d test rows (threshold=%.3f)",
            len(X_test),
            threshold,
        )

        test_proba = model.predict_proba(X_test)[:, 1]
        approved = test_proba < threshold

        metrics_default = training.evaluate_classification_metrics(
            y_test, test_proba, threshold=0.5
        )
        metrics_business = training.evaluate_classification_metrics(
            y_test, test_proba, threshold=threshold
        )

        profit_model = training.portfolio_profit(y_test, approved, monto_test)
        profit_approve_all = training.portfolio_profit(
            y_test, np.ones(len(y_test), dtype=bool), monto_test
        )
        profit_reject_all = training.portfolio_profit(
            y_test, np.zeros(len(y_test), dtype=bool), monto_test
        )

        results = {
            "ROC-AUC": metrics_default["ROC-AUC"],
            "PR-AUC": metrics_default["PR-AUC"],
            "F1 @0.5": metrics_default["F1 @0.5"],
            "F6 @0.5": metrics_default["F6 @0.5"],
            f"F1 @{threshold}": metrics_business[f"F1 @{threshold}"],
            f"F6 @{threshold}": metrics_business[f"F6 @{threshold}"],
            "profit_model": profit_model,
            "profit_approve_all": profit_approve_all,
            "profit_reject_all": profit_reject_all,
            "profit_uplift_vs_approve_all": profit_model - profit_approve_all,
        }
        logger.info(
            "Test profit: model=%.2f, approve_all=%.2f, reject_all=%.2f, uplift=%.2f",
            profit_model,
            profit_approve_all,
            profit_reject_all,
            results["profit_uplift_vs_approve_all"],
        )
        return results
    except Exception as e:
        logger.error("Failed to evaluate model on test set: %s", e)
        raise CustomException(str(e), sys) from e


def get_feature_importance(
    model, feature_columns: list, top_n: int = TOP_N_FEATURES_DEFAULT
) -> pd.Series:
    """Return the top-N most important features, for any of the model types this
    project has trained: tree-based (feature_importances_) or a scikit-learn
    Pipeline wrapping a linear model (absolute coefficients).
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(
                model.feature_importances_, index=feature_columns
            )
        elif hasattr(model, "named_steps"):
            coefs = model.named_steps["clf"].coef_[0]
            importances = pd.Series(np.abs(coefs), index=feature_columns)
        else:
            logger.warning(
                "Model type %s exposes no known importance/coefficient attribute",
                type(model),
            )
            return pd.Series(dtype=float)

        top_importances = importances.sort_values(ascending=False).head(top_n)
        logger.info(
            "Top feature: %s (importance=%.4f)",
            top_importances.index[0],
            top_importances.iloc[0],
        )
        return top_importances
    except Exception as e:
        logger.error("Failed to compute feature importance: %s", e)
        raise CustomException(str(e), sys) from e


def check_threshold_stability(
    model, threshold: float, datasets: dict
) -> pd.DataFrame:
    """Compute the profit that a single, already-fixed threshold would yield across
    several named splits (e.g. train/val/test).

    `datasets` maps a split name to (X, y, amount). If profit at this threshold is
    similar across splits, the chosen decision policy is unlikely to be an artifact
    of a single partition and should hold up, at least in the short term, once in
    production (see notebooks/3_modeling.ipynb, Section 8).
    """
    try:
        rows = []
        for name, (X, y, amount) in datasets.items():
            proba = model.predict_proba(X)[:, 1]
            profit = training.portfolio_profit(y, proba < threshold, amount)
            rows.append({"split": name, "profit_at_threshold": profit})

        stability = pd.DataFrame(rows).set_index("split")
        logger.info(
            "Threshold stability across splits: %s",
            stability.to_dict()["profit_at_threshold"],
        )
        return stability
    except Exception as e:
        logger.error("Failed to check threshold stability: %s", e)
        raise CustomException(str(e), sys) from e


def evaluation_pipeline(
    model_artifact: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    monto_test: pd.Series,
    top_n_features: int = TOP_N_FEATURES_DEFAULT,
) -> dict:
    """End-to-end evaluation stage: final test metrics/profit plus feature importance
    for an already-trained model artifact (as produced by training.save_model /
    loaded with training.load_model).
    """
    try:
        model = model_artifact["model"]
        threshold = model_artifact["threshold"]
        feature_columns = model_artifact["feature_columns"]

        metrics = evaluate_on_test(
            model, threshold, X_test, y_test, monto_test
        )
        feature_importance = get_feature_importance(
            model, feature_columns, top_n=top_n_features
        )

        return {"metrics": metrics, "feature_importance": feature_importance}
    except CustomException:
        raise
    except Exception as e:
        logger.error("evaluation_pipeline failed: %s", e)
        raise CustomException(str(e), sys) from e
