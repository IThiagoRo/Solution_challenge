import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.core.exception import CustomException

logger = logging.getLogger(__name__)

TOP_N_COUNTRIES_DEFAULT = 10


def fit_preprocessing(
    train_df: pd.DataFrame, top_n_countries: int = TOP_N_COUNTRIES_DEFAULT
) -> dict:
    """Compute, from a training set, all statistics needed by transform()."""
    try:
        logger.info(
            "Fitting preprocessing params on %d training rows", len(train_df)
        )

        params = {
            "median_b": train_df["b"].median(),
            "median_c": train_df["c"].median(),
            "median_d": train_df["d"].median(),
            "median_f": train_df["f"].median(),
            "median_l": train_df["l"].median(),
            "median_m": train_df["m"].median(),
            "top_countries": train_df["g"]
            .value_counts()
            .head(top_n_countries)
            .index.tolist(),
        }
        j_counts = train_df["j"].value_counts()
        params["j_freq_map"] = (j_counts / len(train_df)).to_dict()
        params["j_freq_default"] = (
            0.0  # categories of j never seen in training
        )

        logger.info(
            "Preprocessing params fitted: top_countries=%s, j_categories=%d",
            params["top_countries"],
            len(params["j_freq_map"]),
        )
        return params
    except Exception as e:
        logger.error("Failed to fit preprocessing params: %s", e)
        raise CustomException(str(e), sys) from e


def transform(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply feature engineering to a new dataset, using ONLY already-fitted params.
    """
    try:
        logger.info(
            "Transforming %d rows with fitted preprocessing params", len(data)
        )
        out = pd.DataFrame(index=data.index)

        # A single-row batch (e.g. one transaction from the API) can make pandas
        # infer an all-null nullable column as dtype=object instead of float64,
        # which breaks np.log1p below -- coerce these to numeric up front so
        # fillna/log1p always see float64 regardless of batch size.
        data = data.copy()
        numeric_nullable_cols = ["b", "c", "d", "f", "l", "m"]
        data[numeric_nullable_cols] = data[numeric_nullable_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        # a: one-hot
        out = pd.concat(
            [
                out,
                pd.get_dummies(data["a"], prefix="a", drop_first=True).astype(
                    int
                ),
            ],
            axis=1,
        )

        # b, c: identical null pattern -> a single shared flag
        out["bc_is_null"] = data["b"].isna().astype(int)
        out["b"] = data["b"].fillna(params["median_b"])
        out["c_log"] = np.log1p(data["c"].fillna(params["median_c"]))

        # d: imputation + capped-value flag
        out["d"] = data["d"].fillna(params["median_d"])
        out["d_is_capped"] = (data["d"] == 50).astype(int)

        # e: skewed, no nulls
        out["e_log"] = np.log1p(data["e"])

        # f: has negative values -> log1p is not valid
        out["f"] = data["f"].fillna(params["median_f"])

        # g: collapse rare/missing categories, then one-hot
        g_clean = data["g"]
        is_top_or_na = g_clean.isin(params["top_countries"]) | g_clean.isna()
        g_clean = g_clean.where(is_top_or_na, other="OTHER").fillna("MISSING")
        out = pd.concat(
            [out, pd.get_dummies(g_clean, prefix="g").astype(int)], axis=1
        )

        # h: unchanged
        out["h"] = data["h"]

        # j: frequency encoding fitted on training data
        out["j_freq"] = (
            data["j"]
            .map(params["j_freq_map"])
            .fillna(params["j_freq_default"])
        )

        # l, m: imputation + log
        out["l_log"] = np.log1p(data["l"].fillna(params["median_l"]))
        out["m_log"] = np.log1p(data["m"].fillna(params["median_m"]))

        # n, p: binary
        out["n"] = data["n"]
        out["p"] = (data["p"] == "Y").astype(int)

        # o: flags (is_null / is_Y / is_N) instead of imputing
        out["o_is_null"] = data["o"].isna().astype(int)
        out["o_is_Y"] = (data["o"] == "Y").astype(int)
        out["o_is_N"] = (data["o"] == "N").astype(int)

        # score, monto
        out["score"] = data["score"]
        out["monto"] = data["monto"]
        out["monto_log"] = np.log1p(data["monto"])

        # temporal features (fecha is re-parsed in case it arrives as a string)
        fecha = pd.to_datetime(data["fecha"])
        out["hora"] = fecha.dt.hour
        out["es_madrugada"] = out["hora"].between(0, 5).astype(int)
        out["dia_semana"] = fecha.dt.dayofweek

        logger.info(
            "Transform complete: %d rows x %d features",
            out.shape[0],
            out.shape[1],
        )
        return out
    except Exception as e:
        logger.error("Failed to transform data: %s", e)
        raise CustomException(str(e), sys) from e


def align_columns(
    features: pd.DataFrame, reference_columns: list
) -> pd.DataFrame:
    """
    Align a transformed DataFrame's columns to a reference list (e.g. train's columns).

    Any expected column that is missing (due to an unseen category) is filled with 0;
    unexpected extra columns are dropped. Prevents breaking in production on new data.
    """
    try:
        return features.reindex(columns=reference_columns, fill_value=0)
    except Exception as e:
        logger.error("Failed to align columns to reference schema: %s", e)
        raise CustomException(str(e), sys) from e


def save_params(params: dict, feature_columns: list, path: str) -> None:
    """Persist fitted params together with the expected feature-column schema."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"params": params, "feature_columns": feature_columns}, path
        )
        logger.info("Saved preprocessing artifact to %s", path)
    except Exception as e:
        logger.error(
            "Failed to save preprocessing artifact to %s: %s", path, e
        )
        raise CustomException(str(e), sys) from e


def load_params(path: str) -> dict:
    """Load a preprocessing artifact previously saved with save_params()."""
    try:
        artifact = joblib.load(path)
        logger.info("Loaded preprocessing artifact from %s", path)
        return artifact
    except Exception as e:
        logger.error(
            "Failed to load preprocessing artifact from %s: %s", path, e
        )
        raise CustomException(str(e), sys) from e


def preprocess_data(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    reference_columns: Optional[list] = None,
) -> tuple[pd.DataFrame, dict]:
    """Entry point of the preprocessing pipeline.

    - If `params` is None, `df` is assumed to be the training set: it is fitted
      on `df` itself (training mode) before being transformed.
    - If `params` is provided (e.g. loaded with `load_params` from a previous
      run), `df` is treated as a new, incoming dataset: it is only transformed,
      and aligned to `reference_columns` when given (recommended in production).

    Returns (features, params) so the caller can persist params on first use
    (training mode) or simply discard them if it already had them.
    """

    logger.info(
        "Starting preprocessing: %d rows, params=%s, reference_columns=%s",
        len(df),
        "provided" if params else "None",
        "provided" if reference_columns else "None",
    )

    try:
        if params is None:
            logger.info(
                "No params provided - fitting preprocessing on incoming data (training mode)"
            )
            params = fit_preprocessing(df)

        features = transform(df, params)

        if reference_columns is not None:
            features = align_columns(features, reference_columns)

        logger.info(
            "Preprocessing complete: %d rows x %d features",
            features.shape[0],
            features.shape[1],
        )
        return features, params
    except CustomException:
        raise
    except Exception as e:
        logger.error("preprocess_data failed: %s", e)
        raise CustomException(str(e), sys) from e
