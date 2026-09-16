"""CLI entrypoint orchestrating the fraud-detection MLOps pipeline end to end.

Run from the project root:
    python src/main.py --training
    python src/main.py --inference --input path/to/new_transactions.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# isort: off
# Running this file directly (`python src/main.py`) only puts its own directory
# (src/) on sys.path, not the project root -- so `src` itself would not be
# importable as a package without this. Must run before the `src.*` imports
# below -- the isort markers keep isort from moving those imports back above
# this line the next time the file is reformatted.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import logger as _logger_setup  # noqa: E402,F401
from src.core.exception import CustomException  # noqa: E402
from src.pipelines import (  # noqa: E402
    evaluation,
    inference,
    preprocessing,
    training,
)

# isort: on

logger = logging.getLogger(__name__)

RAW_DATA_PATH_DEFAULT = (
    "data/MercadoLibre Data Scientist Technical Challenge - Dataset.csv"
)
ARTIFACTS_DIR_DEFAULT = "artifacts"
PREDICTIONS_PATH_DEFAULT = "artifacts/predictions.csv"

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def temporal_split(
    df: pd.DataFrame, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC
):
    """Chronologically split a raw dataset into train/val/test (see
    notebooks/2_feature_engineering.ipynb): val and test always come after train in
    time, to mimic how the model would only ever see past data in production.
    """
    try:
        df_sorted = df.sort_values("fecha").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * (1 - val_frac - test_frac))
        val_end = int(n * (1 - test_frac))
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        logger.info(
            "Temporal split: train=%d (%s -> %s), val=%d (%s -> %s), test=%d (%s -> %s)",
            len(train_df),
            train_df["fecha"].min(),
            train_df["fecha"].max(),
            len(val_df),
            val_df["fecha"].min(),
            val_df["fecha"].max(),
            len(test_df),
            test_df["fecha"].min(),
            test_df["fecha"].max(),
        )
        return train_df, val_df, test_df
    except Exception as e:
        logger.error("Failed to perform temporal split: %s", e)
        raise CustomException(str(e), sys) from e


def run_training(data_path: str, artifacts_dir: str) -> None:
    """Load raw data, split it temporally, fit preprocessing + the model, and save
    both artifacts to disk -- end to end equivalent of notebooks 2 and 3.
    """
    logger.info("=== Starting TRAINING pipeline ===")
    raw_df = pd.read_csv(data_path, parse_dates=["fecha"])
    train_df, val_df, test_df = temporal_split(raw_df)

    X_train, params = preprocessing.preprocess_data(train_df)
    feature_columns = list(X_train.columns)
    X_val, _ = preprocessing.preprocess_data(
        val_df, params=params, reference_columns=feature_columns
    )
    X_test, _ = preprocessing.preprocess_data(
        test_df, params=params, reference_columns=feature_columns
    )
    preprocessing.save_params(
        params, feature_columns, f"{artifacts_dir}/preprocessing_params.pkl"
    )

    y_train, y_val, y_test = (
        train_df["fraude"],
        val_df["fraude"],
        test_df["fraude"],
    )
    monto_val, monto_test = val_df["monto"], test_df["monto"]

    result = training.train_pipeline(X_train, y_train, X_val, y_val, monto_val)
    training.save_model(
        result["model_name"],
        result["model"],
        result["threshold"],
        feature_columns,
        f"{artifacts_dir}/final_model.pkl",
    )

    # Final held-out check: metrics/profit vs. baselines, plus feature importance.
    model_artifact = {
        "model_name": result["model_name"],
        "model": result["model"],
        "threshold": result["threshold"],
        "feature_columns": feature_columns,
    }
    eval_result = evaluation.evaluation_pipeline(
        model_artifact, X_test, y_test, monto_test
    )
    logger.info("Final test metrics: %s", eval_result["metrics"])
    logger.info(
        "Top features: %s", eval_result["feature_importance"].to_dict()
    )
    logger.info("=== TRAINING pipeline complete ===")


def run_inference(
    input_path: str, output_path: str, artifacts_dir: str
) -> None:
    """Load new, raw transactions and score them with the already-trained pipeline."""
    logger.info("=== Starting INFERENCE pipeline ===")
    raw_df = pd.read_csv(input_path, parse_dates=["fecha"])
    results = inference.inference_pipeline(
        raw_df,
        preprocessing_path=f"{artifacts_dir}/preprocessing_params.pkl",
        model_path=f"{artifacts_dir}/final_model.pkl",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(results), output_path)
    logger.info("=== INFERENCE pipeline complete ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fraud detection MLOps pipeline"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--training",
        action="store_true",
        help="Run the training pipeline end to end",
    )
    mode.add_argument(
        "--inference", action="store_true", help="Score new, raw transactions"
    )

    parser.add_argument(
        "--data",
        default=RAW_DATA_PATH_DEFAULT,
        help="Raw CSV used for --training (default: the challenge dataset)",
    )
    parser.add_argument(
        "--input",
        default=RAW_DATA_PATH_DEFAULT,
        help="Raw CSV of new transactions to score with --inference "
        "(default: the challenge dataset, for a quick smoke test)",
    )
    parser.add_argument(
        "--output",
        default=PREDICTIONS_PATH_DEFAULT,
        help="Where to save --inference predictions",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=ARTIFACTS_DIR_DEFAULT,
        help="Directory to read/write preprocessing and model artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.training:
            run_training(args.data, args.artifacts_dir)
        elif args.inference:
            run_inference(args.input, args.output, args.artifacts_dir)
    except CustomException as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
