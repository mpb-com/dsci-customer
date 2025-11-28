from datetime import datetime, timedelta
from lifetimes.utils import calibration_and_holdout_data
from loguru import logger as log
import pandas as pd
from .features import create_btyd_features
from .config import (
    CALIBRATION_END_DATE,
    TEST_HORIZON_DAYS,
    TEST_SAMPLE_SIZE,
    MIN_TRANSACTION_COUNT,
)
from .model import ParetoEmpiricalSingleTrainSplit
from .eval import evaluate_model_predictions
from pathlib import Path
from .config import DATA_DIR


def fetch_sample_transactions(bq, sample_size: int, observation_end_date: str):
    """Fetch sample transactions for testing with date limit and sampling"""
    # Build HAVING clause if MIN_TRANSACTION_COUNT is set
    having_clause = f"HAVING COUNT(*) > {MIN_TRANSACTION_COUNT}" if MIN_TRANSACTION_COUNT is not None else ""

    sample_query = f"""
    WITH sampled_customers AS (
        SELECT customer_id
        FROM `mpb-data-science-dev-ab-602d.dsci_daw.STV`
        WHERE DATE(transaction_completed_datetime) <= '{observation_end_date}'
        AND transaction_completed_datetime is not null
        GROUP BY customer_id
        {having_clause}
        ORDER BY RAND()
        LIMIT {sample_size}
    )
    SELECT customer_id,
    DATETIME(transaction_completed_datetime) as txn_date
    FROM `mpb-data-science-dev-ab-602d.dsci_daw.STV`
    WHERE DATE(transaction_completed_datetime) <= '{observation_end_date}'
    AND transaction_completed_datetime is not null
    AND customer_id IN (SELECT customer_id FROM sampled_customers)
    """

    transactions = bq.get_string(sample_query)
    log.info(f"Fetched {len(transactions)} transactions for {transactions['customer_id'].nunique()} customers")
    return transactions


def create_temporal_train_test_split(transactions: pd.DataFrame, calibration_end_date: str, test_horizon_days: int):
    """Create temporal train/test split using lifetimes calibration_and_holdout_data"""

    # Calculate dates
    calibration_end = datetime.strptime(calibration_end_date, "%Y-%m-%d")
    observation_end = calibration_end + timedelta(days=test_horizon_days)

    log.info(f"Calibration period: up to {calibration_end_date}")
    log.info(f"Holdout period: {calibration_end_date} to {observation_end.strftime('%Y-%m-%d')}")

    # Use lifetimes for proper temporal split
    cal_holdout_data = calibration_and_holdout_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="txn_date",
        calibration_period_end=calibration_end,
        observation_period_end=observation_end,
        freq="D",
    )

    log.info(f"Created calibration/holdout data with {len(cal_holdout_data)} customers")
    log.info(f"Columns: {cal_holdout_data.columns.tolist()}")

    # Create training features from calibration period
    calibration_transactions = transactions[transactions["txn_date"] <= calibration_end].copy()
    train_features = create_btyd_features(calibration_transactions)

    # Create test features (calibration period features for prediction)
    test_features = create_btyd_features(calibration_transactions)

    # Merge with holdout data for ground truth
    test_features = test_features.merge(cal_holdout_data.reset_index(), on="customer_id", how="left")

    # Fill missing values (customers with no holdout activity)
    test_features["frequency_holdout"] = test_features["frequency_holdout"].fillna(0)
    test_features["duration_holdout"] = test_features["duration_holdout"].fillna(test_horizon_days)

    # Create ground truth labels
    test_features["y_true_alive"] = (test_features["frequency_holdout"] > 0).astype(int)
    test_features["y_true_txns"] = test_features["frequency_holdout"].astype(int)

    log.info(f"Training features: {len(train_features)} customers")
    log.info(f"Test features: {len(test_features)} customers")
    log.info(f"Holdout period activity: {test_features['y_true_alive'].sum()})")
    log.info(f"active customers ({test_features['y_true_alive'].mean():.1%})")
    log.info(f"Holdout period transactions: {test_features['y_true_txns'].sum()} total")

    return {
        "train_features": train_features,
        "test_features": test_features,
        "cal_holdout_data": cal_holdout_data,
    }


def _save_metrics_to_file(metrics: dict, filepath: str):
    metrics_path = Path(__file__).parent / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("LAPSE PROPENSITY MODEL TEST METRICS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset size: {metrics['n_customers']} customers\n")
        f.write(f"Baseline (% active): {metrics['baseline_active_rate']:.1%}\n\n")
        f.write("Classification Metrics:\n")
        f.write(f"  AUC: {metrics['auc']:.4f}\n")
        f.write(f"  Log Loss: {metrics['log_loss']:.4f}\n")
        f.write(f"  Brier Score: {metrics['brier_score']:.4f}\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")


def backtest_pipeline(bq):
    log.info("Starting backtest pipeline for lapse propensity model")
    # Calculate observation end date
    calibration_end = datetime.strptime(CALIBRATION_END_DATE, "%Y-%m-%d")
    observation_end = calibration_end + timedelta(days=TEST_HORIZON_DAYS)
    observation_end_str = observation_end.strftime("%Y-%m-%d")

    # Fetch sample transactions
    transactions = fetch_sample_transactions(bq, TEST_SAMPLE_SIZE, observation_end_str)
    split_data = create_temporal_train_test_split(transactions, CALIBRATION_END_DATE, TEST_HORIZON_DAYS)
    train_features = split_data["train_features"]
    test_features = split_data["test_features"]

    # Fit model
    model = ParetoEmpiricalSingleTrainSplit()
    model.fit(train_features)

    # Make predictions on test set
    test_features["p_alive"] = model.p_alive(test_features)
    test_features["customer_status"] = model.customer_status(test_features)
    test_features.to_parquet(Path(DATA_DIR) / "test_results.parquet")

    # Evaluate predictions
    metrics = evaluate_model_predictions(test_features)
    _save_metrics_to_file(metrics, Path(DATA_DIR) / "test_metrics.txt")
    log.info("Metrics computed and saved to file")
    return {
        "metrics": metrics,
        "test_results": test_features,
        "model": model,
    }
