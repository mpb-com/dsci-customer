from datetime import datetime, timedelta
from lifetimes.utils import calibration_and_holdout_data
from loguru import logger as log
import pandas as pd
from .features import create_btyd_features
from .config import (
    TEST_END_DATE,
    TEST_HORIZON_DAYS,
    TEST_SAMPLE_SIZE,
    MIN_TRANSACTION_COUNT,
    DEAD_LIFT_MULTIPLIER,
    ALIVE_LIFT_MULTIPLIER,
)
from .model import ParetoEmpiricalSingleTrainSplit, _get_customer_status
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


def create_sliding_window_split(transactions: pd.DataFrame, test_end_date: str, test_horizon_days: int):
    """Create sliding window temporal split for production-style calibration

    This implements the production workflow in backtesting:
    1. Train Pareto on data up to (test_end - 2*horizon)
    2. Calibrate isotonic regression on the next horizon period
    3. Train fresh Pareto on data up to (test_end - horizon)
    4. Apply calibration from step 2 to predictions on final test period

    Args:
        transactions: Transaction data with customer_id and txn_date
        test_end_date: End of final test period (we have ground truth up to here)
        test_horizon_days: Length of each window (e.g., 180 days)

    Returns:
        dict with calibrator_train_features, calibrator_cal_features, prod_train_features, test_features
    """

    # Calculate dates by working backwards from test_end
    test_end = datetime.strptime(test_end_date, "%Y-%m-%d")
    calibration_start = test_end - timedelta(days=2 * test_horizon_days)
    calibration_end = test_end - timedelta(days=test_horizon_days)

    log.info("=" * 80)
    log.info("SLIDING WINDOW SPLIT (Production-Style Calibration)")
    log.info("=" * 80)
    log.info(f"Calibrator Training:  [Start] → {calibration_start.strftime('%Y-%m-%d')}")
    log.info(f"Calibration Window:   {calibration_start.strftime('%Y-%m-%d')} → {calibration_end.strftime('%Y-%m-%d')}")
    log.info(f"Production Training:  [Start] → {calibration_end.strftime('%Y-%m-%d')}")
    log.info(f"Test Window:          {calibration_end.strftime('%Y-%m-%d')} → {test_end_date}")
    log.info("")

    # 1. CALIBRATOR TRAINING: Fit "old" model to learn systematic bias
    calibrator_train_txns = transactions[transactions["txn_date"] <= calibration_start].copy()
    calibrator_train_features = create_btyd_features(calibrator_train_txns)
    log.info(f"Calibrator training set: {len(calibrator_train_features)} customers")

    # 2. CALIBRATION WINDOW: Get ground truth to measure bias
    cal_holdout_data_cal = calibration_and_holdout_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="txn_date",
        calibration_period_end=calibration_start,
        observation_period_end=calibration_end,
        freq="D",
    )

    calibrator_cal_features = calibrator_train_features.copy()
    calibrator_cal_features = calibrator_cal_features.merge(
        cal_holdout_data_cal.reset_index(), on="customer_id", how="left"
    )
    calibrator_cal_features["frequency_holdout"] = calibrator_cal_features["frequency_holdout"].fillna(0)
    calibrator_cal_features["y_true_alive"] = (calibrator_cal_features["frequency_holdout"] > 0).astype(int)
    log.info(f"Calibration window: {len(calibrator_cal_features)} customers")
    log.info(
        f"Calibration period activity: {calibrator_cal_features['y_true_alive'].sum()} active "
        f"({calibrator_cal_features['y_true_alive'].mean():.1%})"
    )

    # 3. PRODUCTION TRAINING: Fit "fresh" model on all data up to test window
    prod_train_txns = transactions[transactions["txn_date"] <= calibration_end].copy()
    prod_train_features = create_btyd_features(prod_train_txns)
    log.info(f"Production training set: {len(prod_train_features)} customers")

    # 4. TEST WINDOW: Final evaluation
    cal_holdout_data_test = calibration_and_holdout_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="txn_date",
        calibration_period_end=calibration_end,
        observation_period_end=test_end,
        freq="D",
    )

    test_features = prod_train_features.merge(cal_holdout_data_test.reset_index(), on="customer_id", how="left")
    test_features["frequency_holdout"] = test_features["frequency_holdout"].fillna(0)
    test_features["duration_holdout"] = test_features["duration_holdout"].fillna(test_horizon_days)
    test_features["y_true_alive"] = (test_features["frequency_holdout"] > 0).astype(int)
    test_features["y_true_txns"] = test_features["frequency_holdout"].astype(int)

    log.info(f"Test set: {len(test_features)} customers")
    log.info(
        f"Test period activity: {test_features['y_true_alive'].sum()} active "
        f"({test_features['y_true_alive'].mean():.1%})"
    )
    log.info(f"Test period transactions: {test_features['y_true_txns'].sum()} total")
    log.info("=" * 80)
    log.info("")

    return {
        "calibrator_train_features": calibrator_train_features,
        "calibrator_cal_features": calibrator_cal_features,
        "prod_train_features": prod_train_features,
        "test_features": test_features,
    }


def _save_metrics_to_file(metrics: dict, filepath: str):
    metrics_path = DATA_DIR / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LAPSE PROPENSITY MODEL TEST METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset size: {metrics['n_customers']} customers\n")
        f.write(f"Baseline (% active): {metrics['baseline_active_rate']:.1%}\n\n")

        # Classification metrics
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"  AUC (Ranking Power): {metrics['auc']:.4f}\n\n")
        f.write("Legacy Metrics (0.5 threshold - NOT RECOMMENDED):\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n\n")
        alive_thresh = metrics.get('alive_threshold', 'N/A')
        dead_thresh = metrics.get('dead_threshold', 'N/A')
        f.write("Threshold-Aware Metrics (using lift-based thresholds - RECOMMENDED):\n")
        f.write(f"  ALIVE Precision (p > {alive_thresh:.3f}): {metrics['alive_precision']:.4f}\n")
        f.write(f"  ALIVE Recall (p > {alive_thresh:.3f}):    {metrics['alive_recall']:.4f}\n")
        f.write(f"  LOST Precision (p < {dead_thresh:.3f}):  {metrics['lost_precision']:.4f}\n")
        f.write(f"  LOST Recall (p < {dead_thresh:.3f}):     {metrics['lost_recall']:.4f}\n\n")
        f.write("Calibration Metrics:\n")
        f.write(f"  Log Loss: {metrics['log_loss']:.4f}\n")
        f.write(f"  Brier Score: {metrics['brier_score']:.4f}\n\n")

        # Safety Net threshold comparison
        if metrics.get('safety_net_threshold'):
            sn = metrics['safety_net_threshold']
            f.write("=" * 80 + "\n")
            f.write("THRESHOLD COMPARISON\n")
            f.write("=" * 80 + "\n")
            f.write(f"Lift-Based DEAD:     < {dead_thresh:.4f}\n")
            f.write(f"Safety Net DEAD:     < {sn['threshold']:.4f} ({sn['target_recall']:.0%} recall target)\n\n")
            f.write(f"Safety Net excludes: {sn['pct_excluded']:.1%} of customers ({sn['n_excluded']:,})\n")
            f.write(f"Safety Net captures: {sn['actual_recall']:.1%} of active customers\n")
            f.write(f"Safety Net risk:     {1 - sn['actual_recall']:.1%} of missing sales\n\n")

        # Purchase timing analysis
        if metrics.get('purchase_timing'):
            pt = metrics['purchase_timing']
            f.write("=" * 80 + "\n")
            f.write("PURCHASE TIMING ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write("Time to 2nd Purchase:\n")
            f.write(f"  Mean:   {pt['time_to_2nd_mean']:.1f} days\n")
            f.write(f"  Median: {pt['time_to_2nd_median']:.1f} days\n")
            f.write(f"  P75:    {pt['time_to_2nd_p75']:.1f} days\n")
            f.write(f"  P90:    {pt['time_to_2nd_p90']:.1f} days\n")
            f.write(f"  P95:    {pt['time_to_2nd_p95']:.1f} days\n\n")
            f.write("One-and-Done:\n")
            f.write(f"  Rate: {pt['one_and_done_rate']:.1%}\n")
            f.write(f"  Count: {pt['one_time_buyers']:,} / {pt['total_customers']:,}\n\n")
            if pt['n_whales'] > 0:
                f.write("Whale IPT (5+ purchases):\n")
                f.write(f"  Mean IPT:   {pt['whale_ipt_mean']:.1f} days\n")
                f.write(f"  Median IPT: {pt['whale_ipt_median']:.1f} days\n")
                f.write(f"  N Whales:   {pt['n_whales']:,}\n\n")

        # Bucket performance
        f.write("=" * 80 + "\n")
        f.write("BUCKET PERFORMANCE\n")
        f.write("=" * 80 + "\n")
        bucket_df = metrics.get('bucket_performance')
        if bucket_df is not None:
            f.write(bucket_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            f.write("\n\n")

        # Decile analysis
        f.write("=" * 80 + "\n")
        f.write("DECILE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        decile_df = metrics.get('decile_analysis')
        if decile_df is not None:
            f.write(decile_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            f.write("\n\n")

        # Threshold analysis
        f.write("=" * 80 + "\n")
        f.write("THRESHOLD ANALYSIS\n")
        f.write("=" * 80 + "\n")
        threshold_df = metrics.get('threshold_analysis')
        if threshold_df is not None:
            f.write(threshold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            f.write("\n\n")

        # Segment analysis
        f.write("=" * 80 + "\n")
        f.write("SEGMENT ANALYSIS\n")
        f.write("=" * 80 + "\n")
        segment_df = metrics.get('segment_analysis')
        if segment_df is not None:
            f.write(segment_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("End of metrics report\n")
        f.write("=" * 80 + "\n")


def backtest_pipeline(bq):
    log.info("Starting backtest pipeline for lapse propensity model (production-style calibration)")

    # Fetch sample transactions up to test end date
    transactions = fetch_sample_transactions(bq, TEST_SAMPLE_SIZE, TEST_END_DATE)

    # Create sliding window split (mimics production workflow)
    split_data = create_sliding_window_split(transactions, TEST_END_DATE, TEST_HORIZON_DAYS)
    calibrator_train_features = split_data["calibrator_train_features"]
    calibrator_cal_features = split_data["calibrator_cal_features"]
    prod_train_features = split_data["prod_train_features"]
    test_features = split_data["test_features"]

    # STEP 1: Train "old" model to learn systematic bias
    log.info("=" * 80)
    log.info("STEP 1: Training calibrator model (learns systematic bias)")
    log.info("=" * 80)
    calibrator_model = ParetoEmpiricalSingleTrainSplit()
    calibrator_model.fit(calibrator_train_features)

    # STEP 2: Calibrate on calibration window (learns the "translation layer")
    log.info("=" * 80)
    log.info("STEP 2: Learning calibration (isotonic regression on calibration window)")
    log.info("=" * 80)
    calibrator_model.calibrate(calibrator_cal_features, calibrator_cal_features["y_true_alive"])

    # STEP 3: Train "fresh" production model on all available data
    log.info("=" * 80)
    log.info("STEP 3: Training production model (uses all data up to test window)")
    log.info("=" * 80)
    prod_model = ParetoEmpiricalSingleTrainSplit()
    prod_model.fit(prod_train_features)

    # STEP 4: Apply calibration from old model to fresh model predictions
    log.info("=" * 80)
    log.info("STEP 4: Applying calibration to production model predictions")
    log.info("=" * 80)

    # Get uncalibrated predictions from production model
    uncalibrated_probs = prod_model.p_alive(test_features)

    # Apply the calibration "glasses" learned from the old model
    # We do this manually by extracting the calibrators and applying them
    pareto_mask = test_features["frequency"] > 0  # Same logic as in model
    calibrated_probs = uncalibrated_probs.copy()

    if calibrator_model.pareto.calibrator is not None:
        calibrated_probs[pareto_mask] = calibrator_model.pareto.calibrator.predict(uncalibrated_probs[pareto_mask])

    if calibrator_model.empirical.calibrator is not None:
        calibrated_probs[~pareto_mask] = calibrator_model.empirical.calibrator.predict(uncalibrated_probs[~pareto_mask])

    test_features["p_alive"] = calibrated_probs

    # Calculate dynamic thresholds based on baseline rate in test set
    baseline_rate = test_features["y_true_alive"].mean()
    dead_threshold = baseline_rate * DEAD_LIFT_MULTIPLIER
    alive_threshold = baseline_rate * ALIVE_LIFT_MULTIPLIER

    log.info("=" * 80)
    log.info("DYNAMIC THRESHOLDS (Lift-Based)")
    log.info("=" * 80)
    log.info(f"Baseline active rate: {baseline_rate:.1%}")
    log.info(f"Dead threshold (< {DEAD_LIFT_MULTIPLIER}x baseline):  {dead_threshold:.3f} ({dead_threshold:.1%})")
    log.info(f"Alive threshold (> {ALIVE_LIFT_MULTIPLIER}x baseline): {alive_threshold:.3f} ({alive_threshold:.1%})")
    log.info("")

    # Apply customer status labels using dynamic thresholds
    test_features["customer_status"] = calibrated_probs.apply(
        lambda p: _get_customer_status(p, dead_threshold, alive_threshold)
    )
    test_features.to_parquet(Path(DATA_DIR) / "test_results.parquet")

    # Evaluate predictions (pass transactions for purchase timing analysis)
    log.info("=" * 80)
    log.info("STEP 5: Evaluating predictions on test window")
    log.info("=" * 80)
    # Pass calibrator_model for scenario plots (has calibrators attached)
    metrics = evaluate_model_predictions(test_features, transactions=transactions, trained_model=calibrator_model)
    _save_metrics_to_file(metrics, Path(DATA_DIR) / "test_metrics.txt")
    log.info("Metrics computed and saved to file")

    return {
        "metrics": metrics,
        "test_results": test_features,
        "calibrator_model": calibrator_model,
        "prod_model": prod_model,
    }
