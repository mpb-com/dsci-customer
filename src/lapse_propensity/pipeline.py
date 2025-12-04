from datetime import datetime, timedelta
from loguru import logger as log
from lifetimes.utils import calibration_and_holdout_data
from .features import fetch_transactions, create_btyd_features
from .model import ParetoEmpiricalSingleTrainSplit, _get_customer_status
from .eval import log_dataframe_stats, log_config_constants
from .config import (
    FINAL_COLUMNS,
    TABLE_NAME,
    TEST_HORIZON_DAYS,
)


def pipe(bq, calibration_window_days=None):
    """
    Production pipeline using sliding window calibration approach.

    This mirrors the backtesting workflow:
    1. Fit calibrator model on older data
    2. Calibrate on recent window (where we have ground truth)
    3. Fit production model on all data
    4. Apply calibration from step 2 to production predictions
    5. Use dynamic lift-based thresholds

    Args:
        bq: BigQuery helper instance
        calibration_window_days: Days to use for calibration window (default: TEST_HORIZON_DAYS)
    """
    log.info("=" * 80)
    log.info("Starting lapse propensity model (production-style calibration)")
    log.info("=" * 80)

    if calibration_window_days is None:
        calibration_window_days = TEST_HORIZON_DAYS

    # Fetch all transactions
    log.info("Fetching transactions")
    transactions = fetch_transactions(bq)

    # Calculate dates for calibration window
    today = datetime.now()
    calibration_start = today - timedelta(days=2 * calibration_window_days)
    calibration_end = today - timedelta(days=calibration_window_days)

    log.info("=" * 80)
    log.info("CALIBRATION WINDOW SETUP")
    log.info("=" * 80)
    log.info(f"Today:                {today.strftime('%Y-%m-%d')}")
    log.info(f"Calibrator training:  [Start] → {calibration_start.strftime('%Y-%m-%d')}")
    log.info(f"Calibration window:   {calibration_start.strftime('%Y-%m-%d')} → {calibration_end.strftime('%Y-%m-%d')}")
    log.info(f"Production training:  [Start] → {calibration_end.strftime('%Y-%m-%d')}")
    log.info("")

    # STEP 1: Train calibrator model on older data
    log.info("=" * 80)
    log.info("STEP 1: Training calibrator model (learns systematic bias)")
    log.info("=" * 80)
    calibrator_txns = transactions[transactions["txn_date"] <= calibration_start].copy()
    calibrator_features = create_btyd_features(calibrator_txns)
    log.info(f"Calibrator training: {len(calibrator_features)} customers")

    calibrator_model = ParetoEmpiricalSingleTrainSplit()
    calibrator_model.fit(calibrator_features)

    # STEP 2: Calibrate on calibration window
    log.info("=" * 80)
    log.info("STEP 2: Learning calibration (isotonic regression on calibration window)")
    log.info("=" * 80)

    cal_holdout_data = calibration_and_holdout_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="txn_date",
        calibration_period_end=calibration_start,
        observation_period_end=calibration_end,
        freq="D",
    )

    calibration_features = calibrator_features.copy()
    calibration_features = calibration_features.merge(cal_holdout_data.reset_index(), on="customer_id", how="left")
    calibration_features["frequency_holdout"] = calibration_features["frequency_holdout"].fillna(0)
    calibration_features["y_true_alive"] = (calibration_features["frequency_holdout"] > 0).astype(int)

    log.info(f"Calibration window: {len(calibration_features)} customers")
    log.info(
        f"Calibration period activity: {calibration_features['y_true_alive'].sum()} active "
        f"({calibration_features['y_true_alive'].mean():.1%})"
    )

    calibrator_model.calibrate(calibration_features, calibration_features["y_true_alive"])

    # Add p_alive predictions to calibration_features for threshold calculation
    calibration_features["p_alive"] = calibrator_model.p_alive(calibration_features)

    # STEP 3: Train production model on all data up to calibration_end
    log.info("=" * 80)
    log.info("STEP 3: Training production model (uses all data up to calibration window end)")
    log.info("=" * 80)
    prod_txns = transactions[transactions["txn_date"] <= calibration_end].copy()
    prod_features = create_btyd_features(prod_txns)
    log.info(f"Production training: {len(prod_features)} customers")

    prod_model = ParetoEmpiricalSingleTrainSplit()
    prod_model.fit(prod_features)

    # STEP 4: Score ALL customers with fresh features (including recent activity)
    log.info("=" * 80)
    log.info("STEP 4: Scoring all customers with calibrated model")
    log.info("=" * 80)
    all_features = create_btyd_features(transactions)
    log.info(f"Scoring {len(all_features)} customers")

    # Get uncalibrated predictions from production model
    uncalibrated_probs = prod_model.p_alive(all_features)

    # Apply calibration "glasses" learned from calibrator model
    pareto_mask = all_features["frequency"] > 0
    calibrated_probs = uncalibrated_probs.copy()

    if calibrator_model.pareto.calibrator is not None:
        calibrated_probs[pareto_mask] = calibrator_model.pareto.calibrator.predict(uncalibrated_probs[pareto_mask])

    if calibrator_model.empirical.calibrator is not None:
        calibrated_probs[~pareto_mask] = calibrator_model.empirical.calibrator.predict(uncalibrated_probs[~pareto_mask])

    all_features["p_alive"] = calibrated_probs.round(4)

    # Calculate business thresholds using calibration window (where we have ground truth)
    from .eval import calculate_business_thresholds
    from .config import MAX_REVENUE_RISK, MIN_ALIVE_LIFT

    business_thresholds = calculate_business_thresholds(
        calibration_features, max_revenue_risk=MAX_REVENUE_RISK, min_alive_lift=MIN_ALIVE_LIFT
    )
    dead_threshold = business_thresholds["dead_threshold"]
    alive_threshold = business_thresholds["alive_threshold"]
    baseline_rate = calibration_features["y_true_alive"].mean()

    log.info("=" * 80)
    log.info("BUSINESS THRESHOLDS (from calibration window)")
    log.info("=" * 80)
    log.info(f"Baseline active rate (calibration window): {baseline_rate:.1%}")
    log.info(f"ALIVE threshold (>{MIN_ALIVE_LIFT:.1f}x baseline): {alive_threshold:.3f} ({alive_threshold:.1%})")
    log.info(f"  → Actual lift: {business_thresholds['actual_alive_lift']:.2f}x")
    log.info(f"DEAD threshold (<{MAX_REVENUE_RISK:.1%} revenue risk): {dead_threshold:.3f} ({dead_threshold:.1%})")
    log.info(f"  → Actual risk: {business_thresholds['actual_revenue_risk']:.1%} of active customers excluded")
    log.info(f"LAPSING range:  {dead_threshold:.3f} - {alive_threshold:.3f}")
    log.info("")

    # Apply customer status labels using business thresholds
    all_features["customer_status"] = calibrated_probs.apply(
        lambda p: _get_customer_status(p, dead_threshold, alive_threshold)
    )

    # Select final columns
    final_features = all_features[FINAL_COLUMNS]

    # Log results and config
    log.info("=" * 80)
    log.info("FINAL RESULTS")
    log.info("=" * 80)
    log_dataframe_stats(final_features, "Production scores")
    log_config_constants()

    # Save results
    log.info("=" * 80)
    log.info("SAVING TO BIGQUERY")
    log.info("=" * 80)
    bq.write_to(final_features, TABLE_NAME)
    log.info(f"Successfully saved {len(final_features)} customer scores to {TABLE_NAME}")
    log.info("=" * 80)
    log.info("Production pipeline complete")
    log.info("=" * 80)
