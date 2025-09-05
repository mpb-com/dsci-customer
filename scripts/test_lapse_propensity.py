"""
Test Script for Customer Lapse Propensity Analysis

Tests the production lapse propensity model on a sample dataset using proper temporal
train/test splitting. Uses lifetimes calibration_and_holdout_data for evaluation.

Usage: python scripts/test_lapse_propensity.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, roc_curve
from lifetimes.utils import calibration_and_holdout_data
import matplotlib.pyplot as plt

from src.config import DATA_DIR
from scripts.lapse_propensity import (
    BQ,
    ParetoEmpiricalSingleTrainSplit,
    create_btyd_features,
    log_dataframe_stats,
    log_config_constants,
    DiagnosticsTracker,
    DATE_CUTOFF,
    PROJECT_ID,
    DATABASE_NAME,
)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Test configuration
TEST_SAMPLE_SIZE = 10000000  # Sample size for testing
TEST_HORIZON_DAYS = 365  # Test period for evaluation
TEST_TABLE_NAME = "customer_ltv_analysis_test"
CALIBRATION_END_DATE = "2024-01-01"  # End of training period

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def fetch_sample_transactions(bq: BQ, sample_size: int, observation_end_date: str):
    """Fetch sample transactions for testing"""

    sample_query = f"""
    WITH sampled_customers AS (
        SELECT DISTINCT customer_id 
        FROM `mpb-data-science-dev-ab-602d.dsci_daw.STV` 
        WHERE DATE(transaction_completed_datetime) <= @observation_end_date
        AND transaction_completed_datetime is not null
        ORDER BY RAND()
        LIMIT {sample_size}
    )
    SELECT customer_id, 
    DATETIME(transaction_completed_datetime) as txn_date,
    CASE when DATE(transaction_completed_datetime) < @date_cutoff then 1 
    else 0 end as excluded_from_training
    FROM `mpb-data-science-dev-ab-602d.dsci_daw.STV` 
    WHERE DATE(transaction_completed_datetime) <= @observation_end_date
    AND transaction_completed_datetime is not null
    AND customer_id IN (SELECT customer_id FROM sampled_customers)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "observation_end_date", "DATE", observation_end_date
            ),
            bigquery.ScalarQueryParameter("date_cutoff", "DATE", DATE_CUTOFF),
        ]
    )

    dtypes = {
        "customer_id": "int32",
        "txn_date": "datetime64[ns]",
        "excluded_from_training": "int8",
    }

    transactions = bq.to_dataframe(sample_query, job_config=job_config, dtypes=dtypes)
    log.info(
        f"Fetched {len(transactions)} transactions for {transactions['customer_id'].nunique()} customers"
    )

    return transactions


def create_temporal_train_test_split(
    transactions: pd.DataFrame, calibration_end_date: str, test_horizon_days: int
):
    """Create temporal train/test split using lifetimes calibration_and_holdout_data"""

    # Calculate dates
    calibration_end = datetime.strptime(calibration_end_date, "%Y-%m-%d")
    observation_end = calibration_end + timedelta(days=test_horizon_days)

    log.info(f"Calibration period: up to {calibration_end_date}")
    log.info(
        f"Holdout period: {calibration_end_date} to {observation_end.strftime('%Y-%m-%d')}"
    )

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
    calibration_transactions = transactions[
        transactions["txn_date"] <= calibration_end
    ].copy()
    train_features = create_btyd_features(calibration_transactions, training=True)

    # Create test features (calibration period features for prediction)
    test_features = create_btyd_features(calibration_transactions, training=False)

    # Merge with holdout data for ground truth
    test_features = test_features.merge(
        cal_holdout_data.reset_index(), on="customer_id", how="left"
    )

    # Fill missing values (customers with no holdout activity)
    test_features["frequency_holdout"] = test_features["frequency_holdout"].fillna(0)
    test_features["duration_holdout"] = test_features["duration_holdout"].fillna(
        test_horizon_days
    )

    # Create ground truth labels
    test_features["y_true_alive"] = (test_features["frequency_holdout"] > 0).astype(int)
    test_features["y_true_txns"] = test_features["frequency_holdout"].astype(int)

    log.info(f"Training features: {len(train_features)} customers")
    log.info(f"Test features: {len(test_features)} customers")
    log.info(
        f"Holdout period activity: {test_features['y_true_alive'].sum()} active customers ({test_features['y_true_alive'].mean():.1%})"
    )
    log.info(f"Holdout period transactions: {test_features['y_true_txns'].sum()} total")

    return {
        "train_features": train_features,
        "test_features": test_features,
        "cal_holdout_data": cal_holdout_data,
    }


def create_evaluation_plots(test_features: pd.DataFrame):
    """Create evaluation plots for the model"""

    y_true = test_features["y_true_alive"].values
    y_pred_proba = test_features["p_alive"].values

    # Set up the plot style
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Lapse Propensity Model Evaluation", fontsize=16, fontweight="bold")

    # 1. ROC Curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "Cannot plot ROC\n(only one class)",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title("ROC Curve")

    # 2. P(alive) Distribution by Ground Truth
    active_probs = y_pred_proba[y_true == 1]
    inactive_probs = y_pred_proba[y_true == 0]

    axes[0, 1].hist(
        inactive_probs, bins=30, alpha=0.7, label="Inactive", color="red", density=True
    )
    axes[0, 1].hist(
        active_probs, bins=30, alpha=0.7, label="Active", color="blue", density=True
    )
    axes[0, 1].set_xlabel("P(alive)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("P(alive) Distribution by Actual Status")
    axes[0, 1].legend(title="Holdout Period")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Customer Status Distribution
    status_counts = test_features["customer_status"].value_counts()
    axes[1, 0].pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1, 0].set_title("Customer Status Distribution")

    # 4. P(alive) vs Days Since Last Transaction
    scatter_alpha = min(0.6, 5000 / len(test_features))
    scatter = axes[1, 1].scatter(
        test_features["days_since_last"],
        test_features["p_alive"],
        c=test_features["y_true_alive"],
        alpha=scatter_alpha,
        cmap="RdYlBu_r",
        s=20,
    )
    axes[1, 1].set_xlabel("Days Since Last Transaction")
    axes[1, 1].set_ylabel("P(alive)")
    axes[1, 1].set_title("P(alive) vs Days Since Last Transaction")
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("Active in Holdout")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = Path(__file__).parent / "model_evaluation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved evaluation plots to {plot_path}")


def evaluate_model_predictions(test_features: pd.DataFrame):
    """Evaluate model predictions using standard classification metrics"""

    y_true = test_features["y_true_alive"].values
    y_pred_proba = test_features["p_alive"].values
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = np.nan

    try:
        logloss = log_loss(y_true, y_pred_proba)
    except ValueError:
        logloss = np.nan

    try:
        brier = brier_score_loss(y_true, y_pred_proba)
    except ValueError:
        brier = np.nan

    # Basic classification metrics
    accuracy = (y_true == y_pred).mean()
    precision = (
        ((y_pred == 1) & (y_true == 1)).sum() / (y_pred == 1).sum()
        if (y_pred == 1).sum() > 0
        else np.nan
    )
    recall = (
        ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
        if (y_true == 1).sum() > 0
        else np.nan
    )

    # Customer status distribution
    status_dist = test_features["customer_status"].value_counts(normalize=True)

    # Create plots
    create_evaluation_plots(test_features)

    # Log results
    log.info("=" * 50)
    log.info("MODEL EVALUATION RESULTS")
    log.info("=" * 50)
    log.info(f"Dataset size: {len(test_features)} customers")
    log.info(f"Baseline (% active): {y_true.mean():.1%}")
    log.info("")
    log.info("Classification Metrics:")
    log.info(f"  AUC: {auc:.4f}")
    log.info(f"  Log Loss: {logloss:.4f}")
    log.info(f"  Brier Score: {brier:.4f}")
    log.info(f"  Accuracy: {accuracy:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall: {recall:.4f}")
    log.info("")
    log.info("Customer Status Distribution:")
    for status, pct in status_dist.items():
        log.info(f"  {status}: {pct:.1%}")
    log.info("")
    log.info("P(alive) Distribution:")
    log.info(f"  Min: {y_pred_proba.min():.4f}")
    log.info(f"  25%: {np.percentile(y_pred_proba, 25):.4f}")
    log.info(f"  50%: {np.percentile(y_pred_proba, 50):.4f}")
    log.info(f"  75%: {np.percentile(y_pred_proba, 75):.4f}")
    log.info(f"  Max: {y_pred_proba.max():.4f}")
    log.info("=" * 50)

    return {
        "auc": auc,
        "log_loss": logloss,
        "brier_score": brier,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "baseline_active_rate": y_true.mean(),
        "n_customers": len(test_features),
    }


def test_production_model():
    """Test the production model with proper temporal evaluation"""
    log.info("Testing production lapse propensity model")

    # Initialize
    bq = BQ()
    diagnostics = DiagnosticsTracker()
    diagnostics.checkpoint("Test initialization")

    # Calculate observation end date
    calibration_end = datetime.strptime(CALIBRATION_END_DATE, "%Y-%m-%d")
    observation_end = calibration_end + timedelta(days=TEST_HORIZON_DAYS)
    observation_end_str = observation_end.strftime("%Y-%m-%d")

    # Fetch sample data
    log.info(f"Fetching sample transactions (n={TEST_SAMPLE_SIZE})")
    transactions = fetch_sample_transactions(bq, TEST_SAMPLE_SIZE, observation_end_str)
    diagnostics.checkpoint("Data fetching")

    # Create temporal split
    log.info("Creating temporal train/test split")
    split_data = create_temporal_train_test_split(
        transactions, CALIBRATION_END_DATE, TEST_HORIZON_DAYS
    )
    diagnostics.checkpoint("Train/test split")

    train_features = split_data["train_features"]
    test_features = split_data["test_features"]

    # Filter for reasonable ranges (same as comparison notebook)
    log.info("Filtering features for reasonable ranges")
    train_filtered = train_features.copy()

    test_filtered = test_features.copy()

    log.info(
        f"Filtered to {len(train_filtered)} training and {len(test_filtered)} test customers"
    )

    # Train production model
    log.info("Training production model")
    model = ParetoEmpiricalSingleTrainSplit()
    model.fit(train_filtered)
    diagnostics.checkpoint("Model training")

    # Generate predictions
    log.info("Generating predictions")
    test_filtered["p_alive"] = model.p_alive(test_filtered)
    test_filtered["customer_status"] = model.customer_status(test_filtered)
    
    # Save test results to parquet
    test_filtered.to_parquet(Path(DATA_DIR) / "test_results.parquet")
    log.info("Saved test results to test_results.parquet")
    
    diagnostics.checkpoint("Prediction generation")

    # Evaluate model
    log.info("Evaluating model performance")
    metrics = evaluate_model_predictions(test_filtered)
    diagnostics.checkpoint("Model evaluation")

    # Log feature statistics
    log_dataframe_stats(test_filtered, "Final test results")

    # Save test results
    test_table_id = f"{PROJECT_ID}.{DATABASE_NAME}.{TEST_TABLE_NAME}"
    log.info(f"Saving test results to {test_table_id}")

    # Select relevant columns for saving
    save_columns = [
        "customer_id",
        "p_alive",
        "customer_status",
        "frequency",
        "recency",
        "T",
        "days_since_last",
        "y_true_alive",
        "y_true_txns",
        "frequency_holdout",
    ]
    test_results = test_filtered[save_columns].copy()
    bq.to_bq(test_results, test_table_id)
    diagnostics.checkpoint("Results saving")

    diagnostics.summary()

    return {
        "metrics": metrics,
        "test_results": test_filtered,
        "model": model,
    }


def validate_production_config():
    """Validate that production configuration is reasonable"""
    log.info("Validating production configuration")
    log_config_constants()

    # Basic validation checks
    from scripts.lapse_propensity import (
        ACTIVE_PROBABILITY_CUTOFF,
        LAPSING_PROBABILITY_CUTOFF,
        ALIVE_CUTOFF_DAYS,
        LAPSING_CUTOFF_DAYS,
    )

    assert 0 < LAPSING_PROBABILITY_CUTOFF < ACTIVE_PROBABILITY_CUTOFF < 1, (
        "Probability cutoffs must be ordered: 0 < lapsing < active < 1"
    )
    assert 0 < ALIVE_CUTOFF_DAYS < LAPSING_CUTOFF_DAYS, (
        "Days cutoffs must be ordered: 0 < alive < lapsing"
    )

    log.info("Configuration validation passed")


def main():
    """Run the production model test"""
    log.info("=" * 60)
    log.info("STARTING LAPSE PROPENSITY MODEL TEST")
    log.info("=" * 60)

    try:
        # Validate configuration
        validate_production_config()

        # Run test
        results = test_production_model()

        # Summary
        metrics = results["metrics"]
        log.info("=" * 60)
        log.info("TEST COMPLETED SUCCESSFULLY!")
        log.info(f"Evaluated on {metrics['n_customers']} customers")
        log.info(f"Model AUC: {metrics['auc']:.4f}")
        log.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
        log.info(
            f"Test results saved to: {PROJECT_ID}.{DATABASE_NAME}.{TEST_TABLE_NAME}"
        )
        log.info("=" * 60)

        return results

    except Exception as e:
        log.error("=" * 60)
        log.error(f"TEST FAILED: {e}")
        log.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
