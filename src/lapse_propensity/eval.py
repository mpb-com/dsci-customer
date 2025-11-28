import pandas as pd
from loguru import logger as log
from .config import (
    ACTIVE_PROBABILITY_CUTOFF,
    LAPSING_PROBABILITY_CUTOFF,
    ALIVE_CUTOFF_DAYS,
    LAPSING_CUTOFF_DAYS,
    PARETO_PENALIZER,
    TRANSACTION_EMPIRICAL_CUTOFF,
    MAX_FREQUENCY_CUTOFF,
)
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def log_dataframe_stats(df: pd.DataFrame, name: str):
    """Log dataframe statistics in consistent format"""
    log.info(f"{name} with {df.shape[0]} rows")
    log.info(f"Columns: {df.columns.tolist()}")
    log.info(f"Sample data:\n{df.head()}")
    log.info(f"Data types:\n{df.dtypes}")
    log.info(f"Statistics:\n{df.describe().round(3)}")
    log.info(f"Status distribution:\n{df.get('customer_status', pd.Series()).value_counts(normalize=True).round(3)}")
    deciles = (
        pd.qcut(df.get("p_alive", pd.Series()), 10, duplicates="drop")
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
    )
    log.info(f"P_alive deciles:\n{deciles}")


def log_config_constants():
    """Log all configuration constants"""
    log.info("Configuration constants:")
    log.info(f"  ACTIVE_PROBABILITY_CUTOFF: {ACTIVE_PROBABILITY_CUTOFF}")
    log.info(f"  LAPSING_PROBABILITY_CUTOFF: {LAPSING_PROBABILITY_CUTOFF}")
    log.info(f"  ALIVE_CUTOFF_DAYS: {ALIVE_CUTOFF_DAYS}")
    log.info(f"  LAPSING_CUTOFF_DAYS: {LAPSING_CUTOFF_DAYS}")
    log.info(f"  PARETO_PENALIZER: {PARETO_PENALIZER}")
    log.info(f"  TRANSACTION_EMPIRICAL_CUTOFF: {TRANSACTION_EMPIRICAL_CUTOFF}")
    log.info(f"  MAX_FREQUENCY_CUTOFF: {MAX_FREQUENCY_CUTOFF}")


def create_evaluation_plots(test_features: pd.DataFrame):
    """Create evaluation plots for the model"""

    y_true = test_features["y_true_alive"].to_numpy(dtype=np.float64)
    y_pred_proba = test_features["p_alive"].to_numpy(dtype=np.float64)

    # Set up the plot style
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Lapse Propensity Model Evaluation", fontsize=16, fontweight="bold")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 0].set_title("ROC Curve")

    # 2. P(alive) Distribution by Ground Truth
    active_probs = y_pred_proba[y_true == 1]
    inactive_probs = y_pred_proba[y_true == 0]

    axes[0, 1].hist(inactive_probs, bins=30, alpha=0.7, label="Inactive", color="red", density=True)
    axes[0, 1].hist(active_probs, bins=30, alpha=0.7, label="Active", color="blue", density=True)
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

    y_true = test_features["y_true_alive"].to_numpy(dtype=np.float64)
    y_pred_proba = test_features["p_alive"].to_numpy(dtype=np.float64)
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
    precision = ((y_pred == 1) & (y_true == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else np.nan
    recall = ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else np.nan

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
