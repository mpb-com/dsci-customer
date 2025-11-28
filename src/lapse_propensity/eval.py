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
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
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


def create_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, ax: plt.Axes):
    """Create calibration curve (reliability diagram)"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy="quantile"
    )

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, ax: plt.Axes):
    """Create precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    ax.plot(recall, precision, linewidth=2, label=f"PR Curve (AP = {avg_precision:.3f})")
    ax.axhline(y=y_true.mean(), color="r", linestyle="--", label=f"Baseline = {y_true.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)


def analyze_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray) -> pd.DataFrame:
    """Analyze model performance at different probability thresholds"""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results.append(
            {
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
                "flagged_pct": (y_pred == 1).mean(),
            }
        )

    return pd.DataFrame(results)


def analyze_purchase_timing(transactions: pd.DataFrame) -> dict:
    """Analyze time to 2nd purchase and inter-purchase times (The Calibration Diagnostics)"""

    # 1. Time to 2nd Purchase (Point of No Return)
    repeaters = transactions.groupby("customer_id").filter(lambda x: len(x) > 1)
    repeaters = repeaters.sort_values(["customer_id", "txn_date"])
    first_two = repeaters.groupby("customer_id").head(2)
    first_two["prev_date"] = first_two.groupby("customer_id")["txn_date"].shift(1)
    first_two = first_two.dropna()  # Keep only the 2nd row
    first_two["days_to_return"] = (first_two["txn_date"] - first_two["prev_date"]).dt.days

    time_to_2nd = first_two["days_to_return"].describe(percentiles=[0.5, 0.75, 0.8, 0.9, 0.95])

    # 2. One-and-Done Ratio
    total_customers = transactions["customer_id"].nunique()
    txn_counts = transactions.groupby("customer_id").size()
    one_time_buyers = (txn_counts == 1).sum()
    one_and_done_rate = one_time_buyers / total_customers

    # 3. Whale Gap (High Frequency IPT)
    high_freq = transactions.groupby("customer_id").filter(lambda x: len(x) >= 5)

    whale_ipt_mean = np.nan
    whale_ipt_median = np.nan
    if len(high_freq) > 0:
        high_freq = high_freq.sort_values(["customer_id", "txn_date"])
        high_freq["prev_date"] = high_freq.groupby("customer_id")["txn_date"].shift(1)
        high_freq = high_freq.dropna()
        high_freq["days_between"] = (high_freq["txn_date"] - high_freq["prev_date"]).dt.days
        whale_ipt_mean = high_freq["days_between"].mean()
        whale_ipt_median = high_freq["days_between"].median()

    return {
        "time_to_2nd_mean": time_to_2nd["mean"],
        "time_to_2nd_median": time_to_2nd["50%"],
        "time_to_2nd_p75": time_to_2nd["75%"],
        "time_to_2nd_p80": time_to_2nd["80%"],
        "time_to_2nd_p90": time_to_2nd["90%"],
        "time_to_2nd_p95": time_to_2nd["95%"],
        "one_and_done_rate": one_and_done_rate,
        "one_time_buyers": one_time_buyers,
        "total_customers": total_customers,
        "whale_ipt_mean": whale_ipt_mean,
        "whale_ipt_median": whale_ipt_median,
        "n_whales": high_freq["customer_id"].nunique() if len(high_freq) > 0 else 0,
    }


def analyze_bucket_performance(test_features: pd.DataFrame) -> pd.DataFrame:
    """Analyze actual performance within each customer status bucket (The Acid Test)"""
    buckets = []

    for status in ["lost", "lapsing", "alive"]:
        bucket_data = test_features[test_features["customer_status"] == status]

        if len(bucket_data) == 0:
            continue

        y_true = bucket_data["y_true_alive"].to_numpy(dtype=np.float64)
        y_pred_proba = bucket_data["p_alive"].to_numpy(dtype=np.float64)

        buckets.append(
            {
                "bucket": status.upper(),
                "n_customers": len(bucket_data),
                "pct_of_total": len(bucket_data) / len(test_features),
                "avg_p_alive": y_pred_proba.mean(),
                "actual_active_rate": y_true.mean(),
                "baseline_rate": test_features["y_true_alive"].mean(),
                "lift": y_true.mean() / test_features["y_true_alive"].mean() if test_features["y_true_alive"].mean() > 0 else 0,
            }
        )

    return pd.DataFrame(buckets)


def create_decile_analysis(test_features: pd.DataFrame) -> pd.DataFrame:
    """Analyze actual active rates by predicted probability deciles"""
    df = test_features.copy()

    # Create deciles
    df["p_alive_decile"] = pd.qcut(df["p_alive"], q=10, labels=False, duplicates="drop") + 1

    deciles = []
    for decile in sorted(df["p_alive_decile"].unique()):
        decile_data = df[df["p_alive_decile"] == decile]
        y_true = decile_data["y_true_alive"].to_numpy(dtype=np.float64)
        y_pred_proba = decile_data["p_alive"].to_numpy(dtype=np.float64)

        deciles.append(
            {
                "decile": int(decile),
                "n_customers": len(decile_data),
                "avg_predicted": y_pred_proba.mean(),
                "actual_active_rate": y_true.mean(),
                "min_prob": y_pred_proba.min(),
                "max_prob": y_pred_proba.max(),
            }
        )

    return pd.DataFrame(deciles)


def create_segment_analysis(test_features: pd.DataFrame) -> pd.DataFrame:
    """Analyze model performance by customer segments"""
    segments = []

    # Make a copy to avoid modifying the original
    df = test_features.copy()

    # By frequency bins
    try:
        df["freq_bin"] = pd.qcut(df["frequency"], q=4, duplicates="drop")
    except ValueError:
        # If qcut fails, use cut instead
        df["freq_bin"] = pd.cut(df["frequency"], bins=4)

    for freq_bin in df["freq_bin"].unique():
        segment_data = df[df["freq_bin"] == freq_bin]
        y_true = segment_data["y_true_alive"].to_numpy(dtype=np.float64)
        y_pred_proba = segment_data["p_alive"].to_numpy(dtype=np.float64)

        if len(np.unique(y_true)) > 1:  # Need both classes for AUC
            auc = roc_auc_score(y_true, y_pred_proba)
        else:
            auc = np.nan

        segments.append(
            {
                "segment_type": "Frequency",
                "segment": str(freq_bin),
                "n_customers": len(segment_data),
                "lapse_rate": y_true.mean(),
                "avg_p_alive": y_pred_proba.mean(),
                "auc": auc,
            }
        )

    # By recency bins
    try:
        df["recency_bin"] = pd.qcut(df["days_since_last"], q=4, duplicates="drop")
    except ValueError:
        df["recency_bin"] = pd.cut(df["days_since_last"], bins=4)

    for recency_bin in df["recency_bin"].unique():
        segment_data = df[df["recency_bin"] == recency_bin]
        y_true = segment_data["y_true_alive"].to_numpy(dtype=np.float64)
        y_pred_proba = segment_data["p_alive"].to_numpy(dtype=np.float64)

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_proba)
        else:
            auc = np.nan

        segments.append(
            {
                "segment_type": "Recency",
                "segment": str(recency_bin),
                "n_customers": len(segment_data),
                "lapse_rate": y_true.mean(),
                "avg_p_alive": y_pred_proba.mean(),
                "auc": auc,
            }
        )

    return pd.DataFrame(segments)


def create_evaluation_plots(test_features: pd.DataFrame):
    """Create evaluation plots for the model"""

    y_true = test_features["y_true_alive"].to_numpy(dtype=np.float64)
    y_pred_proba = test_features["p_alive"].to_numpy(dtype=np.float64)

    # Set up the plot style
    plt.style.use("default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Lapse Propensity Model Evaluation", fontsize=16, fontweight="bold")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Calibration Curve
    create_calibration_curve(y_true, y_pred_proba, axes[0, 1])

    # 3. Precision-Recall Curve
    create_precision_recall_curve(y_true, y_pred_proba, axes[0, 2])

    # 4. P(alive) Distribution by Ground Truth
    active_probs = y_pred_proba[y_true == 1]
    inactive_probs = y_pred_proba[y_true == 0]

    axes[1, 0].hist(inactive_probs, bins=30, alpha=0.7, label="Lapsed", color="red", density=True)
    axes[1, 0].hist(active_probs, bins=30, alpha=0.7, label="Active", color="blue", density=True)
    axes[1, 0].set_xlabel("P(alive)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("P(alive) Distribution by Actual Status")
    axes[1, 0].legend(title="Holdout Period")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Customer Status Distribution
    status_counts = test_features["customer_status"].value_counts()
    axes[1, 1].pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1, 1].set_title("Customer Status Distribution")

    # 6. P(alive) vs Days Since Last Transaction
    scatter_alpha = min(0.6, 5000 / len(test_features))
    scatter = axes[1, 2].scatter(
        test_features["days_since_last"],
        test_features["p_alive"],
        c=test_features["y_true_alive"],
        alpha=scatter_alpha,
        cmap="RdYlBu_r",
        s=20,
    )
    axes[1, 2].set_xlabel("Days Since Last Transaction")
    axes[1, 2].set_ylabel("P(alive)")
    axes[1, 2].set_title("P(alive) vs Days Since Last Transaction")
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label("Active in Holdout")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = Path(__file__).parent / "model_evaluation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved evaluation plots to {plot_path}")


def evaluate_model_predictions(test_features: pd.DataFrame, transactions: pd.DataFrame = None):
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

    # Purchase timing analysis (if transactions provided)
    purchase_timing = None
    if transactions is not None:
        purchase_timing = analyze_purchase_timing(transactions)

    # Bucket performance (The Acid Test)
    bucket_df = analyze_bucket_performance(test_features)

    # Decile analysis
    decile_df = create_decile_analysis(test_features)

    # Threshold analysis
    threshold_df = analyze_thresholds(y_true, y_pred_proba)

    # Segment analysis
    segment_df = create_segment_analysis(test_features)

    # Create plots
    create_evaluation_plots(test_features)

    # Log results
    log.info("=" * 80)
    log.info("MODEL EVALUATION RESULTS")
    log.info("=" * 80)
    log.info(f"Dataset size: {len(test_features)} customers")
    log.info(f"Baseline (% active in holdout): {y_true.mean():.1%}")
    log.info("")

    # Purchase timing analysis (THE KEY CALIBRATION DIAGNOSTIC)
    if purchase_timing is not None:
        log.info("=" * 80)
        log.info("PURCHASE TIMING ANALYSIS (CALIBRATION DIAGNOSTIC)")
        log.info("=" * 80)
        log.info("")
        log.info("1. TIME TO 2ND PURCHASE (Point of No Return):")
        log.info(f"   Mean:   {purchase_timing['time_to_2nd_mean']:.1f} days")
        log.info(f"   Median: {purchase_timing['time_to_2nd_median']:.1f} days")
        log.info(f"   P75:    {purchase_timing['time_to_2nd_p75']:.1f} days")
        log.info(f"   P80:    {purchase_timing['time_to_2nd_p80']:.1f} days")
        log.info(f"   P90:    {purchase_timing['time_to_2nd_p90']:.1f} days ⭐ THE CLIFF")
        log.info(f"   P95:    {purchase_timing['time_to_2nd_p95']:.1f} days")
        log.info("")
        log.info("   💡 If P90 < ALIVE_CUTOFF_DAYS, your empirical curve is too flat!")
        log.info(f"      Current ALIVE_CUTOFF_DAYS = {ALIVE_CUTOFF_DAYS}")
        if purchase_timing['time_to_2nd_p90'] < ALIVE_CUTOFF_DAYS * 0.7:
            log.warning(f"   ⚠️  P90 ({purchase_timing['time_to_2nd_p90']:.0f} days) << ALIVE_CUTOFF ({ALIVE_CUTOFF_DAYS} days)")
            log.warning("      Consider reducing ALIVE_CUTOFF_DAYS to make curve steeper")
        log.info("")
        log.info("2. ONE-AND-DONE RATIO:")
        log.info(f"   Rate:           {purchase_timing['one_and_done_rate']:.1%}")
        log.info(f"   One-time buyers: {purchase_timing['one_time_buyers']:,} / {purchase_timing['total_customers']:,}")
        log.info("")
        log.info("   💡 High ratio means new customers should start with lower P(alive)")
        log.info("")
        log.info("3. WHALE GAP (High Frequency IPT):")
        if not np.isnan(purchase_timing['whale_ipt_mean']):
            log.info(f"   Whales (5+ txns): {purchase_timing['n_whales']:,} customers")
            log.info(f"   Mean IPT:   {purchase_timing['whale_ipt_mean']:.1f} days")
            log.info(f"   Median IPT: {purchase_timing['whale_ipt_median']:.1f} days")
            log.info("")
            ratio = purchase_timing['time_to_2nd_median'] / purchase_timing['whale_ipt_median']
            log.info(f"   💡 Whales buy {ratio:.1f}x faster than average")
            if ratio > 2:
                log.info("      ✅ Hybrid model (switching based on frequency) is justified")
        else:
            log.info("   Not enough high-frequency customers for analysis")
        log.info("")

    log.info("=" * 80)
    log.info("BUCKET PERFORMANCE (THE ACID TEST)")
    log.info("=" * 80)
    log.info("Does each bucket have the expected actual active rate?")
    log.info("  LOST (<0.3):    Should be <5%")
    log.info("  LAPSING (0.3-0.6): Should be ~15-25%")
    log.info("  ALIVE (>0.6):   Should be >50%")
    log.info("")
    log.info(bucket_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log.info("")
    # Check for fence-sitter trap
    lapsing_pct = bucket_df[bucket_df["bucket"] == "LAPSING"]["pct_of_total"].values
    if len(lapsing_pct) > 0 and lapsing_pct[0] > 0.6:
        log.warning("⚠️  FENCE-SITTER TRAP: >60% of customers are in LAPSING bucket!")
        log.warning("    Model is hedging its bets. Consider recalibration.")
    log.info("=" * 80)
    log.info("DECILE ANALYSIS")
    log.info("=" * 80)
    log.info("Actual active rate by predicted probability decile:")
    log.info("")
    log.info(decile_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log.info("")
    log.info("=" * 80)
    log.info("CLASSIFICATION METRICS (at 0.5 threshold)")
    log.info("=" * 80)
    log.info(f"  AUC (Ranking Power): {auc:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall: {recall:.4f}")
    log.info(f"  Accuracy: {accuracy:.4f}")
    log.info("")
    log.info("Calibration Metrics (ignore if using buckets):")
    log.info(f"  Log Loss: {logloss:.4f}")
    log.info(f"  Brier Score: {brier:.4f}")
    log.info("")
    log.info("=" * 80)
    log.info("THRESHOLD ANALYSIS")
    log.info("=" * 80)
    log.info("Performance at different probability thresholds:")
    log.info("")
    log.info(threshold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log.info("")
    log.info("=" * 80)
    log.info("SEGMENT ANALYSIS")
    log.info("=" * 80)
    log.info("Model performance by customer segments:")
    log.info("")
    log.info(segment_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log.info("=" * 80)

    return {
        "auc": auc,
        "log_loss": logloss,
        "brier_score": brier,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "baseline_active_rate": y_true.mean(),
        "n_customers": len(test_features),
        "purchase_timing": purchase_timing,
        "bucket_performance": bucket_df,
        "decile_analysis": decile_df,
        "threshold_analysis": threshold_df,
        "segment_analysis": segment_df,
    }
