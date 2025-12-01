import pandas as pd
from loguru import logger as log
from .config import (
    DATA_DIR,
    ALIVE_CUTOFF_DAYS,
    LAPSING_CUTOFF_DAYS,
    PARETO_PENALIZER,
    TRANSACTION_EMPIRICAL_CUTOFF,
    MAX_FREQUENCY_CUTOFF,
    DEAD_LIFT_MULTIPLIER,
    ALIVE_LIFT_MULTIPLIER,
    DEAD_RECALL_TARGET,
    MAX_REVENUE_RISK,
    MIN_ALIVE_LIFT,
    TEST_HORIZON_DAYS,
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
    log.info(f"  DEAD_LIFT_MULTIPLIER: {DEAD_LIFT_MULTIPLIER}x baseline (threshold for LOST)")
    log.info(f"  ALIVE_LIFT_MULTIPLIER: {ALIVE_LIFT_MULTIPLIER}x baseline (threshold for ALIVE)")
    log.info(f"  ALIVE_CUTOFF_DAYS: {ALIVE_CUTOFF_DAYS} (empirical decay parameter)")
    log.info(f"  LAPSING_CUTOFF_DAYS: {LAPSING_CUTOFF_DAYS} (empirical decay parameter)")
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


def calculate_optimal_thresholds(
    test_features: pd.DataFrame,
    baseline_rate: float,
    value_saved: float = 50.0,
    cost_intervention: float = 2.0,
    intervention_effectiveness: float = 0.20,
) -> dict:
    """Calculate optimal thresholds using three methods"""

    # METHOD 1: Baseline Crossover (for ALIVE cutoff)
    # Find where actual active rate crosses baseline
    decile_df = create_decile_analysis(test_features)
    # Find the lowest prob where actual_active_rate >= baseline
    above_baseline = decile_df[decile_df["actual_active_rate"] >= baseline_rate]
    if len(above_baseline) > 0:
        alive_cutoff_baseline = above_baseline["min_prob"].min()
    else:
        alive_cutoff_baseline = 0.5  # fallback

    # METHOD 2: Recall Constraint (for DEAD cutoff - capture 90% of churners)
    df_sorted = test_features.copy()
    df_sorted = df_sorted.sort_values("p_alive", ascending=True)
    total_lapsed = (df_sorted["y_true_alive"] == 0).sum()
    df_sorted["cum_lapsed"] = (df_sorted["y_true_alive"] == 0).cumsum()
    df_sorted["lapsed_recall"] = df_sorted["cum_lapsed"] / total_lapsed if total_lapsed > 0 else 0

    # Find cutoff where we capture 90% of lapsed customers
    recall_90 = df_sorted[df_sorted["lapsed_recall"] >= 0.90]
    if len(recall_90) > 0:
        dead_cutoff_recall = recall_90["p_alive"].iloc[0]
    else:
        dead_cutoff_recall = 0.3  # fallback

    # METHOD 3: Economic Utility (profit maximizing)
    def calculate_profit(cutoff: float) -> float:
        targeted = test_features[test_features["p_alive"] < cutoff]
        if len(targeted) == 0:
            return 0

        # True positives: customers who lapsed and we targeted
        saved_customers = ((targeted["y_true_alive"] == 0).sum()) * intervention_effectiveness
        revenue_gain = saved_customers * value_saved

        # Cost: we pay for everyone we target
        total_cost = len(targeted) * cost_intervention

        return revenue_gain - total_cost

    # Sweep thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    profits = [calculate_profit(t) for t in thresholds]
    max_profit_idx = np.argmax(profits)
    lapsing_cutoff_economic = thresholds[max_profit_idx]
    max_profit = profits[max_profit_idx]

    return {
        "alive_cutoff_baseline": alive_cutoff_baseline,
        "dead_cutoff_recall_90": dead_cutoff_recall,
        "lapsing_cutoff_economic": lapsing_cutoff_economic,
        "max_profit": max_profit,
        "value_saved": value_saved,
        "cost_intervention": cost_intervention,
        "intervention_effectiveness": intervention_effectiveness,
    }


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


def calculate_business_thresholds(
    df: pd.DataFrame, max_revenue_risk: float = 0.05, min_alive_lift: float = 3.0
) -> dict:
    """
    Calculate probability thresholds based on business constraints (revenue risk and lift).

    This function determines cutoffs based on business objectives rather than arbitrary
    probabilities, making it easy to communicate with stakeholders.

    Args:
        df: DataFrame with 'p_alive' and 'y_true_alive' columns
        max_revenue_risk: Maximum % of active customers we're willing to exclude (e.g., 0.05 = 5%)
        min_alive_lift: Minimum lift required for ALIVE bucket (e.g., 3.0 = 3× baseline)

    Returns:
        dict with dead_threshold, alive_threshold, and bucket statistics
    """
    baseline_rate = df["y_true_alive"].mean()
    total_active = df["y_true_alive"].sum()
    total_customers = len(df)

    # DEAD THRESHOLD: Find score where we exclude exactly max_revenue_risk of active customers
    # Sort LOW to HIGH (worst customers first)
    df_sorted_low = df.sort_values("p_alive", ascending=True).reset_index(drop=True)
    df_sorted_low["cum_active"] = df_sorted_low["y_true_alive"].cumsum()
    df_sorted_low["pct_active_excluded"] = df_sorted_low["cum_active"] / total_active

    # Find where we hit the revenue risk limit
    dead_candidates = df_sorted_low[df_sorted_low["pct_active_excluded"] <= max_revenue_risk]
    if len(dead_candidates) > 0:
        dead_idx = dead_candidates.index[-1]  # Last customer before exceeding risk
        dead_threshold = df_sorted_low.loc[dead_idx, "p_alive"]
        actual_revenue_risk = df_sorted_low.loc[dead_idx, "pct_active_excluded"]
        dead_bucket_size = dead_idx + 1
    else:
        # Edge case: can't even exclude one customer without exceeding risk
        dead_threshold = df["p_alive"].min()
        actual_revenue_risk = 0.0
        dead_bucket_size = 0

    # ALIVE THRESHOLD: Find score where lift >= min_alive_lift
    # Try different thresholds and calculate lift for each
    candidate_thresholds = df["p_alive"].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]).values
    best_threshold = None
    best_lift = 0

    for threshold in candidate_thresholds:
        alive_segment = df[df["p_alive"] > threshold]
        if len(alive_segment) > 0:
            segment_active_rate = alive_segment["y_true_alive"].mean()
            lift = segment_active_rate / baseline_rate if baseline_rate > 0 else 0
            if lift >= min_alive_lift and (best_threshold is None or threshold < best_threshold):
                best_threshold = threshold
                best_lift = lift

    # Fallback: if no threshold achieves target lift, use highest lift available
    if best_threshold is None:
        for threshold in sorted(candidate_thresholds, reverse=True):
            alive_segment = df[df["p_alive"] > threshold]
            if len(alive_segment) > 10:  # Need at least 10 customers
                segment_active_rate = alive_segment["y_true_alive"].mean()
                lift = segment_active_rate / baseline_rate if baseline_rate > 0 else 0
                if lift > best_lift:
                    best_threshold = threshold
                    best_lift = lift

    alive_threshold = best_threshold if best_threshold is not None else df["p_alive"].quantile(0.9)

    # Calculate bucket statistics
    lost_bucket = df[df["p_alive"] < dead_threshold]
    alive_bucket = df[df["p_alive"] > alive_threshold]
    lapsing_bucket = df[(df["p_alive"] >= dead_threshold) & (df["p_alive"] <= alive_threshold)]

    return {
        "dead_threshold": dead_threshold,
        "alive_threshold": alive_threshold,
        "max_revenue_risk": max_revenue_risk,
        "actual_revenue_risk": actual_revenue_risk,
        "min_alive_lift": min_alive_lift,
        "actual_alive_lift": best_lift,
        "baseline_rate": baseline_rate,
        # Bucket sizes
        "lost_size": len(lost_bucket),
        "lost_pct": len(lost_bucket) / total_customers,
        "alive_size": len(alive_bucket),
        "alive_pct": len(alive_bucket) / total_customers,
        "lapsing_size": len(lapsing_bucket),
        "lapsing_pct": len(lapsing_bucket) / total_customers,
        # Revenue distribution
        "lost_active_count": lost_bucket["y_true_alive"].sum(),
        "lost_active_pct": lost_bucket["y_true_alive"].sum() / total_active if total_active > 0 else 0,
        "alive_active_count": alive_bucket["y_true_alive"].sum(),
        "alive_active_pct": alive_bucket["y_true_alive"].sum() / total_active if total_active > 0 else 0,
        "lapsing_active_count": lapsing_bucket["y_true_alive"].sum(),
        "lapsing_active_pct": lapsing_bucket["y_true_alive"].sum() / total_active if total_active > 0 else 0,
        # Quality metrics
        "alive_precision": alive_bucket["y_true_alive"].mean() if len(alive_bucket) > 0 else 0,
        "lost_precision": 1 - lost_bucket["y_true_alive"].mean() if len(lost_bucket) > 0 else 0,
    }


def generate_model_summary(
    metrics: dict, dead_threshold: float, alive_threshold: float, horizon_days: int = 570
) -> str:
    """
    Generates a production-grade Model Card explaining logic, risk, and performance.

    This summary is designed for stakeholders (Finance, CRM, Executives) who need to
    understand what the model does and why they should trust it.

    Args:
        metrics: Dict with model performance metrics
        dead_threshold: Probability cutoff for LOST bucket (from calibration)
        alive_threshold: Probability cutoff for ALIVE bucket (from calibration)
        horizon_days: Observation window length (default: 570 days)

    Returns:
        Formatted string suitable for reports or Slack updates
    """
    # Calculate implied Lift for the summary text
    baseline = metrics.get("baseline_active_rate", 0.168)
    alive_lift = (metrics["alive_precision"] / baseline) if baseline > 0 else 0.0

    # Calculate what percentage of active customers fall into LAPSING
    alive_recall = metrics.get("alive_recall", 0)
    lost_false_negative_rate = 1 - metrics.get("lost_precision", 0.95)  # FNR = customers wrongly marked LOST
    lapsing_active_pct = 100 - (alive_recall * 100) - (lost_false_negative_rate * 100)

    # Calculate prior (baseline) performance
    import numpy as np

    p = baseline
    prior_brier = p * (1 - p)  # Var(Y) = p(1-p) - irreducible uncertainty
    prior_logloss = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # Entropy of the prior
    prior_auc = 0.5  # No ranking ability without features

    # Calculate skill improvement over prior
    brier_skill = ((prior_brier - metrics["brier_score"]) / prior_brier * 100) if prior_brier > 0 else 0
    logloss_skill = ((prior_logloss - metrics["log_loss"]) / prior_logloss * 100) if prior_logloss > 0 else 0
    auc_skill = ((metrics["auc"] - prior_auc) / (1 - prior_auc) * 100) if prior_auc < 1 else 0

    summary = f"""
================================================================================
FINAL MODEL CARD: LAPSE PROPENSITY (Hybrid Pareto/NBD + Empirical)
================================================================================

DEFINITION OF "ALIVE":
Customer makes at least ONE transaction within the next {horizon_days} days ({horizon_days / 30:.0f} months).

This is not "alive forever" - it's "alive within the observation window."
Ground truth: y_true_alive = 1 if customer transacted in {horizon_days}-day holdout period.

================================================================================

1. CALIBRATION & RELIABILITY (Why you can trust the score)

   A. BRIER SCORE (Probability Accuracy)
      - Model:       {metrics["brier_score"]:.4f}
      - Prior:       {prior_brier:.4f}  [Formula: p(1-p) = {baseline:.3f} × {1 - baseline:.3f}]
      - Skill:       {brier_skill:.1f}% reduction in error vs prior
      → Measures mean squared error of predicted probabilities.
        Lower is better. 0 = perfect, prior = {prior_brier:.4f}.

   B. LOG LOSS (Penalizes Confident Mistakes)
      - Model:       {metrics["log_loss"]:.4f}
      - Prior:       {prior_logloss:.4f}  [Formula: -[p×log(p) + (1-p)×log(1-p)]]
      - Skill:       {logloss_skill:.1f}% reduction in entropy vs prior
      → Cross-entropy loss. Heavily penalizes confident wrong predictions.
        Lower is better. 0 = perfect calibration.

   C. AUC (Ranking Power)
      - Model:       {metrics["auc"]:.4f}
      - Prior:       {prior_auc:.4f}  [No features = no ranking ability]
      - Skill:       {auc_skill:.1f}% of maximum achievable gain
      → Probability that a random active customer ranks higher than a random
        inactive customer. 0.5 = no discrimination, 1.0 = perfect separation.

   VERDICT: The model is CALIBRATED. If it predicts a 20% chance of activity,
            historically exactly 20% of such customers transacted within the next
            {horizon_days} days ({horizon_days / 30:.0f} months).

2. BUSINESS THRESHOLDS (Dynamic Constraints)
   The model segments customers based on Risk Tolerance and Value Lift.
   Time Horizon: {horizon_days} days ({horizon_days / 30:.0f} months)

   A. THE "LOST" BUCKET (< {dead_threshold:.4f}) → COST SAVINGS
      - Definition: Customers unlikely to transact in next {horizon_days} days.
      - Safety Constraint: Max 5% Revenue Risk.
      - Reality Check: We have successfully identified a group where
        {metrics["lost_precision"]:.1%} did NOT transact in the {horizon_days}-day window.
        We accept that ~{lost_false_negative_rate:.1%} of ACTUALLY ACTIVE customers
        will incorrectly fall into this bucket (False Negatives), but the cost
        savings on the other {metrics["lost_precision"]:.1%} outweigh this loss.
      - Performance: LOST Precision = {metrics["lost_precision"]:.1%}.

   B. THE "ALIVE" BUCKET (> {alive_threshold:.4f}) → VIP TREATMENT
      - Definition: Customers highly likely to transact in next {horizon_days} days.
      - Value Constraint: Minimum {alive_lift:.1f}× Lift vs Average.
      - Reality Check: These customers are {alive_lift:.1f}× more likely to buy
        within {horizon_days} days than the average customer in our database.
      - Performance: ALIVE Precision = {metrics["alive_precision"]:.1%}
        (vs Baseline {baseline:.1%}).

3. OPERATIONAL IMPACT
   - Total Active Customers Captured as VIPs: {metrics["alive_recall"]:.1%}
   - Total Dead Customers Removed from Cost:  {metrics["lost_recall"]:.1%}
   - The Remaining Active Customers ({lapsing_active_pct:.1f}%) are in the
     LAPSING bucket and should be targeted with retention campaigns.

4. MODEL PERFORMANCE SUMMARY
   - AUC (Ranking Power): {metrics["auc"]:.4f}
     → The model can distinguish active from inactive customers
   - Dataset Size: {metrics["n_customers"]:,} customers
   - Baseline Active Rate: {baseline:.1%}

5. RECOMMENDATIONS BY STAKEHOLDER
   - CFO: "Stop marketing to {metrics.get("business_thresholds", {}).get("lost_pct", 0.33):.0%} of database
     (LOST bucket). {metrics["lost_precision"]:.0%} of them are never coming back."
   - CMO: "Focus retention budget on LAPSING bucket
     ({metrics.get("business_thresholds", {}).get("lapsing_pct", 0.55):.0%} of customers).
     This is where spend generates ROI."
   - CRM: "Protect ALIVE bucket
     ({metrics.get("business_thresholds", {}).get("alive_pct", 0.16):.0%} of customers).
     They're {alive_lift:.1f}× more valuable. Don't spam them."

================================================================================
PRODUCTION READY: Thresholds locked from calibration set.
DEAD < {dead_threshold:.4f} | ALIVE > {alive_threshold:.4f}
================================================================================
"""
    return summary


def log_business_threshold_report(thresholds: dict):
    """Generate shareholder-friendly report based on business thresholds"""
    log.info("=" * 80)
    log.info("EXECUTIVE SUMMARY: REVENUE RISK & OPPORTUNITY")
    log.info("=" * 80)
    log.info(
        f"Based on constraints: Max Revenue Risk = {thresholds['max_revenue_risk']:.1%} | "
        f"Min VIP Lift = {thresholds['min_alive_lift']:.1f}×"
    )
    log.info("")

    # THE "LOST" BUCKET
    log.info("-" * 80)
    log.info('THE "LOST" BUCKET (Cost Saving)')
    log.info("-" * 80)
    log.info(f"Definition:     Customers with score < {thresholds['dead_threshold']:.3f}")
    log.info(f"Size:           {thresholds['lost_size']:,} customers ({thresholds['lost_pct']:.1%} of database)")
    log.info(f"Risk:           Contains only {thresholds['lost_active_pct']:.1%} of all active buyers")
    log.info(f"Precision:      {thresholds['lost_precision']:.1%} are truly dead")
    log.info("")
    log.info(
        f'Recommendation: "Safe to stop marketing. We save {thresholds["lost_pct"]:.0%} of budget '
        f'while risking only {thresholds["lost_active_pct"]:.0%} of revenue."'
    )
    log.info("")

    # THE "ALIVE" BUCKET
    log.info("-" * 80)
    log.info('THE "ALIVE" BUCKET (VIPs)')
    log.info("-" * 80)
    log.info(f"Definition:     Customers with score > {thresholds['alive_threshold']:.3f}")
    log.info(f"Size:           {thresholds['alive_size']:,} customers ({thresholds['alive_pct']:.1%} of database)")
    log.info(
        f"Quality:        These customers are {thresholds['actual_alive_lift']:.1f}× more likely to buy than average"
    )
    log.info(
        f"Capture:        Contains {thresholds['alive_active_pct']:.1%} of all active buyers "
        f"({thresholds['alive_active_count']:,} customers)"
    )
    log.info(f"Precision:      {thresholds['alive_precision']:.1%} stay active")
    log.info("")
    log.info('Recommendation: "Protect these High-Intent users. Do not spam."')
    log.info("")

    # THE "LAPSING" BUCKET
    log.info("-" * 80)
    log.info('THE "LAPSING" BUCKET (The Battleground)')
    log.info("-" * 80)
    log.info(
        f"Definition:     Score between {thresholds['dead_threshold']:.3f} and {thresholds['alive_threshold']:.3f}"
    )
    log.info(f"Size:           {thresholds['lapsing_size']:,} customers ({thresholds['lapsing_pct']:.1%} of database)")
    log.info(
        f"Opportunity:    Contains {thresholds['lapsing_active_pct']:.1%} of all active buyers "
        f"({thresholds['lapsing_active_count']:,} customers)"
    )
    log.info("")
    log.info('Recommendation: "This is where marketing spend generates ROI. Target aggressively."')
    log.info("")

    # Summary table
    log.info("-" * 80)
    log.info("REVENUE DISTRIBUTION SUMMARY")
    log.info("-" * 80)
    log.info(f"{'Bucket':<12} {'% of DB':<10} {'Active Buyers':<15} {'% of Revenue':<15}")
    log.info("-" * 80)
    log.info(
        f"{'LOST':<12} {thresholds['lost_pct']:>8.1%}  "
        f"{thresholds['lost_active_count']:>12,}   {thresholds['lost_active_pct']:>12.1%}"
    )
    log.info(
        f"{'LAPSING':<12} {thresholds['lapsing_pct']:>8.1%}  "
        f"{thresholds['lapsing_active_count']:>12,}   {thresholds['lapsing_active_pct']:>12.1%}"
    )
    log.info(
        f"{'ALIVE':<12} {thresholds['alive_pct']:>8.1%}  "
        f"{thresholds['alive_active_count']:>12,}   {thresholds['alive_active_pct']:>12.1%}"
    )
    log.info("=" * 80)
    log.info("")


def find_dead_threshold(df: pd.DataFrame, target_recall: float = DEAD_RECALL_TARGET) -> dict:
    """
    Find DEAD threshold using Safety Net (Cumulative Recall) approach.

    This method determines the probability cutoff below which the remaining population
    contributes negligible revenue. We accept a small risk (1 - target_recall) of
    missing active customers to maximize cost savings by excluding the bottom tail.

    Logic:
    1. Sort customers by p_alive (HIGH to LOW) - prioritize high probability customers
    2. Calculate cumulative recall as we move down the list
    3. Find the probability score where cumulative recall = target_recall (e.g., 95%)
    4. Customers below this threshold are considered DEAD

    Args:
        df: DataFrame with 'p_alive' and 'y_true_alive' columns
        target_recall: Target recall of active customers (default: 0.95 = 95%)

    Returns:
        dict with threshold, pct_excluded, and summary statistics
    """
    # Sort by p_alive (HIGH to LOW) - best customers first
    df_sorted = df.sort_values("p_alive", ascending=False).reset_index(drop=True)

    # Calculate cumulative metrics
    total_active = df_sorted["y_true_alive"].sum()
    df_sorted["cum_active"] = df_sorted["y_true_alive"].cumsum()
    df_sorted["cum_recall"] = df_sorted["cum_active"] / total_active if total_active > 0 else 0

    # Find the cutoff where cumulative recall >= target_recall
    target_rows = df_sorted[df_sorted["cum_recall"] >= target_recall]

    if len(target_rows) == 0:
        # Edge case: can't achieve target recall
        log.warning(f"Cannot achieve {target_recall:.1%} recall - using minimum p_alive")
        threshold = df_sorted["p_alive"].min()
        pct_excluded = 0.0
        n_excluded = 0
        actual_recall = 0.0
        n_active_captured = 0
    else:
        # The threshold is the p_alive of the FIRST customer where we hit target recall
        # (remember: sorted HIGH to LOW, so index increases as p_alive decreases)
        threshold_idx = target_rows.index[0]
        threshold = df_sorted.loc[threshold_idx, "p_alive"]

        # Everyone BELOW this threshold (higher index in sorted list) is excluded
        n_excluded = len(df_sorted) - (threshold_idx + 1)
        pct_excluded = n_excluded / len(df_sorted)

        # Use the actual cumulative recall at this threshold point
        actual_recall = df_sorted.loc[threshold_idx, "cum_recall"]
        n_active_captured = df_sorted.loc[threshold_idx, "cum_active"]

    # Business summary
    log.info("=" * 80)
    log.info("SAFETY NET (CUMULATIVE RECALL) THRESHOLD")
    log.info("=" * 80)
    log.info(f"Target Recall: {target_recall:.1%}")
    log.info(f"DEAD Cutoff:   < {threshold:.4f}")
    log.info(f"Outcome:       Exclude bottom {pct_excluded:.1%} ({n_excluded:,} customers)")
    log.info(f"Guarantee:     Capture {actual_recall:.1%} of active customers")
    log.info(f"Risk:          Accept {1 - actual_recall:.1%} risk of missing sales")
    log.info("=" * 80)
    log.info("")

    return {
        "threshold": threshold,
        "target_recall": target_recall,
        "actual_recall": actual_recall,
        "n_excluded": n_excluded,
        "pct_excluded": pct_excluded,
        "n_total": len(df_sorted),
        "n_active_total": total_active,
        "n_active_captured": n_active_captured,
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
                "lift": y_true.mean() / test_features["y_true_alive"].mean()
                if test_features["y_true_alive"].mean() > 0
                else 0,
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


def create_evaluation_plots(
    test_features: pd.DataFrame, dead_threshold: float, alive_threshold: float, trained_model=None
):
    """Create evaluation plots for the model with dynamic thresholds"""

    y_true = test_features["y_true_alive"].to_numpy(dtype=np.float64)
    y_pred_proba = test_features["p_alive"].to_numpy(dtype=np.float64)
    baseline = y_true.mean()

    # Set up the plot style
    plt.style.use("default")
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
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

    # Use raw counts (no density normalization) to show true imbalance
    # With 16.8% active vs 83.2% inactive, the red bars should be ~5x taller
    axes[1, 0].hist(inactive_probs, bins=50, alpha=0.7, label=f"Lapsed (n={len(inactive_probs):,})", color="red")
    axes[1, 0].hist(active_probs, bins=50, alpha=0.7, label=f"Active (n={len(active_probs):,})", color="blue")

    # Add threshold lines
    axes[1, 0].axvline(
        dead_threshold, color="darkred", linestyle="--", linewidth=2, label=f"LOST < {dead_threshold:.3f}"
    )
    axes[1, 0].axvline(
        alive_threshold, color="darkgreen", linestyle="--", linewidth=2, label=f"ALIVE > {alive_threshold:.3f}"
    )
    axes[1, 0].axvline(baseline, color="gray", linestyle=":", linewidth=2, label=f"Baseline = {baseline:.3f}")

    axes[1, 0].set_xlabel("P(alive)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("P(alive) Distribution by Actual Status (Raw Counts)")
    axes[1, 0].legend(title="Holdout Period", fontsize=8)
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

    # Add threshold lines
    axes[1, 2].axhline(
        dead_threshold, color="darkred", linestyle="--", linewidth=2, label=f"LOST < {dead_threshold:.3f}"
    )
    axes[1, 2].axhline(
        alive_threshold, color="darkgreen", linestyle="--", linewidth=2, label=f"ALIVE > {alive_threshold:.3f}"
    )
    axes[1, 2].axhline(baseline, color="gray", linestyle=":", linewidth=2, label=f"Baseline = {baseline:.3f}")

    axes[1, 2].set_xlabel("Days Since Last Transaction")
    axes[1, 2].set_ylabel("P(alive)")
    axes[1, 2].set_title("P(alive) vs Days Since Last Transaction")
    axes[1, 2].legend(fontsize=8, loc="upper right")
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label("Active in Holdout")
    axes[1, 2].grid(True, alpha=0.3)

    # Bottom row: Hypothetical customer scenarios over 5 years (using full model)
    if trained_model is None:
        # Fallback: fit a simple model if none provided (shouldn't happen in practice)
        from .model import ParetoEmpiricalSingleTrainSplit

        demo_model = ParetoEmpiricalSingleTrainSplit()
        if len(test_features) > 100:
            demo_model.fit(test_features.sample(min(1000, len(test_features))))
    else:
        demo_model = trained_model

    def simulate_customer_journey(purchase_days, max_days=1825):
        """Simulate P(alive) over time given purchase pattern"""
        days = np.arange(0, max_days, 1)
        p_alive_history = []

        for current_day in days:
            # Calculate BTYD features at this point in time
            past_purchases = [p for p in purchase_days if p <= current_day]

            if len(past_purchases) == 0:
                p_alive_history.append(0.5)  # Before first purchase
                continue

            frequency = len(past_purchases) - 1  # Repeat purchases
            recency = past_purchases[-1]  # Last purchase day
            T = current_day  # Customer age
            days_since_last = T - recency

            # Create feature row
            features = pd.DataFrame(
                {"frequency": [frequency], "recency": [recency], "T": [T], "days_since_last": [days_since_last]}
            )

            # Get prediction from model
            p_alive = demo_model.p_alive(features).iloc[0]
            p_alive_history.append(p_alive)

        return days, p_alive_history

    # Scenario 1: Loyal customer (purchases every 60 days)
    purchase_days_loyal = list(np.arange(0, 1825, 60))
    days, p_alive_loyal = simulate_customer_journey(purchase_days_loyal)

    axes[2, 0].plot(days, p_alive_loyal, linewidth=2, color="blue")
    axes[2, 0].scatter(
        purchase_days_loyal,
        [1.0] * len(purchase_days_loyal),
        marker="^",
        s=50,
        color="green",
        alpha=0.6,
        label="Purchase",
        zorder=5,
    )
    axes[2, 0].axhline(
        dead_threshold, color="darkred", linestyle="--", linewidth=1.5, label=f"LOST < {dead_threshold:.3f}"
    )
    axes[2, 0].axhline(
        alive_threshold, color="darkgreen", linestyle="--", linewidth=1.5, label=f"ALIVE > {alive_threshold:.3f}"
    )
    axes[2, 0].axhline(baseline, color="gray", linestyle=":", linewidth=1.5, label=f"Baseline = {baseline:.3f}")
    axes[2, 0].set_xlabel("Days Since First Purchase")
    axes[2, 0].set_ylabel("P(alive)")
    axes[2, 0].set_title("Scenario 1: Loyal Customer (Every 60 Days)")
    axes[2, 0].legend(fontsize=7, loc="lower left")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim([0, 1.05])

    # Scenario 2: Declining customer (frequent → infrequent → lapsed)
    purchase_days_decline = [0, 30, 70, 120, 200, 350, 600]
    days, p_alive_decline = simulate_customer_journey(purchase_days_decline)

    axes[2, 1].plot(days, p_alive_decline, linewidth=2, color="orange")
    axes[2, 1].scatter(
        purchase_days_decline,
        [1.0] * len(purchase_days_decline),
        marker="^",
        s=50,
        color="green",
        alpha=0.6,
        label="Purchase",
        zorder=5,
    )
    axes[2, 1].axhline(
        dead_threshold, color="darkred", linestyle="--", linewidth=1.5, label=f"LOST < {dead_threshold:.3f}"
    )
    axes[2, 1].axhline(
        alive_threshold, color="darkgreen", linestyle="--", linewidth=1.5, label=f"ALIVE > {alive_threshold:.3f}"
    )
    axes[2, 1].axhline(baseline, color="gray", linestyle=":", linewidth=1.5, label=f"Baseline = {baseline:.3f}")
    axes[2, 1].set_xlabel("Days Since First Purchase")
    axes[2, 1].set_ylabel("P(alive)")
    axes[2, 1].set_title("Scenario 2: Declining Customer (Lapsing)")
    axes[2, 1].legend(fontsize=7, loc="upper right")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim([0, 1.05])

    # Scenario 3: One-and-done (single purchase, never returns)
    purchase_days_once = [0]
    days, p_alive_once = simulate_customer_journey(purchase_days_once)

    axes[2, 2].plot(days, p_alive_once, linewidth=2, color="red")
    axes[2, 2].scatter(
        purchase_days_once, [1.0], marker="^", s=50, color="green", alpha=0.6, label="Purchase", zorder=5
    )
    axes[2, 2].axhline(
        dead_threshold, color="darkred", linestyle="--", linewidth=1.5, label=f"LOST < {dead_threshold:.3f}"
    )
    axes[2, 2].axhline(
        alive_threshold, color="darkgreen", linestyle="--", linewidth=1.5, label=f"ALIVE > {alive_threshold:.3f}"
    )
    axes[2, 2].axhline(baseline, color="gray", linestyle=":", linewidth=1.5, label=f"Baseline = {baseline:.3f}")
    axes[2, 2].set_xlabel("Days Since First Purchase")
    axes[2, 2].set_ylabel("P(alive)")
    axes[2, 2].set_title("Scenario 3: One-and-Done Customer")
    axes[2, 2].legend(fontsize=7, loc="upper right")
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].set_ylim([0, 1.05])

    plt.tight_layout()

    # Save plot
    plot_path = DATA_DIR / "model_evaluation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved evaluation plots to {plot_path}")


def evaluate_model_predictions(
    test_features: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    trained_model=None,
    horizon_days: int = TEST_HORIZON_DAYS,
):
    """Evaluate model predictions using standard classification metrics

    Args:
        test_features: DataFrame with predictions and ground truth
        transactions: Optional transaction history for purchase timing analysis
        trained_model: Optional trained model for scenario plots
        horizon_days: Observation window length in days (defines what "alive" means)
    """

    y_true = test_features["y_true_alive"].to_numpy(dtype=np.float64)
    y_pred_proba = test_features["p_alive"].to_numpy(dtype=np.float64)

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

    # Calculate dynamic thresholds for threshold-aware metrics
    baseline_rate = y_true.mean()
    dead_threshold = baseline_rate * DEAD_LIFT_MULTIPLIER
    alive_threshold = baseline_rate * ALIVE_LIFT_MULTIPLIER

    # Threshold-aware classification metrics (using lift-based thresholds)
    y_pred_alive = (y_pred_proba > alive_threshold).astype(int)  # Predicted as ALIVE
    y_pred_lost = (y_pred_proba < dead_threshold).astype(int)  # Predicted as LOST

    # ALIVE metrics (for retention targeting)
    alive_precision = (
        ((y_pred_alive == 1) & (y_true == 1)).sum() / (y_pred_alive == 1).sum()
        if (y_pred_alive == 1).sum() > 0
        else np.nan
    )
    alive_recall = (
        ((y_pred_alive == 1) & (y_true == 1)).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else np.nan
    )

    # LOST metrics (for intervention targeting)
    lost_precision = (
        ((y_pred_lost == 1) & (y_true == 0)).sum() / (y_pred_lost == 1).sum()
        if (y_pred_lost == 1).sum() > 0
        else np.nan
    )
    lost_recall = (
        ((y_pred_lost == 1) & (y_true == 0)).sum() / (y_true == 0).sum() if (y_true == 0).sum() > 0 else np.nan
    )

    # Purchase timing analysis (if transactions provided)
    purchase_timing = None
    if transactions is not None:
        purchase_timing = analyze_purchase_timing(transactions)

    # Safety Net threshold (Cumulative Recall approach)
    safety_net_threshold = find_dead_threshold(test_features, target_recall=DEAD_RECALL_TARGET)

    # Business constraint thresholds (Shareholder-friendly)
    business_thresholds = calculate_business_thresholds(
        test_features, max_revenue_risk=MAX_REVENUE_RISK, min_alive_lift=MIN_ALIVE_LIFT
    )

    # Bucket performance (The Acid Test)
    bucket_df = analyze_bucket_performance(test_features)

    # Decile analysis
    decile_df = create_decile_analysis(test_features)

    # Threshold analysis
    threshold_df = analyze_thresholds(y_true, y_pred_proba)

    # Segment analysis
    segment_df = create_segment_analysis(test_features)

    # Create plots (using thresholds already calculated above)
    create_evaluation_plots(test_features, dead_threshold, alive_threshold, trained_model)

    # Log results
    log.info("=" * 80)
    log.info("MODEL EVALUATION RESULTS")
    log.info("=" * 80)
    log.info(f"Observation Window: {horizon_days} days ({horizon_days / 30:.0f} months)")
    log.info(f"Definition of 'ALIVE': Customer made ≥1 transaction within {horizon_days}-day window")
    log.info(f"Dataset size: {len(test_features)} customers")
    log.info(f"Baseline (% active in holdout): {y_true.mean():.1%}")
    log.info("")

    # Business threshold report (Shareholder-friendly)
    log_business_threshold_report(business_thresholds)

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
        if purchase_timing["time_to_2nd_p90"] < ALIVE_CUTOFF_DAYS * 0.7:
            log.warning(
                f"   ⚠️  P90 ({purchase_timing['time_to_2nd_p90']:.0f} days) << ALIVE_CUTOFF ({ALIVE_CUTOFF_DAYS} days)"
            )
            log.warning("      Consider reducing ALIVE_CUTOFF_DAYS to make curve steeper")
        log.info("")
        log.info("2. ONE-AND-DONE RATIO:")
        log.info(f"   Rate:           {purchase_timing['one_and_done_rate']:.1%}")
        log.info(f"   One-time buyers: {purchase_timing['one_time_buyers']:,} / {purchase_timing['total_customers']:,}")
        log.info("")
        log.info("   💡 High ratio means new customers should start with lower P(alive)")
        log.info("")
        log.info("3. WHALE GAP (High Frequency IPT):")
        if not np.isnan(purchase_timing["whale_ipt_mean"]):
            log.info(f"   Whales (5+ txns): {purchase_timing['n_whales']:,} customers")
            log.info(f"   Mean IPT:   {purchase_timing['whale_ipt_mean']:.1f} days")
            log.info(f"   Median IPT: {purchase_timing['whale_ipt_median']:.1f} days")
            log.info("")
            ratio = purchase_timing["time_to_2nd_median"] / purchase_timing["whale_ipt_median"]
            log.info(f"   💡 Whales buy {ratio:.1f}x faster than average")
            if ratio > 2:
                log.info("      ✅ Hybrid model (switching based on frequency) is justified")
        else:
            log.info("   Not enough high-frequency customers for analysis")
        log.info("")

    log.info("=" * 80)
    log.info("DYNAMIC THRESHOLDS (Lift-Based)")
    log.info("=" * 80)
    baseline = y_true.mean()
    dead_val = baseline * DEAD_LIFT_MULTIPLIER
    alive_val = baseline * ALIVE_LIFT_MULTIPLIER
    log.info(f"Baseline active rate: {baseline:.1%}")
    log.info(f"  LOST cutoff:    < {DEAD_LIFT_MULTIPLIER}x baseline = {dead_val:.3f}")
    log.info(
        f"  LAPSING range:  {DEAD_LIFT_MULTIPLIER}x - {ALIVE_LIFT_MULTIPLIER}x baseline = {dead_val:.3f} - {alive_val:.3f}"
    )
    log.info(f"  ALIVE cutoff:   > {ALIVE_LIFT_MULTIPLIER}x baseline = {alive_val:.3f}")
    log.info("")
    log.info("=" * 80)
    log.info("THRESHOLD COMPARISON")
    log.info("=" * 80)
    sn_threshold = safety_net_threshold["threshold"]
    sn_pct_excluded = safety_net_threshold["pct_excluded"]
    log.info(f"Lift-Based DEAD:     < {dead_val:.4f} ({DEAD_LIFT_MULTIPLIER}x baseline)")
    log.info(f"Safety Net DEAD:     < {sn_threshold:.4f} ({safety_net_threshold['target_recall']:.0%} recall)")
    log.info("")
    log.info(f"Safety Net excludes: {sn_pct_excluded:.1%} of customers ({safety_net_threshold['n_excluded']:,})")
    log.info(f"Safety Net captures: {safety_net_threshold['actual_recall']:.1%} of active customers")
    if sn_threshold < dead_val:
        log.info("💡 Safety Net is MORE aggressive (lower threshold) than Lift-Based")
    elif sn_threshold > dead_val:
        log.info("💡 Safety Net is LESS aggressive (higher threshold) than Lift-Based")
    else:
        log.info("💡 Safety Net and Lift-Based thresholds are identical")
    log.info("")
    log.info("=" * 80)
    log.info("BUCKET PERFORMANCE (THE ACID TEST)")
    log.info("=" * 80)
    log.info("Does each bucket have the expected actual active rate?")
    log.info(f"  LOST (< {DEAD_LIFT_MULTIPLIER}x baseline):     Should be below baseline ({baseline:.1%})")
    log.info(
        f"  LAPSING ({DEAD_LIFT_MULTIPLIER}x - {ALIVE_LIFT_MULTIPLIER}x): Should be near baseline ({baseline:.1%})"
    )
    log.info(f"  ALIVE (> {ALIVE_LIFT_MULTIPLIER}x baseline):    Should be above baseline ({baseline:.1%})")
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
    log.info("CLASSIFICATION METRICS")
    log.info("=" * 80)
    log.info(f"  AUC (Ranking Power): {auc:.4f}")
    log.info("")
    log.info("-" * 80)
    log.info("OPERATION 1: COST CUTTING (LOST Customers)")
    log.info("-" * 80)
    log.info(f"  LOST Precision (p < {dead_threshold:.3f}): {lost_precision:.1%}")
    log.info(f"    → When we mark someone as DEAD, {lost_precision:.1%} are truly gone")
    log.info(f"    → Risk: Only {1 - lost_precision:.1%} false positives (acceptable)")
    log.info(f"  LOST Recall (p < {dead_threshold:.3f}):    {lost_recall:.1%}")
    log.info(f"    → We've identified {lost_recall:.1%} of all dead customers")
    log.info("")
    log.info("-" * 80)
    log.info("OPERATION 2: REVENUE PROTECTION (ALIVE Customers)")
    log.info("-" * 80)
    baseline = y_true.mean()
    alive_lift = alive_precision / baseline if baseline > 0 else 0
    log.info(f"  ALIVE Precision (p > {alive_threshold:.3f}): {alive_precision:.1%}")
    log.info(f"    → When we mark someone as VIP, {alive_precision:.1%} stay active")
    log.info(f"    → This is {alive_lift:.1f}× better than baseline ({baseline:.1%})")
    log.info(f"  ALIVE Recall (p > {alive_threshold:.3f}):    {alive_recall:.1%}")
    log.info(f"    → We've captured {alive_recall:.1%} of all active customers as VIPs")
    log.info("")
    log.info("Calibration Metrics:")
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

    # Generate and log final model card
    metrics_dict = {
        "auc": auc,
        "log_loss": logloss,
        "brier_score": brier,
        "alive_precision": alive_precision,
        "alive_recall": alive_recall,
        "lost_precision": lost_precision,
        "lost_recall": lost_recall,
        "baseline_active_rate": y_true.mean(),
        "n_customers": len(test_features),
        "business_thresholds": business_thresholds,
    }
    model_card = generate_model_summary(metrics_dict, dead_threshold, alive_threshold, horizon_days)
    log.info("\n" + model_card)

    return {
        "auc": auc,
        "log_loss": logloss,
        "brier_score": brier,
        "alive_precision": alive_precision,
        "alive_recall": alive_recall,
        "lost_precision": lost_precision,
        "lost_recall": lost_recall,
        "dead_threshold": dead_threshold,
        "alive_threshold": alive_threshold,
        "safety_net_threshold": safety_net_threshold,
        "business_thresholds": business_thresholds,
        "baseline_active_rate": y_true.mean(),
        "n_customers": len(test_features),
        "purchase_timing": purchase_timing,
        "bucket_performance": bucket_df,
        "decile_analysis": decile_df,
        "threshold_analysis": threshold_df,
        "segment_analysis": segment_df,
    }
