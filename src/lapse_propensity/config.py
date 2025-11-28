from pathlib import Path


# Lift-based threshold multipliers (relative to baseline - window-agnostic)
DEAD_LIFT_MULTIPLIER = 0.5  # Dead if < 0.5x baseline (worse than average)
ALIVE_LIFT_MULTIPLIER = 2.0  # Alive if > 2.0x baseline (better than average)

# Safety Net threshold (cumulative recall approach for DEAD cutoff)
DEAD_RECALL_TARGET = 0.95  # Capture 95% of active customers (accept 5% risk of missing sales)

# Empirical model decay parameters (for calculating p_alive curves, NOT for classification)
ACTIVE_PROBABILITY_CUTOFF = 0.31  # Empirical decay curve shape parameter (NOT a classification threshold)
ALIVE_CUTOFF_DAYS = 270
LAPSING_CUTOFF_DAYS = 540


# Update these!
PROJECT_ID = "mpb-data-science-dev-ab-602d"
DATABASE_NAME = "sandbox"
TABLE_NAME = "customer_ltv_analysis"

# For testing and backtesting
TEST_SAMPLE_SIZE = 1000000  # Sample size for testing
TEST_END_DATE = "2025-10-01"  # End of final test period (we have ground truth up to here)
TEST_HORIZON_DAYS = 570  # Length of each test window (6 months - covers median IPT, keeps training fresh)
MIN_TRANSACTION_COUNT = None

# Timeline is back-calculated from TEST_END_DATE:
# Training:     [Start] → (TEST_END_DATE - 2*TEST_HORIZON_DAYS)
# Calibration:  (TEST_END_DATE - 2*TEST_HORIZON_DAYS) → (TEST_END_DATE - TEST_HORIZON_DAYS)
# Test:         (TEST_END_DATE - TEST_HORIZON_DAYS) → TEST_END_DATE

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Don't touch!
PARETO_PENALIZER = 0.001
TRANSACTION_EMPIRICAL_CUTOFF = 1
MAX_FREQUENCY_CUTOFF = 100  # Higher than this and we get numerical issues


# Queries
TRANSACTION_QUERY = """
    SELECT customer_id,
    DATETIME(transaction_completed_datetime) as txn_date
    FROM `mpb-data-science-dev-ab-602d.dsci_daw.STV`
    WHERE transaction_completed_datetime is not null
    """

FINAL_COLUMNS = [
    "customer_id",
    "p_alive",
    "customer_status",
    "frequency",
    "recency",
    "T",
    "days_since_last",
]
