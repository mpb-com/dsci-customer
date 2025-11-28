from pathlib import Path


ACTIVE_PROBABILITY_CUTOFF = 0.6
LAPSING_PROBABILITY_CUTOFF = 0.3
ALIVE_CUTOFF_DAYS = 270
LAPSING_CUTOFF_DAYS = 540


# Update these!
PROJECT_ID = "mpb-data-science-dev-ab-602d"
DATABASE_NAME = "sandbox"
TABLE_NAME = "customer_ltv_analysis"

# For testing and backtesting
TEST_SAMPLE_SIZE = 1000000  # Sample size for testing
TEST_HORIZON_DAYS = 540  # Test period for evaluation
MIN_TRANSACTION_COUNT = None
CALIBRATION_END_DATE = "2024-01-01"  # End of training period
DATA_DIR = Path(__file__).parent.parent / "data"

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
