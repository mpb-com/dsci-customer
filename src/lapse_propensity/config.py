from pathlib import Path


# Business constraint thresholds
MAX_REVENUE_RISK = 0.05  # Maximum % of active customers excluded from DEAD bucket (5% revenue risk)
MIN_ALIVE_LIFT = 2.0  # Probability threshold for ALIVE bucket (2× baseline)

# Safety Net threshold (alternative approach for comparison - not used for classification)
DEAD_RECALL_TARGET = 0.90  # Capture 90% of active customers (for comparison/logging only)

# Empirical model decay parameters (for calculating p_alive curves, NOT for classification)
ACTIVE_PROBABILITY_CUTOFF = 0.31  # Empirical decay curve shape parameter (NOT a classification threshold)
ALIVE_CUTOFF_DAYS = 270
LAPSING_CUTOFF_DAYS = 540


# Update these!
PROJECT_ID = "mpb-data-science-dev-ab-602d"
DATABASE_NAME = "sandbox"
TABLE_NAME = "customer_ltv_analysis"

# For testing and backtesting
TEST_SAMPLE_SIZE = 10000000  # Sample size for testing
TEST_END_DATE = "2025-10-01"  # End of final test period (we have ground truth up to here)
TEST_HORIZON_DAYS = 570  # Length of each test window (6 months - covers median IPT, keeps training fresh)
MIN_TRANSACTION_COUNT = None

# Timeline is back-calculated from TEST_END_DATE:
# Training:     [Start] → (TEST_END_DATE - 2*TEST_HORIZON_DAYS)
# Calibration:  (TEST_END_DATE - 2*TEST_HORIZON_DAYS) → (TEST_END_DATE - TEST_HORIZON_DAYS)
# Test:         (TEST_END_DATE - TEST_HORIZON_DAYS) → TEST_END_DATE

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Model and feature engineering configuration
# ==============================================
# To test a new model or feature set:
# 1. Create your class in model.py (inherit from BaseLapseModel)
# 2. Create your feature engineer in features.py (inherit from BaseFeatureEngineer)
# 3. Update the class names below
# 4. Run backtests or production pipeline as normal
#
# Available models: See model.py for available classes
# Available feature engineers: See features.py for available classes
MODEL_CLASS_NAME = "XGBoostLapseModel"  # "ParetoEmpiricalSingleTrainSplit"  # Name of model class to use
FEATURE_ENGINEER_CLASS_NAME ="XGBoostFeatureEngineer" #  "BTYDFeatureEngineer"  # Name of feature engineer class to use

# Don't touch!
PARETO_PENALIZER = 0.001
TRANSACTION_EMPIRICAL_CUTOFF = 1
MAX_FREQUENCY_CUTOFF = 100  # Higher than this and we get numerical issues


# Queries
TRANSACTION_QUERY = """
    SELECT
        customer_id,
        DATETIME(transaction_completed_datetime) as txn_date,
        transaction_type,
        sold_sell_items,
        total_buy_items,
        delivery_charge_market,
        buy_value_market,
        total_sold_sell_value,
        market,
        ga_sub_channel
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
