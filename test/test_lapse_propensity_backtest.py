import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from dsci_utils import BQHelper
from src.lapse_propensity.backtest import backtest_pipeline
from src.utils import get_bq_helper


@pytest.fixture
def bq_dry_run() -> BQHelper:
    return get_bq_helper(validate=True)


@pytest.fixture
def mock_transaction_data():
    """Create mock transaction data spanning calibration and holdout periods."""
    # Customer 1: Active in both periods
    customer_1 = pd.DataFrame(
        {
            "customer_id": [1] * 8,
            "txn_date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-03-20",
                    "2023-05-10",
                    "2023-07-05",
                    "2023-09-20",
                    "2023-11-15",
                    "2024-02-10",  # Holdout period
                    "2024-04-20",  # Holdout period
                ]
            ),
        }
    )

    # Customer 2: Active in calibration, lapsed in holdout
    customer_2 = pd.DataFrame(
        {
            "customer_id": [2] * 4,
            "txn_date": pd.to_datetime(
                [
                    "2023-02-10",
                    "2023-05-15",
                    "2023-08-10",
                    "2023-11-20",
                ]
            ),
        }
    )

    # Customer 3: Single transaction in calibration period
    customer_3 = pd.DataFrame(
        {
            "customer_id": [3] * 1,
            "txn_date": pd.to_datetime(["2023-06-01"]),
        }
    )

    # Customer 4: Active in calibration, active in holdout
    customer_4 = pd.DataFrame(
        {
            "customer_id": [4] * 6,
            "txn_date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-03-01",
                    "2023-05-01",
                    "2023-07-01",
                    "2023-09-01",
                    "2024-03-01",  # Holdout period
                ]
            ),
        }
    )

    # Customer 5: High frequency in calibration, inactive in holdout
    customer_5 = pd.DataFrame(
        {
            "customer_id": [5] * 12,
            "txn_date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01",
                    "2023-05-01",
                    "2023-06-01",
                    "2023-07-01",
                    "2023-08-01",
                    "2023-09-01",
                    "2023-10-01",
                    "2023-11-01",
                    "2023-12-01",
                ]
            ),
        }
    )

    # Combine all customers
    transactions = pd.concat([customer_1, customer_2, customer_3, customer_4, customer_5], ignore_index=True)

    # Ensure correct data types
    transactions["customer_id"] = transactions["customer_id"].astype("int32")
    transactions["txn_date"] = pd.to_datetime(transactions["txn_date"])

    return transactions


@patch("src.lapse_propensity.backtest.Path")
@patch.object(BQHelper, "get_string")
def test_backtest_pipeline_default_params(mock_get_string, mock_path, bq_dry_run, mock_transaction_data):
    """Test backtest pipeline with default parameters."""
    # Mock the get_string call to return our fixture data
    mock_get_string.return_value = mock_transaction_data

    # Mock Path operations to avoid file I/O
    mock_parquet_path = MagicMock()
    mock_txt_path = MagicMock()
    mock_path.return_value.__truediv__.side_effect = [mock_parquet_path, mock_txt_path]

    # Mock the to_parquet method
    with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
        # Mock file writing
        with patch("builtins.open", MagicMock()):
            # Run the backtest pipeline
            results = backtest_pipeline(bq_dry_run)

    # Verify that get_string was called (fetching transactions)
    assert mock_get_string.call_count == 1

    # Verify results structure
    assert "metrics" in results
    assert "test_results" in results
    assert "model" in results

    # Verify metrics
    metrics = results["metrics"]
    expected_metric_keys = [
        "auc",
        "log_loss",
        "brier_score",
        "accuracy",
        "precision",
        "recall",
        "baseline_active_rate",
        "n_customers",
    ]
    assert all(key in metrics for key in expected_metric_keys)

    # Verify test results
    test_results = results["test_results"]
    assert len(test_results) == mock_transaction_data["customer_id"].nunique()

    # Check that required columns are present
    required_columns = [
        "customer_id",
        "p_alive",
        "customer_status",
        "frequency",
        "recency",
        "T",
        "days_since_last",
        "y_true_alive",
        "y_true_txns",
    ]
    assert all(col in test_results.columns for col in required_columns)

    # Verify predictions are valid
    assert (test_results["p_alive"] >= 0).all()
    assert (test_results["p_alive"] <= 1).all()

    # Verify customer status is valid
    valid_statuses = {"alive", "lapsing", "lost"}
    assert set(test_results["customer_status"].unique()).issubset(valid_statuses)

    # Verify ground truth labels are binary
    assert set(test_results["y_true_alive"].unique()).issubset({0, 1})

    # Verify that to_parquet was called
    assert mock_to_parquet.call_count == 1
