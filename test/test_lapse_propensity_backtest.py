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
    customer_1 = pd.DataFrame(
        {
            "customer_id": [1] * 10,
            "txn_date": pd.to_datetime(
                [
                    "2022-01-15",
                    "2022-03-20",
                    "2022-05-10",
                    "2022-07-05",
                    "2022-09-20",
                    "2022-11-15",
                    "2023-02-10",
                    "2023-04-20",
                    "2024-06-15",
                    "2024-09-10",
                ]
            ),
        }
    )

    customer_2 = pd.DataFrame(
        {
            "customer_id": [2] * 6,
            "txn_date": pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-05-15",
                    "2022-08-10",
                    "2022-11-20",
                    "2023-03-15",
                    "2024-07-20",
                ]
            ),
        }
    )

    customer_3 = pd.DataFrame(
        {
            "customer_id": [3] * 1,
            "txn_date": pd.to_datetime(["2022-06-01"]),
        }
    )

    customer_4 = pd.DataFrame(
        {
            "customer_id": [4] * 8,
            "txn_date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-03-01",
                    "2022-05-01",
                    "2022-07-01",
                    "2022-09-01",
                    "2023-03-01",
                    "2024-05-01",
                    "2024-08-15",
                ]
            ),
        }
    )

    customer_5 = pd.DataFrame(
        {
            "customer_id": [5] * 15,
            "txn_date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-02-01",
                    "2022-03-01",
                    "2022-04-01",
                    "2022-05-01",
                    "2022-06-01",
                    "2022-07-01",
                    "2022-08-01",
                    "2022-09-01",
                    "2022-10-01",
                    "2022-11-01",
                    "2022-12-01",
                    "2023-06-01",
                    "2024-04-01",
                    "2024-09-15",
                ]
            ),
        }
    )

    transactions = pd.concat([customer_1, customer_2, customer_3, customer_4, customer_5], ignore_index=True)
    transactions["customer_id"] = transactions["customer_id"].astype("int32")
    transactions["txn_date"] = pd.to_datetime(transactions["txn_date"])

    return transactions


@patch("src.lapse_propensity.backtest.TEST_HORIZON_DAYS", 180)
@patch("src.lapse_propensity.backtest.TEST_END_DATE", "2024-10-01")
@patch("src.lapse_propensity.backtest.Path")
@patch.object(BQHelper, "get_string")
def test_backtest_pipeline_default_params(mock_get_string, mock_path, bq_dry_run, mock_transaction_data):
    """Test backtest pipeline with default parameters."""
    mock_get_string.return_value = mock_transaction_data

    mock_parquet_path = MagicMock()
    mock_txt_path = MagicMock()
    mock_path.return_value.__truediv__.side_effect = [mock_parquet_path, mock_txt_path]

    with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
        with patch("builtins.open", MagicMock()):
            results = backtest_pipeline(bq_dry_run)

    assert mock_get_string.call_count == 1

    assert "metrics" in results
    assert "test_results" in results
    assert "calibrator_model" in results
    assert "prod_model" in results

    metrics = results["metrics"]
    expected_metric_keys = [
        "auc",
        "log_loss",
        "brier_score",
        "alive_precision",
        "alive_recall",
        "lost_precision",
        "lost_recall",
        "baseline_active_rate",
        "n_customers",
        "dead_threshold",
        "alive_threshold",
    ]
    assert all(key in metrics for key in expected_metric_keys)

    test_results = results["test_results"]
    assert len(test_results) == mock_transaction_data["customer_id"].nunique()

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

    assert (test_results["p_alive"] >= 0).all()
    assert (test_results["p_alive"] <= 1).all()

    valid_statuses = {"alive", "lapsing", "lost"}
    assert set(test_results["customer_status"].unique()).issubset(valid_statuses)

    assert set(test_results["y_true_alive"].unique()).issubset({0, 1})

    assert mock_to_parquet.call_count == 1
