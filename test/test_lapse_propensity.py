import pytest
import pandas as pd
from unittest.mock import patch
from datetime import datetime
from dsci_utils import BQHelper
from src.lapse_propensity.pipeline import pipe
from src.utils import get_bq_helper


@pytest.fixture
def bq_dry_run() -> BQHelper:
    return get_bq_helper(validate=True)


@pytest.fixture
def mock_transaction_data():
    """Create mock transaction data spanning calibration and test periods."""
    customer_1 = pd.DataFrame(
        {
            "customer_id": [1] * 8,
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
                ]
            ),
        }
    )

    customer_2 = pd.DataFrame(
        {
            "customer_id": [2] * 4,
            "txn_date": pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-05-15",
                    "2022-08-10",
                    "2022-11-20",
                ]
            ),
        }
    )

    customer_3 = pd.DataFrame({"customer_id": [3] * 1, "txn_date": pd.to_datetime(["2022-06-01"])})

    customer_4 = pd.DataFrame(
        {
            "customer_id": [4] * 6,
            "txn_date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-03-01",
                    "2022-05-01",
                    "2022-07-01",
                    "2022-09-01",
                    "2023-03-01",
                ]
            ),
        }
    )

    customer_5 = pd.DataFrame(
        {
            "customer_id": [5] * 12,
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
                ]
            ),
        }
    )

    transactions = pd.concat([customer_1, customer_2, customer_3, customer_4, customer_5], ignore_index=True)
    transactions["customer_id"] = transactions["customer_id"].astype("int32")
    transactions["txn_date"] = pd.to_datetime(transactions["txn_date"])

    return transactions


@patch("src.lapse_propensity.pipeline.datetime")
@patch.object(BQHelper, "write_to")
@patch.object(BQHelper, "get_string")
def test_lapse_propensity_pipeline_default_params(
    mock_get_string, mock_write_to, mock_datetime, bq_dry_run, mock_transaction_data
):
    """Test lapse propensity pipeline with default parameters."""
    mock_get_string.return_value = mock_transaction_data
    mock_write_to.return_value = None

    mock_now = datetime(2023, 6, 1)
    mock_datetime.now.return_value = mock_now

    # Use shorter calibration window for test data (180 days instead of 570)
    pipe(bq_dry_run, calibration_window_days=180)

    assert mock_get_string.call_count == 1
    assert mock_write_to.call_count == 1

    call_args = mock_write_to.call_args
    saved_data = call_args[0][0]
    table_id = call_args[0][1]

    expected_columns = [
        "customer_id",
        "p_alive",
        "customer_status",
        "frequency",
        "recency",
        "T",
        "days_since_last",
    ]
    assert all(col in saved_data.columns for col in expected_columns)
    assert len(saved_data) == mock_transaction_data["customer_id"].nunique()
    assert (saved_data["p_alive"] >= 0).all()
    assert (saved_data["p_alive"] <= 1).all()

    valid_statuses = {"alive", "lapsing", "lost"}
    assert set(saved_data["customer_status"].unique()).issubset(valid_statuses)
    assert "customer_ltv_analysis" in table_id
