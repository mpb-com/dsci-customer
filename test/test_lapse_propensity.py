import pytest
import pandas as pd
from unittest.mock import patch
from dsci_utils import BQHelper
from src.lapse_propensity.pipeline import pipe
from src.utils import get_bq_helper


@pytest.fixture
def bq_dry_run() -> BQHelper:
    return get_bq_helper(validate=True)


@pytest.fixture
def mock_transaction_data():
    """Create mock transaction data for testing."""
    # Customer 1: Active customer (recent transactions)
    customer_1 = pd.DataFrame(
        {
            "customer_id": [1] * 5,
            "txn_date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-04-20",
                    "2023-07-10",
                    "2023-10-05",
                    "2023-12-20",
                ]
            ),
        }
    )

    # Customer 2: Lapsing customer (last transaction 200 days ago)
    customer_2 = pd.DataFrame(
        {
            "customer_id": [2] * 3,
            "txn_date": pd.to_datetime(
                [
                    "2023-01-10",
                    "2023-04-15",
                    "2023-06-10",
                ]
            ),
        }
    )

    # Customer 3: Lost customer (last transaction 400 days ago)
    customer_3 = pd.DataFrame(
        {
            "customer_id": [3] * 2,
            "txn_date": pd.to_datetime(
                [
                    "2022-10-01",
                    "2022-11-20",
                ]
            ),
        }
    )

    # Customer 4: Single transaction customer
    customer_4 = pd.DataFrame({"customer_id": [4] * 1, "txn_date": pd.to_datetime(["2023-11-01"])})

    # Customer 5: High frequency customer
    customer_5 = pd.DataFrame(
        {
            "customer_id": [5] * 10,
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


@patch.object(BQHelper, "write_to")
@patch.object(BQHelper, "get_string")
def test_lapse_propensity_pipeline_default_params(mock_get_string, mock_write_to, bq_dry_run, mock_transaction_data):
    """Test lapse propensity pipeline with default parameters."""
    # Mock the get_string call to return our fixture data
    mock_get_string.return_value = mock_transaction_data

    # Mock the write_to call to avoid actual BigQuery writes
    mock_write_to.return_value = None

    # Run the pipeline
    pipe(bq_dry_run)

    # Verify that get_string was called (fetching transactions)
    assert mock_get_string.call_count == 1

    # Verify that write_to was called (saving results)
    assert mock_write_to.call_count == 1

    # Verify the saved results have the expected structure
    call_args = mock_write_to.call_args
    saved_data = call_args[0][0]  # First positional argument
    table_id = call_args[0][1]  # Second positional argument

    # Check that all expected columns are present
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

    # Check that we have predictions for all customers
    assert len(saved_data) == mock_transaction_data["customer_id"].nunique()

    # Check that p_alive is between 0 and 1
    assert (saved_data["p_alive"] >= 0).all()
    assert (saved_data["p_alive"] <= 1).all()

    # Check that customer_status contains valid values
    valid_statuses = {"alive", "lapsing", "lost"}
    assert set(saved_data["customer_status"].unique()).issubset(valid_statuses)

    # Check that the table ID is correct
    assert "customer_ltv_analysis" in table_id
