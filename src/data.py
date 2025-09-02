import numpy as np
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from src.utils.bq import BQ
from google.cloud import bigquery
from .config import (
    TRANSACTION_QUERY,
    CUSTOMER_DATA_QUERY,
    TRANSACTIONS_DATASET,
    CUSTOMER_DATA_DATASET,
    BTYD_FEATURES_DATASET,
    ACTIVE,
    LAPSING,
    LOST,
)


def label_predictions(probs, alive_min=0.6, lapsed_max=0.3):
    labels = np.full(len(probs), LAPSING)
    labels[probs >= alive_min] = ACTIVE
    labels[probs <= lapsed_max] = LOST
    return labels


def fetch_transactions():
    """Fetch transaction data from BigQuery and save to parquet"""
    bq = BQ()

    job_config = bigquery.QueryJobConfig()
    dtypes = {"id": "int32", "txn_date": "datetime64[ns]", "value": "int32"}

    transactions = bq.to_dataframe(
        TRANSACTION_QUERY, job_config=job_config, dtypes=dtypes
    )

    print(f"Fetched {len(transactions)} transactions from {TRANSACTIONS_DATASET}")

    return transactions


def save_customer_data():
    bq = BQ()

    customer_data = bq.to_dataframe(CUSTOMER_DATA_QUERY)

    customer_data.to_parquet(CUSTOMER_DATA_DATASET, index=False)
    print(f"Saved {len(customer_data)} customer records to {CUSTOMER_DATA_DATASET}")


def save_btyd_features(
    transactions: pd.DataFrame,
):
    """Create BTYD features (frequency, recency, T) from transaction data"""
    # Create summary data for BTYD models
    summary = summary_data_from_transaction_data(
        transactions=transactions,
        customer_id_col="id",
        datetime_col="txn_date",
        monetary_value_col="value",
    )

    # Calculate days since last transaction
    max_date = transactions["txn_date"].max()
    last_transaction_date = transactions.groupby("id")["txn_date"].max()
    days_since_last = (max_date - last_transaction_date).dt.days

    # Add days_since_last to summary
    summary["days_since_last"] = days_since_last.reindex(summary.index)

    # Reset index to make customer_id a column
    summary = summary.reset_index()
    summary.rename(columns={"id": "customer_id"}, inplace=True)

    int_cols = ["frequency", "recency", "T", "monetary_value", "days_since_last"]
    for col in int_cols:
        if col in summary.columns:
            summary[col] = summary[col].astype("int32")

    summary.to_parquet(BTYD_FEATURES_DATASET, index=False)

    print(f"Saved BTYD features to {BTYD_FEATURES_DATASET}")
    print(f"No of customers: {summary.shape[0]}")
