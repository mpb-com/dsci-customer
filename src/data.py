import numpy as np
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from src.utils.bq import BQ
from google.cloud import bigquery
from .config import (
    CUTOFF_DAYS,
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


def save_transactions():
    """Fetch transaction data from BigQuery and save to parquet"""
    bq = BQ()

    job_config = bigquery.QueryJobConfig()
    dtypes = {"id": "int32", "txn_date": "datetime64[ns]", "value": "int32"}

    transactions = bq.to_dataframe(
        TRANSACTION_QUERY, job_config=job_config, dtypes=dtypes
    )

    print(f"Fetched {len(transactions)} transactions from {TRANSACTIONS_DATASET}")

    transactions.to_parquet(TRANSACTIONS_DATASET, index=False)


def save_customer_data():
    bq = BQ()

    customer_data = bq.to_dataframe(CUSTOMER_DATA_QUERY)

    customer_data.to_parquet(CUSTOMER_DATA_DATASET, index=False)
    print(f"Saved {len(customer_data)} customer records to {CUSTOMER_DATA_DATASET}")


def save_btyd_features_with_survival(cutoff_days: int = CUTOFF_DAYS):
    """Create BTYD and survival features (frequency, recency, T, monetary_value, days_since_last, event_observed)"""

    transactions = pd.read_parquet(TRANSACTIONS_DATASET)

    summary = summary_data_from_transaction_data(
        transactions=transactions,
        customer_id_col="id",
        datetime_col="txn_date",
        monetary_value_col="value",
    )

    max_date = transactions["txn_date"].max()
    last_transaction_date = transactions.groupby("id")["txn_date"].max()
    first_transaction_date = transactions.groupby("id")["txn_date"].min()

    days_since_last = (max_date - last_transaction_date).dt.days

    # Map back to summary
    summary["days_since_last"] = days_since_last.reindex(summary.index)
    summary = summary.reset_index()
    summary.rename(columns={"id": "customer_id"}, inplace=True)

    # ✅ Correct survival label: churned = long inactivity
    summary["event_observed"] = summary["days_since_last"] > cutoff_days

    # Optional: derive cohort year
    cohort_year = first_transaction_date.dt.year
    summary["cohort_year"] = summary["customer_id"].map(cohort_year)

    # Include duration explicitly
    summary["duration"] = summary["T"]

    # Cast
    int_cols = [
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "days_since_last",
        "duration",
    ]
    for col in int_cols:
        if col in summary.columns:
            summary[col] = summary[col].astype("int32")

    summary["event_observed"] = summary["event_observed"].astype("bool")

    summary.to_parquet(BTYD_FEATURES_DATASET, index=False)

    print(f"Saved BTYD + survival features to {BTYD_FEATURES_DATASET}")
    print(f"No of customers: {summary.shape[0]}")
