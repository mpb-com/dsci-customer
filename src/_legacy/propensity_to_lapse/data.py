import numpy as np
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
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


def save_transactions(bq):
    """Fetch transaction data from BigQuery and save to parquet"""

    job_config = bigquery.QueryJobConfig()
    dtypes = {"id": "int32", "txn_date": "datetime64[ns]", "value": "int32"}

    transactions = bq.to_dataframe(
        TRANSACTION_QUERY, job_config=job_config, dtypes=dtypes
    )

    print(f"Fetched {len(transactions)} transactions from {TRANSACTIONS_DATASET}")

    transactions.to_parquet(TRANSACTIONS_DATASET, index=False)


def save_customer_data(bq):
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


def create_train_test_split(test_period_days=90, random_state=42, sample_size=10000):
    """Create proper temporal train/test split"""
    np.random.seed(random_state)

    # Dates
    max_complete_date = pd.Timestamp("2025-06-30")
    cutoff_date = max_complete_date - pd.Timedelta(days=test_period_days)

    # Load transactions
    transactions = pd.read_parquet(TRANSACTIONS_DATASET)
    transactions["txn_date"] = pd.to_datetime(transactions["txn_date"])

    # Get ALL customers with sufficient pre-cutoff history (no future bias)
    pre_cutoff_txns = transactions[transactions["txn_date"] <= cutoff_date]
    customer_counts = pre_cutoff_txns.groupby("id").size()
    eligible_customers = customer_counts[
        customer_counts >= 1
    ].index.tolist()  # Include all customers with at least 1 transaction

    sample_customers = np.random.choice(
        eligible_customers,
        size=min(sample_size, len(eligible_customers)),
        replace=False,
    )
    sample_txns = pre_cutoff_txns[pre_cutoff_txns["id"].isin(sample_customers)]

    # Calculate BTYD features (pre-cutoff only)
    btyd_summary = summary_data_from_transaction_data(
        sample_txns, "id", "txn_date", "value", observation_period_end=cutoff_date
    )

    # Add required columns
    btyd_summary["customer_id"] = btyd_summary.index
    btyd_summary.reset_index(drop=True, inplace=True)

    # Add survival analysis columns
    last_txn_dates = sample_txns.groupby("id")["txn_date"].max()
    btyd_summary["days_since_last"] = (
        (cutoff_date - last_txn_dates.reindex(btyd_summary["customer_id"]))
        .dt.days.fillna(0)
        .values
    )
    btyd_summary["event_observed"] = True
    first_txn_dates = sample_txns.groupby("id")["txn_date"].min()
    btyd_summary["cohort_year"] = first_txn_dates.reindex(
        btyd_summary["customer_id"]
    ).dt.year.fillna(2020)
    btyd_summary["duration"] = btyd_summary["T"]

    # Cast to proper types (fill NaN first)
    int_cols = [
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "days_since_last",
        "duration",
        "cohort_year",
    ]
    for col in int_cols:
        if col in btyd_summary.columns:
            btyd_summary[col] = btyd_summary[col].fillna(0).astype("int32")
    btyd_summary["event_observed"] = btyd_summary["event_observed"].astype("bool")

    # Calculate holdout actuals for test set only
    holdout_txns = transactions[
        (transactions["id"].isin(sample_customers))
        & (transactions["txn_date"] > cutoff_date)
        & (transactions["txn_date"] <= max_complete_date)
    ]

    actual_counts = (
        holdout_txns.groupby("id").size().reindex(sample_customers, fill_value=0)
    )

    # Create train set (no ground truth labels)
    train_df = btyd_summary.copy()
    train_df["test_period_days"] = (max_complete_date - cutoff_date).days

    # Create test set (with ground truth for evaluation)
    test_df = btyd_summary.copy()
    # CRITICAL FIX: Match holdout counts to customer_id order in test_df
    test_df["y_true_txns"] = (
        test_df["customer_id"].map(actual_counts).fillna(0).astype(int)
    )
    test_df["y_true_alive"] = (test_df["y_true_txns"] > 0).astype(int)
    test_df["test_period_days"] = (max_complete_date - cutoff_date).days

    test_df = test_df[test_df["y_true_txns"] < 15]

    return {"train_df": train_df, "test_df": test_df}
