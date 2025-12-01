from .config import MAX_FREQUENCY_CUTOFF, TRANSACTION_QUERY
from lifetimes.utils import summary_data_from_transaction_data
from loguru import logger as log
import pandas as pd


def fetch_transactions(bq, query=TRANSACTION_QUERY):
    """Fetch transaction data from BigQuery"""
    transactions = bq.get_string(query)
    n_customers = transactions['customer_id'].nunique()
    n_transactions = len(transactions)
    log.info(f"Fetched {n_transactions:,} transactions for {n_customers:,} customers")

    return transactions


def create_btyd_features(transactions: pd.DataFrame):
    """
    Create BTYD features: frequency, recency, T, days_since_last
    """
    _transactions = transactions.copy()

    summary = summary_data_from_transaction_data(
        transactions=_transactions,
        customer_id_col="customer_id",
        datetime_col="txn_date",
    )

    max_date = _transactions["txn_date"].max()
    last_transaction_date = _transactions.groupby("customer_id")["txn_date"].max()
    days_since_last = (max_date - last_transaction_date).dt.days
    summary["days_since_last"] = days_since_last.reindex(summary.index)

    summary = summary.reset_index()

    # Cap frequency to prevent numerical issues
    high_freq_count = (summary["frequency"] > MAX_FREQUENCY_CUTOFF).sum()
    if high_freq_count > 0:
        summary.loc[summary["frequency"] > MAX_FREQUENCY_CUTOFF, "frequency"] = MAX_FREQUENCY_CUTOFF
        log.info(f"Capped {high_freq_count} customers with frequency > {MAX_FREQUENCY_CUTOFF}")

    # Cast to int32 to save memory
    int_cols = [
        "frequency",
        "recency",
        "T",
        "days_since_last",
    ]
    for col in int_cols:
        if col in summary.columns:
            summary[col] = summary[col].astype("int32")

    log.info(f"Created BTYD features with {summary.shape[0]} customers")
    log.info(f"Columns: {summary.columns.tolist()}")
    log.info(f"Sample data:\n{summary.head()}")
    log.info(f"Data types:\n{summary.dtypes}")
    log.info(summary.describe().round(1))

    return summary
