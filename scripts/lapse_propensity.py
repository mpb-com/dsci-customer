"""
Customer Lapse Propensity Analysis Script

Calculates customer lapse propensity using hybrid Pareto/NBD and empirical models.
Processes BigQuery transaction data to predict customer status and survival probability.

- Trains on post-cutoff transactions, applies to all customers
- Outputs customer_id, p_alive, status, and BTYD features

Usage: python lapse_propensity.py
"""

from dataclasses import dataclass
from lifetimes import ParetoNBDFitter
import pandas as pd
import numpy as np
from google.cloud import bigquery
import dotenv
from lifetimes.utils import summary_data_from_transaction_data
import logging


# --- Config -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class CustomerStatus:
    """Customer status labels. Change as desired"""

    ACTIVE = "active"
    LAPSING = "lapsing"
    LOST = "lost"


ACTIVE_PROBABILITY_CUTOFF = 0.6
LAPSING_PROBABILITY_CUTOFF = 0.3
ALIVE_CUTOFF_DAYS = 270
LAPSING_CUTOFF_DAYS = 540

# Don't touch!
PARETO_PENALIZER = 0.001
TRANSACTION_EMPIRICAL_CUTOFF = 1
MAX_FREQUENCY_CUTOFF = 100  # Higher than this and we get numerical issues

# Update these!
PROJECT_ID = "mpb-data-science-dev-ab-602d"
DATABASE_NAME = "sandbox"
TABLE_NAME = "customer_ltv_analysis"


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


# --- Data Access ------------------------------------------------------------
class BQ:
    def __init__(self, project_id=PROJECT_ID):
        dotenv.load_dotenv()
        self.client = bigquery.Client(project=project_id)

    def to_dataframe(
        self,
        query: str,
        job_config: bigquery.QueryJobConfig = None,
        dtypes: dict = None,
    ) -> pd.DataFrame:
        """
        Execute a query and return the results as a pandas DataFrame.
        """
        df = self.client.query(query, job_config=job_config).result().to_dataframe()
        if dtypes:
            df = df.astype(dtypes)
        return df

    def to_bq(self, df: pd.DataFrame, table_id: str) -> None:
        """
        Write a pandas DataFrame to a BigQuery table.
        """
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()


# --- Models -----------------------------------------------------------------
class Empirical:
    def __init__(
        self,
        alive_cutoff_days=ALIVE_CUTOFF_DAYS,
        lapsing_cutoff_days=LAPSING_CUTOFF_DAYS,
    ):
        self.name = f"Empirical_{alive_cutoff_days}_{lapsing_cutoff_days}"
        self.model = None
        self.alive_cutoff_days = alive_cutoff_days
        self.lapsing_cutoff_days = lapsing_cutoff_days

    def fit(self) -> None:
        self.decay_rate = -np.log(ACTIVE_PROBABILITY_CUTOFF) / self.alive_cutoff_days

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        days = df["days_since_last"]
        probs = np.exp(-self.decay_rate * days)
        return pd.Series(probs, index=df.index)


class ParetoNBD:
    def __init__(self):
        self.name = "ParetoNBD"
        self.model = ParetoNBDFitter(penalizer_coef=PARETO_PENALIZER)

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df["frequency"], df["recency"], df["T"], verbose=True)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        probs = self.model.conditional_probability_alive(
            df["frequency"], df["recency"], df["T"]
        )
        return pd.Series(probs, index=df.index)


class ParetoEmpiricalSingleTrainSplit:
    def __init__(
        self,
        alive_cutoff_days=ALIVE_CUTOFF_DAYS,
        lapsing_cutoff_days=LAPSING_CUTOFF_DAYS,
        transaction_cutoff=TRANSACTION_EMPIRICAL_CUTOFF,
    ):
        ParetoNBD.__init__(self)
        self.pareto = ParetoNBD()
        self.empirical = Empirical(alive_cutoff_days, lapsing_cutoff_days)
        self.name = f"ParetoESTNS_{alive_cutoff_days}_{lapsing_cutoff_days}"
        self.transaction_cutoff = transaction_cutoff

    def fit(self, df: pd.DataFrame) -> None:
        df_train = df[df["frequency"] > 0]
        log.info(f"Training on {len(df_train)} customers with frequency > 0")
        ParetoNBD.fit(self, df_train)
        self.empirical.fit()

    def _handle_nan_fallback(
        self, pareto_probs: pd.Series, empirical_probs: pd.Series
    ) -> pd.Series:
        """Handle NaN values from Pareto model by falling back to empirical predictions"""
        nan_count = pareto_probs.isnull().sum()
        if nan_count > 0:
            log.info(
                f"Found {nan_count} NaN predictions from Pareto model, falling back to empirical"
            )
        return pareto_probs.fillna(empirical_probs)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        pareto_probs = ParetoNBD.p_alive(self, df)
        empirical_probs = self.empirical.p_alive(df)

        # Handle NaN values from Pareto model by falling back to empirical
        pareto_probs = self._handle_nan_fallback(pareto_probs, empirical_probs)

        probs = np.where(
            df["frequency"] > TRANSACTION_EMPIRICAL_CUTOFF - 1,
            pareto_probs,
            empirical_probs,
        )
        return pd.Series(probs, index=df.index)

    def customer_status(self, df: pd.DataFrame) -> pd.Series:
        probs = self.p_alive(df)
        return probs.apply(_get_customer_status)


# --- Data Processing --------------------------------------------------------
def fetch_transactions(bq: BQ, query=TRANSACTION_QUERY):
    """Fetch transaction data from BigQuery"""
    dtypes = {
        "customer_id": "int32",
        "txn_date": "datetime64[ns]",
    }

    transactions = bq.to_dataframe(query, dtypes=dtypes)
    log.info(f"Fetched {len(transactions)} transactions")

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
        summary.loc[summary["frequency"] > MAX_FREQUENCY_CUTOFF, "frequency"] = (
            MAX_FREQUENCY_CUTOFF
        )
        log.info(
            f"Capped {high_freq_count} customers with frequency > {MAX_FREQUENCY_CUTOFF}"
        )

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


def _get_customer_status(p_alive: float) -> CustomerStatus:
    """Determine customer status based on p_alive probability."""
    if p_alive <= LAPSING_PROBABILITY_CUTOFF:
        return CustomerStatus.LOST
    elif p_alive < ACTIVE_PROBABILITY_CUTOFF:
        return CustomerStatus.LAPSING
    else:
        return CustomerStatus.ACTIVE


# --- Utilities --------------------------------------------------------------
def log_dataframe_stats(df: pd.DataFrame, name: str):
    """Log dataframe statistics in consistent format"""
    log.info(f"{name} with {df.shape[0]} rows")
    log.info(f"Columns: {df.columns.tolist()}")
    log.info(f"Sample data:\n{df.head()}")
    log.info(f"Data types:\n{df.dtypes}")
    log.info(f"Statistics:\n{df.describe().round(3)}")
    log.info(
        f"Status distribution:\n{df.get('customer_status', pd.Series()).value_counts(normalize=True).round(3)}"
    )
    log.info(
        f"P_alive deciles:\n{pd.qcut(df.get('p_alive', pd.Series()), 10, duplicates='drop').value_counts(normalize=True).sort_index().round(3)}"
    )


def log_config_constants():
    """Log all configuration constants"""
    log.info("Configuration constants:")
    log.info(f"  ACTIVE_PROBABILITY_CUTOFF: {ACTIVE_PROBABILITY_CUTOFF}")
    log.info(f"  LAPSING_PROBABILITY_CUTOFF: {LAPSING_PROBABILITY_CUTOFF}")
    log.info(f"  ALIVE_CUTOFF_DAYS: {ALIVE_CUTOFF_DAYS}")
    log.info(f"  LAPSING_CUTOFF_DAYS: {LAPSING_CUTOFF_DAYS}")
    log.info(f"  PARETO_PENALIZER: {PARETO_PENALIZER}")
    log.info(f"  TRANSACTION_EMPIRICAL_CUTOFF: {TRANSACTION_EMPIRICAL_CUTOFF}")
    log.info(f"  MAX_FREQUENCY_CUTOFF: {MAX_FREQUENCY_CUTOFF}")


# --- Main -------------------------------------------------------------------
def main():
    log.info("Starting lapse propensity model")
    bq = BQ()

    # Fetch and process data
    log.info("Fetching transactions and calculating features")
    transactions = fetch_transactions(bq)
    features = create_btyd_features(transactions)

    # Train model
    log.info("Fitting model")
    model = ParetoEmpiricalSingleTrainSplit()
    model.fit(features)

    # Generate predictions
    log.info("Calculating probabilities and customer status")
    features["p_alive"] = model.p_alive(features).round(2)
    features["customer_status"] = model.customer_status(features)

    # Select final columns
    final_features = features[FINAL_COLUMNS]

    # Log results and config
    log_dataframe_stats(final_features, "Final results")
    log_config_constants()

    # Save results
    bq.to_bq(final_features, f"{PROJECT_ID}.{DATABASE_NAME}.{TABLE_NAME}")
    log.info("Lapse propensity model completed and data saved to BigQuery")


if __name__ == "__main__":
    main()
