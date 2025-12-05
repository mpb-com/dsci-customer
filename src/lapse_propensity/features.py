from abc import ABC, abstractmethod
from .config import MAX_FREQUENCY_CUTOFF, TRANSACTION_QUERY
from lifetimes.utils import summary_data_from_transaction_data
from loguru import logger as log
import pandas as pd
import numpy as np


class BaseFeatureEngineer(ABC):
    """Abstract base class for feature engineering.

    All feature engineers must implement these methods to be compatible
    with the pipeline and backtesting infrastructure.
    """

    @abstractmethod
    def create_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw transaction data.

        Args:
            transactions: DataFrame with columns ['customer_id', 'txn_date']

        Returns:
            DataFrame with customer-level features, must include 'customer_id' column
        """
        pass

    @abstractmethod
    def get_feature_columns(self) -> list[str]:
        """Return list of feature column names (excluding customer_id).

        Returns:
            List of feature column names
        """
        pass


def fetch_transactions(bq, query=TRANSACTION_QUERY):
    """Fetch transaction data from BigQuery"""
    transactions = bq.get_string(query)
    n_customers = transactions["customer_id"].nunique()
    n_transactions = len(transactions)
    log.info(f"Fetched {n_transactions:,} transactions for {n_customers:,} customers")

    return transactions


class BTYDFeatureEngineer(BaseFeatureEngineer):
    """Standard BTYD feature engineering: frequency, recency, T, days_since_last.

    These are the classic Buy Till You Die features used by Pareto/NBD and similar models.
    """

    def __init__(self):
        self.name = "BTYDFeatureEngineer"

    def get_feature_columns(self) -> list[str]:
        """Return BTYD feature column names."""
        return ["frequency", "recency", "T", "days_since_last"]

    def create_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create BTYD features: frequency, recency, T, days_since_last.

        Args:
            transactions: DataFrame with ['customer_id', 'txn_date']

        Returns:
            DataFrame with BTYD features
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

        # Only log for production datasets (suppress for simulations/scenarios)
        if summary.shape[0] > 5000:
            log.info(f"Created BTYD features with {summary.shape[0]} customers")
            log.info(f"Columns: {summary.columns.tolist()}")
            log.info(f"Sample data:\n{summary.head()}")
            log.info(f"Data types:\n{summary.dtypes}")
            log.info(summary.describe().round(1))

        return summary


def create_btyd_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create BTYD features: frequency, recency, T, days_since_last.

    This is a convenience wrapper for backward compatibility.
    For new code, prefer using BTYDFeatureEngineer directly.

    Args:
        transactions: DataFrame with ['customer_id', 'txn_date']

    Returns:
        DataFrame with BTYD features
    """
    engineer = BTYDFeatureEngineer()
    return engineer.create_features(transactions)


def get_feature_engineer_class(class_name: str):
    """Get feature engineer class by name.

    Args:
        class_name: Name of the feature engineer class (e.g., 'BTYDFeatureEngineer')

    Returns:
        Feature engineer class that inherits from BaseFeatureEngineer

    Raises:
        ValueError: If class name not found or doesn't inherit from BaseFeatureEngineer
    """
    # Get the class from the current module's globals
    engineer_class = globals().get(class_name)

    if engineer_class is None:
        available_engineers = [
            name for name, obj in globals().items()
            if isinstance(obj, type) and issubclass(obj, BaseFeatureEngineer) and obj != BaseFeatureEngineer
        ]
        raise ValueError(
            f"Feature engineer class '{class_name}' not found. "
            f"Available engineers: {', '.join(available_engineers)}"
        )

    if not issubclass(engineer_class, BaseFeatureEngineer):
        raise ValueError(
            f"Feature engineer class '{class_name}' must inherit from BaseFeatureEngineer"
        )

    return engineer_class


def create_feature_engineer_from_config():
    """Create feature engineer instance from config.

    Returns:
        Feature engineer instance based on FEATURE_ENGINEER_CLASS_NAME in config
    """
    from .config import FEATURE_ENGINEER_CLASS_NAME

    engineer_class = get_feature_engineer_class(FEATURE_ENGINEER_CLASS_NAME)
    return engineer_class()


class XGBoostFeatureEngineer(BaseFeatureEngineer):
    """Rich feature engineering for XGBoost models.

    Extends BTYD features with additional engineered features:
    - Transaction velocity and trends
    - Time-based patterns
    - Normalized metrics
    """

    def __init__(self):
        self.name = "XGBoostFeatureEngineer"
        self.btyd_engineer = BTYDFeatureEngineer()

    def get_feature_columns(self) -> list[str]:
        """Return all feature column names."""
        base_cols = self.btyd_engineer.get_feature_columns()
        extra_cols = [
            "avg_days_between_purchases",
            "purchase_acceleration",
            "days_since_first_pct",
            "frequency_per_month",
            "is_new_customer",
            "recency_to_T_ratio",
            "days_since_last_normalized",
        ]
        return base_cols + extra_cols

    def create_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create rich feature set for XGBoost."""
        # Get base BTYD features
        features = self.btyd_engineer.create_features(transactions)

        # Add transaction aggregations
        txn_agg = self._create_transaction_aggregates(transactions)
        features = features.merge(txn_agg, on="customer_id", how="left")

        # Add derived features
        features = self._add_derived_features(features)

        # Only log for production datasets (suppress for simulations/scenarios)
        if len(features) > 5000:
            log.info(f"Created XGBoost features: {len(features)} customers, {len(features.columns)} columns")

        return features

    def _create_transaction_aggregates(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregates from transaction history."""
        txns = transactions.sort_values(["customer_id", "txn_date"]).copy()
        txns["prev_txn_date"] = txns.groupby("customer_id")["txn_date"].shift(1)
        txns["days_between"] = (txns["txn_date"] - txns["prev_txn_date"]).dt.days

        # Add missing columns with default values
        default_cols = {
            "buy_value_market": 0,
            "total_buy_items": 0,
            "sold_sell_items": 0,
            "delivery_charge_market": 0,
            "total_sold_sell_value": 0,
        }
        for col, default_val in default_cols.items():
            if col not in txns.columns:
                txns[col] = default_val

        # Monetary aggregations
        agg = txns.groupby("customer_id").agg(
            avg_days_between_purchases=("days_between", "mean"),
            total_revenue=("buy_value_market", "sum"),
            avg_transaction_value=("buy_value_market", "mean"),
            total_items_bought=("total_buy_items", "sum"),
            total_items_sold=("sold_sell_items", "sum"),
            avg_delivery_charge=("delivery_charge_market", "mean"),
            total_sold_value=("total_sold_sell_value", "sum"),
            transaction_count=("customer_id", "count")
        ).reset_index()

        # Fill NaN values
        for col in agg.columns:
            if col != "customer_id":
                agg[col] = agg[col].fillna(0)

        # Categorical one-hot encoding
        categorical_features = self._create_categorical_features(txns)
        agg = agg.merge(categorical_features, on="customer_id", how="left")

        # Fill NaN values for categorical features
        categorical_cols = [col for col in agg.columns if col.startswith(("transaction_type_", "market_", "ga_sub_channel_"))]
        agg[categorical_cols] = agg[categorical_cols].fillna(0)

        return agg

    def _create_categorical_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create one-hot encoded features for categorical variables."""
        results = []

        # For each categorical column
        for col in ['transaction_type', 'market', 'ga_sub_channel']:
            # Skip if column doesn't exist in transactions
            if col not in transactions.columns:
                continue

            # Count frequency per customer
            cat_counts = transactions.groupby(['customer_id', col]).size().reset_index(name='count')

            # Pivot to wide format (one column per category)
            cat_pivot = cat_counts.pivot_table(
                index='customer_id',
                columns=col,
                values='count',
                fill_value=0
            )

            # Rename columns to include feature name
            cat_pivot.columns = [f"{col}_{val}" for val in cat_pivot.columns]

            results.append(cat_pivot)

        # Combine all categorical features
        if len(results) > 0:
            categorical_df = pd.concat(results, axis=1).reset_index()
        else:
            # If no categorical features, return empty dataframe with customer_id
            categorical_df = pd.DataFrame({'customer_id': []})

        return categorical_df

    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features derived from base features."""
        df = features.copy()

        # Purchase acceleration (negative = slowing down)
        df["purchase_acceleration"] = np.where(
            df["avg_days_between_purchases"] > 0,
            (df["avg_days_between_purchases"] - df["days_since_last"]) / df["avg_days_between_purchases"],
            0,
        )

        # Days since last as % of customer lifetime
        df["days_since_first_pct"] = np.where(df["T"] > 0, df["days_since_last"] / df["T"], 0)

        # Normalized purchase frequency (purchases per month)
        df["frequency_per_month"] = np.where(df["T"] > 0, (df["frequency"] + 1) / (df["T"] / 30), 0)

        # Binary indicator for new customers
        df["is_new_customer"] = (df["frequency"] == 0).astype(np.int32)

        # Recency ratio (0 = at start, 1 = at end/recent)
        df["recency_to_T_ratio"] = np.where(df["T"] > 0, df["recency"] / df["T"], 0)

        # Days since last normalized by average IPT
        df["days_since_last_normalized"] = np.where(
            df["avg_days_between_purchases"] > 0,
            df["days_since_last"] / df["avg_days_between_purchases"],
            0,
        )

        # Buy/sell behavior
        df["buy_sell_ratio"] = np.where(
            df["total_items_sold"] > 0,
            df["total_items_bought"] / df["total_items_sold"],
            df["total_items_bought"]
        )

        df["avg_items_per_transaction"] = (
            (df["total_items_bought"] + df["total_items_sold"]) / df["transaction_count"]
        ).fillna(0)

        df["revenue_per_item"] = np.where(
            (df["total_items_bought"] + df["total_items_sold"]) > 0,
            df["total_revenue"] / (df["total_items_bought"] + df["total_items_sold"]),
            0
        )

        return df
