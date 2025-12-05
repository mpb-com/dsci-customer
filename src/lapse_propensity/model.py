import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from lifetimes import ParetoNBDFitter
from loguru import logger as log
from sklearn.isotonic import IsotonicRegression

from .config import (
    ACTIVE_PROBABILITY_CUTOFF,
    ALIVE_CUTOFF_DAYS,
    LAPSING_CUTOFF_DAYS,
    PARETO_PENALIZER,
    TRANSACTION_EMPIRICAL_CUTOFF,
)


class BaseLapseModel(ABC):
    """Abstract base class for lapse propensity models.

    All lapse models must implement these core methods to be compatible
    with the pipeline and backtesting infrastructure.
    """

    # Class attribute: Does this model need labels for training?
    requires_labels_for_training: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on customer features.

        Args:
            df: DataFrame with features (and y_true_alive if requires_labels_for_training=True)
        """
        pass

    @abstractmethod
    def calibrate(self, df: pd.DataFrame, y_true: pd.Series) -> None:
        """Calibrate model predictions using isotonic regression.

        Args:
            df: DataFrame with features
            y_true: Ground truth labels (1 = active, 0 = inactive)
        """
        pass

    @abstractmethod
    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        """Predict probability that each customer is alive.

        Args:
            df: DataFrame with features

        Returns:
            Series of probabilities (0-1) with same index as df
        """
        pass


class CustomerStatus:
    """Customer status labels. Change as desired"""

    ACTIVE = "alive"
    LAPSING = "lapsing"
    LOST = "lost"


def _get_customer_status(p_alive: float, dead_threshold: float, alive_threshold: float) -> str:
    """Determine customer status based on p_alive and business thresholds.

    Args:
        p_alive: Probability customer is alive
        dead_threshold: Threshold below which customer is considered LOST (max 5% revenue risk)
        alive_threshold: Threshold above which customer is considered ALIVE (2× baseline)

    Returns:
        Customer status: 'lost', 'lapsing', or 'alive'
    """
    if p_alive < dead_threshold:
        return CustomerStatus.LOST
    elif p_alive < alive_threshold:
        return CustomerStatus.LAPSING
    else:
        return CustomerStatus.ACTIVE


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
        self.calibrator = None

    def fit(self) -> None:
        self.decay_rate = -np.log(ACTIVE_PROBABILITY_CUTOFF) / self.alive_cutoff_days

    def calibrate(self, df: pd.DataFrame, y_true: pd.Series) -> None:
        """Calibrate probabilities using isotonic regression"""
        uncalibrated_probs = self._p_alive_uncalibrated(df)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(uncalibrated_probs, y_true)
        log.info(f"Empirical model calibrated on {len(df)} samples")

    def _p_alive_uncalibrated(self, df: pd.DataFrame) -> pd.Series:
        """Get uncalibrated probabilities"""
        days = df["days_since_last"]
        probs = np.exp(-self.decay_rate * days)
        return pd.Series(probs, index=df.index)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        """Get probabilities (calibrated if calibrator exists)"""
        uncalibrated = self._p_alive_uncalibrated(df)
        if self.calibrator is not None:
            calibrated = self.calibrator.predict(uncalibrated)
            return pd.Series(calibrated, index=df.index)
        return uncalibrated


class ParetoNBD:
    def __init__(self):
        self.name = "ParetoNBD"
        self.model = ParetoNBDFitter(penalizer_coef=PARETO_PENALIZER)
        self.calibrator = None

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df["frequency"], df["recency"], df["T"], verbose=True)

    def calibrate(self, df: pd.DataFrame, y_true: pd.Series) -> None:
        """Calibrate probabilities using isotonic regression"""
        uncalibrated_probs = self._p_alive_uncalibrated(df)
        # Remove NaN values for calibration
        mask = ~uncalibrated_probs.isna()
        if mask.sum() > 0:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(uncalibrated_probs[mask], y_true[mask])
            log.info(f"Pareto model calibrated on {mask.sum()} samples ({(~mask).sum()} NaN excluded)")
        else:
            log.warning("All Pareto predictions are NaN, skipping calibration")

    def _p_alive_uncalibrated(self, df: pd.DataFrame) -> pd.Series:
        """Get uncalibrated probabilities"""
        probs = self.model.conditional_probability_alive(df["frequency"], df["recency"], df["T"])
        return pd.Series(probs, index=df.index)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        """Get probabilities (calibrated if calibrator exists)"""
        uncalibrated = self._p_alive_uncalibrated(df)
        if self.calibrator is not None:
            # Only calibrate non-NaN values
            mask = ~uncalibrated.isna()
            calibrated = uncalibrated.copy()
            if mask.sum() > 0:
                calibrated[mask] = self.calibrator.predict(uncalibrated[mask])
            return calibrated
        return uncalibrated


class ParetoEmpiricalSingleTrainSplit(BaseLapseModel):
    """Hybrid model combining Pareto/NBD for repeat buyers and empirical decay for new customers.

    This model switches between two sub-models based on transaction frequency:
    - Pareto/NBD for customers with frequency > transaction_cutoff
    - Empirical exponential decay for customers with frequency <= transaction_cutoff
    """

    def __init__(
        self,
        alive_cutoff_days=ALIVE_CUTOFF_DAYS,
        lapsing_cutoff_days=LAPSING_CUTOFF_DAYS,
        transaction_cutoff=TRANSACTION_EMPIRICAL_CUTOFF,
    ):
        self.pareto = ParetoNBD()
        self.empirical = Empirical(alive_cutoff_days, lapsing_cutoff_days)
        self.name = f"ParetoESTNS_{alive_cutoff_days}_{lapsing_cutoff_days}"
        self.transaction_cutoff = transaction_cutoff

    def fit(self, df: pd.DataFrame) -> None:
        df_train = df[df["frequency"] > 0]
        log.info(f"Training on {len(df_train)} customers with frequency > 0")
        self.pareto.fit(df_train)
        self.empirical.fit()

    def calibrate(self, df: pd.DataFrame, y_true: pd.Series) -> None:
        """Calibrate both sub-models separately on their respective segments"""
        # Calibrate Pareto on high-frequency customers
        pareto_mask = df["frequency"] > TRANSACTION_EMPIRICAL_CUTOFF - 1
        if pareto_mask.sum() > 0:
            self.pareto.calibrate(df[pareto_mask], y_true[pareto_mask])
        else:
            log.warning("No high-frequency customers for Pareto calibration")

        # Calibrate Empirical on low-frequency customers
        empirical_mask = ~pareto_mask
        if empirical_mask.sum() > 0:
            self.empirical.calibrate(df[empirical_mask], y_true[empirical_mask])
        else:
            log.warning("No low-frequency customers for Empirical calibration")

        log.info("Hybrid model calibration complete")

    def _handle_nan_fallback(self, pareto_probs: pd.Series, empirical_probs: pd.Series) -> pd.Series:
        """Handle NaN values from Pareto model by falling back to empirical predictions"""
        nan_count = pareto_probs.isnull().sum()
        if nan_count > 0:
            log.info(f"Found {nan_count} NaN predictions from Pareto model, falling back to empirical")
        return pareto_probs.fillna(empirical_probs)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        pareto_probs = self.pareto.p_alive(df)
        empirical_probs = self.empirical.p_alive(df)

        # Handle NaN values from Pareto model by falling back to empirical
        pareto_probs = self._handle_nan_fallback(pareto_probs, empirical_probs)

        probs = np.where(
            df["frequency"] > TRANSACTION_EMPIRICAL_CUTOFF - 1,
            pareto_probs,
            empirical_probs,
        )
        return pd.Series(probs, index=df.index)

    def customer_status(self, df: pd.DataFrame, dead_threshold: float, alive_threshold: float) -> pd.Series:
        """Apply customer status labels using dynamic thresholds.

        Args:
            df: DataFrame with customer features
            dead_threshold: Threshold below which customer is LOST
            alive_threshold: Threshold above which customer is ALIVE

        Returns:
            Series of customer status labels
        """
        probs = self.p_alive(df)
        return probs.apply(lambda p: _get_customer_status(p, dead_threshold, alive_threshold))


def get_model_class(model_class_name: str):
    """Get model class by name.

    Args:
        model_class_name: Name of the model class (e.g., 'ParetoEmpiricalSingleTrainSplit')

    Returns:
        Model class that inherits from BaseLapseModel

    Raises:
        ValueError: If model class name is not found or doesn't inherit from BaseLapseModel
    """
    # Get the class from the current module's globals
    model_class = globals().get(model_class_name)

    if model_class is None:
        available_models = [
            name for name, obj in globals().items()
            if isinstance(obj, type) and issubclass(obj, BaseLapseModel) and obj != BaseLapseModel
        ]
        raise ValueError(
            f"Model class '{model_class_name}' not found. "
            f"Available models: {', '.join(available_models)}"
        )

    if not issubclass(model_class, BaseLapseModel):
        raise ValueError(
            f"Model class '{model_class_name}' must inherit from BaseLapseModel"
        )

    return model_class


def create_model_from_config():
    """Create model instance from config.

    Returns:
        Model instance based on MODEL_CLASS_NAME in config
    """
    from .config import MODEL_CLASS_NAME

    model_class = get_model_class(MODEL_CLASS_NAME)
    return model_class()


class XGBoostLapseModel(BaseLapseModel):
    """XGBoost model for lapse propensity prediction.

    This is a supervised model that requires labeled data for training.
    Unlike BTYD models, it learns from ground truth labels.
    """

    requires_labels_for_training = True

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_child_weight: int = 50,
    ):
        self.name = "XGBoostLapseModel"
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                random_state=42,
                eval_metric="logloss",
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.feature_columns = None
        self.feature_medians = None  # Store median values for imputation

    def fit(self, df: pd.DataFrame) -> None:
        """Train XGBoost model.

        Args:
            df: DataFrame with features AND 'y_true_alive' column
        """
        if "y_true_alive" not in df.columns:
            raise ValueError(
                "XGBoostLapseModel requires 'y_true_alive' column for training. "
                "This is a supervised model."
            )

        # Identify feature columns (exclude metadata, targets, and calibration columns)
        excluded_cols = [
            "customer_id", "y_true_alive", "y_true_txns",
            "frequency_holdout", "duration_holdout",
            # Calibration columns from lifetimes library
            "frequency_cal", "recency_cal", "T_cal", "duration_cal"
        ]

        self.feature_columns = [
            col for col in df.columns
            if col not in excluded_cols
        ]

        X = df[self.feature_columns]
        y = df["y_true_alive"]

        # Store feature medians for imputation
        self.feature_medians = X.median()

        log.info(f"Training XGBoost on {len(df)} samples with {len(self.feature_columns)} features")
        log.info(f"Features: {self.feature_columns}")
        log.info(f"Class distribution: {y.value_counts().to_dict()}")

        self.model.fit(X, y)

        # Log feature importance
        feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        log.info("Top 10 features by importance:")
        log.info(f"\n{feature_importance.head(10).to_string(index=False)}")

    def calibrate(self, df: pd.DataFrame, y_true: pd.Series) -> None:
        """No calibration needed for XGBoost - probabilities are already well-calibrated."""
        log.info("Skipping calibration for XGBoost (not needed)")
        pass

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        """Predict probability that each customer is alive.

        Args:
            df: DataFrame with features

        Returns:
            Series of probabilities (0-1)
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Add missing features using training medians
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            df_copy = df.copy()
            for feature in missing_features:
                impute_value = self.feature_medians[feature] if self.feature_medians is not None else 0
                df_copy[feature] = impute_value
            X = df_copy[self.feature_columns]
        else:
            X = df[self.feature_columns]

        probs = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (alive)

        return pd.Series(probs, index=df.index)
