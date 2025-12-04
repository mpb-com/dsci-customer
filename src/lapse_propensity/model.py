import numpy as np
import pandas as pd
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
