import numpy as np
import pandas as pd
from lifetimes import ParetoNBDFitter
from loguru import logger as log

from .config import (
    ACTIVE_PROBABILITY_CUTOFF,
    LAPSING_PROBABILITY_CUTOFF,
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


def _get_customer_status(p_alive: float) -> str:
    """Determine customer status based on p_alive probability."""
    if p_alive <= LAPSING_PROBABILITY_CUTOFF:
        return CustomerStatus.LOST
    elif p_alive < ACTIVE_PROBABILITY_CUTOFF:
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
        probs = self.model.conditional_probability_alive(df["frequency"], df["recency"], df["T"])
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
        self.pareto.fit(df_train)
        self.empirical.fit()

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

    def customer_status(self, df: pd.DataFrame) -> pd.Series:
        probs = self.p_alive(df)
        return probs.apply(_get_customer_status)
