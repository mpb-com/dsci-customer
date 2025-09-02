import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, ParetoNBDFitter


class BaseModel:
    def __init__(self):
        self.name = "BaseModel"
        self.model = None

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses should implement this method")

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        raise NotImplementedError("Subclasses should implement this method")

    def _get_customer_status(self, p_alive: float) -> str:
        if p_alive <= 0.3:
            return "Lost"
        elif p_alive < 0.6:
            return "Lapsing"
        else:
            return "Active"

    def customer_status(self, df: pd.DataFrame) -> pd.Series:
        probs = self.p_alive(df)
        return probs.apply(self._get_customer_status)


class BGNBD(BaseModel):
    def __init__(self):
        self.name = "BGNBD"
        self.model = BetaGeoFitter(penalizer_coef=0.001)

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df["frequency"], df["recency"], df["T"])

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        probs = self.model.conditional_probability_alive(
            df["frequency"], df["recency"], df["T"]
        )
        return pd.Series(probs, index=df.index)

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        preds = self.model.predict(horizon, df["frequency"], df["recency"], df["T"])
        return pd.Series(preds, index=df.index)


class ParetoNBD(BaseModel):
    def __init__(self):
        self.name = "ParetoNBD"
        self.model = ParetoNBDFitter(penalizer_coef=0.001)

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df["frequency"], df["recency"], df["T"])

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        probs = self.model.conditional_probability_alive(
            df["frequency"], df["recency"], df["T"]
        )
        return pd.Series(probs, index=df.index)

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        preds = self.model.predict(horizon, df["frequency"], df["recency"], df["T"])
        return pd.Series(preds, index=df.index)


class Empirical(BaseModel):
    def __init__(self, alive_cutoff_days=270, lapsing_cutoff_days=540):
        self.name = f"Empirical_{alive_cutoff_days}_{lapsing_cutoff_days}"
        self.model = None
        self.alive_cutoff_days = alive_cutoff_days
        self.lapsing_cutoff_days = lapsing_cutoff_days

    def fit(self, df: pd.DataFrame) -> None:
        self.decay_rate = -np.log(0.6) / self.alive_cutoff_days

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        days = df["days_since_last"]
        probs = np.exp(-self.decay_rate * days)
        return pd.Series(probs, index=df.index)

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        p_alive = self.p_alive(df)
        # Use minimum observation period to avoid inflated rates for new customers
        min_T = np.maximum(df["T"], 365)  # At least 1 year observation
        transaction_rate = df["frequency"] / min_T
        return p_alive * transaction_rate * horizon


class BGNBDEmpiricalSingle(BaseModel):
    def __init__(self, alive_cutoff_days=270, lapsing_cutoff_days=540):
        BGNBD.__init__(self)
        self.bgnbd = BGNBD()
        self.empirical = Empirical(alive_cutoff_days, lapsing_cutoff_days)
        self.name = f"BGNBDES_{alive_cutoff_days}_{lapsing_cutoff_days}"

    def fit(self, df: pd.DataFrame) -> None:
        BGNBD.fit(self, df)
        self.empirical.fit(df)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        bgnbd_probs = BGNBD.p_alive(self, df)
        empirical_probs = self.empirical.p_alive(df)
        probs = np.where(df["frequency"] > 0, bgnbd_probs, empirical_probs)
        return pd.Series(probs, index=df.index)

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        bgnbd_preds = BGNBD.predict_future_transactions(self, df, horizon)
        empirical_preds = self.empirical.predict_future_transactions(df, horizon)
        preds = np.where(df["frequency"] > 0, bgnbd_preds, empirical_preds)
        return pd.Series(preds, index=df.index)


class BGNBDEmpiricalSingleTrainNoSingle(BaseModel):
    def __init__(self, alive_cutoff_days=270, lapsing_cutoff_days=540):
        BGNBD.__init__(self)
        self.bgnbd = BGNBD()
        self.empirical = Empirical(alive_cutoff_days, lapsing_cutoff_days)
        self.name = f"BGNBDESTNS_{alive_cutoff_days}_{lapsing_cutoff_days}"

    def fit(self, df: pd.DataFrame) -> None:
        BGNBD.fit(self, df[df["frequency"] > 1])
        self.empirical.fit(df)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        bgnbd_probs = BGNBD.p_alive(self, df)
        empirical_probs = self.empirical.p_alive(df)
        probs = np.where(df["frequency"] > 0, bgnbd_probs, empirical_probs)
        return pd.Series(probs, index=df.index)

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        bgnbd_preds = BGNBD.predict_future_transactions(self, df, horizon)
        empirical_preds = self.empirical.predict_future_transactions(df, horizon)
        preds = np.where(df["frequency"] > 0, bgnbd_preds, empirical_preds)
        return pd.Series(preds, index=df.index)


class ParetoEmpiricalSingleTrainNoSingle(BaseModel):
    def __init__(self, alive_cutoff_days=270, lapsing_cutoff_days=540):
        ParetoNBD.__init__(self)
        self.pareto = ParetoNBD()
        self.empirical = Empirical(alive_cutoff_days, lapsing_cutoff_days)
        self.name = f"ParetoESTNS_{alive_cutoff_days}_{lapsing_cutoff_days}"

    def fit(self, df: pd.DataFrame) -> None:
        ParetoNBD.fit(self, df[df["frequency"] > 1])
        self.empirical.fit(df)

    def p_alive(self, df: pd.DataFrame) -> pd.Series:
        pareto_probs = ParetoNBD.p_alive(self, df)
        empirical_probs = self.empirical.p_alive(df)
        probs = np.where(df["frequency"] > 0, pareto_probs, empirical_probs)
        return pd.Series(probs, index=df.index)

    def predict_future_transactions(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        pareto_preds = ParetoNBD.predict_future_transactions(self, df, horizon)
        empirical_preds = self.empirical.predict_future_transactions(df, horizon)
        preds = np.where(df["frequency"] > 0, pareto_preds, empirical_preds)
        return pd.Series(preds, index=df.index)
