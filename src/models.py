"""
ML Models — Profitability Simulator
Implements:
  - DemandForecaster    : XGBoost multi-category demand prediction
  - ElasticityEstimator : Log-log OLS price elasticity per category
  - MarginPredictor     : Ridge regression for gross/EBIT margin
  - TimeSeriesForecaster: SARIMA-based weekly revenue/EBIT forecast
  - FeatureEngineer     : Shared preprocessing pipeline
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
class FeatureEngineer:
    """Transform raw retail DataFrame into ML-ready feature matrix."""

    FEATURE_COLS = [
        "effective_price", "unit_cost", "discount_rate_pct",
        "fixed_costs", "promo_flag",
        "inflation_rate_pct", "fuel_price_index",
        "consumer_confidence_index", "competitor_price_index",
        "exchange_rate_eur_usd", "interest_rate_pct",
        "unemployment_rate_pct", "gdp_growth_pct",
        "month_sin", "month_cos", "week_sin", "week_cos",
        "cat_encoded",
    ]

    def __init__(self):
        self.cat_encoder = LabelEncoder()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_cyclical(df)
        df["cat_encoded"] = self.cat_encoder.fit_transform(df["category"])
        self._fitted = True
        return df[self.FEATURE_COLS]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = df.copy()
        df = self._add_cyclical(df)
        df["cat_encoded"] = self.cat_encoder.transform(df["category"])
        return df[self.FEATURE_COLS]

    @staticmethod
    def _add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["week_sin"]  = np.sin(2 * np.pi * df["week_num"] / 52)
        df["week_cos"]  = np.cos(2 * np.pi * df["week_num"] / 52)
        return df


# ──────────────────────────────────────────────────────────────
# DEMAND FORECASTER  (XGBoost)
# ──────────────────────────────────────────────────────────────
class DemandForecaster:
    """
    XGBoost model predicting weekly demand_units per category.
    Captures non-linear interactions: price × seasonality × macro.
    """

    def __init__(self):
        self.fe = FeatureEngineer()
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.metrics: dict = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "DemandForecaster":
        X = self.fe.fit_transform(df)
        y = df["demand_units"].values
        self.model.fit(X, y)

        # Cross-validated metrics (3-fold time-aware)
        split = int(len(y) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y[:split], y[split:]
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)

        self.metrics = {
            "MAE":   round(mean_absolute_error(y_te, preds), 1),
            "RMSE":  round(np.sqrt(mean_squared_error(y_te, preds)), 1),
            "R2":    round(r2_score(y_te, preds), 3),
            "MAPE":  round(np.mean(np.abs((y_te - preds) / (y_te + 1e-6))) * 100, 2),
        }
        # Re-fit on full data
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.fe.transform(df)
        return np.maximum(0, self.model.predict(X))

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.fe.FEATURE_COLS,
        ).sort_values(ascending=False)


# ──────────────────────────────────────────────────────────────
# PRICE ELASTICITY ESTIMATOR  (Log-Log OLS per category)
# ──────────────────────────────────────────────────────────────
class ElasticityEstimator:
    """
    Estimates price elasticity: ln(Q) = α + β·ln(P) + controls.
    β ≈ price elasticity of demand (expected: negative).
    """

    def __init__(self):
        self.results: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> "ElasticityEstimator":
        for cat, grp in df.groupby("category"):
            grp = grp.copy()
            ln_q = np.log(grp["demand_units"].clip(1))
            ln_p = np.log(grp["effective_price"].clip(0.01))

            # Controls: log(competitor price), inflation, seasonality
            ln_comp = np.log(grp["competitor_price_index"].clip(1))
            conf    = (grp["consumer_confidence_index"] - 58) / 10
            season  = np.sin(2 * np.pi * grp["month"] / 12)
            promo   = grp["promo_flag"]
            gdp     = grp["gdp_growth_pct"] / 100

            X = sm.add_constant(
                pd.DataFrame({
                    "ln_price":    ln_p,
                    "ln_comp":     ln_comp,
                    "confidence":  conf,
                    "seasonality": season,
                    "promo":       promo,
                    "gdp":         gdp,
                })
            )
            ols = sm.OLS(ln_q, X).fit()
            self.results[cat] = {
                "elasticity":   round(ols.params.get("ln_price", np.nan), 3),
                "std_err":      round(ols.bse.get("ln_price", np.nan), 3),
                "p_value":      round(ols.pvalues.get("ln_price", np.nan), 4),
                "r_squared":    round(ols.rsquared, 3),
                "cross_elast":  round(ols.params.get("ln_comp", np.nan), 3),
                "promo_lift":   round(ols.params.get("promo", np.nan), 3),
            }
        return self

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).T.reset_index().rename(columns={"index": "category"})


# ──────────────────────────────────────────────────────────────
# MARGIN PREDICTOR  (Ridge regression)
# ──────────────────────────────────────────────────────────────
class MarginPredictor:
    """Ridge regression predicting gross_margin_pct and ebit_margin_pct."""

    def __init__(self):
        self.fe        = FeatureEngineer()
        self.gross_mdl = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        self.ebit_mdl  = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        self.metrics: dict = {}

    def fit(self, df: pd.DataFrame) -> "MarginPredictor":
        X = self.fe.fit_transform(df)
        y_gross = df["gross_margin_pct"].values
        y_ebit  = df["ebit_margin_pct"].values

        split = int(len(y_gross) * 0.8)
        for name, mdl, y in [("gross", self.gross_mdl, y_gross), ("ebit", self.ebit_mdl, y_ebit)]:
            mdl.fit(X.iloc[:split], y[:split])
            preds = mdl.predict(X.iloc[split:])
            self.metrics[name] = {
                "MAE":  round(mean_absolute_error(y[split:], preds), 3),
                "R2":   round(r2_score(y[split:], preds), 3),
            }
            mdl.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.fe.transform(df)
        return pd.DataFrame({
            "pred_gross_margin_pct": self.gross_mdl.predict(X).round(2),
            "pred_ebit_margin_pct":  self.ebit_mdl.predict(X).round(2),
        })


# ──────────────────────────────────────────────────────────────
# TIME SERIES FORECASTER  (SARIMA)
# ──────────────────────────────────────────────────────────────
class TimeSeriesForecaster:
    """
    SARIMA(1,1,1)(1,1,1,52) per category for weekly revenue / EBIT.
    Forecasts the next `horizon` weeks.
    """

    def __init__(self, horizon: int = 12):
        self.horizon = horizon
        self.models:    dict[str, SARIMAX] = {}
        self.fitted:    dict = {}
        self.last_date: dict[str, pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame, target: str = "revenue") -> "TimeSeriesForecaster":
        self._target = target
        agg = (
            df.groupby(["category", "date"])[target]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        for cat, grp in agg.groupby("category"):
            series = grp.set_index("date")[target].asfreq("W-MON")
            self.last_date[cat] = series.index[-1]
            try:
                mdl = SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 52),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.fitted[cat] = mdl.fit(disp=False, maxiter=60)
            except Exception:
                # Fallback: simple ARIMA(1,1,1)
                mdl = SARIMAX(series, order=(1, 1, 1))
                self.fitted[cat] = mdl.fit(disp=False, maxiter=60)
        return self

    def forecast(self, category: str) -> pd.DataFrame:
        if category not in self.fitted:
            raise ValueError(f"Category '{category}' not trained.")
        res   = self.fitted[category]
        fcast = res.get_forecast(steps=self.horizon)
        idx   = pd.date_range(
            start=self.last_date[category] + pd.Timedelta(weeks=1),
            periods=self.horizon,
            freq="W-MON",
        )
        return pd.DataFrame({
            "date":     idx,
            "forecast": fcast.predicted_mean.values,
            "lower_ci": fcast.conf_int().iloc[:, 0].values,
            "upper_ci": fcast.conf_int().iloc[:, 1].values,
        })

    def forecast_all(self) -> dict[str, pd.DataFrame]:
        return {cat: self.forecast(cat) for cat in self.fitted}


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: train all models at once
# ──────────────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame) -> dict:
    """Train all models and return a dict of fitted objects + metrics."""
    demand_model    = DemandForecaster().fit(df)
    elasticity_mdl  = ElasticityEstimator().fit(df)
    margin_model    = MarginPredictor().fit(df)
    ts_revenue      = TimeSeriesForecaster(horizon=12).fit(df, target="revenue")
    ts_ebit         = TimeSeriesForecaster(horizon=12).fit(df, target="ebit")

    return {
        "demand":     demand_model,
        "elasticity": elasticity_mdl,
        "margin":     margin_model,
        "ts_revenue": ts_revenue,
        "ts_ebit":    ts_ebit,
    }
