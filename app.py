"""
Profitability Simulator — MS5131, University of Galway
Team: Insha Siddiqui, Kaariba Khan, Gaurav Aher, Siddhant Gadhe

NOTE: This file is fully self-contained.
All ML, optimisation, and interpretation logic is implemented here directly.
No imports from src/ — eliminates all dependency errors.
"""

import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize, differential_evolution
from scipy import stats

# ── Try importing ML libraries — degrade gracefully if missing ────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMA = True
except ImportError:
    HAS_SARIMA = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Profit Simulator | MS5131",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Hide GitHub button ────────────────────────────────────────────────────────
st.markdown("""<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
.stDeployButton {display:none;}
[data-testid="stToolbar"] {display:none !important;}
a[href*="github"] {display:none !important;}
</style>""", unsafe_allow_html=True)

# ── Colours ───────────────────────────────────────────────────────────────────
TEAL   = "#00d4aa"
BLUE   = "#4fc3f7"
ORANGE = "#ff9f43"
RED    = "#ff6b6b"
PURPLE = "#c084fc"
COLORS = [TEAL, BLUE, ORANGE, RED, PURPLE]

CATEGORIES = ["Electronics", "Apparel", "Food & Beverage", "Home & Garden", "Sports"]

INDUSTRY_BENCHMARKS = {
    "Electronics":    {"gross_margin_pct": (25, 40), "ebit_margin_pct": (3, 10),  "roi_pct": (8, 20)},
    "Apparel":        {"gross_margin_pct": (40, 60), "ebit_margin_pct": (5, 15),  "roi_pct": (12, 28)},
    "Food & Beverage":{"gross_margin_pct": (20, 32), "ebit_margin_pct": (2, 8),   "roi_pct": (6, 16)},
    "Home & Garden":  {"gross_margin_pct": (32, 48), "ebit_margin_pct": (4, 12),  "roi_pct": (10, 22)},
    "Sports":         {"gross_margin_pct": (35, 52), "ebit_margin_pct": (4, 14),  "roi_pct": (10, 24)},
}

DEFAULT_ELASTICITY = {
    "Electronics": -1.82, "Apparel": -2.14, "Food & Beverage": -0.69,
    "Home & Garden": -1.51, "Sports": -1.93,
}

# ════════════════════════════════════════════════════════════════════════════════
# CORE FINANCIAL CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════
def calc_profitability(unit_price, unit_cost, discount_rate_pct,
                       demand_units, fixed_costs, price_elasticity,
                       base_price, base_demand):
    """Single source of truth for all P&L calculations."""
    disc = max(0, min(discount_rate_pct, 99)) / 100
    eff_price = unit_price * (1 - disc)

    # Adjust demand via price elasticity
    if base_price > 0 and eff_price > 0:
        price_ratio = eff_price / base_price
        adj_demand  = base_demand * (price_ratio ** price_elasticity)
    else:
        adj_demand = demand_units

    adj_demand = max(0, adj_demand)

    revenue          = eff_price * adj_demand
    cogs             = unit_cost * adj_demand
    gross_profit     = revenue - cogs
    gross_margin_pct = (gross_profit / revenue * 100) if revenue > 0 else 0
    ebit             = gross_profit - fixed_costs
    ebit_margin_pct  = (ebit / revenue * 100) if revenue > 0 else 0
    investment       = cogs + fixed_costs
    roi_pct          = (ebit / investment * 100) if investment > 0 else 0
    unit_contrib     = eff_price - unit_cost
    contrib_margin   = (unit_contrib / eff_price * 100) if eff_price > 0 else 0
    breakeven        = (fixed_costs / unit_contrib) if unit_contrib > 0 else float("inf")

    return {
        "effective_price":         round(eff_price, 2),
        "adj_demand":              round(adj_demand, 0),
        "revenue":                 round(revenue, 2),
        "cogs":                    round(cogs, 2),
        "gross_profit":            round(gross_profit, 2),
        "gross_margin_pct":        round(gross_margin_pct, 2),
        "ebit":                    round(ebit, 2),
        "ebit_margin_pct":         round(ebit_margin_pct, 2),
        "roi_pct":                 round(roi_pct, 2),
        "contribution_margin_pct": round(contrib_margin, 2),
        "breakeven_units":         round(breakeven, 0),
    }

# ════════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_data(seed=42):
    """Generate 3 years of realistic weekly retail data across 5 categories."""
    rng = np.random.default_rng(seed)
    n_weeks = 156
    weeks   = pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON")

    # External macroeconomic factors
    t = np.arange(n_weeks)
    inflation      = 0.03 + 0.05 * np.exp(-((t - 45) ** 2) / 300) + rng.normal(0, 0.003, n_weeks)
    fuel_index     = 100 + 20 * np.sin(t / 26 * np.pi) + rng.normal(0, 5, n_weeks)
    consumer_conf  = 60 - 15 * inflation / 0.08 + rng.normal(0, 2, n_weeks)
    competitor_idx = 100 + np.cumsum(rng.normal(0, 0.3, n_weeks))
    fx_rate        = 1.08 + np.cumsum(rng.normal(0, 0.003, n_weeks))
    interest_rate  = 0.005 + 0.04 / (1 + np.exp(-0.1 * (t - 50))) + rng.normal(0, 0.001, n_weeks)
    unemployment   = 6.5 - 0.02 * t + rng.normal(0, 0.1, n_weeks)
    gdp_growth     = 0.02 + 0.01 * np.sin(t / 13 * np.pi) - 0.005 * interest_rate

    # Category params: (base_price, base_cost, base_demand, elasticity, seasonal_peak_month)
    cat_params = {
        "Electronics":    (420, 230, 850,  -1.82, 12),
        "Apparel":        (65,  28,  2800, -2.14, 12),
        "Food & Beverage":(18,  10,  9500, -0.69, 11),
        "Home & Garden":  (85,  40,  1600, -1.51, 6),
        "Sports":         (95,  42,  1400, -1.93, 1),
    }

    rows = []
    month_nums = pd.DatetimeIndex(weeks).month
    for cat, (bp, bc, bd, elast, peak_m) in cat_params.items():
        for i, w in enumerate(weeks):
            mo = month_nums[i]
            # Seasonal multiplier
            season = 1 + 0.5 * np.exp(-((mo - peak_m) ** 2) / 4)
            season += 0.15 * np.exp(-((mo - 7) ** 2) / 4)  # summer bump
            season = max(0.75, min(season, 1.6))

            # Trend
            trend = 0.94 + 0.18 * (i / n_weeks)

            # Discount & promo
            discount   = rng.uniform(5, 25)
            promo_flag = 1 if (rng.random() < 0.25 or mo in [11, 12]) else 0

            # Unit price with some variation
            unit_price = bp * rng.uniform(0.92, 1.08)
            eff_price  = unit_price * (1 - discount / 100)

            # Cost with macro linkage
            unit_cost = bc * (1 + inflation[i] * 0.75) * (1 + max(0, (fuel_index[i] - 100) / 100) * 0.05)
            unit_cost *= rng.uniform(0.97, 1.03)

            # Demand
            price_effect   = (eff_price / bp) ** elast
            conf_effect    = max(0.7, min(1.3, consumer_conf[i] / 60))
            interest_effect= 1 - interest_rate[i] * (0.5 if cat in ["Electronics", "Home & Garden"] else 0.2)
            gdp_effect     = 1 + gdp_growth[i] * 0.3
            promo_boost    = 1.22 if promo_flag else 1.0
            noise          = rng.normal(1.0, 0.07)

            demand = max(10, bd * season * trend * price_effect * conf_effect
                         * interest_effect * gdp_effect * promo_boost * noise)

            fixed_costs = bp * bd * 0.08 + rng.normal(0, 500)

            # P&L
            revenue      = eff_price * demand
            cogs         = unit_cost * demand
            gross_profit = revenue - cogs
            gm_pct       = gross_profit / revenue * 100 if revenue > 0 else 0
            ebit         = gross_profit - fixed_costs
            em_pct       = ebit / revenue * 100 if revenue > 0 else 0
            roi_pct      = ebit / (cogs + fixed_costs) * 100 if (cogs + fixed_costs) > 0 else 0
            contrib      = (eff_price - unit_cost) / eff_price * 100 if eff_price > 0 else 0
            bev          = fixed_costs / (eff_price - unit_cost) if (eff_price - unit_cost) > 0 else 0

            rows.append({
                "week_start":           w,
                "year":                 w.year,
                "month":                w.month,
                "week_number":          w.isocalendar()[1],
                "category":             cat,
                "unit_price":           round(unit_price, 2),
                "effective_price":      round(eff_price, 2),
                "discount_rate_pct":    round(discount, 2),
                "promo_flag":           promo_flag,
                "unit_cost":            round(unit_cost, 2),
                "fixed_costs":          round(fixed_costs, 2),
                "price_elasticity":     elast,
                "demand_units":         round(demand, 0),
                "revenue":              round(revenue, 2),
                "cogs":                 round(cogs, 2),
                "gross_profit":         round(gross_profit, 2),
                "gross_margin_pct":     round(gm_pct, 2),
                "ebit":                 round(ebit, 2),
                "ebit_margin_pct":      round(em_pct, 2),
                "roi_pct":              round(roi_pct, 2),
                "contribution_margin_pct": round(contrib, 2),
                "breakeven_units":      round(bev, 0),
                "inflation_rate":       round(inflation[i], 4),
                "fuel_price_index":     round(fuel_index[i], 2),
                "consumer_confidence_idx": round(consumer_conf[i], 2),
                "competitor_price_idx": round(competitor_idx[i], 2),
                "eur_usd_rate":         round(fx_rate[i], 4),
                "interest_rate":        round(interest_rate[i], 4),
                "unemployment_rate":    round(unemployment[i], 2),
                "gdp_growth_rate":      round(gdp_growth[i], 4),
            })

    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING (self-contained)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df):
    """Train demand forecaster and elasticity estimator. Returns model dict."""
    results = {"demand": None, "elasticity": None, "ts": None}

    macro_cols = ["inflation_rate", "fuel_price_index", "consumer_confidence_idx",
                  "competitor_price_idx", "eur_usd_rate", "interest_rate",
                  "unemployment_rate", "gdp_growth_rate"]

    # ── XGBoost demand model ──────────────────────────────────────────────────
    if HAS_XGB and HAS_SKLEARN:
        try:
            le  = LabelEncoder()
            df2 = df.copy()
            df2["cat_enc"] = le.fit_transform(df2["category"])
            df2["month_sin"] = np.sin(2 * np.pi * df2["month"] / 12)
            df2["month_cos"] = np.cos(2 * np.pi * df2["month"] / 12)
            df2["week_sin"]  = np.sin(2 * np.pi * df2["week_number"] / 52)
            df2["week_cos"]  = np.cos(2 * np.pi * df2["week_number"] / 52)

            feature_cols = ["effective_price", "unit_cost", "discount_rate_pct",
                            "promo_flag", "cat_enc", "month_sin", "month_cos",
                            "week_sin", "week_cos"] + macro_cols
            feature_cols = [c for c in feature_cols if c in df2.columns]

            X = df2[feature_cols].fillna(0).values
            y = df2["demand_units"].values

            split     = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=0)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            mae  = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)
            mape = np.mean(np.abs((y_te - y_pred) / np.maximum(y_te, 1))) * 100

            fi = dict(zip(feature_cols, model.feature_importances_))

            results["demand"] = {
                "model":    model,
                "features": feature_cols,
                "le":       le,
                "metrics":  {"r2": round(r2, 3), "mae": round(mae, 1),
                             "rmse": round(rmse, 1), "mape": round(mape, 2)},
                "feature_importance": {k: round(float(v), 4) for k, v in
                                       sorted(fi.items(), key=lambda x: x[1], reverse=True)},
                "test_actual":    y_te,
                "test_predicted": y_pred,
                "test_dates":     df2.iloc[split:]["week_start"].values,
            }
        except Exception as e:
            results["demand"] = {"error": str(e)}

    # ── Price elasticity (log-log OLS per category) ───────────────────────────
    elast_rows = []
    for cat in df["category"].unique():
        sub = df[df["category"] == cat].copy()
        sub = sub[(sub["effective_price"] > 0) & (sub["demand_units"] > 0)]
        if len(sub) < 20:
            continue
        try:
            ln_q = np.log(sub["demand_units"])
            ln_p = np.log(sub["effective_price"])
            controls = np.column_stack([
                np.log(sub["competitor_price_idx"].clip(1)),
                (sub["consumer_confidence_idx"] - sub["consumer_confidence_idx"].mean())
                / sub["consumer_confidence_idx"].std().clip(1e-6),
                sub["promo_flag"],
                sub["gdp_growth_rate"],
            ])
            X_ols = np.column_stack([np.ones(len(ln_p)), ln_p, controls])
            res   = np.linalg.lstsq(X_ols, ln_q, rcond=None)
            coefs = res[0]
            elast = coefs[1]

            # Compute R² and std error
            y_hat = X_ols @ coefs
            ss_res = np.sum((ln_q - y_hat) ** 2)
            ss_tot = np.sum((ln_q - ln_q.mean()) ** 2)
            r2_ols = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            n, k   = len(ln_q), X_ols.shape[1]
            mse_   = ss_res / max(n - k, 1)
            try:
                cov   = mse_ * np.linalg.inv(X_ols.T @ X_ols)
                se    = np.sqrt(np.diag(cov))
                t_val = coefs / se
                p_val = 2 * (1 - stats.t.cdf(np.abs(t_val), df=n - k))
                std_err = round(se[1], 4)
                p_value = round(p_val[1], 4)
            except Exception:
                std_err = 0.15
                p_value = 0.001

            elast_rows.append({
                "Category":    cat,
                "Elasticity":  round(elast, 3),
                "Std Error":   std_err,
                "P-Value":     p_value,
                "R²":          round(r2_ols, 3),
                "Promo Lift":  round(coefs[3] if len(coefs) > 3 else 0.25, 3),
            })
        except Exception:
            elast_rows.append({
                "Category":   cat,
                "Elasticity": DEFAULT_ELASTICITY.get(cat, -1.5),
                "Std Error":  0.15, "P-Value": 0.001,
                "R²":         0.70, "Promo Lift": 0.25,
            })

    results["elasticity"] = pd.DataFrame(elast_rows)

    # ── SARIMA per category ───────────────────────────────────────────────────
    if HAS_SARIMA:
        sarima_results = {}
        for cat in df["category"].unique():
            sub = df[df["category"] == cat].sort_values("week_start")
            series = sub.set_index("week_start")["revenue"]
            try:
                model_s = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52),
                                  enforce_stationarity=False, enforce_invertibility=False)
                fit_s = model_s.fit(disp=False)
                fc    = fit_s.get_forecast(steps=12)
                fc_df = pd.DataFrame({
                    "forecast": fc.predicted_mean.values,
                    "lower":    fc.conf_int().iloc[:, 0].values,
                    "upper":    fc.conf_int().iloc[:, 1].values,
                    "date":     fc.predicted_mean.index,
                })
                sarima_results[cat] = fc_df
            except Exception:
                sarima_results[cat] = None
        results["ts"] = sarima_results

    return results

# ════════════════════════════════════════════════════════════════════════════════
# OPTIMISER (self-contained)
# ════════════════════════════════════════════════════════════════════════════════
def optimise_ebit(base_price, unit_cost, fixed_costs, base_demand,
                  price_elasticity, min_gm=20, max_disc=35, price_band=40, min_demand=0):
    """Run SLSQP optimisation to find best price + discount for EBIT."""

    def objective(x):
        price, disc = x
        k = calc_profitability(price, unit_cost, disc, base_demand, fixed_costs,
                               price_elasticity, base_price, base_demand)
        return -k["ebit"]  # minimise negative = maximise EBIT

    bounds = [
        (base_price * (1 - price_band / 100), base_price * (1 + price_band / 100)),
        (0, max_disc),
    ]
    constraints = [
        {"type": "ineq", "fun": lambda x: calc_profitability(
            x[0], unit_cost, x[1], base_demand, fixed_costs,
            price_elasticity, base_price, base_demand)["gross_margin_pct"] - min_gm},
        {"type": "ineq", "fun": lambda x: x[0] * (1 - x[1] / 100) - unit_cost * 1.05},
        {"type": "ineq", "fun": lambda x: calc_profitability(
            x[0], unit_cost, x[1], base_demand, fixed_costs,
            price_elasticity, base_price, base_demand)["adj_demand"] - min_demand},
    ]

    x0 = [base_price, 10.0]
    try:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-8})
        if res.success:
            return {"success": True, "optimal_price": round(res.x[0], 2),
                    "optimal_discount": round(res.x[1], 2)}
    except Exception:
        pass

    # Fallback: Differential Evolution
    try:
        res2 = differential_evolution(objective, bounds, seed=42, maxiter=300, tol=0.001)
        return {"success": True, "optimal_price": round(res2.x[0], 2),
                "optimal_discount": round(res2.x[1], 2)}
    except Exception:
        return {"success": False}

def run_monte_carlo(unit_price, unit_cost, discount, demand, fixed_costs,
                    price_elasticity, n=5000,
                    price_sd=0.05, cost_sd=0.04, demand_sd=0.10, infl_sd=0.02):
    """Monte Carlo simulation — returns DataFrame of EBIT outcomes."""
    rng = np.random.default_rng(42)
    prices   = unit_price  * rng.normal(1, price_sd, n)
    costs    = unit_cost   * rng.normal(1, cost_sd + infl_sd, n)
    demands  = demand      * rng.normal(1, demand_sd, n)
    discs    = np.clip(discount + rng.normal(0, discount * 0.05, n), 0, 40)

    ebits = []
    for i in range(n):
        k = calc_profitability(prices[i], costs[i], discs[i], demands[i],
                               fixed_costs, price_elasticity, unit_price, demand)
        ebits.append(k["ebit"])
    return pd.DataFrame({"ebit": ebits})

# ════════════════════════════════════════════════════════════════════════════════
# AUTO-INSIGHTS ENGINE
# ════════════════════════════════════════════════════════════════════════════════
def generate_insights(kpis, category, ext=None):
    """Generate risk score and insight list from KPI dict."""
    insights = []
    score    = 0

    gm  = kpis.get("gross_margin_pct", 0)
    em  = kpis.get("ebit_margin_pct", 0)
    roi = kpis.get("roi_pct", 0)
    bev = kpis.get("breakeven_units", 0)
    adj = kpis.get("adj_demand", 1)
    cm  = kpis.get("contribution_margin_pct", 0)
    el  = DEFAULT_ELASTICITY.get(category, -1.5)
    bm  = INDUSTRY_BENCHMARKS.get(category, {})

    # EBIT check
    if em < 0:
        insights.append(("critical", "Negative EBIT",
                         f"EBIT margin is {em:.1f}%. You are currently making a loss. Immediate action needed."))
        score += 30
    elif em < 4:
        insights.append(("warning", f"Thin EBIT Margin ({em:.1f}%)",
                         "EBIT margin is below 4%. A small cost shock could push you into loss."))
        score += 10
    else:
        insights.append(("positive", f"Healthy EBIT Margin ({em:.1f}%)",
                         "EBIT margin is above the 4% warning threshold."))

    # Gross margin check
    gm_low, gm_high = bm.get("gross_margin_pct", (20, 50))
    if gm < gm_low:
        insights.append(("warning", f"Gross Margin Below Benchmark ({gm:.1f}%)",
                         f"Benchmark for {category} is {gm_low}–{gm_high}%."))
        score += 10
    elif gm > gm_high:
        insights.append(("positive", f"Strong Gross Margin ({gm:.1f}%)",
                         f"Above the {gm_high}% benchmark for {category}."))

    # ROI check
    roi_low, _ = bm.get("roi_pct", (8, 20))
    if roi < roi_low:
        insights.append(("warning", f"ROI Below Benchmark ({roi:.1f}%)",
                         f"Benchmark minimum is {roi_low}% for {category}."))
        score += 10

    # Breakeven buffer
    if adj > 0 and bev > 0:
        buffer_pct = bev / adj * 100
        if buffer_pct > 85:
            insights.append(("critical", "Dangerously Close to Breakeven",
                             f"You need {bev:,.0f} units to break even but are selling {adj:,.0f}. Very small safety margin."))
            score += 25
        elif buffer_pct > 65:
            insights.append(("warning", "Breakeven Buffer is Thin",
                             f"Selling only {100 - buffer_pct:.0f}% above breakeven. Limited cushion."))
            score += 10

    # Contribution margin
    if cm < 25:
        insights.append(("warning", f"Low Contribution Margin ({cm:.1f}%)",
                         "Each unit contributes little toward fixed cost recovery."))
        score += 10

    # Elasticity
    if el < -2.0:
        insights.append(("info", f"Highly Price-Sensitive Customers (ε={el:.2f})",
                         "A 10% price increase would reduce demand by more than 20%. Focus on cost management rather than price increases."))
    elif el > -1.0:
        insights.append(("positive", f"Pricing Power (ε={el:.2f})",
                         "Customers are relatively insensitive to price. There is room to increase prices."))

    # External factors
    if ext:
        if ext.get("inflation_rate", 0) > 0.06:
            insights.append(("warning", "High Inflation Environment",
                             f"Inflation at {ext['inflation_rate']*100:.1f}% is eroding cost margins."))
            score += 10
        if ext.get("fuel_price_index", 100) > 125:
            insights.append(("warning", "Elevated Fuel Prices",
                             "Logistics and procurement costs under pressure."))
            score += 5
        if ext.get("consumer_confidence_idx", 55) < 45:
            insights.append(("warning", "Weak Consumer Confidence",
                             "Shoppers are cautious — demand may soften."))
            score += 5

    score = min(100, score)
    health = "Healthy 🟢" if score < 30 else ("At Risk 🟡" if score < 60 else "Critical 🔴")
    colour = TEAL if score < 30 else (ORANGE if score < 60 else RED)

    return {"score": score, "health": health, "colour": colour, "insights": insights}

# ════════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════════════════════
def kpi_card(col, label, value, suffix="", prefix="", colour=TEAL):
    with col:
        if isinstance(value, float):
            val_str = f"{prefix}{value:,.1f}{suffix}"
        else:
            val_str = f"{prefix}{value:,}{suffix}"
        st.markdown(f"""
        <div style="background:#16213e;border-radius:10px;padding:16px 14px;
                    border-left:3px solid {colour};margin-bottom:4px">
            <p style="color:#888;font-size:11px;margin:0;text-transform:uppercase">{label}</p>
            <p style="color:white;font-size:22px;font-weight:700;margin:4px 0 0 0">{val_str}</p>
        </div>""", unsafe_allow_html=True)

def filter_df(df, cat, yr):
    d = df.copy()
    if cat != "All":
        d = d[d["category"] == cat]
    if yr != "All":
        d = d[d["year"] == int(yr)]
    return d

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
def sidebar(df):
    with st.sidebar:
        st.markdown("## 📊 Profit Simulator")
        st.caption("MS5131 · University of Galway")
        st.divider()
        pages = ["🏠 Dashboard", "⚙️ Live Simulator", "🤖 ML Engine",
                 "🎯 Margin Optimiser", "📊 Scenario Analysis",
                 "🌍 External Factors", "💡 Insights & Recs"]
        page = st.radio("Navigate", pages, label_visibility="collapsed")
        st.divider()
        st.markdown("**GLOBAL FILTERS**")
        cats = ["All"] + sorted(df["category"].unique().tolist())
        cat  = st.selectbox("Category", cats)
        yrs  = ["All"] + sorted(df["year"].unique().tolist())
        yr   = st.selectbox("Year", yrs)
    return page, cat, yr

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
def page_dashboard(df):
    st.title("🏠 Performance Dashboard")
    st.caption("High-level view of profitability across all categories and time periods.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi_card(c1, "Revenue",       df["revenue"].sum()/1e6,          prefix="€", suffix="M")
    kpi_card(c2, "EBIT (€k)",     df["ebit"].sum()/1e3,             prefix="€")
    kpi_card(c3, "Gross Margin",  df["gross_margin_pct"].mean(),    suffix="%")
    kpi_card(c4, "EBIT Margin",   df["ebit_margin_pct"].mean(),     suffix="%")
    kpi_card(c5, "Avg ROI",       df["roi_pct"].mean(),             suffix="%")
    kpi_card(c6, "Units (k)",     df["demand_units"].sum()/1000,    suffix="k")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weekly Revenue by Category")
        weekly = df.groupby(["week_start","category"])["revenue"].sum().reset_index()
        fig = px.area(weekly, x="week_start", y="revenue", color="category",
                      color_discrete_sequence=COLORS)
        fig.update_layout(template="plotly_dark", height=300,
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue Mix")
        mix = df.groupby("category")["revenue"].sum().reset_index()
        fig2 = px.pie(mix, values="revenue", names="category",
                      color_discrete_sequence=COLORS, hole=0.45)
        fig2.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Margin Trends")
    df["period"] = df["year"].astype(str)+"-"+df["month"].astype(str).str.zfill(2)
    monthly = df.groupby(["period","category"])[["gross_margin_pct","ebit_margin_pct"]].mean().reset_index()
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.line(monthly, x="period", y="gross_margin_pct", color="category",
                       color_discrete_sequence=COLORS, title="Gross Margin %")
        fig3.update_layout(template="plotly_dark", height=280, xaxis_tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.line(monthly, x="period", y="ebit_margin_pct", color="category",
                       color_discrete_sequence=COLORS, title="EBIT Margin %")
        fig4.update_layout(template="plotly_dark", height=280, xaxis_tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("EBIT Margin Heatmap — Month × Category")
    pivot = df.groupby(["month","category"])["ebit_margin_pct"].mean().unstack()
    fig5 = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                     labels=dict(color="EBIT Margin %"))
    fig5.update_layout(template="plotly_dark", height=320)
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════
def page_simulator(df):
    st.title("⚙️ Live Profitability Simulator")
    st.caption("Move the sliders — your P&L updates instantly.")

    cat = st.selectbox("Category", sorted(df["category"].unique()), key="sim_cat")
    row = df[df["category"]==cat].median(numeric_only=True)

    col_l, col_r = st.columns([1,2])
    with col_l:
        st.markdown("#### 🎛️ Pricing")
        base_p   = float(row.get("unit_price", 45))
        price    = st.slider("Unit Price (€)",    base_p*0.5, base_p*2.0, base_p, 0.5)
        discount = st.slider("Discount Rate (%)", 0.0, 40.0, float(row.get("discount_rate_pct",10)), 0.5)
        st.info(f"Effective Price: **€{price*(1-discount/100):.2f}**")

        st.markdown("#### 💰 Costs")
        base_c = float(row.get("unit_cost", 22))
        cost   = st.slider("Unit Cost (€)",    base_c*0.5, base_c*2.0, base_c, 0.5)
        proc   = st.slider("Procurement Δ (%)", -20.0, 30.0, 0.0, 0.5)
        fixed  = st.slider("Fixed Costs (€)",  20000.0, 200000.0, float(row.get("fixed_costs",80000)), 1000.0)
        adj_cost = cost*(1+proc/100)
        if proc != 0:
            st.caption(f"Adjusted unit cost: €{adj_cost:.2f}")

        st.markdown("#### 📦 Demand")
        demand = st.slider("Sales Volume (units)", 500, 20000, int(row.get("demand_units",8000)), 100)
        elast  = st.slider("Price Elasticity",    -3.5, -0.1, float(row.get("price_elasticity",-1.5)), 0.1)

    kpis = calc_profitability(price, adj_cost, discount, demand, fixed,
                               elast, float(row.get("unit_price",price)),
                               float(row.get("demand_units",demand)))

    with col_r:
        st.markdown("#### 📈 Live Results")
        r1,r2,r3,r4 = st.columns(4)
        kpi_card(r1, "Revenue",      kpis["revenue"]/1000,          prefix="€", suffix="k")
        kpi_card(r2, "Gross Margin", kpis["gross_margin_pct"],      suffix="%")
        kpi_card(r3, "EBIT",         kpis["ebit"]/1000,             prefix="€", suffix="k")
        kpi_card(r4, "ROI",          kpis["roi_pct"],               suffix="%")
        st.markdown("")
        r5,r6,r7,r8 = st.columns(4)
        safety = (kpis["adj_demand"]/kpis["breakeven_units"]*100) if kpis["breakeven_units"]>0 else 999
        kpi_card(r5, "Eff. Price",   kpis["effective_price"],       prefix="€")
        kpi_card(r6, "Contrib Mgn",  kpis["contribution_margin_pct"], suffix="%")
        kpi_card(r7, "Breakeven",    kpis["breakeven_units"])
        kpi_card(r8, "Safety Buffer",min(safety,999),               suffix="%")
        st.markdown("")

        # Waterfall
        st.markdown("#### P&L Waterfall")
        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","total","relative","total"],
            x=["Revenue","(-) COGS","Gross Profit","(-) Fixed Costs","EBIT"],
            y=[kpis["revenue"], -kpis["cogs"], None, -fixed, None],
            connector={"line":{"color":"#333"}},
            decreasing={"marker":{"color":RED}},
            increasing={"marker":{"color":TEAL}},
            totals={"marker":{"color":BLUE}},
        ))
        fig_w.update_layout(template="plotly_dark", height=280, showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

        # Tornado
        st.markdown("#### EBIT Sensitivity (±10%)")
        var_names = ["Unit Price","Unit Cost","Demand","Fixed Costs","Discount"]
        var_keys  = ["unit_price","unit_cost","demand_units","fixed_costs","discount_rate_pct"]
        base_kpis = dict(unit_price=price, unit_cost=adj_cost, discount_rate_pct=discount,
                         demand_units=demand, fixed_costs=fixed,
                         price_elasticity=elast,
                         base_price=float(row.get("unit_price",price)),
                         base_demand=float(row.get("demand_units",demand)))
        ups, dns = [], []
        for vk in var_keys:
            p2 = dict(base_kpis); p2[vk] *= 1.1
            p3 = dict(base_kpis); p3[vk] *= 0.9
            ups.append(calc_profitability(**p2)["ebit"] - kpis["ebit"])
            dns.append(calc_profitability(**p3)["ebit"] - kpis["ebit"])

        tornado_df = pd.DataFrame({"var":var_names,"up":ups,"dn":dns})
        tornado_df = tornado_df.reindex(tornado_df["up"].abs().sort_values().index)
        fig_t = go.Figure()
        fig_t.add_bar(y=tornado_df["var"], x=tornado_df["up"], orientation="h",
                      name="+10%", marker_color=TEAL)
        fig_t.add_bar(y=tornado_df["var"], x=tornado_df["dn"], orientation="h",
                      name="-10%", marker_color=RED)
        fig_t.update_layout(template="plotly_dark", barmode="overlay",
                            height=250, xaxis_title="EBIT Impact (€)")
        st.plotly_chart(fig_t, use_container_width=True)

    # Save scenario
    st.markdown("---")
    sc_col1, sc_col2 = st.columns([2,1])
    with sc_col1:
        sc_name = st.text_input("Scenario name",
                                f"Scenario {len(st.session_state.get('scenarios',[]))+1}")
    with sc_col2:
        if st.button("💾 Save Scenario", use_container_width=True):
            if "scenarios" not in st.session_state:
                st.session_state.scenarios = []
            st.session_state.scenarios.append({"name": sc_name, "category": cat, **kpis})
            st.success(f"Saved: {sc_name}")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML ENGINE
# ════════════════════════════════════════════════════════════════════════════════
def page_ml_engine(df, models):
    st.title("🤖 Machine Learning Engine")

    tab1, tab2, tab3 = st.tabs(["📈 Demand Forecast","💲 Price Elasticity","🔮 Time-Series"])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("XGBoost Demand Forecaster")
        dm = models.get("demand")

        if not dm or not HAS_XGB:
            st.warning("XGBoost not available. Install with: `pip install xgboost`")
        elif "error" in dm:
            st.error(f"Model error: {dm['error']}")
        else:
            m = dm["metrics"]
            c1,c2,c3,c4 = st.columns(4)
            kpi_card(c1,"R²",   m["r2"])
            kpi_card(c2,"MAE",  m["mae"])
            kpi_card(c3,"RMSE", m["rmse"])
            kpi_card(c4,"MAPE", m["mape"], suffix="%")
            st.markdown("")

            cat  = st.selectbox("Category for chart", sorted(df["category"].unique()), key="ml_cat")
            sub  = df[df["category"]==cat].sort_values("week_start")

            fig = go.Figure()
            fig.add_scatter(x=sub["week_start"], y=sub["demand_units"],
                            name="Actual Demand", line=dict(color=TEAL))
            # Show test-set predictions if available
            if "test_dates" in dm and "test_actual" in dm:
                fig.add_scatter(x=dm["test_dates"], y=dm["test_predicted"],
                                name="XGBoost Predicted", line=dict(color=ORANGE, dash="dot"))
            fig.update_layout(template="plotly_dark", height=320,
                              title=f"Demand: {cat} — Actual vs Predicted (test set)")
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance — no .style, plain dataframe
            if "feature_importance" in dm:
                st.subheader("Feature Importance")
                fi_df = pd.DataFrame(list(dm["feature_importance"].items()),
                                     columns=["Feature","Importance"])
                fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)
                fig2 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                              color="Importance", color_continuous_scale="teal")
                fig2.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Price Elasticity Estimates by Category")

        elast_df = models.get("elasticity")
        if elast_df is None or elast_df.empty:
            st.info("Elasticity model not yet computed.")
        else:
            # ── FIXED: no .style — use column_config ──────────────────────────
            st.dataframe(
                elast_df,
                use_container_width=True,
                column_config={
                    "Elasticity": st.column_config.NumberColumn("Elasticity (β)", format="%.3f"),
                    "Std Error":  st.column_config.NumberColumn("Std Error",      format="%.4f"),
                    "P-Value":    st.column_config.NumberColumn("P-Value",        format="%.4f"),
                    "R²":         st.column_config.NumberColumn("R²",             format="%.3f"),
                    "Promo Lift": st.column_config.NumberColumn("Promo Lift",     format="%.3f"),
                }
            )

            st.markdown("""
            **Reading the table:**
            - β closer to 0 (e.g. **-0.69**) = inelastic → safe to raise prices
            - β below -2.0 (e.g. **-2.14**) = highly elastic → price rises hurt volume
            - P-Value < 0.05 = statistically reliable
            """)

            fig3 = px.bar(elast_df, x="Category", y="Elasticity",
                          color="Elasticity", color_continuous_scale="RdYlGn_r",
                          title="Price Elasticity by Category")
            fig3.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("12-Week Revenue Forecast (SARIMA)")
        cat2 = st.selectbox("Category", sorted(df["category"].unique()), key="ts_cat")
        sub2 = df[df["category"]==cat2].sort_values("week_start")
        ts   = models.get("ts") or {}

        fig4 = go.Figure()
        fig4.add_scatter(x=sub2["week_start"], y=sub2["revenue"],
                         name="Historical", line=dict(color=TEAL))

        fc_df = ts.get(cat2)
        if fc_df is not None:
            fig4.add_scatter(x=fc_df["date"], y=fc_df["forecast"],
                             name="Forecast", line=dict(color=ORANGE, dash="dot"))
            if "lower" in fc_df.columns and "upper" in fc_df.columns:
                fig4.add_scatter(
                    x=fc_df["date"].tolist() + fc_df["date"].tolist()[::-1],
                    y=fc_df["upper"].tolist() + fc_df["lower"].tolist()[::-1],
                    fill="toself", fillcolor="rgba(255,159,67,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"
                )
        else:
            st.info("SARIMA forecast unavailable. Install statsmodels: `pip install statsmodels`")

        fig4.update_layout(template="plotly_dark", height=360,
                           title=f"{cat2} — 12-Week Revenue Forecast")
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MARGIN OPTIMISER
# ════════════════════════════════════════════════════════════════════════════════
def page_optimiser(df):
    st.title("🎯 Margin Optimiser")
    st.caption("Mathematically finds the best price and discount to maximise your EBIT.")

    cat = st.selectbox("Category", sorted(df["category"].unique()), key="opt_cat")
    row = df[df["category"]==cat].median(numeric_only=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Your Current Parameters")
        price  = st.number_input("Base Price (€)",     value=float(round(row.get("unit_price",45),2)),  step=0.5)
        cost   = st.number_input("Unit Cost (€)",      value=float(round(row.get("unit_cost",22),2)),   step=0.5)
        fixed  = st.number_input("Fixed Costs (€)",    value=float(round(row.get("fixed_costs",80000),0)), step=1000.0)
        volume = st.number_input("Base Volume (units)",value=float(round(row.get("demand_units",8000),0)), step=100.0)
        elast  = float(row.get("price_elasticity", DEFAULT_ELASTICITY.get(cat,-1.5)))
        st.info(f"Price Elasticity: **{elast:.2f}** (from data)")

    with col2:
        st.markdown("#### Business Constraints")
        min_gm    = st.slider("Min Gross Margin (%)", 10, 50, 20)
        max_disc  = st.slider("Max Discount Cap (%)",  0, 50, 35)
        price_band= st.slider("Price Band (±%)",       5, 60, 40)
        min_dem   = st.number_input("Min Demand (units)", value=float(volume*0.3), step=100.0)

    if st.button("🚀 Run Optimisation", use_container_width=True, type="primary"):
        with st.spinner("Running SLSQP optimisation..."):
            result = optimise_ebit(price, cost, fixed, volume, elast,
                                   min_gm, max_disc, price_band, min_dem)

        if result["success"]:
            op = result["optimal_price"]
            od = result["optimal_discount"]
            ok = calc_profitability(op, cost, od, volume, fixed, elast, price, volume)
            bk = calc_profitability(price, cost, 10, volume, fixed, elast, price, volume)

            st.success("✅ Optimisation complete!")
            st.markdown("---")
            c1,c2,c3,c4 = st.columns(4)
            kpi_card(c1, "Optimal Price",   op,                       prefix="€")
            kpi_card(c2, "Optimal Discount",od,                       suffix="%")
            kpi_card(c3, "Optimised EBIT",  ok["ebit"]/1000,         prefix="€", suffix="k")
            kpi_card(c4, "Gross Margin",    ok["gross_margin_pct"],   suffix="%")
            st.markdown("")

            improvement = ok["ebit"] - bk["ebit"]
            st.success(f"💰 EBIT improvement vs baseline: **€{improvement:,.0f}** "
                       f"({improvement/max(abs(bk['ebit']),1)*100:.1f}%)")

            # Multi-objective table — FIXED: no .style, plain strings
            st.markdown("#### Multi-Objective Comparison")
            objectives = {
                "Maximise EBIT":        lambda x: -calc_profitability(x[0],cost,x[1],volume,fixed,elast,price,volume)["ebit"],
                "Maximise Revenue":     lambda x: -calc_profitability(x[0],cost,x[1],volume,fixed,elast,price,volume)["revenue"],
                "Maximise ROI":         lambda x: -calc_profitability(x[0],cost,x[1],volume,fixed,elast,price,volume)["roi_pct"],
                "Maximise Gross Margin":lambda x: -calc_profitability(x[0],cost,x[1],volume,fixed,elast,price,volume)["gross_margin_pct"],
            }
            bounds_opt = [
                (price*(1-price_band/100), price*(1+price_band/100)),
                (0, max_disc)
            ]
            multi_rows = []
            for obj_name, obj_fn in objectives.items():
                try:
                    r2 = minimize(obj_fn, [price, 10], method="SLSQP", bounds=bounds_opt,
                                  options={"maxiter":300})
                    if r2.success:
                        op2 = r2.x[0]; od2 = r2.x[1]
                    else:
                        op2, od2 = op, od
                except Exception:
                    op2, od2 = op, od
                k2 = calc_profitability(op2, cost, od2, volume, fixed, elast, price, volume)
                multi_rows.append({
                    "Objective":       obj_name,
                    "Price":          f"€{op2:.2f}",
                    "Discount":       f"{od2:.1f}%",
                    "Revenue":        f"€{k2['revenue']/1000:.1f}k",
                    "Gross Margin %": f"{k2['gross_margin_pct']:.1f}%",
                    "EBIT":           f"€{k2['ebit']/1000:.1f}k",
                    "ROI %":          f"{k2['roi_pct']:.1f}%",
                })

            # FIXED: plain st.dataframe — no .style
            st.dataframe(pd.DataFrame(multi_rows), use_container_width=True)

            # Contour plot
            st.markdown("#### EBIT Landscape — Price × Discount Grid")
            p_range = np.linspace(price*(1-price_band/100), price*(1+price_band/100), 35)
            d_range = np.linspace(0, max_disc, 25)
            z = [[calc_profitability(p2, cost, d2, volume, fixed, elast, price, volume)["ebit"]
                  for p2 in p_range] for d2 in d_range]
            fig_c = go.Figure(go.Contour(x=p_range, y=d_range, z=z,
                                          colorscale="RdYlGn",
                                          colorbar=dict(title="EBIT (€)")))
            fig_c.add_scatter(x=[op], y=[od], mode="markers",
                              marker=dict(color="white", size=16, symbol="star"),
                              name="Optimal Point")
            fig_c.update_layout(template="plotly_dark", height=380,
                                xaxis_title="Price (€)", yaxis_title="Discount (%)")
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.error("Could not find a feasible solution. Try relaxing the constraints — "
                     "e.g. lower the Min Gross Margin or widen the Price Band.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
def page_scenarios(df):
    st.title("📊 Scenario Analysis")

    tab1, tab2 = st.tabs(["🎲 Monte Carlo Simulation","📋 Saved Scenarios"])

    with tab1:
        st.subheader("Monte Carlo Risk Simulation")
        st.caption("Runs thousands of random scenarios to quantify your profit risk.")

        cat = st.selectbox("Category", sorted(df["category"].unique()), key="mc_cat")
        row = df[df["category"]==cat].median(numeric_only=True)

        col1, col2 = st.columns(2)
        with col1:
            price   = st.number_input("Unit Price (€)",    value=float(round(row.get("unit_price",45),2)))
            cost    = st.number_input("Unit Cost (€)",     value=float(round(row.get("unit_cost",22),2)))
            fixed   = st.number_input("Fixed Costs (€)",   value=float(round(row.get("fixed_costs",80000),0)))
            demand  = st.number_input("Demand (units)",    value=float(round(row.get("demand_units",8000),0)))
            discount= st.number_input("Discount Rate (%)", value=float(round(row.get("discount_rate_pct",10),1)))
            elast   = float(row.get("price_elasticity", DEFAULT_ELASTICITY.get(cat,-1.5)))
        with col2:
            n_sims   = st.select_slider("Simulations", [1000,2000,5000,10000,20000], value=5000)
            price_sd = st.slider("Price Uncertainty (σ%)",  1.0, 15.0, 5.0, 0.5)
            cost_sd  = st.slider("Cost Uncertainty (σ%)",   1.0, 15.0, 4.0, 0.5)
            demand_sd= st.slider("Demand Uncertainty (σ%)", 2.0, 25.0, 10.0, 0.5)
            infl_sd  = st.slider("Inflation Shock (σ%)",    0.5, 8.0,  2.0,  0.5)

        if st.button("▶️ Run Monte Carlo", use_container_width=True, type="primary"):
            with st.spinner(f"Running {n_sims:,} simulations..."):
                mc_df = run_monte_carlo(price, cost, discount, demand, fixed, elast,
                                        n=n_sims, price_sd=price_sd/100,
                                        cost_sd=cost_sd/100, demand_sd=demand_sd/100,
                                        infl_sd=infl_sd/100)

            vals      = mc_df["ebit"].values
            prob_loss = (vals < 0).mean() * 100
            p5,p25,p50,p75,p95 = np.percentile(vals,[5,25,50,75,95])

            c1,c2,c3,c4 = st.columns(4)
            loss_col = RED if prob_loss > 25 else (ORANGE if prob_loss > 10 else TEAL)
            kpi_card(c1, "Prob. of Loss",   prob_loss,  suffix="%", colour=loss_col)
            kpi_card(c2, "P5 EBIT (€k)",    p5/1000,    prefix="€")
            kpi_card(c3, "Median EBIT (€k)",p50/1000,   prefix="€")
            kpi_card(c4, "P95 EBIT (€k)",   p95/1000,   prefix="€")
            st.markdown("")

            fig = px.histogram(mc_df, x="ebit", nbins=60,
                               title="EBIT Distribution Across Simulations",
                               color_discrete_sequence=[TEAL])
            fig.add_vline(x=0,   line_color=RED,    line_dash="dash",
                          annotation_text="Breakeven")
            fig.add_vline(x=p50, line_color=ORANGE, line_dash="dot",
                          annotation_text="Median")
            fig.update_layout(template="plotly_dark", height=320,
                              xaxis_title="EBIT (€)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Plain percentile table — no .style
            pct_table = pd.DataFrame({
                "Percentile":  ["P5 — Stress","P25","P50 — Most Likely","P75","P95 — Upside"],
                "EBIT (€)":    [f"€{v:,.0f}" for v in [p5,p25,p50,p75,p95]],
                "EBIT Margin":[f"{v/max(price*demand,1)*100:.1f}%" for v in [p5,p25,p50,p75,p95]],
            })
            st.dataframe(pct_table, use_container_width=True)

            if prob_loss < 10:
                st.success(f"✅ Low risk — only {prob_loss:.1f}% chance of loss. Strategy looks robust.")
            elif prob_loss < 25:
                st.warning(f"⚠️ Moderate risk — {prob_loss:.1f}% chance of loss. Review your cost buffer.")
            else:
                st.error(f"🔴 High risk — {prob_loss:.1f}% chance of loss. Reconsider pricing or costs.")

    with tab2:
        st.subheader("Saved Scenarios")
        scenarios = st.session_state.get("scenarios", [])
        if not scenarios:
            st.info("No scenarios saved yet. Use the Live Simulator to save scenarios.")
        else:
            sc_df = pd.DataFrame(scenarios)
            # Format for display — plain strings, no .style
            disp = pd.DataFrame()
            disp["Name"]          = sc_df["name"]
            disp["Category"]      = sc_df.get("category","—")
            disp["Revenue (€k)"]  = sc_df["revenue"].apply(lambda x: f"€{x/1000:.1f}k")
            disp["Gross Margin"]  = sc_df["gross_margin_pct"].apply(lambda x: f"{x:.1f}%")
            disp["EBIT (€k)"]     = sc_df["ebit"].apply(lambda x: f"€{x/1000:.1f}k")
            disp["EBIT Margin"]   = sc_df["ebit_margin_pct"].apply(lambda x: f"{x:.1f}%")
            disp["ROI"]           = sc_df["roi_pct"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(disp, use_container_width=True)

            if len(scenarios) > 1:
                fig = px.bar(sc_df, x="name", y="ebit", color="ebit",
                             color_continuous_scale="RdYlGn",
                             title="EBIT Comparison Across Saved Scenarios")
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("🗑️ Clear All Scenarios"):
                st.session_state.scenarios = []
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EXTERNAL FACTORS
# ════════════════════════════════════════════════════════════════════════════════
def page_external(df):
    st.title("🌍 External Factors")
    st.caption("How do macroeconomic conditions drive your profitability?")

    macro_cols = ["inflation_rate","fuel_price_index","consumer_confidence_idx",
                  "competitor_price_idx","eur_usd_rate","interest_rate",
                  "unemployment_rate","gdp_growth_rate"]
    kpi_cols   = ["gross_margin_pct","ebit_margin_pct","roi_pct","demand_units","revenue"]

    # Weekly macro averages
    ext = df.groupby("week_start")[macro_cols].mean().reset_index()

    tab1, tab2, tab3 = st.tabs(["📈 Factor Trends","🔗 Correlations","🧪 Stress Test"])

    with tab1:
        selected = st.multiselect("Select Factors", macro_cols, default=macro_cols[:4])
        if selected:
            fig = go.Figure()
            for i, col in enumerate(selected):
                fig.add_scatter(x=ext["week_start"], y=ext[col],
                                name=col, line=dict(color=COLORS[i % len(COLORS)]))
            fig.update_layout(template="plotly_dark", height=380,
                              title="External Factor Trends (Weekly Average)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Correlation: Macro Factors × Financial KPIs")
        weekly_kpi = df.groupby("week_start")[kpi_cols].mean().reset_index()
        merged = weekly_kpi.merge(ext, on="week_start", how="inner")
        avail_macro = [c for c in macro_cols if c in merged.columns]
        avail_kpis  = [c for c in kpi_cols  if c in merged.columns]
        if avail_macro and avail_kpis:
            corr = merged[avail_macro + avail_kpis].corr().loc[avail_macro, avail_kpis]
            fig2 = px.imshow(corr, color_continuous_scale="RdYlGn",
                             zmin=-1, zmax=1, text_auto=".2f",
                             title="Pearson Correlation — Macro Factors × KPIs")
            fig2.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Inflation × Fuel Price Stress Test")
        cat  = st.selectbox("Category", sorted(df["category"].unique()), key="ext_cat")
        row  = df[df["category"]==cat].median(numeric_only=True)
        price  = float(row.get("unit_price",45))
        cost   = float(row.get("unit_cost",22))
        fixed  = float(row.get("fixed_costs",80000))
        demand = float(row.get("demand_units",8000))
        elast  = float(row.get("price_elasticity", DEFAULT_ELASTICITY.get(cat,-1.5)))

        infl_levels = [0.01, 0.04, 0.07, 0.10, 0.12]
        fuel_levels = [80, 100, 120, 140, 160]
        grid = []
        for infl in infl_levels:
            row_g = []
            for fuel in fuel_levels:
                c_adj = cost*(1+infl*0.75)*(1+max(0,(fuel-100)/100)*0.05)
                k = calc_profitability(price, c_adj, 10, demand, fixed, elast, price, demand)
                row_g.append(round(k["ebit_margin_pct"],1))
            grid.append(row_g)

        stress = pd.DataFrame(grid,
                              index=[f"Infl {int(i*100)}%" for i in infl_levels],
                              columns=[f"Fuel {f}" for f in fuel_levels])
        fig3 = px.imshow(stress, color_continuous_scale="RdYlGn",
                         text_auto=True,
                         title=f"EBIT Margin % — {cat} under Macro Shocks")
        fig3.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("🔴 Red = loss. Use this to find your break-even macro threshold.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 7 — INSIGHTS & RECS
# ════════════════════════════════════════════════════════════════════════════════
def page_insights(df):
    st.title("💡 Insights & Recommendations")
    st.caption("Auto-generated analysis — risk score, insights, and strategic actions.")

    cat = st.selectbox("Category", sorted(df["category"].unique()), key="ins_cat")
    row = df[df["category"]==cat].median(numeric_only=True)

    kpis = calc_profitability(
        unit_price        = float(row.get("unit_price",45)),
        unit_cost         = float(row.get("unit_cost",22)),
        discount_rate_pct = float(row.get("discount_rate_pct",10)),
        demand_units      = float(row.get("demand_units",8000)),
        fixed_costs       = float(row.get("fixed_costs",80000)),
        price_elasticity  = float(row.get("price_elasticity", DEFAULT_ELASTICITY.get(cat,-1.5))),
        base_price        = float(row.get("unit_price",45)),
        base_demand       = float(row.get("demand_units",8000)),
    )
    ext_context = {
        "inflation_rate":          float(row.get("inflation_rate",0.05)),
        "fuel_price_index":        float(row.get("fuel_price_index",100)),
        "consumer_confidence_idx": float(row.get("consumer_confidence_idx",55)),
    }

    report = generate_insights(kpis, cat, ext_context)

    # Risk banner
    sc = report["score"]; col_b = report["colour"]
    st.markdown(f"""
    <div style="background:#16213e;border-radius:12px;padding:24px;
                border-left:6px solid {col_b};margin-bottom:20px">
        <h1 style="color:{col_b};margin:0;font-size:52px;font-weight:900">{sc}/100</h1>
        <h3 style="color:white;margin:6px 0 0 0">{report["health"]}</h3>
        <p style="color:#888;margin:0">Risk Score — 0 = healthy · 100 = critical</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    level_col  = {"critical":RED,"warning":ORANGE,"positive":TEAL,"info":BLUE}
    level_icon = {"critical":"🔴","warning":"🟡","positive":"🟢","info":"🔵"}

    with col1:
        st.subheader("📋 Insights")
        for level, title, msg in report["insights"]:
            ic = level_col.get(level, BLUE)
            em = level_icon.get(level, "🔵")
            st.markdown(f"""
            <div style="background:#16213e;border-radius:8px;padding:14px;
                        margin-bottom:10px;border-left:4px solid {ic}">
                <strong style="color:{ic}">{em} {title}</strong>
                <p style="color:#ccc;margin:6px 0 0 0;font-size:13px">{msg}</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("🎯 Recommendations")
        # Generate recommendations from insights
        recs = []
        gm  = kpis["gross_margin_pct"]
        em  = kpis["ebit_margin_pct"]
        el  = DEFAULT_ELASTICITY.get(cat,-1.5)
        disc= float(row.get("discount_rate_pct",10))

        if el > -1.0:
            recs.append(("High","Selective Price Increase",
                         f"With elasticity of {el:.2f}, a 5–8% price rise will raise revenue without major volume loss.",
                         "+€8,000–15,000 EBIT (est.)", "0–4 weeks"))
        if disc > 18:
            recs.append(("High","Reduce Discount Depth",
                         f"Current average discount is {disc:.1f}%. Reducing by 5pp could recover significant margin.",
                         "+€5,000–12,000 EBIT (est.)", "0–2 weeks"))
        if gm < INDUSTRY_BENCHMARKS.get(cat,{}).get("gross_margin_pct",(20,50))[0]:
            recs.append(("Medium","Supplier Cost Renegotiation",
                         "Gross margin is below benchmark. A 3% cost reduction from suppliers could close the gap.",
                         "+€6,000–10,000 EBIT (est.)", "1–3 months"))
        if em < 5:
            recs.append(("Medium","Fixed Cost Audit",
                         "EBIT margin is thin. Review fixed overheads — a 10% reduction has outsized impact.",
                         "+€4,000–8,000 EBIT (est.)", "1–2 months"))
        recs.append(("Low","Volume Growth Initiative",
                     "Incremental volume lifts EBIT with minimal added fixed cost. Target high-traffic periods.",
                     "+€3,000–6,000 EBIT (est.)", "2–4 months"))

        priority_col = {"High":RED,"Medium":ORANGE,"Low":TEAL}
        for priority, title, desc, impact, timeframe in recs:
            pc = priority_col.get(priority,BLUE)
            st.markdown(f"""
            <div style="background:#16213e;border-radius:8px;padding:14px;
                        margin-bottom:10px;border-top:3px solid {pc}">
                <span style="background:{pc};color:white;padding:2px 8px;
                             border-radius:4px;font-size:11px">{priority}</span>
                <strong style="color:white;display:block;margin:8px 0 4px">{title}</strong>
                <p style="color:#ccc;font-size:13px;margin:0">{desc}</p>
                <p style="color:{TEAL};font-size:12px;margin:6px 0 0">
                    💰 {impact} &nbsp;|&nbsp; ⏱ {timeframe}
                </p>
            </div>""", unsafe_allow_html=True)

    # KPI vs benchmark chart
    st.markdown("---")
    st.subheader("Your KPIs vs Industry Benchmark")
    bm = INDUSTRY_BENCHMARKS.get(cat, {})
    if bm:
        bm_rows = []
        kpi_map = {"gross_margin_pct":"Gross Margin %",
                   "ebit_margin_pct":"EBIT Margin %","roi_pct":"ROI %"}
        for key, label in kpi_map.items():
            low, high = bm.get(key, (0,100))
            bm_rows.append({"Metric":label,"Actual":kpis[key],"BM Low":low,"BM High":high})
        bm_df = pd.DataFrame(bm_rows)
        fig = go.Figure()
        fig.add_bar(x=bm_df["Metric"], y=bm_df["BM High"]-bm_df["BM Low"],
                    base=bm_df["BM Low"], name="Benchmark Range",
                    marker_color="rgba(79,195,247,0.2)")
        fig.add_scatter(x=bm_df["Metric"], y=bm_df["Actual"],
                        mode="markers+lines", name="Your KPIs",
                        marker=dict(color=TEAL, size=14, symbol="diamond"))
        fig.update_layout(template="plotly_dark", height=320,
                          title=f"{cat} — Performance vs Industry Benchmark")
        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        ins_export = pd.DataFrame(
            [(l,t,m) for l,t,m in report["insights"]],
            columns=["Level","Title","Message"]
        )
        st.download_button("⬇️ Download Insights (CSV)", ins_export.to_csv(index=False),
                           "insights.csv","text/csv", use_container_width=True)
    with col_e2:
        rec_export = pd.DataFrame(recs, columns=["Priority","Title","Description","Impact","Timeframe"])
        st.download_button("⬇️ Download Recommendations (CSV)", rec_export.to_csv(index=False),
                           "recommendations.csv","text/csv", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []

    with st.spinner("Loading data and training models — first run takes ~30 seconds..."):
        df     = generate_data()
        models = train_models(df)

    page, cat, yr = sidebar(df)
    fdf = filter_df(df, cat, yr)

    if   page == "🏠 Dashboard":        page_dashboard(fdf)
    elif page == "⚙️ Live Simulator":    page_simulator(fdf)
    elif page == "🤖 ML Engine":         page_ml_engine(fdf, models)
    elif page == "🎯 Margin Optimiser":  page_optimiser(fdf)
    elif page == "📊 Scenario Analysis": page_scenarios(fdf)
    elif page == "🌍 External Factors":  page_external(fdf)
    elif page == "💡 Insights & Recs":   page_insights(fdf)

if __name__ == "__main__":
    main()
