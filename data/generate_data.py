"""
Data Generator — Profitability Simulator
Generates 3 years of weekly retail data (156 weeks) across 5 product categories,
incorporating realistic external macro factors, price elasticity, seasonality,
promotions, and cost dynamics.

External factors modelled:
  - Inflation rate (CPI proxy)
  - Fuel / Energy price index
  - Consumer Confidence Index
  - Competitor price index
  - EUR/USD exchange rate
  - Interest rate (ECB proxy)
  - Unemployment rate
  - GDP growth rate
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# CATEGORY DEFINITIONS
# ──────────────────────────────────────────────────────────────
CATEGORIES = {
    "Electronics": {
        "base_price": 350,
        "base_cost": 195,
        "base_units": 800,
        "price_elasticity": -1.8,
        "fixed_cost_base": 28_000,
        "fuel_sensitivity": 0.12,   # how much fuel price affects cost
        "inflation_cost_pass": 0.75, # fraction of inflation passed to cost
    },
    "Apparel": {
        "base_price": 65,
        "base_cost": 26,
        "base_units": 2_500,
        "price_elasticity": -2.1,
        "fixed_cost_base": 12_000,
        "fuel_sensitivity": 0.08,
        "inflation_cost_pass": 0.60,
    },
    "Food & Beverage": {
        "base_price": 18,
        "base_cost": 11,
        "base_units": 8_000,
        "price_elasticity": -0.7,
        "fixed_cost_base": 9_000,
        "fuel_sensitivity": 0.18,   # logistics-heavy
        "inflation_cost_pass": 0.90,
    },
    "Home & Garden": {
        "base_price": 95,
        "base_cost": 46,
        "base_units": 1_200,
        "price_elasticity": -1.5,
        "fixed_cost_base": 14_000,
        "fuel_sensitivity": 0.10,
        "inflation_cost_pass": 0.65,
    },
    "Sports": {
        "base_price": 120,
        "base_cost": 55,
        "base_units": 1_500,
        "price_elasticity": -1.9,
        "fixed_cost_base": 16_000,
        "fuel_sensitivity": 0.09,
        "inflation_cost_pass": 0.68,
    },
}

# Month-level seasonality multipliers
SEASONALITY = {
    1: 0.78, 2: 0.76, 3: 0.85, 4: 0.90,
    5: 0.94, 6: 0.97, 7: 0.88, 8: 0.92,
    9: 0.98, 10: 1.08, 11: 1.30, 12: 1.50,
}

# Category-specific seasonal overrides
CATEGORY_SEASON = {
    "Home & Garden": {4: 1.35, 5: 1.50, 6: 1.45, 3: 1.15},
    "Sports":        {1: 1.10, 2: 1.12, 7: 1.20, 8: 1.18},
    "Food & Beverage": {12: 1.30, 11: 1.15},
}


def _simulate_external_factors(n: int, seed: int = 42) -> pd.DataFrame:
    """Simulate macro external factors for n weeks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Inflation: peaks around week 35–55 (replicates 2022 energy crisis spike)
    inflation = (
        0.028
        + 0.055 * np.exp(-((t - 45) ** 2) / (2 * 25 ** 2))
        + rng.normal(0, 0.003, n)
    ).clip(0.01, 0.12)

    # Fuel/energy index: base 100, volatile cycle
    fuel_base = 100 + 28 * np.sin(t / 26 * np.pi) + np.cumsum(rng.normal(0, 0.4, n))
    fuel_index = (fuel_base + rng.normal(0, 3, n)).clip(70, 180)

    # Consumer Confidence: inversely related to inflation & rates
    conf_base = 58 + 12 * np.sin(t / 52 * np.pi) - inflation * 120
    consumer_confidence = (conf_base + rng.normal(0, 1.5, n)).clip(30, 90)

    # Competitor price index: drifts with inflation, mean-reverts
    comp_drift = np.cumsum(rng.normal(0.03, 0.4, n))
    competitor_price_index = (100 + comp_drift + inflation * 80).clip(85, 130)

    # EUR/USD exchange rate: random walk around 1.08
    fx_walk = np.cumsum(rng.normal(0, 0.003, n))
    exchange_rate = (1.08 + 0.08 * np.sin(t / 78 * np.pi) + fx_walk).clip(0.92, 1.25)

    # Interest rate (ECB path): near zero early, rising after week 40
    interest_rate = (
        0.005 + 0.038 * (1 / (1 + np.exp(-(t - 50) / 10)))
        + rng.normal(0, 0.001, n)
    ).clip(0, 0.05)

    # Unemployment: slowly declining
    unemployment = (5.8 - 1.2 * (t / n) + rng.normal(0, 0.18, n)).clip(3.5, 8.0)

    # GDP growth: quarterly seasonal wave
    gdp_growth = (
        0.022 + 0.012 * np.sin(t / 13 * np.pi)
        - interest_rate * 0.8
        + rng.normal(0, 0.004, n)
    )

    return pd.DataFrame({
        "inflation_rate_pct":        np.round(inflation * 100, 2),
        "fuel_price_index":          np.round(fuel_index, 2),
        "consumer_confidence_index": np.round(consumer_confidence, 2),
        "competitor_price_index":    np.round(competitor_price_index, 2),
        "exchange_rate_eur_usd":     np.round(exchange_rate, 4),
        "interest_rate_pct":         np.round(interest_rate * 100, 3),
        "unemployment_rate_pct":     np.round(unemployment, 2),
        "gdp_growth_pct":            np.round(gdp_growth * 100, 3),
    })


def generate_retail_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate 156 weeks × 5 categories of retail data.
    Returns a flat DataFrame with all financial + external columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-03", periods=156, freq="W-MON")
    n = len(dates)

    ext = _simulate_external_factors(n, seed=seed)
    trend = np.linspace(0.94, 1.12, n)  # slight long-run growth

    records = []

    for week_idx, date in enumerate(dates):
        month = date.month
        base_seasonal = SEASONALITY[month]

        inflation = ext.loc[week_idx, "inflation_rate_pct"] / 100
        fuel_idx   = ext.loc[week_idx, "fuel_price_index"]
        confidence = ext.loc[week_idx, "consumer_confidence_index"]
        fx_rate    = ext.loc[week_idx, "exchange_rate_eur_usd"]
        interest   = ext.loc[week_idx, "interest_rate_pct"] / 100

        for cat_name, cp in CATEGORIES.items():
            # Category seasonal override
            cat_season_override = CATEGORY_SEASON.get(cat_name, {})
            seasonal = cat_season_override.get(month, base_seasonal)

            # ── PRICING ─────────────────────────────────────────
            price_drift = rng.normal(1.0, 0.04)
            # Prices track inflation partially
            unit_price = cp["base_price"] * price_drift * (1 + inflation * 0.55)

            # Promotion logic: Black Friday (Nov), Christmas (Dec), random 15%
            promo_flag = int(month in (11, 12) or rng.random() < 0.15)
            discount_rate = (
                rng.uniform(0.10, 0.35) if promo_flag
                else rng.uniform(0.00, 0.08)
            )
            effective_price = unit_price * (1 - discount_rate)

            # ── COSTS ────────────────────────────────────────────
            fuel_cost_impact = cp["fuel_sensitivity"] * (fuel_idx - 100) / 100
            infl_cost_impact = cp["inflation_cost_pass"] * inflation
            fx_impact = 0.04 * (fx_rate - 1.08)  # import cost sensitivity
            unit_cost = (
                cp["base_cost"]
                * (1 + infl_cost_impact + fuel_cost_impact + fx_impact)
                * rng.normal(1.0, 0.025)
            )

            # Fixed costs also grow with inflation
            fixed_costs = cp["fixed_cost_base"] * (1 + inflation * 0.35) * rng.normal(1, 0.03)

            # ── DEMAND ───────────────────────────────────────────
            # Price elasticity effect
            price_ratio = effective_price / cp["base_price"]
            price_effect = price_ratio ** cp["price_elasticity"]

            # Confidence multiplier (discretionary spend)
            conf_effect = (confidence / 58) ** 0.35

            # Interest rate dampens big-ticket purchases
            interest_effect = 1 - interest * (1.8 if cat_name == "Electronics" else 0.8)

            # GDP growth effect
            gdp = ext.loc[week_idx, "gdp_growth_pct"] / 100
            gdp_effect = 1 + gdp * 0.4

            demand = (
                cp["base_units"]
                * seasonal
                * trend[week_idx]
                * price_effect
                * conf_effect
                * interest_effect
                * gdp_effect
                * (1 + 0.22 * promo_flag)   # promo lift
                * rng.normal(1.0, 0.07)
            )
            demand_units = max(30, int(demand))

            # ── FINANCIALS ───────────────────────────────────────
            revenue      = effective_price * demand_units
            cogs         = unit_cost * demand_units
            gross_profit = revenue - cogs
            gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0.0
            ebit         = gross_profit - fixed_costs
            ebit_margin  = (ebit / revenue * 100) if revenue > 0 else 0.0
            total_invest = cogs + fixed_costs
            roi          = (ebit / total_invest * 100) if total_invest > 0 else 0.0
            contrib_margin = ((effective_price - unit_cost) / effective_price * 100) if effective_price > 0 else 0.0
            unit_contrib = effective_price - unit_cost
            breakeven    = fixed_costs / unit_contrib if unit_contrib > 0 else np.nan

            records.append({
                # Time
                "date":          date,
                "week_num":      week_idx + 1,
                "year":          date.year,
                "quarter":       f"Q{date.quarter}",
                "month":         month,
                "month_name":    date.strftime("%b"),
                # Category
                "category":      cat_name,
                # Pricing
                "unit_price":          round(unit_price, 2),
                "discount_rate_pct":   round(discount_rate * 100, 1),
                "effective_price":     round(effective_price, 2),
                "promo_flag":          promo_flag,
                # Costs
                "unit_cost":           round(unit_cost, 2),
                "fixed_costs":         round(fixed_costs, 2),
                # Demand
                "demand_units":        demand_units,
                # Revenue & P&L
                "revenue":             round(revenue, 2),
                "cogs":                round(cogs, 2),
                "gross_profit":        round(gross_profit, 2),
                "gross_margin_pct":    round(gross_margin, 2),
                "ebit":                round(ebit, 2),
                "ebit_margin_pct":     round(ebit_margin, 2),
                "roi_pct":             round(roi, 2),
                "contribution_margin_pct": round(contrib_margin, 2),
                "breakeven_units":     round(breakeven, 0) if not np.isnan(breakeven) else np.nan,
                # External factors
                "inflation_rate_pct":        ext.loc[week_idx, "inflation_rate_pct"],
                "fuel_price_index":          ext.loc[week_idx, "fuel_price_index"],
                "consumer_confidence_index": ext.loc[week_idx, "consumer_confidence_index"],
                "competitor_price_index":    ext.loc[week_idx, "competitor_price_index"],
                "exchange_rate_eur_usd":     ext.loc[week_idx, "exchange_rate_eur_usd"],
                "interest_rate_pct":         ext.loc[week_idx, "interest_rate_pct"],
                "unemployment_rate_pct":     ext.loc[week_idx, "unemployment_rate_pct"],
                "gdp_growth_pct":            ext.loc[week_idx, "gdp_growth_pct"],
            })

    return pd.DataFrame(records)


def generate_external_factors_summary(seed: int = 42) -> pd.DataFrame:
    """Return standalone weekly external factors table (for display / export)."""
    dates = pd.date_range(start="2022-01-03", periods=156, freq="W-MON")
    ext = _simulate_external_factors(156, seed=seed)
    ext.insert(0, "date", dates)
    ext.insert(1, "week_num", range(1, 157))
    return ext


# ──────────────────────────────────────────────────────────────
# ENTRYPOINT — run directly to generate CSV files
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    print("Generating retail_data.csv …")
    df_retail = generate_retail_data()
    df_retail.to_csv(os.path.join(out_dir, "retail_data.csv"), index=False)
    print(f"  ✓  {len(df_retail):,} rows  ×  {len(df_retail.columns)} columns")

    print("Generating external_factors.csv …")
    df_ext = generate_external_factors_summary()
    df_ext.to_csv(os.path.join(out_dir, "external_factors.csv"), index=False)
    print(f"  ✓  {len(df_ext):,} rows  ×  {len(df_ext.columns)} columns")

    print("\nDone. Files saved in:", out_dir)
