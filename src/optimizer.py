"""
Optimiser — Profitability Simulator
Implements:
  - MarginOptimiser   : scipy SLSQP constrained optimisation of price/cost levers
  - ScenarioEngine    : Monte Carlo simulation of profit distributions
  - SensitivityEngine : One-at-a-time and grid sensitivity analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────
# PROFITABILITY CALCULATOR  (pure function, no ML dependency)
# ──────────────────────────────────────────────────────────────
def calc_profitability(
    unit_price: float,
    unit_cost: float,
    discount_rate_pct: float,
    demand_units: int,
    fixed_costs: float,
    price_elasticity: float = -1.5,
    base_price: Optional[float] = None,
    base_units: Optional[int] = None,
) -> dict:
    """
    Core financial calculator.  Computes a full P&L given levers.
    If base_price / base_units provided, adjusts demand via elasticity.
    """
    base_price = base_price or unit_price
    base_units = base_units or demand_units

    eff_price = unit_price * (1 - discount_rate_pct / 100)

    # Adjust demand via price elasticity
    price_ratio = eff_price / max(base_price, 0.01)
    adj_units = int(base_units * (price_ratio ** price_elasticity))
    adj_units = max(1, adj_units)

    revenue      = eff_price * adj_units
    cogs         = unit_cost * adj_units
    gross_profit = revenue - cogs
    gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0.0
    ebit         = gross_profit - fixed_costs
    ebit_margin  = (ebit / revenue * 100) if revenue > 0 else 0.0
    total_invest = cogs + fixed_costs
    roi          = (ebit / total_invest * 100) if total_invest > 0 else 0.0
    contrib_margin = ((eff_price - unit_cost) / eff_price * 100) if eff_price > 0 else 0.0
    unit_contrib = eff_price - unit_cost
    breakeven    = fixed_costs / unit_contrib if unit_contrib > 0 else float("inf")

    return {
        "unit_price":          round(unit_price, 4),
        "effective_price":     round(eff_price, 4),
        "unit_cost":           round(unit_cost, 4),
        "discount_rate_pct":   round(discount_rate_pct, 2),
        "demand_units":        adj_units,
        "revenue":             round(revenue, 2),
        "cogs":                round(cogs, 2),
        "gross_profit":        round(gross_profit, 2),
        "gross_margin_pct":    round(gross_margin, 2),
        "fixed_costs":         round(fixed_costs, 2),
        "ebit":                round(ebit, 2),
        "ebit_margin_pct":     round(ebit_margin, 2),
        "roi_pct":             round(roi, 2),
        "contribution_margin_pct": round(contrib_margin, 2),
        "breakeven_units":     round(breakeven, 0),
    }


# ──────────────────────────────────────────────────────────────
# MARGIN OPTIMISER
# ──────────────────────────────────────────────────────────────
class MarginOptimiser:
    """
    Finds optimal (unit_price, discount_rate) that maximises EBIT
    subject to business constraints.

    Constraints (configurable):
      - Minimum gross margin floor
      - Price band (±% from current price)
      - Max discount cap
      - Minimum demand volume
    """

    def __init__(
        self,
        unit_cost: float,
        base_price: float,
        base_units: int,
        fixed_costs: float,
        price_elasticity: float = -1.5,
        min_gross_margin_pct: float = 20.0,
        max_discount_pct: float = 35.0,
        price_range_pct: float = 40.0,
        min_demand: int = 100,
    ):
        self.unit_cost           = unit_cost
        self.base_price          = base_price
        self.base_units          = base_units
        self.fixed_costs         = fixed_costs
        self.price_elasticity    = price_elasticity
        self.min_gross_margin    = min_gross_margin_pct
        self.max_discount        = max_discount_pct
        self.price_range         = price_range_pct
        self.min_demand          = min_demand

    def _objective(self, x):
        price, discount = x
        r = calc_profitability(
            unit_price=price,
            unit_cost=self.unit_cost,
            discount_rate_pct=discount,
            demand_units=self.base_units,
            fixed_costs=self.fixed_costs,
            price_elasticity=self.price_elasticity,
            base_price=self.base_price,
            base_units=self.base_units,
        )
        return -r["ebit"]   # minimise negative EBIT = maximise EBIT

    def _constraints(self):
        return [
            # Gross margin floor
            {
                "type": "ineq",
                "fun": lambda x: self._eval(x)["gross_margin_pct"] - self.min_gross_margin,
            },
            # Demand minimum
            {
                "type": "ineq",
                "fun": lambda x: self._eval(x)["demand_units"] - self.min_demand,
            },
            # Price > 1.05 × unit_cost  (avoid selling below cost with margin)
            {
                "type": "ineq",
                "fun": lambda x: x[0] * (1 - x[1] / 100) - self.unit_cost * 1.05,
            },
        ]

    def _eval(self, x):
        return calc_profitability(
            unit_price=x[0],
            unit_cost=self.unit_cost,
            discount_rate_pct=x[1],
            demand_units=self.base_units,
            fixed_costs=self.fixed_costs,
            price_elasticity=self.price_elasticity,
            base_price=self.base_price,
            base_units=self.base_units,
        )

    def optimise(self, method: str = "SLSQP") -> dict:
        """Run constrained optimisation. Returns optimal params + financials."""
        lo_price = self.base_price * (1 - self.price_range / 100)
        hi_price = self.base_price * (1 + self.price_range / 100)
        lo_disc  = 0.0
        hi_disc  = self.max_discount

        bounds = [(lo_price, hi_price), (lo_disc, hi_disc)]
        x0     = [self.base_price, 10.0]   # starting guess

        result = minimize(
            self._objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=self._constraints(),
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            # Fallback: global differential evolution
            result_de = differential_evolution(
                self._objective,
                bounds=bounds,
                maxiter=200,
                seed=42,
                tol=1e-6,
            )
            result = result_de

        opt_price, opt_disc = result.x
        opt_financials = self._eval(result.x)

        return {
            "optimal_price":    round(opt_price, 2),
            "optimal_discount": round(opt_disc, 2),
            "converged":        result.success,
            **opt_financials,
        }

    def optimise_multiple_objectives(self) -> pd.DataFrame:
        """
        Sweep across objectives: max EBIT, max Revenue, max ROI, max Gross Margin.
        Returns a comparison DataFrame.
        """
        objectives = {
            "Maximise EBIT":         lambda x: -self._eval(x)["ebit"],
            "Maximise Revenue":      lambda x: -self._eval(x)["revenue"],
            "Maximise ROI":          lambda x: -self._eval(x)["roi_pct"],
            "Maximise Gross Margin": lambda x: -self._eval(x)["gross_margin_pct"],
        }
        rows = []
        for obj_name, obj_fn in objectives.items():
            lo_p = self.base_price * 0.6
            hi_p = self.base_price * 1.5
            r = minimize(obj_fn, [self.base_price, 10.0], method="SLSQP",
                         bounds=[(lo_p, hi_p), (0, self.max_discount)],
                         options={"maxiter": 300})
            fin = self._eval(r.x)
            rows.append({
                "Objective": obj_name,
                "Price (€)":     round(r.x[0], 2),
                "Discount (%)":  round(r.x[1], 2),
                "Revenue (€)":   fin["revenue"],
                "Gross Margin %": fin["gross_margin_pct"],
                "EBIT (€)":      fin["ebit"],
                "ROI %":         fin["roi_pct"],
                "Units Sold":    fin["demand_units"],
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# SCENARIO ENGINE  (Monte Carlo)
# ──────────────────────────────────────────────────────────────
class ScenarioEngine:
    """
    Monte Carlo simulation: sample uncertainty in price, cost, demand,
    and external factors to produce a distribution of EBIT outcomes.
    """

    def __init__(
        self,
        base_params: dict,
        n_simulations: int = 5_000,
        seed: int = 42,
    ):
        self.base   = base_params
        self.n_sims = n_simulations
        self.rng    = np.random.default_rng(seed)

    def run(
        self,
        price_std_pct:      float = 5.0,
        cost_std_pct:       float = 4.0,
        demand_std_pct:     float = 10.0,
        discount_std_pct:   float = 3.0,
        inflation_shock_pct: float = 2.0,
    ) -> pd.DataFrame:
        """
        Runs Monte Carlo and returns a DataFrame with per-simulation results.
        """
        n = self.n_sims
        bp = self.base

        prices    = bp["unit_price"]    * (1 + self.rng.normal(0, price_std_pct   / 100, n))
        costs     = bp["unit_cost"]     * (1 + self.rng.normal(0, cost_std_pct    / 100, n))
        discounts = np.clip(bp["discount_rate_pct"] + self.rng.normal(0, discount_std_pct, n), 0, 40)
        demand    = (bp["demand_units"] * (1 + self.rng.normal(0, demand_std_pct  / 100, n))).astype(int)
        infl_cost = 1 + self.rng.normal(0, inflation_shock_pct / 100, n)
        costs     = costs * infl_cost

        records = []
        for i in range(n):
            eff = prices[i] * (1 - discounts[i] / 100)
            rev = eff * demand[i]
            cog = costs[i] * demand[i]
            gp  = rev - cog
            eb  = gp - bp["fixed_costs"]
            records.append({
                "simulation":     i,
                "unit_price":     round(prices[i], 2),
                "unit_cost":      round(costs[i], 2),
                "discount_pct":   round(discounts[i], 2),
                "demand_units":   demand[i],
                "revenue":        round(rev, 2),
                "gross_profit":   round(gp, 2),
                "gross_margin_pct": round(gp / rev * 100 if rev > 0 else 0, 2),
                "ebit":           round(eb, 2),
                "ebit_margin_pct": round(eb / rev * 100 if rev > 0 else 0, 2),
                "roi_pct":        round(eb / (cog + bp["fixed_costs"]) * 100 if (cog + bp["fixed_costs"]) > 0 else 0, 2),
            })
        return pd.DataFrame(records)

    def percentile_table(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        """Return P5 / P25 / P50 / P75 / P95 summary statistics."""
        pcts = [5, 25, 50, 75, 95]
        cols = ["revenue", "gross_profit", "ebit", "gross_margin_pct", "roi_pct"]
        rows = []
        for p in pcts:
            row = {"Percentile": f"P{p}"}
            for c in cols:
                row[c] = round(np.percentile(sim_df[c], p), 2)
            rows.append(row)
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# SENSITIVITY ENGINE
# ──────────────────────────────────────────────────────────────
class SensitivityEngine:
    """
    One-at-a-time (OAT) and two-way grid sensitivity analysis.
    """

    def __init__(self, base_params: dict):
        self.base = base_params

    def one_way(
        self,
        variable: str,
        lo_pct: float = -30.0,
        hi_pct: float = 30.0,
        steps: int = 13,
    ) -> pd.DataFrame:
        """Vary one variable ±30% (default) and record EBIT, margin, ROI."""
        bp     = self.base
        values = np.linspace(bp[variable] * (1 + lo_pct / 100),
                             bp[variable] * (1 + hi_pct / 100), steps)
        rows = []
        for v in values:
            p = {**bp, variable: v}
            r = calc_profitability(**p)
            rows.append({
                variable:          round(v, 4),
                "pct_change":      round((v / bp[variable] - 1) * 100, 1),
                "ebit":            r["ebit"],
                "gross_margin_pct": r["gross_margin_pct"],
                "roi_pct":         r["roi_pct"],
                "revenue":         r["revenue"],
                "demand_units":    r["demand_units"],
            })
        return pd.DataFrame(rows)

    def two_way(
        self,
        var1: str,
        var2: str,
        target: str = "ebit",
        lo_pct: float = -25.0,
        hi_pct: float = 25.0,
        steps: int = 9,
    ) -> pd.DataFrame:
        """Grid of var1 × var2 → target metric."""
        bp = self.base
        v1_vals = np.linspace(bp[var1] * (1 + lo_pct / 100),
                              bp[var1] * (1 + hi_pct / 100), steps)
        v2_vals = np.linspace(bp[var2] * (1 + lo_pct / 100),
                              bp[var2] * (1 + hi_pct / 100), steps)
        rows = []
        for v1 in v1_vals:
            for v2 in v2_vals:
                p = {**bp, var1: v1, var2: v2}
                r = calc_profitability(**p)
                rows.append({var1: round(v1, 4), var2: round(v2, 4), target: r[target]})
        return pd.DataFrame(rows)
