"""
Interpreter — Profitability Simulator
Auto-generates data-driven insights, risk flags, and strategic recommendations
based on current KPIs, trend analysis, and industry benchmarks.

Industry benchmarks used (retail sector):
  Gross Margin:  28 – 45 %   (food 20–30, apparel 40–60, electronics 25–35)
  EBIT Margin:   4  – 12 %
  ROI:           10 – 25 %
  Contribution:  30 – 55 %
  Breakeven:     should be < 70% of actual volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────────────────────
# BENCHMARKS
# ──────────────────────────────────────────────────────────────
INDUSTRY_BENCHMARKS = {
    "Electronics":     {"gross_margin": (25, 40), "ebit_margin": (3, 10), "roi": (8, 20)},
    "Apparel":         {"gross_margin": (40, 60), "ebit_margin": (5, 15), "roi": (12, 28)},
    "Food & Beverage": {"gross_margin": (20, 32), "ebit_margin": (2, 8),  "roi": (6, 16)},
    "Home & Garden":   {"gross_margin": (32, 48), "ebit_margin": (4, 12), "roi": (10, 22)},
    "Sports":          {"gross_margin": (35, 52), "ebit_margin": (4, 14), "roi": (10, 24)},
}

RISK_THRESHOLDS = {
    "ebit_negative":         0,
    "gross_margin_low":      20,
    "roi_low":               8,
    "discount_high":         25,
    "breakeven_danger_pct":  85,
    "ebit_margin_warning":   4,
}


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────
@dataclass
class Insight:
    level: str        # "critical" | "warning" | "positive" | "info"
    category: str     # KPI area
    title: str
    body: str
    action: str = ""

    @property
    def icon(self) -> str:
        return {"critical": "🔴", "warning": "🟡", "positive": "🟢", "info": "🔵"}[self.level]


@dataclass
class InterpretationReport:
    kpi_summary: dict
    insights: List[Insight] = field(default_factory=list)
    recommendations: List[dict] = field(default_factory=list)
    risk_score: float = 0.0        # 0 = low risk … 100 = critical
    health_label: str = "Unknown"

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for ins in self.insights:
            rows.append({
                "Level":    ins.level.upper(),
                "Area":     ins.category,
                "Title":    ins.title,
                "Detail":   ins.body,
                "Action":   ins.action,
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# INTERPRETER
# ──────────────────────────────────────────────────────────────
class Interpreter:
    """
    Given current KPIs and historical DataFrame, produces a structured
    interpretation report with risks, positives, and actionable recommendations.
    """

    def __init__(self, category: str = "all"):
        self.category = category

    # ── MAIN ENTRY POINT ─────────────────────────────────────
    def interpret(
        self,
        kpis: dict,
        hist_df: pd.DataFrame,
        sim_result: dict | None = None,
        elasticity: float | None = None,
    ) -> InterpretationReport:
        """
        kpis       : dict from calc_profitability()
        hist_df    : full retail DataFrame for trend context
        sim_result : optional Monte Carlo results dict
        elasticity : price elasticity coefficient
        """
        report = InterpretationReport(kpi_summary=kpis)
        bench = INDUSTRY_BENCHMARKS.get(self.category, INDUSTRY_BENCHMARKS["Apparel"])

        self._check_ebit(kpis, report)
        self._check_gross_margin(kpis, bench, report)
        self._check_roi(kpis, bench, report)
        self._check_breakeven(kpis, report)
        self._check_discount(kpis, report)
        self._check_contribution_margin(kpis, report)
        self._check_external_risks(hist_df, report)

        if elasticity is not None:
            self._check_elasticity(elasticity, kpis, report)

        if sim_result is not None:
            self._check_monte_carlo_risk(sim_result, report)

        # Trend analysis
        self._trend_analysis(hist_df, report)

        # Build recommendations
        report.recommendations = self._build_recommendations(kpis, bench, elasticity, hist_df)

        # Risk score: 0–100
        n_crit = sum(1 for i in report.insights if i.level == "critical")
        n_warn = sum(1 for i in report.insights if i.level == "warning")
        report.risk_score = min(100, n_crit * 30 + n_warn * 10)
        report.health_label = (
            "Critical" if report.risk_score >= 60 else
            "At Risk"  if report.risk_score >= 30 else
            "Healthy"
        )
        return report

    # ── INDIVIDUAL CHECKS ─────────────────────────────────────
    def _check_ebit(self, kpis: dict, report: InterpretationReport):
        ebit = kpis["ebit"]
        em   = kpis["ebit_margin_pct"]
        if ebit < 0:
            report.insights.append(Insight(
                level="critical", category="Profitability",
                title="Negative EBIT — Operating at a Loss",
                body=f"Current EBIT is €{ebit:,.0f} ({em:.1f}% margin). "
                     f"Fixed costs of €{kpis['fixed_costs']:,.0f} cannot be covered by gross profit.",
                action="Immediate action required: raise prices, reduce fixed costs, or increase volume."
            ))
        elif em < RISK_THRESHOLDS["ebit_margin_warning"]:
            report.insights.append(Insight(
                level="warning", category="Profitability",
                title=f"EBIT Margin Thin ({em:.1f}%)",
                body=f"EBIT margin of {em:.1f}% is below the {RISK_THRESHOLDS['ebit_margin_warning']}% "
                     f"warning threshold. A small cost increase or volume drop could turn profit negative.",
                action="Reduce discretionary fixed costs; review discount depth."
            ))
        else:
            report.insights.append(Insight(
                level="positive", category="Profitability",
                title=f"Healthy EBIT Margin ({em:.1f}%)",
                body=f"EBIT of €{ebit:,.0f} at {em:.1f}% margin is solid.",
                action=""
            ))

    def _check_gross_margin(self, kpis: dict, bench: dict, report: InterpretationReport):
        gm = kpis["gross_margin_pct"]
        lo, hi = bench["gross_margin"]
        if gm < lo:
            report.insights.append(Insight(
                level="warning", category="Cost Structure",
                title=f"Gross Margin Below Industry Benchmark ({gm:.1f}% vs {lo}–{hi}%)",
                body=f"Gross margin of {gm:.1f}% is below the retail benchmark of {lo}–{hi}% for this category. "
                     f"COGS = €{kpis['cogs']:,.0f} vs Revenue = €{kpis['revenue']:,.0f}.",
                action="Renegotiate supplier costs, reduce procurement expenses, or increase price."
            ))
        elif gm > hi:
            report.insights.append(Insight(
                level="positive", category="Cost Structure",
                title=f"Gross Margin Above Benchmark ({gm:.1f}%)",
                body=f"Gross margin of {gm:.1f}% exceeds the typical {lo}–{hi}% range — strong pricing power.",
                action="Monitor competitor pricing; avoid becoming uncompetitive."
            ))
        else:
            report.insights.append(Insight(
                level="positive", category="Cost Structure",
                title=f"Gross Margin Within Benchmark ({gm:.1f}%)",
                body=f"Gross margin is within the healthy {lo}–{hi}% range.", action=""
            ))

    def _check_roi(self, kpis: dict, bench: dict, report: InterpretationReport):
        roi = kpis["roi_pct"]
        lo, hi = bench["roi"]
        if roi < lo:
            report.insights.append(Insight(
                level="warning", category="Returns",
                title=f"ROI Below Benchmark ({roi:.1f}% vs {lo}–{hi}%)",
                body=f"Return on investment of {roi:.1f}% falls short of the {lo}–{hi}% benchmark. "
                     "Capital efficiency is weak.",
                action="Reduce working capital tied up in COGS; optimise inventory turns."
            ))
        else:
            report.insights.append(Insight(
                level="positive", category="Returns",
                title=f"ROI Healthy ({roi:.1f}%)",
                body=f"ROI of {roi:.1f}% is within or above the {lo}–{hi}% benchmark.", action=""
            ))

    def _check_breakeven(self, kpis: dict, report: InterpretationReport):
        bev = kpis.get("breakeven_units", float("inf"))
        actual = kpis["demand_units"]
        if bev == float("inf") or actual == 0:
            return
        pct = bev / actual * 100
        if pct > RISK_THRESHOLDS["breakeven_danger_pct"]:
            report.insights.append(Insight(
                level="critical", category="Volume Risk",
                title=f"Breakeven at {pct:.0f}% of Current Volume — Very High Risk",
                body=f"You need {bev:,.0f} units to break even, but currently selling {actual:,.0f}. "
                     f"Only a {100-pct:.0f}% volume decline separates you from losses.",
                action="Increase demand buffer through marketing, expand distribution, or reduce fixed cost base."
            ))
        elif pct > 70:
            report.insights.append(Insight(
                level="warning", category="Volume Risk",
                title=f"Breakeven at {pct:.0f}% of Volume — Moderate Risk",
                body=f"Breakeven point is {bev:,.0f} units ({pct:.0f}% of current {actual:,.0f} units).",
                action="Build demand resilience; diversify customer base."
            ))
        else:
            report.insights.append(Insight(
                level="positive", category="Volume Risk",
                title=f"Comfortable Volume Buffer ({100-pct:.0f}% above breakeven)",
                body=f"Breakeven at {bev:,.0f} units — well below current sales of {actual:,.0f}.",
                action=""
            ))

    def _check_discount(self, kpis: dict, report: InterpretationReport):
        disc = kpis["discount_rate_pct"]
        if disc > RISK_THRESHOLDS["discount_high"]:
            revenue_lost = kpis["unit_price"] * kpis["demand_units"] - kpis["revenue"]
            report.insights.append(Insight(
                level="warning", category="Pricing",
                title=f"Deep Discount ({disc:.1f}%) — Margin Erosion Risk",
                body=f"At {disc:.1f}% discount, you are forgoing ~€{revenue_lost:,.0f} in potential revenue. "
                     "High discounts can anchor customer price expectations downward.",
                action="Use targeted/segmented discounts instead of blanket promotions; add value bundles."
            ))
        elif disc > 15:
            report.insights.append(Insight(
                level="info", category="Pricing",
                title=f"Moderate Discount ({disc:.1f}%)",
                body=f"Discount level is moderate. Monitor whether demand uplift justifies the margin trade-off.",
                action=""
            ))

    def _check_contribution_margin(self, kpis: dict, report: InterpretationReport):
        cm = kpis["contribution_margin_pct"]
        if cm < 25:
            report.insights.append(Insight(
                level="warning", category="Unit Economics",
                title=f"Low Contribution Margin ({cm:.1f}%)",
                body=f"Each unit contributes only {cm:.1f}% of its price toward covering fixed costs. "
                     "Low contribution margins limit operational leverage.",
                action="Reduce variable costs or increase effective selling price."
            ))

    def _check_external_risks(self, df: pd.DataFrame, report: InterpretationReport):
        if df.empty:
            return
        recent = df.tail(4)  # last 4 weeks
        avg_inflation = recent["inflation_rate_pct"].mean()
        avg_fuel      = recent["fuel_price_index"].mean()
        avg_conf      = recent["consumer_confidence_index"].mean()

        if avg_inflation > 6:
            report.insights.append(Insight(
                level="warning", category="External Risk",
                title=f"High Inflation Environment ({avg_inflation:.1f}%)",
                body="Recent inflation is elevated, which is likely driving up procurement and fixed costs. "
                     "Real margins may be eroding faster than nominal figures suggest.",
                action="Lock in supplier contracts; review pricing quarterly; hedge fuel exposure."
            ))
        if avg_fuel > 125:
            report.insights.append(Insight(
                level="warning", category="External Risk",
                title=f"Elevated Fuel/Energy Costs (Index: {avg_fuel:.0f})",
                body="Fuel prices are significantly above baseline, increasing logistics and energy costs.",
                action="Audit supply chain for fuel-sensitive costs; explore local sourcing."
            ))
        if avg_conf < 45:
            report.insights.append(Insight(
                level="warning", category="Demand Risk",
                title=f"Low Consumer Confidence ({avg_conf:.1f})",
                body="Weak consumer sentiment typically precedes demand contraction, "
                     "particularly for discretionary categories.",
                action="Increase value-for-money messaging; consider bundle pricing to protect volume."
            ))

    def _check_elasticity(self, elasticity: float, kpis: dict, report: InterpretationReport):
        if elasticity < -2.5:
            report.insights.append(Insight(
                level="warning", category="Price Sensitivity",
                title=f"High Price Sensitivity (ε = {elasticity:.2f})",
                body=f"Demand is highly elastic: a 10% price increase reduces demand by ~{abs(elasticity)*10:.0f}%. "
                     "Pricing power is limited.",
                action="Focus on cost reduction rather than price increases; differentiate product to reduce elasticity."
            ))
        elif -1.0 < elasticity < 0:
            report.insights.append(Insight(
                level="positive", category="Price Sensitivity",
                title=f"Inelastic Demand (ε = {elasticity:.2f}) — Pricing Power",
                body=f"Demand is relatively inelastic. A 10% price increase reduces demand by only ~{abs(elasticity)*10:.0f}%. "
                     "This indicates strong brand or necessity-driven purchasing.",
                action="Incrementally test price increases; capture additional margin without major volume loss."
            ))

    def _check_monte_carlo_risk(self, sim_result: dict, report: InterpretationReport):
        p5_ebit = sim_result.get("p5_ebit", 0)
        p50_ebit = sim_result.get("p50_ebit", 0)
        prob_loss = sim_result.get("prob_loss_pct", 0)
        if prob_loss > 30:
            report.insights.append(Insight(
                level="critical", category="Simulation Risk",
                title=f"High Probability of Loss in Simulation ({prob_loss:.0f}%)",
                body=f"Monte Carlo simulations show a {prob_loss:.0f}% probability of negative EBIT "
                     f"under realistic uncertainty. P5 EBIT: €{p5_ebit:,.0f}.",
                action="Increase margin buffer; reduce fixed cost exposure before scaling volume."
            ))
        elif prob_loss > 10:
            report.insights.append(Insight(
                level="warning", category="Simulation Risk",
                title=f"Moderate Loss Probability ({prob_loss:.0f}%)",
                body=f"{prob_loss:.0f}% of simulations yield negative EBIT. Median EBIT: €{p50_ebit:,.0f}.",
                action="Monitor cost volatility closely; maintain contingency reserves."
            ))

    def _trend_analysis(self, df: pd.DataFrame, report: InterpretationReport):
        if df.empty or len(df) < 8:
            return
        cat_df = df if self.category == "all" else df[df["category"] == self.category]
        if cat_df.empty:
            return
        weekly = cat_df.groupby("date")[["ebit", "gross_margin_pct", "revenue"]].mean().tail(12)
        if len(weekly) < 4:
            return

        # EBIT trend (linear slope)
        x = np.arange(len(weekly))
        slope_ebit = np.polyfit(x, weekly["ebit"].values, 1)[0]
        slope_gm   = np.polyfit(x, weekly["gross_margin_pct"].values, 1)[0]

        if slope_ebit < -100:
            report.insights.append(Insight(
                level="warning", category="Trend",
                title="Declining EBIT Trend Over Last 12 Weeks",
                body=f"EBIT has been declining at ~€{abs(slope_ebit):.0f}/week on average.",
                action="Investigate root cause: rising costs, falling volumes, or pricing pressure."
            ))
        elif slope_ebit > 100:
            report.insights.append(Insight(
                level="positive", category="Trend",
                title="Improving EBIT Trend",
                body=f"EBIT is growing at ~€{slope_ebit:.0f}/week — positive momentum.", action=""
            ))

        if slope_gm < -0.1:
            report.insights.append(Insight(
                level="warning", category="Trend",
                title="Gross Margin Compression Trend",
                body=f"Gross margin has been declining by ~{abs(slope_gm):.2f}pp per week.",
                action="Review cost trends and pricing strategy immediately."
            ))

    def _build_recommendations(
        self,
        kpis: dict,
        bench: dict,
        elasticity: float | None,
        df: pd.DataFrame,
    ) -> list[dict]:
        recs = []

        # 1. Pricing recommendation
        gm = kpis["gross_margin_pct"]
        em = kpis["ebit_margin_pct"]
        if elasticity is not None and elasticity > -1.5 and gm < bench["gross_margin"][1]:
            price_upside = (bench["gross_margin"][1] - gm) / 100 * kpis["revenue"]
            recs.append({
                "Priority": "High",
                "Category": "Pricing Strategy",
                "Recommendation": "Implement a 5–8% selective price increase on low-elasticity SKUs.",
                "Estimated Impact": f"+€{price_upside * 0.4:,.0f} EBIT improvement (indicative)",
                "Rationale": f"Price elasticity of {elasticity:.2f} suggests demand will not decline proportionately.",
                "Timeframe": "0–4 weeks",
            })

        # 2. Discount reduction
        if kpis["discount_rate_pct"] > 20:
            disc_saving = kpis["unit_price"] * kpis["demand_units"] * 0.05
            recs.append({
                "Priority": "High",
                "Category": "Discount Management",
                "Recommendation": "Reduce blanket discounts by 5pp; shift to targeted loyalty discounts.",
                "Estimated Impact": f"+€{disc_saving:,.0f} revenue uplift",
                "Rationale": "Deep discounts erode margin faster than they drive incremental demand at scale.",
                "Timeframe": "0–2 weeks",
            })

        # 3. Fixed cost reduction
        if em < bench["ebit_margin"][0]:
            fc_target = kpis["fixed_costs"] * 0.10
            recs.append({
                "Priority": "Medium",
                "Category": "Fixed Cost Optimisation",
                "Recommendation": "Audit fixed overhead for 10% reduction opportunity.",
                "Estimated Impact": f"+€{fc_target:,.0f} EBIT improvement",
                "Rationale": f"Current EBIT margin of {em:.1f}% is below the {bench['ebit_margin'][0]}% benchmark floor.",
                "Timeframe": "4–8 weeks",
            })

        # 4. External factor hedging
        if not df.empty:
            avg_infl = df.tail(4)["inflation_rate_pct"].mean()
            if avg_infl > 5:
                recs.append({
                    "Priority": "Medium",
                    "Category": "Cost Hedging",
                    "Recommendation": "Negotiate 6–12 month fixed-price supplier contracts.",
                    "Estimated Impact": "Reduce cost volatility exposure by 15–25%",
                    "Rationale": f"Inflation averaging {avg_infl:.1f}% is materially eroding unit costs.",
                    "Timeframe": "4–12 weeks",
                })

        # 5. Volume / demand
        bev = kpis.get("breakeven_units", float("inf"))
        actual = kpis["demand_units"]
        if bev != float("inf") and actual > 0 and bev / actual > 0.75:
            recs.append({
                "Priority": "High",
                "Category": "Volume Growth",
                "Recommendation": "Invest in demand generation to increase volume by 15–20%.",
                "Estimated Impact": f"+€{kpis['contribution_margin_pct'] / 100 * kpis['effective_price'] * actual * 0.15:,.0f} contribution gain",
                "Rationale": f"Breakeven at {bev/actual*100:.0f}% of volume — headroom is dangerously thin.",
                "Timeframe": "4–12 weeks",
            })

        # 6. Mix optimisation
        if not df.empty and "category" in df.columns:
            top_cat = (
                df.groupby("category")["ebit_margin_pct"].mean()
                .sort_values(ascending=False)
                .index[0]
            )
            recs.append({
                "Priority": "Low",
                "Category": "Product Mix",
                "Recommendation": f"Allocate more shelf/marketing spend to '{top_cat}' — highest EBIT margin category.",
                "Estimated Impact": "Portfolio margin improvement of 0.5–2pp",
                "Rationale": "Shifting mix toward higher-margin categories improves blended profitability without price changes.",
                "Timeframe": "4–8 weeks",
            })

        return recs
