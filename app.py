"""
Profitability Simulator & Margin Optimisation Model
MS5131 — Major Business Analytics Project
University of Galway, AY 25-26

Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.generate_data import generate_retail_data, generate_external_factors_summary
from src.optimizer import calc_profitability, MarginOptimiser, ScenarioEngine, SensitivityEngine
from src.interpreter import Interpreter, INDUSTRY_BENCHMARKS

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Profitability Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# THEME & CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp { background: #0b1622; color: #c8d8e8; }
section[data-testid="stSidebar"] { background: #07111f !important; border-right: 1px solid #162030; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#0f1f30,#0a1628);
    border: 1px solid #1a2d42;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #6688aa !important; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 700; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

.kpi-positive  { color: #00d4aa !important; }
.kpi-negative  { color: #ff6b6b !important; }
.kpi-neutral   { color: #4fc3f7 !important; }

.insight-card {
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 4px solid;
    font-size: 13px;
    line-height: 1.6;
}
.insight-critical { background:#200e0e; border-color:#ff4444; }
.insight-warning  { background:#1e1a08; border-color:#ffbb33; }
.insight-positive { background:#091e15; border-color:#00d4aa; }
.insight-info     { background:#0a1825; border-color:#4fc3f7; }

.rec-card {
    background: #0f1f30;
    border: 1px solid #1a2d42;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.rec-high   { border-left: 4px solid #ff6b6b; }
.rec-medium { border-left: 4px solid #ffbb33; }
.rec-low    { border-left: 4px solid #4fc3f7;  }

.section-header {
    font-size: 11px; color: #445566;
    text-transform: uppercase; letter-spacing: .10em;
    border-bottom: 1px solid #162030;
    padding-bottom: 6px; margin-bottom: 16px; margin-top: 4px;
}
.stSlider > div > div { background: #1a2d42 !important; }
.stSelectbox > div, .stMultiSelect > div { background: #0f1f30 !important; }
div.stButton > button {
    background: #0f1f30; border: 1px solid #2a4060;
    color: #c8d8e8; border-radius: 6px; font-size: 12px;
    transition: all .2s;
}
div.stButton > button:hover { background: #1a3550; border-color: #00d4aa; color: #00d4aa; }
.stDataFrame { background: #0a1525; }
div[data-testid="stExpander"] { background: #0f1f30; border: 1px solid #1a2d42; border-radius: 8px; }
hr { border-color: #162030 !important; }
</style>
""", unsafe_allow_html=True)

PALETTE = {
    "teal":    "#00d4aa", "blue":  "#4fc3f7", "orange": "#ff9f43",
    "red":     "#ff6b6b", "purple":"#c084fc",  "yellow": "#ffd93d",
    "bg":      "#0b1622", "bg2":   "#0f1f30",
}
CAT_COLORS = {
    "Electronics":   PALETTE["teal"],
    "Apparel":       PALETTE["blue"],
    "Food & Beverage": PALETTE["orange"],
    "Home & Garden": PALETTE["purple"],
    "Sports":        PALETTE["yellow"],
}
PLOTLY_THEME = dict(
    paper_bgcolor="#0b1622", plot_bgcolor="#0b1622",
    font_color="#8899aa", font_family="Inter",
    xaxis=dict(gridcolor="#162030", zerolinecolor="#162030"),
    yaxis=dict(gridcolor="#162030", zerolinecolor="#162030"),
)


# ──────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating dataset…")
def load_data():
    df     = generate_retail_data(seed=42)
    df_ext = generate_external_factors_summary(seed=42)
    return df, df_ext


@st.cache_resource(show_spinner="Training ML models (first run only)…")
def load_models(df: pd.DataFrame):
    from src.models import train_all_models
    return train_all_models(df)


def fmt_eur(val: float, compact=False) -> str:
    if compact:
        if abs(val) >= 1_000_000: return f"€{val/1_000_000:.2f}M"
        if abs(val) >= 1_000:     return f"€{val/1_000:.1f}k"
    return f"€{val:,.0f}"


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
def sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style='padding:16px 0 4px'>
          <div style='font-size:17px;font-weight:700;color:#c8d8e8;letter-spacing:.03em'>📊 Profit Simulator</div>
          <div style='font-size:10px;color:#445566;margin-top:2px'>MS5131 · University of Galway</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠  Dashboard", "⚙️  Live Simulator", "🤖  ML Engine",
             "🎯  Margin Optimiser", "📊  Scenario Analysis",
             "🌍  External Factors", "💡  Insights & Recs"],
            label_visibility="collapsed",
        )

        st.markdown("<hr><div class='section-header'>Global Filters</div>", unsafe_allow_html=True)
        cats = ["All"] + sorted(df["category"].unique().tolist())
        sel_cat = st.selectbox("Category", cats)
        years   = ["All"] + sorted(df["year"].unique().tolist())
        sel_yr  = st.selectbox("Year", years)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:10px;color:#334455;line-height:1.6'>"
            "Data: 3 years · 156 weeks · 5 categories<br>"
            "Models: XGBoost · Ridge · SARIMA<br>"
            "Optimiser: SLSQP · Differential Evolution"
            "</div>", unsafe_allow_html=True
        )

    return page.split("  ")[1], sel_cat, sel_yr


def filter_df(df, cat, yr):
    if cat != "All": df = df[df["category"] == cat]
    if yr  != "All": df = df[df["year"] == int(yr)]
    return df


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
def page_dashboard(df: pd.DataFrame, df_filt: pd.DataFrame):
    st.markdown("## 🏠 Dashboard")
    st.markdown("*Aggregated KPIs and trends across selected filters.*")

    # KPIs
    total_rev  = df_filt["revenue"].sum()
    total_ebit = df_filt["ebit"].sum()
    avg_gm     = df_filt["gross_margin_pct"].mean()
    avg_roi    = df_filt["roi_pct"].mean()
    avg_em     = df_filt["ebit_margin_pct"].mean()
    total_units= df_filt["demand_units"].sum()

    prev = df_filt.head(len(df_filt) // 2)
    curr = df_filt.tail(len(df_filt) // 2)
    rev_delta  = (curr["revenue"].sum() / max(prev["revenue"].sum(), 1) - 1) * 100
    ebit_delta = (curr["ebit"].sum()    / max(abs(prev["ebit"].sum()), 1) - 1) * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue",    fmt_eur(total_rev, True),  f"{rev_delta:+.1f}% vs prior half")
    c2.metric("Total EBIT",       fmt_eur(total_ebit, True), f"{ebit_delta:+.1f}% vs prior half")
    c3.metric("Avg Gross Margin", f"{avg_gm:.1f}%")
    c4.metric("Avg EBIT Margin",  f"{avg_em:.1f}%")
    c5.metric("Avg ROI",          f"{avg_roi:.1f}%")
    c6.metric("Total Units Sold", f"{total_units:,}")

    st.markdown("---")

    # ── Row 1: Revenue over time + Category mix ──────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        weekly = df_filt.groupby(["date", "category"])[["revenue", "ebit"]].sum().reset_index()
        fig = go.Figure()
        for cat, color in CAT_COLORS.items():
            cat_data = weekly[weekly["category"] == cat]
            if cat_data.empty: continue
            fig.add_trace(go.Scatter(
                x=cat_data["date"], y=cat_data["revenue"],
                mode="lines", name=cat, line=dict(color=color, width=1.8),
                stackgroup="one", hovertemplate=f"<b>{cat}</b><br>€%{{y:,.0f}}<extra></extra>"
            ))
        fig.update_layout(**PLOTLY_THEME, title="Weekly Revenue by Category (Stacked)",
                          height=320, margin=dict(t=40, b=20, l=0, r=0),
                          legend=dict(bgcolor="#0f1f30", font_size=11))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_rev = df_filt.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat_rev, values="revenue", names="category",
                     color="category",
                     color_discrete_map=CAT_COLORS,
                     hole=0.55)
        fig.update_traces(textfont_size=11, hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<extra></extra>")
        fig.update_layout(**PLOTLY_THEME, title="Revenue Mix", height=320,
                          margin=dict(t=40, b=0, l=0, r=0),
                          showlegend=True, legend=dict(font_size=10, bgcolor="#0f1f30"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: EBIT margin trend + Gross Margin benchmark ────
    col3, col4 = st.columns(2)
    with col3:
        monthly = df_filt.groupby(["year", "month_name", "month"])[["ebit_margin_pct","gross_margin_pct"]].mean().reset_index()
        monthly = monthly.sort_values(["year", "month"])
        monthly["label"] = monthly["month_name"] + " " + monthly["year"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["label"], y=monthly["gross_margin_pct"],
                                 mode="lines+markers", name="Gross Margin %",
                                 line=dict(color=PALETTE["teal"], width=2),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=monthly["label"], y=monthly["ebit_margin_pct"],
                                 mode="lines+markers", name="EBIT Margin %",
                                 line=dict(color=PALETTE["blue"], width=2, dash="dot"),
                                 marker=dict(size=4)))
        fig.add_hline(y=0, line_color=PALETTE["red"], line_dash="dash", line_width=1)
        fig.update_layout(**PLOTLY_THEME, title="Monthly Margin Trends", height=300,
                          margin=dict(t=40, b=20, l=0, r=0), hovermode="x unified",
                          legend=dict(bgcolor="#0f1f30", font_size=11),
                          xaxis_tickangle=-45, xaxis_nticks=12)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        cat_kpis = df_filt.groupby("category")[["gross_margin_pct","ebit_margin_pct","roi_pct"]].mean().reset_index()
        fig = go.Figure()
        for col, color, name in [
            ("gross_margin_pct", PALETTE["teal"],   "Gross Margin %"),
            ("ebit_margin_pct",  PALETTE["blue"],   "EBIT Margin %"),
            ("roi_pct",          PALETTE["orange"], "ROI %"),
        ]:
            fig.add_trace(go.Bar(x=cat_kpis["category"], y=cat_kpis[col],
                                 name=name, marker_color=color))
        fig.update_layout(**PLOTLY_THEME, title="KPIs by Category", height=300,
                          barmode="group", margin=dict(t=40, b=20, l=0, r=0),
                          legend=dict(bgcolor="#0f1f30", font_size=11))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Heatmap ────────────────────────────────────────
    st.markdown("##### EBIT Margin Heatmap — Month × Category")
    pivot = df_filt.pivot_table(index="month_name", columns="category",
                                values="ebit_margin_pct", aggfunc="mean")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex([m for m in month_order if m in pivot.index])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#3d0000"],[0.4,"#1a2d42"],[0.7,"#005a4a"],[1,"#00d4aa"]],
        text=pivot.round(1).astype(str).values,
        texttemplate="%{text}%",
        hovertemplate="<b>%{x}</b> · %{y}<br>EBIT Margin: %{z:.1f}%<extra></extra>",
        colorbar=dict(tickfont_color="#8899aa"),
    ))
    fig.update_layout(**PLOTLY_THEME, height=260, margin=dict(t=10, b=20, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — LIVE SIMULATOR
# ══════════════════════════════════════════════════════════════
def page_simulator(df: pd.DataFrame):
    st.markdown("## ⚙️ Live Profitability Simulator")
    st.markdown("*Adjust levers in real-time and see immediate P&L impact.*")

    categories = sorted(df["category"].unique().tolist())

    # ── Controls ─────────────────────────────────────────────
    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.markdown("<div class='section-header'>Category</div>", unsafe_allow_html=True)
        cat = st.selectbox("Select Category", categories, key="sim_cat", label_visibility="collapsed")
        cat_df = df[df["category"] == cat]
        defaults = cat_df[["unit_price","unit_cost","discount_rate_pct","demand_units","fixed_costs"]].median()

        st.markdown("<div class='section-header'>Pricing Levers</div>", unsafe_allow_html=True)
        unit_price    = st.slider("Unit Price (€)",      float(defaults["unit_price"]   * 0.5),
                                  float(defaults["unit_price"]   * 2.0), float(defaults["unit_price"]),   step=0.5)
        discount_rate = st.slider("Discount Rate (%)",   0.0, 40.0, float(defaults["discount_rate_pct"]), step=0.5)

        st.markdown("<div class='section-header'>Cost Structure</div>", unsafe_allow_html=True)
        unit_cost     = st.slider("Unit Cost (€)",       float(defaults["unit_cost"]    * 0.5),
                                  float(unit_price * 0.95), float(defaults["unit_cost"]), step=0.5)
        fixed_costs   = st.slider("Fixed Costs (€)",     5_000.0, 200_000.0, float(defaults["fixed_costs"]), step=1_000.0)
        proc_change   = st.slider("Procurement Δ (%)",   -30.0, 50.0, 0.0, step=1.0,
                                  help="Simulates supplier cost increase/decrease")
        adj_unit_cost = unit_cost * (1 + proc_change / 100)

        st.markdown("<div class='section-header'>Demand</div>", unsafe_allow_html=True)
        demand_units  = st.slider("Sales Volume (units)", 100, 30_000, int(defaults["demand_units"]), step=100)

        st.markdown("<div class='section-header'>Elasticity</div>", unsafe_allow_html=True)
        elasticity    = st.slider("Price Elasticity",    -3.5, -0.1, -1.5, step=0.1,
                                  help="How sensitive demand is to price changes")

        # Scenario saving
        st.markdown("<div class='section-header'>Save Scenario</div>", unsafe_allow_html=True)
        sc_name = st.text_input("Scenario name", placeholder="e.g. Q4 Base Case")
        if st.button("💾 Save Scenario") and sc_name.strip():
            if "scenarios" not in st.session_state:
                st.session_state.scenarios = []
            r_save = calc_profitability(unit_price, adj_unit_cost, discount_rate,
                                        demand_units, fixed_costs, elasticity,
                                        unit_price, demand_units)
            st.session_state.scenarios.append({
                "name": sc_name, "category": cat,
                "unit_price": unit_price, "unit_cost": adj_unit_cost,
                "discount": discount_rate, "demand": demand_units, "fixed_costs": fixed_costs,
                **r_save
            })
            st.success(f"Saved: {sc_name}")

    # ── Live calculation ─────────────────────────────────────
    results = calc_profitability(
        unit_price, adj_unit_cost, discount_rate,
        demand_units, fixed_costs, elasticity,
        unit_price, demand_units
    )
    base_results = calc_profitability(
        float(defaults["unit_price"]), float(defaults["unit_cost"]),
        float(defaults["discount_rate_pct"]), int(defaults["demand_units"]),
        float(defaults["fixed_costs"]), elasticity,
        float(defaults["unit_price"]), int(defaults["demand_units"])
    )

    with col_results:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenue",       fmt_eur(results["revenue"], True),
                  f"{(results['revenue']/max(base_results['revenue'],1)-1)*100:+.1f}%")
        k2.metric("Gross Margin",  f"{results['gross_margin_pct']:.1f}%",
                  f"{results['gross_margin_pct']-base_results['gross_margin_pct']:+.1f}pp")
        k3.metric("EBIT",          fmt_eur(results["ebit"], True),
                  f"{(results['ebit']-base_results['ebit']):+,.0f}")
        k4.metric("ROI",           f"{results['roi_pct']:.1f}%",
                  f"{results['roi_pct']-base_results['roi_pct']:+.1f}pp")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Effective Price", f"€{results['effective_price']:.2f}")
        k6.metric("Contribution Margin", f"{results['contribution_margin_pct']:.1f}%")
        k7.metric("Breakeven Units", f"{results['breakeven_units']:,.0f}")
        k8.metric("Actual vs BEV", f"{results['demand_units']/max(results['breakeven_units'],1)*100:.0f}%",
                  help="Actual units as % of breakeven — higher is safer")

        # P&L waterfall
        pnl_labels = ["Revenue", "(-) COGS", "Gross Profit", "(-) Fixed Costs", "EBIT"]
        pnl_values = [
            results["revenue"], -results["cogs"],
            results["gross_profit"], -results["fixed_costs"], results["ebit"]
        ]
        pnl_colors = [PALETTE["teal"], PALETTE["red"], PALETTE["blue"], PALETTE["orange"],
                      PALETTE["teal"] if results["ebit"] >= 0 else PALETTE["red"]]
        measure = ["absolute", "relative", "total", "relative", "total"]

        fig = go.Figure(go.Waterfall(
            x=pnl_labels, y=pnl_values, measure=measure,
            connector=dict(line=dict(color=PALETTE["bg2"], width=2)),
            decreasing=dict(marker_color=PALETTE["red"]),
            increasing=dict(marker_color=PALETTE["teal"]),
            totals=dict(marker_color=PALETTE["blue"]),
            text=[fmt_eur(v) for v in pnl_values],
            textposition="outside",
            textfont=dict(size=11, color="#c8d8e8"),
        ))
        fig.update_layout(**PLOTLY_THEME, title="P&L Waterfall", height=300,
                          margin=dict(t=40, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # Breakeven chart
        vol_range = np.arange(max(100, demand_units // 3), demand_units * 2, max(50, demand_units // 40))
        bev_data = []
        for v in vol_range:
            r2 = calc_profitability(unit_price, adj_unit_cost, discount_rate, v,
                                    fixed_costs, elasticity, unit_price, demand_units)
            bev_data.append({"units": v, "ebit": r2["ebit"], "revenue": r2["revenue"], "cogs_fc": r2["cogs"] + r2["fixed_costs"]})
        bev_df = pd.DataFrame(bev_data)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bev_df["units"], y=bev_df["revenue"],
                                  name="Revenue", line=dict(color=PALETTE["teal"], width=2)))
        fig2.add_trace(go.Scatter(x=bev_df["units"], y=bev_df["cogs_fc"],
                                  name="Total Costs", line=dict(color=PALETTE["red"], width=2)))
        fig2.add_vline(x=results["breakeven_units"], line_color=PALETTE["yellow"],
                       line_dash="dash", annotation_text=f"BEV: {results['breakeven_units']:,.0f}",
                       annotation_font_color=PALETTE["yellow"])
        fig2.add_vline(x=demand_units, line_color=PALETTE["blue"],
                       line_dash="dot", annotation_text=f"Current: {demand_units:,}",
                       annotation_font_color=PALETTE["blue"])
        fig2.update_layout(**PLOTLY_THEME, title="Breakeven Analysis", height=280,
                           margin=dict(t=40, b=20, l=0, r=0),
                           xaxis_title="Units Sold", yaxis_title="€",
                           legend=dict(bgcolor="#0f1f30", font_size=11))
        st.plotly_chart(fig2, use_container_width=True)

        # Sensitivity tornado
        vars_to_test = {
            "unit_price": unit_price, "unit_cost": adj_unit_cost,
            "discount_rate_pct": discount_rate, "demand_units": demand_units, "fixed_costs": fixed_costs
        }
        base_ebit = results["ebit"]
        tornado_rows = []
        for var, val in vars_to_test.items():
            for chg, label in [(0.1, "+10%"), (-0.1, "-10%")]:
                ptest = {**vars_to_test, var: val * (1 + chg),
                         "price_elasticity": elasticity, "base_price": unit_price, "base_units": demand_units}
                r = calc_profitability(**ptest)
                tornado_rows.append({"Variable": var.replace("_", " ").title(),
                                     "Change": label, "EBIT Delta": r["ebit"] - base_ebit})
        tornado_df = pd.DataFrame(tornado_rows)
        pivot_t = tornado_df.pivot(index="Variable", columns="Change", values="EBIT Delta").fillna(0)
        pivot_t["range"] = pivot_t["+10%"] - pivot_t["-10%"]
        pivot_t = pivot_t.sort_values("range", ascending=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(y=pivot_t.index, x=pivot_t["-10%"], name="-10%",
                              orientation="h", marker_color=PALETTE["red"]))
        fig3.add_trace(go.Bar(y=pivot_t.index, x=pivot_t["+10%"], name="+10%",
                              orientation="h", marker_color=PALETTE["teal"]))
        fig3.add_vline(x=0, line_color="#445566")
        fig3.update_layout(**PLOTLY_THEME, title="Sensitivity Tornado — EBIT Impact of ±10% Variable Change",
                           barmode="relative", height=260, margin=dict(t=40, b=20, l=0, r=0),
                           legend=dict(bgcolor="#0f1f30", font_size=11))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — ML ENGINE
# ══════════════════════════════════════════════════════════════
def page_ml_engine(df: pd.DataFrame, models: dict):
    st.markdown("## 🤖 ML Forecasting Engine")

    demand_mdl  = models["demand"]
    elast_mdl   = models["elasticity"]
    ts_rev_mdl  = models["ts_revenue"]
    ts_ebit_mdl = models["ts_ebit"]

    tab1, tab2, tab3 = st.tabs(["📈 Demand Forecast", "📐 Price Elasticity", "🔮 Time Series Forecast"])

    # ── Demand forecast tab ───────────────────────────────────
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c2:
            st.markdown("**Model Metrics (Hold-out)**")
            metrics = demand_mdl.metrics
            st.metric("R²",   f"{metrics['R2']:.3f}")
            st.metric("MAE",  f"{metrics['MAE']:,.1f} units")
            st.metric("RMSE", f"{metrics['RMSE']:,.1f} units")
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")

        with c1:
            preds = demand_mdl.predict(df)
            pred_df = df.copy()
            pred_df["predicted"] = preds
            weekly_pred = (
                pred_df.groupby(["date", "category"])
                .agg(actual=("demand_units", "sum"), predicted=("predicted", "sum"))
                .reset_index()
            )
            cat_pred = st.selectbox("Category", sorted(df["category"].unique()), key="ml_cat")
            sub = weekly_pred[weekly_pred["category"] == cat_pred]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["actual"],
                                     name="Actual", line=dict(color=PALETTE["teal"], width=1.5)))
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["predicted"],
                                     name="XGBoost Predicted", line=dict(color=PALETTE["orange"], width=1.5, dash="dot")))
            fig.update_layout(**PLOTLY_THEME, title=f"Demand Forecast — {cat_pred}",
                              height=320, margin=dict(t=40, b=20, l=0, r=0),
                              legend=dict(bgcolor="#0f1f30"))
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        fi = demand_mdl.feature_importance.head(12)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker=dict(color=fi.values, colorscale=[[0, "#1a2d42"], [1, "#00d4aa"]]),
        ))
        fig_fi.update_layout(**PLOTLY_THEME, title="XGBoost Feature Importance (Top 12)",
                             height=350, margin=dict(t=40, b=20, l=0, r=0),
                             xaxis_title="Importance Score")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Price elasticity tab ──────────────────────────────────
    with tab2:
        elast_df = elast_mdl.summary()
        st.markdown("**Price Elasticity Estimates by Category (Log-Log OLS)**")
        styled = elast_df.copy()
        # Colour-code elasticity column manually using st.dataframe column config
st.dataframe(
    styled,
    use_container_width=True,
    column_config={
        "elasticity": st.column_config.NumberColumn(
            "Price Elasticity",
            help="How sensitive demand is to price changes",
            format="%.2f",
        ),
        "std_err": st.column_config.NumberColumn("Std Error", format="%.3f"),
        "p_value": st.column_config.NumberColumn("P-Value", format="%.4f"),
        "r_squared": st.column_config.NumberColumn("R²", format="%.2f"),
        "promo_lift": st.column_config.NumberColumn("Promo Lift", format="%.2f"),
    }
)

        # Elasticity arc chart
        fig_e = go.Figure()
        colors = [PALETTE["red"] if abs(e) > 2 else PALETTE["orange"] if abs(e) > 1.5
                  else PALETTE["teal"] for e in elast_df["elasticity"]]
        fig_e.add_trace(go.Bar(
            x=elast_df["category"], y=elast_df["elasticity"],
            marker_color=colors,
            error_y=dict(type="data", array=elast_df["std_err"], color="#445566"),
            text=elast_df["elasticity"].round(3).astype(str),
            textposition="outside",
        ))
        fig_e.add_hline(y=-1, line_color=PALETTE["yellow"], line_dash="dash",
                        annotation_text="Elastic threshold (ε=-1)")
        fig_e.update_layout(**PLOTLY_THEME, title="Price Elasticity by Category (±1 SE)",
                            height=340, margin=dict(t=40, b=20, l=0, r=0),
                            yaxis_title="Elasticity Coefficient")
        st.plotly_chart(fig_e, use_container_width=True)

    # ── Time series forecast tab ──────────────────────────────
    with tab3:
        cat_ts = st.selectbox("Category for Forecast", sorted(df["category"].unique()), key="ts_cat")
        col_ts1, col_ts2 = st.columns(2)
        for col, ts_mdl, title, color in [
            (col_ts1, ts_rev_mdl, "Revenue Forecast (12 Weeks)", PALETTE["teal"]),
            (col_ts2, ts_ebit_mdl, "EBIT Forecast (12 Weeks)", PALETTE["blue"]),
        ]:
            with col:
                try:
                    fcast = ts_mdl.forecast(cat_ts)
                    hist  = df[df["category"] == cat_ts].groupby("date")[
                        "revenue" if "Revenue" in title else "ebit"
                    ].sum().reset_index()
                    hist.columns = ["date", "value"]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist["date"], y=hist["value"],
                        name="Historical", line=dict(color=color, width=1.8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=fcast["date"], y=fcast["forecast"],
                        name="Forecast", line=dict(color=PALETTE["yellow"], width=2, dash="dot")
                    ))
                    fig.add_traces([
                        go.Scatter(x=pd.concat([fcast["date"], fcast["date"][::-1]]),
                                   y=pd.concat([fcast["upper_ci"], fcast["lower_ci"][::-1]]),
                                   fill="toself", fillcolor="rgba(255,187,51,0.10)",
                                   line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True)
                    ])
                    fig.update_layout(**PLOTLY_THEME, title=f"{title} — {cat_ts}",
                                      height=310, margin=dict(t=40, b=20, l=0, r=0),
                                      legend=dict(bgcolor="#0f1f30", font_size=11))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as ex:
                    st.warning(f"Forecast unavailable: {ex}")


# ══════════════════════════════════════════════════════════════
# PAGE 4 — MARGIN OPTIMISER
# ══════════════════════════════════════════════════════════════
def page_optimiser(df: pd.DataFrame):
    st.markdown("## 🎯 Margin Optimiser")
    st.markdown("*SLSQP constrained optimisation to find the profit-maximising price/discount strategy.*")

    cats = sorted(df["category"].unique().tolist())
    cat  = st.selectbox("Category", cats, key="opt_cat")
    cat_df = df[df["category"] == cat]
    defaults = cat_df[["unit_price","unit_cost","fixed_costs","demand_units"]].median()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Current Parameters</div>", unsafe_allow_html=True)
        base_price  = st.number_input("Base Price (€)",     value=float(defaults["unit_price"]), step=1.0)
        unit_cost   = st.number_input("Unit Cost (€)",      value=float(defaults["unit_cost"]),  step=1.0)
        fixed_costs = st.number_input("Fixed Costs (€)",    value=float(defaults["fixed_costs"]), step=500.0)
        base_units  = st.number_input("Base Volume (units)",value=int(defaults["demand_units"]),  step=100)
        elasticity  = st.slider("Price Elasticity", -3.5, -0.1, -1.5, step=0.1, key="opt_elast")

    with c2:
        st.markdown("<div class='section-header'>Constraint Settings</div>", unsafe_allow_html=True)
        min_gm      = st.slider("Min Gross Margin Floor (%)", 5.0, 50.0, 20.0)
        max_disc    = st.slider("Max Discount Cap (%)",       0.0, 50.0, 35.0)
        price_range = st.slider("Price Band (±%)",            5.0, 60.0, 40.0)
        min_demand  = st.number_input("Min Acceptable Demand", value=max(100, int(base_units * 0.3)), step=50)

    if st.button("🚀 Run Optimisation"):
        with st.spinner("Optimising…"):
            opt = MarginOptimiser(
                unit_cost=unit_cost, base_price=base_price, base_units=int(base_units),
                fixed_costs=fixed_costs, price_elasticity=elasticity,
                min_gross_margin_pct=min_gm, max_discount_pct=max_disc,
                price_range_pct=price_range, min_demand=int(min_demand),
            )
            result = opt.optimise()
            multi  = opt.optimise_multiple_objectives()

        # Base vs Optimised
        base_calc = calc_profitability(base_price, unit_cost, 5.0, int(base_units),
                                       fixed_costs, elasticity, base_price, int(base_units))
        st.markdown("---")
        st.markdown("#### ✅ Optimal vs. Base Case")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Optimal Price",    f"€{result['optimal_price']:.2f}",
                  f"{(result['optimal_price']/base_price-1)*100:+.1f}% vs base")
        m2.metric("Optimal Discount", f"{result['optimal_discount']:.1f}%")
        m3.metric("EBIT",             fmt_eur(result["ebit"], True),
                  f"{result['ebit']-base_calc['ebit']:+,.0f} vs base")
        m4.metric("Gross Margin",     f"{result['gross_margin_pct']:.1f}%",
                  f"{result['gross_margin_pct']-base_calc['gross_margin_pct']:+.1f}pp")
        m5.metric("ROI",              f"{result['roi_pct']:.1f}%",
                  f"{result['roi_pct']-base_calc['roi_pct']:+.1f}pp")

        # Multi-objective comparison
        st.markdown("#### Multi-Objective Trade-off Analysis")
        # Format columns manually before display — avoids Styler crash on Streamlit Cloud
            fmt_multi = multi.copy()
            for col in fmt_multi.columns:
                if "price" in col.lower() or "€" in col.lower():
                    try:
                        fmt_multi[col] = fmt_multi[col].apply(
                            lambda x: f"€{x:.2f}" if isinstance(x, (int, float)) else x
                        )
                    except Exception:
                        pass
                elif "%" in col or "margin" in col.lower() or "ebit" in col.lower() or "roi" in col.lower():
                    try:
                        fmt_multi[col] = fmt_multi[col].apply(
                            lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
                        )
                    except Exception:
                        pass
            
            st.dataframe(fmt_multi, use_container_width=True)

        # Price–EBIT surface
        prices    = np.linspace(base_price * 0.6, base_price * 1.5, 30)
        discounts = np.linspace(0, max_disc, 20)
        z_ebit = np.zeros((len(discounts), len(prices)))
        for i, d in enumerate(discounts):
            for j, p in enumerate(prices):
                r = calc_profitability(p, unit_cost, d, int(base_units),
                                       fixed_costs, elasticity, base_price, int(base_units))
                z_ebit[i, j] = r["ebit"]

        fig = go.Figure(go.Contour(
            z=z_ebit, x=prices.round(1), y=discounts.round(1),
            colorscale=[[0, "#3d0000"], [0.4, "#1a2d42"], [0.7, "#005a4a"], [1, "#00d4aa"]],
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title="EBIT (€)", tickfont_color="#8899aa"),
            hovertemplate="Price: €%{x}<br>Discount: %{y}%<br>EBIT: €%{z:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[result["optimal_price"]], y=[result["optimal_discount"]],
            mode="markers+text", marker=dict(color=PALETTE["yellow"], size=14, symbol="star"),
            text=["Optimal"], textposition="top center", textfont=dict(color=PALETTE["yellow"]),
        ))
        fig.update_layout(**PLOTLY_THEME, title="EBIT Contour — Price × Discount Space",
                          xaxis_title="Unit Price (€)", yaxis_title="Discount Rate (%)",
                          height=400, margin=dict(t=50, b=30, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_scenarios(df: pd.DataFrame):
    st.markdown("## 📊 Scenario Analysis")

    tab1, tab2 = st.tabs(["🎲 Monte Carlo Simulation", "📋 Saved Scenarios"])

    with tab1:
        st.markdown("*Simulate uncertainty in costs, prices, and demand to model profit distributions.*")
        cats = sorted(df["category"].unique())
        cat  = st.selectbox("Category", cats, key="mc_cat")
        cat_df = df[df["category"] == cat]
        d = cat_df[["unit_price","unit_cost","discount_rate_pct","demand_units","fixed_costs"]].median()

        c1, c2 = st.columns(2)
        with c1:
            n_sims      = st.slider("Simulations", 1_000, 20_000, 5_000, step=1_000)
            price_std   = st.slider("Price Uncertainty (σ%)",    1.0, 15.0, 5.0)
            cost_std    = st.slider("Cost Uncertainty (σ%)",     1.0, 15.0, 4.0)
        with c2:
            demand_std  = st.slider("Demand Uncertainty (σ%)",   2.0, 25.0, 10.0)
            infl_shock  = st.slider("Inflation Shock (σ%)",      0.5, 8.0,  2.0)

        base_params = {
            "unit_price": float(d["unit_price"]), "unit_cost": float(d["unit_cost"]),
            "discount_rate_pct": float(d["discount_rate_pct"]),
            "demand_units": int(d["demand_units"]), "fixed_costs": float(d["fixed_costs"]),
        }

        if st.button("▶️  Run Monte Carlo"):
            with st.spinner(f"Running {n_sims:,} simulations…"):
                eng = ScenarioEngine(base_params, n_simulations=n_sims)
                sim_df = eng.run(price_std, cost_std, demand_std, 3.0, infl_shock)
                pct_tbl = eng.percentile_table(sim_df)

            prob_loss = (sim_df["ebit"] < 0).mean() * 100
            p5_ebit   = np.percentile(sim_df["ebit"], 5)
            p50_ebit  = np.percentile(sim_df["ebit"], 50)
            p95_ebit  = np.percentile(sim_df["ebit"], 95)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prob. of Loss",   f"{prob_loss:.1f}%",
                      delta_color="inverse")
            m2.metric("P5 EBIT",  fmt_eur(p5_ebit, True))
            m3.metric("Median EBIT", fmt_eur(p50_ebit, True))
            m4.metric("P95 EBIT", fmt_eur(p95_ebit, True))

            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=sim_df["ebit"], nbinsx=80,
                    marker_color=PALETTE["teal"], opacity=0.8, name="EBIT Distribution",
                ))
                fig.add_vline(x=0, line_color=PALETTE["red"], line_dash="dash",
                              annotation_text="Breakeven", annotation_font_color=PALETTE["red"])
                fig.add_vline(x=p5_ebit,  line_color=PALETTE["orange"], line_dash="dot",
                              annotation_text="P5", annotation_font_color=PALETTE["orange"])
                fig.add_vline(x=p50_ebit, line_color=PALETTE["yellow"], line_dash="dot",
                              annotation_text="Median", annotation_font_color=PALETTE["yellow"])
                fig.update_layout(**PLOTLY_THEME, title="EBIT Distribution (Monte Carlo)",
                                  xaxis_title="EBIT (€)", yaxis_title="Frequency",
                                  height=320, margin=dict(t=40, b=20, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=sorted(sim_df["ebit"]),
                    y=np.linspace(0, 100, len(sim_df)),
                    mode="lines", line=dict(color=PALETTE["blue"], width=2),
                    name="CDF",
                ))
                fig2.add_hline(y=prob_loss, line_color=PALETTE["red"], line_dash="dash",
                               annotation_text=f"Loss prob: {prob_loss:.1f}%",
                               annotation_font_color=PALETTE["red"])
                fig2.add_vline(x=0, line_color=PALETTE["red"], line_dash="dash")
                fig2.update_layout(**PLOTLY_THEME, title="Cumulative Probability of EBIT",
                                   xaxis_title="EBIT (€)", yaxis_title="Percentile",
                                   height=320, margin=dict(t=40, b=20, l=0, r=0))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Percentile Summary Table**")
            st.dataframe(pct_tbl.style.format({c: "€{:,.0f}" if c != "Percentile" and c in
                ["revenue","gross_profit","ebit"] else "{:.2f}%" if "pct" in c else "{}" for c in pct_tbl.columns}),
                use_container_width=True)

    with tab2:
        scenarios = st.session_state.get("scenarios", [])
        if not scenarios:
            st.info("No saved scenarios yet. Use the Live Simulator to save scenarios.")
        else:
            sc_df = pd.DataFrame(scenarios)
            st.dataframe(sc_df[["name","category","unit_price","unit_cost","discount",
                                 "demand","ebit","gross_margin_pct","roi_pct"]].style.format({
                "unit_price": "€{:.2f}", "unit_cost": "€{:.2f}", "discount": "{:.1f}%",
                "demand": "{:,.0f}", "ebit": "€{:,.0f}", "gross_margin_pct": "{:.1f}%", "roi_pct": "{:.1f}%"
            }).background_gradient(subset=["ebit"], cmap="RdYlGn"), use_container_width=True)

            fig = go.Figure()
            for metric, color in [("ebit","ebit (€)"),("gross_margin_pct","gross margin %"),("roi_pct","roi %")]:
                fig.add_trace(go.Bar(x=sc_df["name"], y=sc_df[metric], name=metric,
                                     marker_color=CAT_COLORS.get(sc_df["category"].iloc[0], PALETTE["teal"])))
            fig.update_layout(**PLOTLY_THEME, barmode="group", title="Scenario Comparison",
                              height=340, margin=dict(t=40,b=20,l=0,r=0),
                              legend=dict(bgcolor="#0f1f30"))
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🗑️ Clear Scenarios"):
                st.session_state.scenarios = []
                st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 6 — EXTERNAL FACTORS
# ══════════════════════════════════════════════════════════════
def page_external(df: pd.DataFrame, df_ext: pd.DataFrame):
    st.markdown("## 🌍 External Factors Impact")
    st.markdown("*Analyse how macroeconomic variables drive margin dynamics.*")

    tab1, tab2, tab3 = st.tabs(["📉 Factor Trends", "🔗 Correlation Analysis", "📊 Sensitivity Grid"])

    with tab1:
        ext_cols = {
            "inflation_rate_pct": ("Inflation Rate", PALETTE["red"]),
            "fuel_price_index":   ("Fuel Price Index", PALETTE["orange"]),
            "consumer_confidence_index": ("Consumer Confidence", PALETTE["teal"]),
            "competitor_price_index":    ("Competitor Price Index", PALETTE["blue"]),
            "interest_rate_pct":  ("Interest Rate", PALETTE["purple"]),
            "gdp_growth_pct":     ("GDP Growth", PALETTE["yellow"]),
        }
        selected = st.multiselect(
            "Select factors to display",
            list(ext_cols.keys()),
            default=list(ext_cols.keys())[:4],
            format_func=lambda x: ext_cols[x][0],
        )
        if selected:
            fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True,
                                subplot_titles=[ext_cols[c][0] for c in selected],
                                vertical_spacing=0.06)
            for i, col in enumerate(selected, 1):
                label, color = ext_cols[col]
                agg = df_ext.groupby("date")[col].mean().reset_index()
                fig.add_trace(
                    go.Scatter(x=agg["date"], y=agg[col], mode="lines",
                               line=dict(color=color, width=1.8), name=label,
                               showlegend=False),
                    row=i, col=1,
                )
                fig.update_yaxes(gridcolor="#162030", row=i, col=1)
                fig.update_xaxes(gridcolor="#162030", row=i, col=1)
            fig.update_layout(**PLOTLY_THEME, height=120 * len(selected) + 40,
                              margin=dict(t=40, b=20, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        factors = ["inflation_rate_pct","fuel_price_index","consumer_confidence_index",
                   "competitor_price_index","exchange_rate_eur_usd","interest_rate_pct",
                   "unemployment_rate_pct","gdp_growth_pct"]
        kpis_c = ["gross_margin_pct","ebit_margin_pct","roi_pct","demand_units","revenue"]

        corr_data = df[factors + kpis_c].corr().loc[factors, kpis_c]
        fig = go.Figure(go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns.tolist(),
            y=[f.replace("_pct","").replace("_"," ").title() for f in corr_data.index],
            colorscale=[[0,"#3d0000"],[0.3,"#1a2d42"],[0.5,"#162030"],[0.7,"#005a4a"],[1,"#00d4aa"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_data.round(2).astype(str).values,
            texttemplate="%{text}",
            hovertemplate="Factor: %{y}<br>KPI: %{x}<br>Correlation: %{z:.3f}<extra></extra>",
            colorbar=dict(title="r", tickfont_color="#8899aa"),
        ))
        fig.update_layout(**PLOTLY_THEME, title="Pearson Correlation: External Factors × Financial KPIs",
                          height=420, margin=dict(t=50, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key Findings**: Inflation negatively correlates with margins as cost pass-through is partial. "
            "Consumer Confidence positively correlates with demand and revenue. "
            "Fuel index has the strongest cost-side impact on Food & Beverage and Electronics."
        )

    with tab3:
        st.markdown("**Sensitivity Grid: Inflation × Fuel Impact on EBIT Margin**")
        cats_s = sorted(df["category"].unique())
        cat_s  = st.selectbox("Category", cats_s, key="ext_cat")
        cat_df_s = df[df["category"] == cat_s]
        d = cat_df_s[["unit_price","unit_cost","discount_rate_pct","demand_units","fixed_costs"]].median()

        infl_range = np.arange(1, 12, 1.5)
        fuel_range = np.arange(80, 160, 12)
        z = np.zeros((len(infl_range), len(fuel_range)))
        base_cost = float(d["unit_cost"])

        for i, infl in enumerate(infl_range):
            for j, fuel in enumerate(fuel_range):
                adj_cost = base_cost * (1 + infl / 100 * 0.65 + (fuel - 100) / 100 * 0.10)
                r = calc_profitability(
                    float(d["unit_price"]), adj_cost, float(d["discount_rate_pct"]),
                    int(d["demand_units"]), float(d["fixed_costs"]),
                )
                z[i, j] = r["ebit_margin_pct"]

        fig = go.Figure(go.Heatmap(
            z=z, x=fuel_range.round(0).astype(int).tolist(),
            y=[f"{v:.1f}%" for v in infl_range],
            colorscale=[[0,"#3d0000"],[0.4,"#1a2d42"],[0.7,"#005a4a"],[1,"#00d4aa"]],
            text=np.round(z, 1).astype(str),
            texttemplate="%{text}%",
            colorbar=dict(title="EBIT Margin %"),
        ))
        fig.update_layout(**PLOTLY_THEME, title=f"EBIT Margin % — Inflation × Fuel Stress Test ({cat_s})",
                          xaxis_title="Fuel Price Index", yaxis_title="Inflation Rate",
                          height=360, margin=dict(t=50, b=30, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 7 — INSIGHTS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def page_insights(df: pd.DataFrame, models: dict):
    st.markdown("## 💡 Insights & Recommendations")

    cats = ["All"] + sorted(df["category"].unique().tolist())
    cat  = st.selectbox("Analyse Category", cats, key="ins_cat")

    cat_df = df if cat == "All" else df[df["category"] == cat]
    cat_key = cat if cat != "All" else list(df["category"].unique())[0]
    medians = cat_df[["unit_price","unit_cost","discount_rate_pct","demand_units","fixed_costs",
                       "effective_price","gross_margin_pct","ebit_margin_pct","roi_pct",
                       "contribution_margin_pct","breakeven_units","ebit","revenue","cogs"]].median()

    kpis = {
        "unit_price":       medians["unit_price"],   "effective_price": medians["effective_price"],
        "unit_cost":        medians["unit_cost"],     "discount_rate_pct": medians["discount_rate_pct"],
        "demand_units":     int(medians["demand_units"]), "revenue": medians["revenue"],
        "cogs":             medians["cogs"],          "gross_profit": medians["revenue"] - medians["cogs"],
        "gross_margin_pct": medians["gross_margin_pct"], "fixed_costs": float(cat_df["fixed_costs"].median()),
        "ebit":             medians["ebit"],          "ebit_margin_pct": medians["ebit_margin_pct"],
        "roi_pct":          medians["roi_pct"],       "contribution_margin_pct": medians["contribution_margin_pct"],
        "breakeven_units":  medians["breakeven_units"],
    }

    # Elasticity
    elast = models["elasticity"].results.get(cat_key, {}).get("elasticity", None)

    interpreter = Interpreter(category=cat_key)
    report = interpreter.interpret(kpis, cat_df, elasticity=elast)

    # ── Risk score banner ─────────────────────────────────────
    risk_color = {"Critical": "#ff4444", "At Risk": "#ffbb33", "Healthy": "#00d4aa"}[report.health_label]
    st.markdown(f"""
    <div style='background:#0f1f30;border:1px solid {risk_color};border-radius:10px;
                padding:16px 24px;display:flex;align-items:center;gap:20px;margin-bottom:20px'>
      <div style='font-size:36px;font-weight:800;color:{risk_color};font-family:JetBrains Mono,monospace'>
        {report.risk_score:.0f}
      </div>
      <div>
        <div style='font-size:16px;font-weight:700;color:{risk_color}'>{report.health_label}</div>
        <div style='font-size:12px;color:#667788'>Risk Score (0 = low risk · 100 = critical) · 
        Category: {cat}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_ins, col_recs = st.columns([1, 1])

    with col_ins:
        st.markdown("#### 📋 Auto-Generated Insights")
        for ins in report.insights:
            css_class = f"insight-{ins.level}"
            st.markdown(f"""
            <div class='insight-card {css_class}'>
              <strong>{ins.icon} {ins.title}</strong><br>
              <span style='color:#8899aa'>{ins.body}</span>
              {"<br><em style='color:#aabbcc'>→ " + ins.action + "</em>" if ins.action else ""}
            </div>
            """, unsafe_allow_html=True)

    with col_recs:
        st.markdown("#### 🎯 Strategic Recommendations")
        for rec in report.recommendations:
            pri_class = {"High": "rec-high", "Medium": "rec-medium", "Low": "rec-low"}[rec["Priority"]]
            pri_color = {"High": "#ff6b6b", "Medium": "#ffbb33", "Low": "#4fc3f7"}[rec["Priority"]]
            st.markdown(f"""
            <div class='rec-card {pri_class}'>
              <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
                <strong style='color:#c8d8e8'>{rec['Category']}</strong>
                <span style='font-size:11px;color:{pri_color};font-weight:600'>{rec['Priority']} Priority</span>
              </div>
              <div style='font-size:13px;margin-bottom:6px'>{rec['Recommendation']}</div>
              <div style='font-size:11px;color:#00d4aa'>💰 {rec['Estimated Impact']}</div>
              <div style='font-size:11px;color:#667788;margin-top:4px'>⏱ {rec['Timeframe']} · {rec['Rationale']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Benchmark radar ───────────────────────────────────────
    st.markdown("#### 📡 KPI vs Industry Benchmark")
    bench = INDUSTRY_BENCHMARKS.get(cat_key, INDUSTRY_BENCHMARKS["Apparel"])
    metrics_r = ["Gross Margin %", "EBIT Margin %", "ROI %"]
    actual_v  = [kpis["gross_margin_pct"], kpis["ebit_margin_pct"], kpis["roi_pct"]]
    bench_lo  = [bench["gross_margin"][0], bench["ebit_margin"][0], bench["roi"][0]]
    bench_hi  = [bench["gross_margin"][1], bench["ebit_margin"][1], bench["roi"][1]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics_r, y=actual_v, name="Actual", marker_color=PALETTE["teal"]))
    fig.add_trace(go.Scatter(x=metrics_r, y=bench_lo, mode="lines+markers", name="Benchmark Low",
                             line=dict(color=PALETTE["red"], dash="dash"), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=metrics_r, y=bench_hi, mode="lines+markers", name="Benchmark High",
                             line=dict(color=PALETTE["yellow"], dash="dash"), marker=dict(size=8)))
    fig.update_layout(**PLOTLY_THEME, title=f"Actual KPIs vs Industry Benchmark — {cat}",
                      height=340, margin=dict(t=50,b=20,l=0,r=0),
                      yaxis_title="%", legend=dict(bgcolor="#0f1f30"))
    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    rec_df = pd.DataFrame(report.recommendations)
    ins_df = report.as_dataframe()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download Recommendations (CSV)",
                           rec_df.to_csv(index=False), "recommendations.csv", "text/csv")
    with c2:
        st.download_button("⬇️ Download Insights Log (CSV)",
                           ins_df.to_csv(index=False), "insights.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    df, df_ext = load_data()
    models     = load_models(df)

    page, sel_cat, sel_yr = sidebar(df)
    df_filt = filter_df(df.copy(), sel_cat, sel_yr)

    if   page == "Dashboard":         page_dashboard(df, df_filt)
    elif page == "Live Simulator":    page_simulator(df)
    elif page == "ML Engine":         page_ml_engine(df, models)
    elif page == "Margin Optimiser":  page_optimiser(df)
    elif page == "Scenario Analysis": page_scenarios(df)
    elif page == "External Factors":  page_external(df, df_ext)
    elif page == "Insights & Recs":   page_insights(df, models)


if __name__ == "__main__":
    main()
