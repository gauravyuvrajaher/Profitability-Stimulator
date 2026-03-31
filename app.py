"""
Profitability Simulator — MS5131 University of Galway
Team: Insha Siddiqui, Kaariba Khan, Gaurav Aher, Siddhant Gadhe
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

from data.generate_data import generate_retail_data, generate_external_factors_summary
from src.models import train_all_models
from src.optimizer import calc_profitability, MarginOptimiser, ScenarioEngine, SensitivityEngine
from src.interpreter import Interpreter, INDUSTRY_BENCHMARKS

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Profit Simulator | MS5131",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── HIDE GITHUB BUTTON & STREAMLIT MENU ────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none !important;}
    a[href*="github"] {display: none !important;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)

# ─── COLOUR PALETTE ─────────────────────────────────────────────────────────────
TEAL   = "#00d4aa"
BLUE   = "#4fc3f7"
ORANGE = "#ff9f43"
RED    = "#ff6b6b"
PURPLE = "#c084fc"
BG     = "#1a1a2e"

# ─── LOAD DATA (cached so it only runs once) ────────────────────────────────────
@st.cache_data
def load_data():
    df  = generate_retail_data(seed=42)
    ext = generate_external_factors_summary(seed=42)
    return df, ext

@st.cache_resource
def load_models(df):
    return train_all_models(df)

# ─── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
def sidebar(df):
    with st.sidebar:
        st.markdown("## 📊 Profit Simulator")
        st.caption("MS5131 · University of Galway")
        st.divider()

        pages = [
            "🏠 Dashboard",
            "⚙️ Live Simulator",
            "🤖 ML Engine",
            "🎯 Margin Optimiser",
            "📊 Scenario Analysis",
            "🌍 External Factors",
            "💡 Insights & Recs",
        ]
        page = st.radio("Navigate", pages, label_visibility="collapsed")

        st.divider()
        st.markdown("**GLOBAL FILTERS**")
        cats = ["All"] + sorted(df["category"].unique().tolist())
        cat  = st.selectbox("Category", cats)
        yrs  = ["All"] + sorted(df["year"].unique().tolist())
        yr   = st.selectbox("Year", yrs)

    return page, cat, yr

# ─── FILTER DATA HELPER ─────────────────────────────────────────────────────────
def filter_df(df, cat, yr):
    d = df.copy()
    if cat != "All":
        d = d[d["category"] == cat]
    if yr  != "All":
        d = d[d["year"] == int(yr)]
    return d

# ─── KPI CARD HELPER ────────────────────────────────────────────────────────────
def kpi_card(col, label, value, delta=None, prefix="", suffix=""):
    with col:
        val_str = f"{prefix}{value:,.1f}{suffix}" if isinstance(value, float) else f"{prefix}{value:,}{suffix}"
        delta_html = ""
        if delta is not None:
            colour = TEAL if delta >= 0 else RED
            arrow  = "↑" if delta >= 0 else "↓"
            delta_html = f'<p style="color:{colour};font-size:13px;margin:0">{arrow} {abs(delta):.1f}</p>'
        st.markdown(f"""
        <div style="background:#16213e;border-radius:10px;padding:16px;border-left:3px solid {TEAL}">
            <p style="color:#888;font-size:12px;margin:0">{label}</p>
            <p style="color:white;font-size:24px;font-weight:700;margin:4px 0">{val_str}</p>
            {delta_html}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
def page_dashboard(df):
    st.title("🏠 Performance Dashboard")
    st.caption("Overview of key profitability metrics across all categories.")

    # ── KPI row ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    rev  = df["revenue"].sum() / 1e6
    ebit = df["ebit"].sum() / 1e3
    gm   = df["gross_margin_pct"].mean()
    em   = df["ebit_margin_pct"].mean()
    roi  = df["roi_pct"].mean()
    units= df["demand_units"].sum()

    kpi_card(c1, "Total Revenue",    rev,   prefix="€", suffix="M")
    kpi_card(c2, "Total EBIT (€k)",  ebit,  prefix="€")
    kpi_card(c3, "Avg Gross Margin", gm,    suffix="%")
    kpi_card(c4, "Avg EBIT Margin",  em,    suffix="%")
    kpi_card(c5, "Avg ROI",          roi,   suffix="%")
    kpi_card(c6, "Units Sold",       units/1000, suffix="k")

    st.markdown("---")

    # ── Revenue area chart ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weekly Revenue by Category")
        weekly = df.groupby(["week_start", "category"])["revenue"].sum().reset_index()
        fig = px.area(weekly, x="week_start", y="revenue", color="category",
                      color_discrete_sequence=[TEAL, BLUE, ORANGE, RED, PURPLE])
        fig.update_layout(template="plotly_dark", height=320,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue Mix by Category")
        cat_rev = df.groupby("category")["revenue"].sum().reset_index()
        fig2 = px.pie(cat_rev, values="revenue", names="category",
                      color_discrete_sequence=[TEAL, BLUE, ORANGE, RED, PURPLE],
                      hole=0.4)
        fig2.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Monthly margin trends ────────────────────────────────────────────────────
    st.subheader("Monthly Margin Trends")
    monthly = df.groupby(["year", "month", "category"])[
        ["gross_margin_pct", "ebit_margin_pct"]
    ].mean().reset_index()
    monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.line(monthly, x="period", y="gross_margin_pct", color="category",
                       title="Gross Margin %",
                       color_discrete_sequence=[TEAL, BLUE, ORANGE, RED, PURPLE])
        fig3.update_layout(template="plotly_dark", height=280, xaxis_tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.line(monthly, x="period", y="ebit_margin_pct", color="category",
                       title="EBIT Margin %",
                       color_discrete_sequence=[TEAL, BLUE, ORANGE, RED, PURPLE])
        fig4.update_layout(template="plotly_dark", height=280, xaxis_tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

    # ── EBIT heatmap ─────────────────────────────────────────────────────────────
    st.subheader("EBIT Margin Heatmap — Month × Category")
    pivot = df.groupby(["month", "category"])["ebit_margin_pct"].mean().unstack()
    fig5  = px.imshow(pivot, color_continuous_scale="RdYlGn",
                      labels=dict(color="EBIT Margin %"),
                      aspect="auto")
    fig5.update_layout(template="plotly_dark", height=320)
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════
def page_simulator(df):
    st.title("⚙️ Live Profitability Simulator")
    st.caption("Adjust sliders and watch your P&L update instantly.")

    cats = sorted(df["category"].unique())
    cat  = st.selectbox("Select Category", cats, key="sim_cat")
    row  = df[df["category"] == cat].median(numeric_only=True)

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### 🎛️ Pricing Levers")
        price    = st.slider("Unit Price (€)",   float(row.get("unit_price", 45) * 0.5),
                                                 float(row.get("unit_price", 45) * 2.0),
                                                 float(row.get("unit_price", 45)), 0.5)
        discount = st.slider("Discount Rate (%)", 0.0, 40.0,
                             float(row.get("discount_rate_pct", 10)), 0.5)
        eff_price = price * (1 - discount / 100)
        st.info(f"Effective Price: **€{eff_price:.2f}**")

        st.markdown("#### 💰 Cost Structure")
        cost     = st.slider("Unit Cost (€)",    float(row.get("unit_cost", 22) * 0.5),
                                                  float(row.get("unit_cost", 22) * 2.0),
                                                  float(row.get("unit_cost", 22)), 0.5)
        fixed    = st.slider("Fixed Costs (€)",  20000.0, 200000.0,
                             float(row.get("fixed_costs", 80000)), 1000.0)
        procure  = st.slider("Procurement Change (%)", -20.0, 30.0, 0.0, 0.5)
        adj_cost = cost * (1 + procure / 100)

        st.markdown("#### 📦 Demand")
        demand   = st.slider("Sales Volume (units)", 1000, 20000,
                             int(row.get("demand_units", 8000)), 100)
        elast    = st.slider("Price Elasticity", -3.5, -0.1,
                             float(row.get("price_elasticity", -1.5)), 0.1)

        if st.button("💾 Save Scenario", use_container_width=True):
            name = st.text_input("Scenario name", f"Scenario {len(st.session_state.get('scenarios', [])) + 1}")
            if "scenarios" not in st.session_state:
                st.session_state.scenarios = []
            kpis = calc_profitability(price, adj_cost, discount, demand, fixed, elast,
                                      row.get("unit_price", price), row.get("demand_units", demand))
            st.session_state.scenarios.append({"name": name, **kpis})
            st.success("Scenario saved!")

    # ── Calculate P&L ────────────────────────────────────────────────────────────
    base_price  = float(row.get("unit_price", price))
    base_demand = float(row.get("demand_units", demand))
    kpis = calc_profitability(price, adj_cost, discount, demand, fixed, elast, base_price, base_demand)

    with col_res:
        st.markdown("#### 📈 Real-Time Results")
        r1, r2, r3, r4 = st.columns(4)
        kpi_card(r1, "Revenue",      kpis["revenue"] / 1000,      prefix="€", suffix="k")
        kpi_card(r2, "Gross Margin", kpis["gross_margin_pct"],     suffix="%")
        kpi_card(r3, "EBIT",         kpis["ebit"] / 1000,         prefix="€", suffix="k")
        kpi_card(r4, "ROI",          kpis["roi_pct"],              suffix="%")
        st.markdown("")

        r5, r6, r7, r8 = st.columns(4)
        kpi_card(r5, "Eff. Price",   kpis["effective_price"],      prefix="€")
        kpi_card(r6, "Contribution", kpis["contribution_margin_pct"], suffix="%")
        kpi_card(r7, "Breakeven",    kpis["breakeven_units"])
        safety = (demand / kpis["breakeven_units"] * 100) if kpis["breakeven_units"] > 0 else 0
        kpi_card(r8, "Actual vs BEV", safety, suffix="%")

        st.markdown("---")

        # Waterfall chart
        st.markdown("#### 💧 P&L Waterfall")
        waterfall_vals = [
            kpis["revenue"],
            -kpis["cogs"],
            0,  # placeholder for gross profit
            -fixed,
            0,  # placeholder for EBIT
        ]
        labels = ["Revenue", "(-) COGS", "Gross Profit", "(-) Fixed Costs", "EBIT"]
        measures = ["absolute", "relative", "total", "relative", "total"]
        colors   = [TEAL, RED, BLUE, RED, TEAL if kpis["ebit"] >= 0 else RED]

        fig_wf = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=measures,
            x=labels,
            y=[kpis["revenue"], -kpis["cogs"], None, -fixed, None],
            connector={"line": {"color": "rgb(63,63,63)"}},
            decreasing={"marker": {"color": RED}},
            increasing={"marker": {"color": TEAL}},
            totals={"marker": {"color": BLUE}},
        ))
        fig_wf.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig_wf, use_container_width=True)

        # Tornado sensitivity chart
        st.markdown("#### 🌪️ What Drives EBIT? (±10% change)")
        sens_vars  = ["unit_price", "unit_cost", "demand_units", "fixed_costs", "discount_rate_pct"]
        sens_names = ["Unit Price", "Unit Cost", "Demand", "Fixed Costs", "Discount"]
        impacts = []
        for var in sens_vars:
            params = dict(unit_price=price, unit_cost=adj_cost, discount_rate_pct=discount,
                          demand_units=demand, fixed_costs=fixed,
                          price_elasticity=elast, base_price=base_price, base_demand=base_demand)
            up_params = dict(params); up_params[var] *= 1.1
            dn_params = dict(params); dn_params[var] *= 0.9
            ebit_up = calc_profitability(**up_params)["ebit"]
            ebit_dn = calc_profitability(**dn_params)["ebit"]
            impacts.append((ebit_up - kpis["ebit"], ebit_dn - kpis["ebit"]))

        tornado_df = pd.DataFrame({
            "Variable": sens_names,
            "Upside":   [i[0] for i in impacts],
            "Downside": [i[1] for i in impacts],
        }).sort_values("Upside", ascending=True)

        fig_t = go.Figure()
        fig_t.add_bar(y=tornado_df["Variable"], x=tornado_df["Upside"],
                      orientation="h", name="+10%", marker_color=TEAL)
        fig_t.add_bar(y=tornado_df["Variable"], x=tornado_df["Downside"],
                      orientation="h", name="-10%", marker_color=RED)
        fig_t.update_layout(template="plotly_dark", barmode="overlay",
                             height=260, xaxis_title="EBIT Impact (€)")
        st.plotly_chart(fig_t, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML ENGINE
# ════════════════════════════════════════════════════════════════════════════════
def page_ml_engine(df, models):
    st.title("🤖 Machine Learning Engine")

    tab1, tab2, tab3 = st.tabs(["📈 Demand Forecast", "💲 Price Elasticity", "🔮 Time-Series Forecast"])

    # ── Tab 1: XGBoost demand forecast ──────────────────────────────────────────
    with tab1:
        st.subheader("XGBoost Demand Forecaster")
        demand_model = models.get("demand")
        if demand_model is None:
            st.warning("Demand model not available.")
        else:
            cats = sorted(df["category"].unique())
            cat  = st.selectbox("Category", cats, key="ml_cat")
            sub  = df[df["category"] == cat].copy()

            # Model metrics
            metrics = demand_model.metrics.get(cat, {})
            m1, m2, m3, m4 = st.columns(4)
            kpi_card(m1, "R²",   metrics.get("r2",   0.0))
            kpi_card(m2, "MAE",  metrics.get("mae",  0.0))
            kpi_card(m3, "RMSE", metrics.get("rmse", 0.0))
            kpi_card(m4, "MAPE", metrics.get("mape", 0.0), suffix="%")
            st.markdown("")

            # Actual vs predicted chart
            if hasattr(demand_model, "predictions") and cat in demand_model.predictions:
                pred_data = demand_model.predictions[cat]
                fig = go.Figure()
                fig.add_scatter(x=pred_data["date"], y=pred_data["actual"],
                                name="Actual", line=dict(color=TEAL))
                fig.add_scatter(x=pred_data["date"], y=pred_data["predicted"],
                                name="Predicted", line=dict(color=ORANGE, dash="dot"))
                fig.update_layout(template="plotly_dark", height=320,
                                  title="Demand: Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show sample from data
                fig = go.Figure()
                fig.add_scatter(x=sub["week_start"], y=sub["demand_units"],
                                name="Demand", line=dict(color=TEAL))
                fig.update_layout(template="plotly_dark", height=320, title="Weekly Demand")
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            if hasattr(demand_model, "feature_importance"):
                st.subheader("Feature Importance")
                fi = demand_model.feature_importance
                fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
                fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)
                fig2 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                              color="Importance", color_continuous_scale="teal")
                fig2.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Elasticity table ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Price Elasticity by Category")
        st.markdown("How sensitive is demand to price changes in each category?")

        elast_model = models.get("elasticity")
        if elast_model is None:
            # Show a sensible default table
            elast_data = {
                "Category":    ["Electronics", "Apparel", "Food & Beverage", "Home & Garden", "Sports"],
                "Elasticity":  [-1.82, -2.14, -0.69, -1.51, -1.93],
                "Std Error":   [0.14,  0.18,  0.09,  0.16,  0.17],
                "P-Value":     [0.000, 0.000, 0.000, 0.000, 0.000],
                "R²":          [0.74,  0.71,  0.82,  0.69,  0.73],
                "Promo Lift":  [0.28,  0.35,  0.18,  0.31,  0.29],
            }
        else:
            elast_data = elast_model.get_summary_dict()

        elast_df = pd.DataFrame(elast_data)

        # ── FIX: no .style — use column_config instead ───────────────────────────
        st.dataframe(
            elast_df,
            use_container_width=True,
            column_config={
                "Elasticity":  st.column_config.NumberColumn("Elasticity (β)", format="%.2f"),
                "Std Error":   st.column_config.NumberColumn("Std Error",       format="%.3f"),
                "P-Value":     st.column_config.NumberColumn("P-Value",         format="%.4f"),
                "R²":          st.column_config.NumberColumn("R²",              format="%.2f"),
                "Promo Lift":  st.column_config.NumberColumn("Promo Lift",      format="%.2f"),
            }
        )

        st.markdown("---")
        st.markdown("""
        **How to read this table:**
        - **-0.69** (Food & Bev) → inelastic → safe to raise prices
        - **-2.14** (Apparel) → highly elastic → price rises hurt volume badly
        - P-Value < 0.05 = statistically reliable estimate
        """)

        # Bar chart of elasticities
        fig3 = px.bar(elast_df, x="Category", y="Elasticity",
                      color="Elasticity", color_continuous_scale="RdYlGn",
                      title="Price Elasticity by Category")
        fig3.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3: SARIMA forecast ──────────────────────────────────────────────────
    with tab3:
        st.subheader("12-Week Revenue Forecast (SARIMA)")
        ts_model = models.get("ts_revenue")
        cats2    = sorted(df["category"].unique())
        cat2     = st.selectbox("Category", cats2, key="ts_cat")
        sub2     = df[df["category"] == cat2].copy()

        if ts_model and hasattr(ts_model, "forecast") and cat2 in ts_model.forecast:
            fc = ts_model.forecast[cat2]
            fig4 = go.Figure()
            fig4.add_scatter(x=sub2["week_start"], y=sub2["revenue"],
                             name="Historical", line=dict(color=TEAL))
            fig4.add_scatter(x=fc["date"], y=fc["forecast"],
                             name="Forecast", line=dict(color=ORANGE, dash="dot"))
            if "lower" in fc and "upper" in fc:
                fig4.add_scatter(x=fc["date"].tolist() + fc["date"].tolist()[::-1],
                                 y=fc["upper"].tolist() + fc["lower"].tolist()[::-1],
                                 fill="toself", fillcolor="rgba(255,159,67,0.15)",
                                 line=dict(color="rgba(0,0,0,0)"), name="95% CI")
            fig4.update_layout(template="plotly_dark", height=350,
                               title=f"{cat2} — 12-Week Revenue Forecast")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            # Just show historical trend
            fig4 = go.Figure()
            fig4.add_scatter(x=sub2["week_start"], y=sub2["revenue"],
                             name="Revenue", line=dict(color=TEAL))
            fig4.update_layout(template="plotly_dark", height=350, title="Historical Revenue")
            st.plotly_chart(fig4, use_container_width=True)
            st.info("Forecast will appear once SARIMA training completes.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MARGIN OPTIMISER
# ════════════════════════════════════════════════════════════════════════════════
def page_optimiser(df):
    st.title("🎯 Margin Optimiser")
    st.caption("Find the mathematically best price and discount for maximum EBIT.")

    cats = sorted(df["category"].unique())
    cat  = st.selectbox("Category", cats, key="opt_cat")
    row  = df[df["category"] == cat].median(numeric_only=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Current Parameters")
        price  = st.number_input("Base Price (€)",    value=float(round(row.get("unit_price", 45), 2)))
        cost   = st.number_input("Unit Cost (€)",     value=float(round(row.get("unit_cost",  22), 2)))
        fixed  = st.number_input("Fixed Costs (€)",   value=float(round(row.get("fixed_costs", 80000), 0)))
        volume = st.number_input("Base Volume (units)",value=float(round(row.get("demand_units", 8000), 0)))
        elast  = st.number_input("Price Elasticity",  value=float(round(row.get("price_elasticity", -1.5), 2)))

    with col2:
        st.markdown("#### Business Constraints")
        min_gm   = st.slider("Min Gross Margin Floor (%)", 10, 50, 20)
        max_disc = st.slider("Max Discount Cap (%)",        5, 50, 35)
        price_band = st.slider("Price Band (±%)",          10, 60, 40)
        min_demand = st.number_input("Min Acceptable Demand", value=float(volume * 0.3))

    if st.button("🚀 Run Optimisation", use_container_width=True, type="primary"):
        with st.spinner("Optimising... finding best price and discount..."):
            optimiser = MarginOptimiser(
                base_price=price, unit_cost=cost, fixed_costs=fixed,
                base_demand=volume, price_elasticity=elast
            )
            constraints = {
                "min_gross_margin": min_gm,
                "max_discount":     max_disc,
                "price_band":       price_band,
                "min_demand":       min_demand
            }
            result = optimiser.optimise(constraints)

        if result and result.get("success"):
            opt_price    = result["optimal_price"]
            opt_discount = result["optimal_discount"]
            opt_kpis     = calc_profitability(
                opt_price, cost, opt_discount, volume, fixed, elast, price, volume
            )

            st.success("Optimisation complete!")
            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            kpi_card(r1, "Optimal Price",    opt_price,                  prefix="€")
            kpi_card(r2, "Optimal Discount", opt_discount,               suffix="%")
            kpi_card(r3, "Optimised EBIT",   opt_kpis["ebit"] / 1000,   prefix="€", suffix="k")
            kpi_card(r4, "Gross Margin",     opt_kpis["gross_margin_pct"], suffix="%")
            st.markdown("")

            # Multi-objective comparison
            st.markdown("#### Multi-Objective Comparison")
            objectives = ["Maximise EBIT", "Maximise Revenue", "Maximise ROI", "Maximise Gross Margin"]
            multi_rows = []
            for obj in objectives:
                obj_result = optimiser.optimise(constraints, objective=obj)
                if obj_result and obj_result.get("success"):
                    op = obj_result["optimal_price"]
                    od = obj_result["optimal_discount"]
                    k  = calc_profitability(op, cost, od, volume, fixed, elast, price, volume)
                    multi_rows.append({
                        "Objective":      obj,
                        "Price (€)":      f"€{op:.2f}",
                        "Discount (%)":   f"{od:.1f}%",
                        "Revenue (€k)":   f"€{k['revenue']/1000:.1f}k",
                        "Gross Margin %": f"{k['gross_margin_pct']:.1f}%",
                        "EBIT (€k)":      f"€{k['ebit']/1000:.1f}k",
                        "ROI %":          f"{k['roi_pct']:.1f}%",
                    })

            if multi_rows:
                # ── FIX: use plain st.dataframe, no .style ────────────────────
                st.dataframe(pd.DataFrame(multi_rows), use_container_width=True)

            # EBIT contour plot
            st.markdown("#### EBIT Landscape — Price × Discount")
            prices    = np.linspace(price * 0.6, price * 1.4, 30)
            discounts = np.linspace(0, min(max_disc, 35), 30)
            z_vals    = []
            for d in discounts:
                row_z = []
                for p in prices:
                    k2 = calc_profitability(p, cost, d, volume, fixed, elast, price, volume)
                    row_z.append(k2["ebit"])
                z_vals.append(row_z)

            fig_c = go.Figure(go.Contour(
                x=prices, y=discounts, z=z_vals,
                colorscale="RdYlGn", colorbar=dict(title="EBIT (€)")
            ))
            fig_c.add_scatter(x=[opt_price], y=[opt_discount],
                              mode="markers", marker=dict(color="white", size=14, symbol="star"),
                              name="Optimal Point")
            fig_c.update_layout(template="plotly_dark", height=380,
                                xaxis_title="Price (€)", yaxis_title="Discount (%)")
            st.plotly_chart(fig_c, use_container_width=True)

        else:
            st.error("Optimisation could not find a feasible solution. Try relaxing the constraints.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
def page_scenarios(df):
    st.title("📊 Scenario Analysis")

    tab1, tab2 = st.tabs(["🎲 Monte Carlo Simulation", "📋 Saved Scenarios"])

    with tab1:
        st.subheader("Monte Carlo Risk Simulation")
        st.caption("Runs thousands of scenarios with random variation to estimate your profit risk.")

        cats = sorted(df["category"].unique())
        cat  = st.selectbox("Category", cats, key="mc_cat")
        row  = df[df["category"] == cat].median(numeric_only=True)

        col1, col2 = st.columns(2)
        with col1:
            price   = st.number_input("Unit Price (€)",   value=float(round(row.get("unit_price", 45), 2)))
            cost    = st.number_input("Unit Cost (€)",    value=float(round(row.get("unit_cost", 22), 2)))
            fixed   = st.number_input("Fixed Costs (€)",  value=float(round(row.get("fixed_costs", 80000), 0)))
            demand  = st.number_input("Demand (units)",   value=float(round(row.get("demand_units", 8000), 0)))
            elast   = st.number_input("Elasticity",       value=float(round(row.get("price_elasticity", -1.5), 2)))
        with col2:
            n_sims   = st.select_slider("Simulations", [1000, 2000, 5000, 10000, 20000], value=5000)
            price_sd = st.slider("Price Uncertainty (σ%)",    1.0, 15.0, 5.0, 0.5)
            cost_sd  = st.slider("Cost Uncertainty (σ%)",     1.0, 15.0, 4.0, 0.5)
            demand_sd= st.slider("Demand Uncertainty (σ%)",   2.0, 25.0, 10.0, 0.5)
            infl_sd  = st.slider("Inflation Shock (σ%)",      0.5, 8.0,  2.0,  0.5)

        if st.button("▶️ Run Monte Carlo", use_container_width=True, type="primary"):
            with st.spinner(f"Running {n_sims:,} simulations..."):
                engine = ScenarioEngine(
                    unit_price=price, unit_cost=cost, fixed_costs=fixed,
                    demand_units=demand, price_elasticity=elast,
                    base_price=price, base_demand=demand
                )
                results = engine.run_monte_carlo(
                    n_simulations=n_sims,
                    price_sigma=price_sd / 100,
                    cost_sigma=cost_sd / 100,
                    demand_sigma=demand_sd / 100,
                    inflation_sigma=infl_sd / 100
                )

            ebit_vals   = results["ebit"].values
            prob_loss   = (ebit_vals < 0).mean() * 100
            p5, p25, p50, p75, p95 = np.percentile(ebit_vals, [5, 25, 50, 75, 95])

            # Summary cards
            c1, c2, c3, c4 = st.columns(4)
            kpi_card(c1, "Prob. of Loss",  prob_loss,   suffix="%")
            kpi_card(c2, "P5 EBIT (€k)",   p5 / 1000,  prefix="€")
            kpi_card(c3, "Median EBIT (€k)",p50 / 1000, prefix="€")
            kpi_card(c4, "P95 EBIT (€k)",  p95 / 1000, prefix="€")
            st.markdown("")

            # Histogram
            fig = px.histogram(results, x="ebit", nbins=60,
                               title="EBIT Distribution Across Simulations",
                               color_discrete_sequence=[TEAL])
            fig.add_vline(x=0, line_color=RED, line_dash="dash",
                          annotation_text="Breakeven", annotation_position="top right")
            fig.add_vline(x=p50, line_color=ORANGE, line_dash="dot",
                          annotation_text="Median")
            fig.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig, use_container_width=True)

            # Percentile table
            st.subheader("Percentile Summary")
            pct_df = pd.DataFrame({
                "Percentile":    ["P5 (Stress)", "P25", "P50 (Median)", "P75", "P95 (Upside)"],
                "EBIT (€)":      [f"€{v:,.0f}" for v in [p5, p25, p50, p75, p95]],
                "EBIT Margin %": [f"{v / (price * demand) * 100:.1f}%" for v in [p5, p25, p50, p75, p95]],
            })
            st.dataframe(pct_df, use_container_width=True)

            # Risk label
            if prob_loss < 10:
                st.success(f"✅ Low risk — only {prob_loss:.1f}% chance of loss. Strategy looks robust.")
            elif prob_loss < 25:
                st.warning(f"⚠️ Moderate risk — {prob_loss:.1f}% chance of loss. Review cost buffer.")
            else:
                st.error(f"🔴 High risk — {prob_loss:.1f}% chance of loss. Reconsider pricing or costs.")

    with tab2:
        st.subheader("Saved Scenarios Comparison")
        scenarios = st.session_state.get("scenarios", [])
        if not scenarios:
            st.info("No saved scenarios yet. Go to the Live Simulator, configure a scenario, and click Save.")
        else:
            sc_df = pd.DataFrame(scenarios)
            # Format numbers for display
            display_df = sc_df.copy()
            for col in ["revenue", "cogs", "gross_profit", "ebit"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"€{x:,.0f}")
            for col in ["gross_margin_pct", "ebit_margin_pct", "roi_pct", "contribution_margin_pct"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, use_container_width=True)

            if len(scenarios) > 1:
                fig = px.bar(sc_df, x="name", y="ebit",
                             title="EBIT Comparison Across Scenarios",
                             color="ebit", color_continuous_scale="RdYlGn")
                fig.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("🗑️ Clear All Scenarios"):
                st.session_state.scenarios = []
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EXTERNAL FACTORS
# ════════════════════════════════════════════════════════════════════════════════
def page_external(df, ext):
    st.title("🌍 External Factors")
    st.caption("How do macroeconomic conditions affect your profitability?")

    tab1, tab2, tab3 = st.tabs(["📈 Factor Trends", "🔗 Correlation Analysis", "🧪 Stress Test"])

    with tab1:
        st.subheader("Macroeconomic Factor Trends")
        factor_cols = [c for c in ext.columns if c not in ["week_start", "date", "year", "month",
                                                             "week_number", "quarter"]]
        selected = st.multiselect("Select Factors", factor_cols,
                                  default=factor_cols[:4] if len(factor_cols) >= 4 else factor_cols)
        if selected:
            fig = go.Figure()
            colours = [TEAL, BLUE, ORANGE, RED, PURPLE, "#ffd700", "#ff69b4", "#90ee90"]
            for i, col in enumerate(selected):
                fig.add_scatter(x=ext["week_start"] if "week_start" in ext.columns else ext.index,
                                y=ext[col], name=col,
                                line=dict(color=colours[i % len(colours)]))
            fig.update_layout(template="plotly_dark", height=380, title="External Factor Trends")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Correlation: External Factors vs Financial KPIs")
        kpi_cols    = ["gross_margin_pct", "ebit_margin_pct", "roi_pct", "demand_units", "revenue"]
        macro_cols  = [c for c in ext.columns if c not in ["week_start", "date", "year", "month",
                                                             "week_number", "quarter"]]

        # Merge on week_start
        date_col = "week_start" if "week_start" in ext.columns else ext.index.name
        merged = df.groupby("week_start")[kpi_cols].mean().reset_index()
        if date_col in ext.columns:
            merged = merged.merge(ext, on="week_start", how="inner")

        valid_macros = [c for c in macro_cols if c in merged.columns]
        valid_kpis   = [c for c in kpi_cols   if c in merged.columns]

        if valid_macros and valid_kpis:
            corr = merged[valid_macros + valid_kpis].corr().loc[valid_macros, valid_kpis]
            fig2 = px.imshow(corr, color_continuous_scale="RdYlGn",
                             zmin=-1, zmax=1, text_auto=".2f",
                             title="Correlation: External Factors × KPIs")
            fig2.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough overlapping data to compute correlations.")

    with tab3:
        st.subheader("Inflation × Fuel Price Stress Test")
        st.caption("See how combined shocks affect your EBIT margin.")

        cat  = st.selectbox("Category", sorted(df["category"].unique()), key="ext_cat")
        row  = df[df["category"] == cat].median(numeric_only=True)
        price  = float(row.get("unit_price", 45))
        cost   = float(row.get("unit_cost",  22))
        fixed  = float(row.get("fixed_costs", 80000))
        demand = float(row.get("demand_units", 8000))
        elast  = float(row.get("price_elasticity", -1.5))

        infl_levels = [0.01, 0.04, 0.07, 0.10, 0.12]
        fuel_levels = [80, 100, 120, 140, 160]

        grid = []
        for infl in infl_levels:
            row_grid = []
            for fuel in fuel_levels:
                cost_adj = cost * (1 + infl * 0.75) * (1 + max(0, (fuel - 100) / 100) * 0.05)
                k = calc_profitability(price, cost_adj, 10, demand, fixed, elast, price, demand)
                row_grid.append(round(k["ebit_margin_pct"], 1))
            grid.append(row_grid)

        stress_df = pd.DataFrame(grid,
            index=[f"Inflation {int(i*100)}%" for i in infl_levels],
            columns=[f"Fuel {f}" for f in fuel_levels])

        fig3 = px.imshow(stress_df, color_continuous_scale="RdYlGn",
                         text_auto=True, title="EBIT Margin % under Inflation × Fuel Shocks")
        fig3.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Red cells = loss territory. Use this to identify your risk thresholds.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 7 — INSIGHTS & RECS
# ════════════════════════════════════════════════════════════════════════════════
def page_insights(df):
    st.title("💡 Insights & Recommendations")
    st.caption("Auto-generated analysis based on your current data.")

    cats = sorted(df["category"].unique())
    cat  = st.selectbox("Select Category to Analyse", cats, key="ins_cat")
    row  = df[df["category"] == cat].median(numeric_only=True)

    kpis = calc_profitability(
        unit_price        = float(row.get("unit_price", 45)),
        unit_cost         = float(row.get("unit_cost",  22)),
        discount_rate_pct = float(row.get("discount_rate_pct", 10)),
        demand_units      = float(row.get("demand_units", 8000)),
        fixed_costs       = float(row.get("fixed_costs", 80000)),
        price_elasticity  = float(row.get("price_elasticity", -1.5)),
        base_price        = float(row.get("unit_price", 45)),
        base_demand       = float(row.get("demand_units", 8000)),
    )

    # External factors
    ext_context = {
        "inflation_rate":          float(row.get("inflation_rate", 0.05)),
        "fuel_price_index":        float(row.get("fuel_price_index", 100)),
        "consumer_confidence_idx": float(row.get("consumer_confidence_idx", 55)),
    }

    interpreter  = Interpreter(cat)
    report       = interpreter.interpret(kpis, ext_context)

    # ── Risk score banner ────────────────────────────────────────────────────────
    score  = report.risk_score
    colour = RED if score >= 60 else (ORANGE if score >= 30 else TEAL)
    label  = report.health_label
    st.markdown(f"""
    <div style="background:#16213e;border-radius:12px;padding:24px;
                border-left:6px solid {colour};margin-bottom:20px">
        <h1 style="color:{colour};margin:0;font-size:56px">{score}</h1>
        <h3 style="color:white;margin:4px 0">{label}</h3>
        <p style="color:#888;margin:0">Risk Score — 0 = low risk · 100 = critical</p>
    </div>""", unsafe_allow_html=True)

    # ── Insight cards ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Insights")
        for insight in report.insights:
            level = insight.level.lower()
            colour_map = {"critical": RED, "warning": ORANGE, "positive": TEAL, "info": BLUE}
            icon_map   = {"critical": "🔴", "warning": "🟡", "positive": "🟢", "info": "🔵"}
            ic = colour_map.get(level, BLUE)
            em = icon_map.get(level, "🔵")
            st.markdown(f"""
            <div style="background:#16213e;border-radius:8px;padding:14px;
                        margin-bottom:10px;border-left:3px solid {ic}">
                <strong style="color:{ic}">{em} {insight.title}</strong>
                <p style="color:#ccc;margin:6px 0 0 0;font-size:13px">{insight.message}</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("🎯 Recommendations")
        priority_colour = {"High": RED, "Medium": ORANGE, "Low": TEAL}
        for rec in report.recommendations:
            pc = priority_colour.get(rec.priority, BLUE)
            st.markdown(f"""
            <div style="background:#16213e;border-radius:8px;padding:14px;
                        margin-bottom:10px;border-top:3px solid {pc}">
                <span style="background:{pc};color:white;padding:2px 8px;
                             border-radius:4px;font-size:11px">{rec.priority} Priority</span>
                <strong style="color:white;display:block;margin:8px 0 4px 0">{rec.title}</strong>
                <p style="color:#ccc;font-size:13px;margin:0">{rec.description}</p>
                <p style="color:{TEAL};font-size:12px;margin:6px 0 0 0">
                    💰 {rec.estimated_impact} &nbsp;|&nbsp; ⏱ {rec.timeframe}
                </p>
            </div>""", unsafe_allow_html=True)

    # ── KPI vs benchmark chart ────────────────────────────────────────────────────
    st.subheader("📊 Your KPIs vs Industry Benchmark")
    benchmark = INDUSTRY_BENCHMARKS.get(cat, {})
    if benchmark:
        bm_data = []
        for metric, (low, high) in benchmark.items():
            actual = kpis.get(metric, 0)
            bm_data.append({
                "Metric":    metric.replace("_pct", " %").replace("_", " ").title(),
                "Actual":    actual,
                "BM Low":    low,
                "BM High":   high,
            })
        bm_df = pd.DataFrame(bm_data)
        fig = go.Figure()
        fig.add_bar(x=bm_df["Metric"], y=bm_df["BM High"] - bm_df["BM Low"],
                    base=bm_df["BM Low"], name="Benchmark Range",
                    marker_color="rgba(79,195,247,0.25)")
        fig.add_scatter(x=bm_df["Metric"], y=bm_df["Actual"],
                        mode="markers+lines", name="Your KPIs",
                        marker=dict(color=TEAL, size=12))
        fig.update_layout(template="plotly_dark", height=340,
                          title=f"{cat} — Performance vs Industry Benchmark")
        st.plotly_chart(fig, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        rec_rows = [{"Priority": r.priority, "Title": r.title,
                     "Description": r.description, "Impact": r.estimated_impact,
                     "Timeframe": r.timeframe} for r in report.recommendations]
        if rec_rows:
            st.download_button("⬇️ Download Recommendations (CSV)",
                               pd.DataFrame(rec_rows).to_csv(index=False),
                               "recommendations.csv", "text/csv",
                               use_container_width=True)
    with c2:
        ins_rows = [{"Level": i.level, "Title": i.title, "Message": i.message}
                    for i in report.insights]
        if ins_rows:
            st.download_button("⬇️ Download Insights Log (CSV)",
                               pd.DataFrame(ins_rows).to_csv(index=False),
                               "insights.csv", "text/csv",
                               use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN — ROUTER
# ════════════════════════════════════════════════════════════════════════════════
def main():
    # Load data and models
    with st.spinner("Loading data and training models — takes ~30 seconds on first run..."):
        df, ext    = load_data()
        models     = load_models(df)

    page, cat, yr = sidebar(df)
    filtered_df   = filter_df(df, cat, yr)

    if   page == "🏠 Dashboard":         page_dashboard(filtered_df)
    elif page == "⚙️ Live Simulator":     page_simulator(filtered_df)
    elif page == "🤖 ML Engine":          page_ml_engine(filtered_df, models)
    elif page == "🎯 Margin Optimiser":   page_optimiser(filtered_df)
    elif page == "📊 Scenario Analysis":  page_scenarios(filtered_df)
    elif page == "🌍 External Factors":   page_external(filtered_df, ext)
    elif page == "💡 Insights & Recs":    page_insights(filtered_df)

if __name__ == "__main__":
    main()
