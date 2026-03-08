# 📊 Profitability Simulator & Margin Optimisation Model
**MS5131 – Major Business Analytics Project | University of Galway | AY 25-26**

> An end-to-end, expert-level interactive analytics platform for retail profitability
> simulation, ML-driven forecasting, constrained optimisation, and automated insight generation.

---

## 🏗️ Architecture

```
profitability_simulator/
├── app.py                        ← Main Streamlit application (7 pages)
├── requirements.txt
├── .streamlit/
│   └── config.toml               ← Dark theme + server settings
├── data/
│   ├── generate_data.py          ← Synthetic data generator (156 weeks × 5 categories)
│   ├── retail_data.csv           ← Auto-generated on first run
│   └── external_factors.csv      ← Auto-generated on first run
└── src/
    ├── __init__.py
    ├── models.py                 ← XGBoost · Ridge · SARIMA · Elasticity OLS
    ├── optimizer.py              ← SLSQP + Differential Evolution + Monte Carlo
    └── interpreter.py           ← Auto insights · Risk scoring · Recommendations
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```
The app auto-generates all data and trains models on first launch (≈30–60 seconds).

---

## 📊 Application Pages

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | KPI overview, revenue trends, margin heatmaps |
| ⚙️ **Live Simulator** | Real-time P&L with sliders, waterfall chart, tornado analysis |
| 🤖 **ML Engine** | XGBoost demand forecasting, price elasticity OLS, SARIMA time series |
| 🎯 **Margin Optimiser** | SLSQP constrained optimisation, multi-objective trade-off analysis |
| 📊 **Scenario Analysis** | Monte Carlo simulation, saved scenario comparison |
| 🌍 **External Factors** | Macro factor trends, correlation heatmap, stress testing |
| 💡 **Insights & Recs** | Auto-generated insights, risk score, prioritised recommendations |

---

## 📦 Data Model

### `retail_data.csv` — 156 weeks × 5 categories = 780 rows

| Column | Description |
|--------|-------------|
| `date`, `week_num`, `year`, `quarter`, `month` | Time dimensions |
| `category` | Electronics · Apparel · Food & Beverage · Home & Garden · Sports |
| `unit_price`, `effective_price`, `discount_rate_pct` | Pricing |
| `unit_cost`, `fixed_costs`, `promo_flag` | Cost structure |
| `demand_units`, `revenue`, `cogs`, `gross_profit` | Volume & revenue |
| `gross_margin_pct`, `ebit`, `ebit_margin_pct`, `roi_pct` | Profitability KPIs |
| `contribution_margin_pct`, `breakeven_units` | Unit economics |
| `inflation_rate_pct`, `fuel_price_index` | External — cost-side |
| `consumer_confidence_index`, `gdp_growth_pct` | External — demand-side |
| `competitor_price_index`, `exchange_rate_eur_usd` | External — competitive |
| `interest_rate_pct`, `unemployment_rate_pct` | External — macro |

### `external_factors.csv` — 156 weekly macro observations

Standalone table of all 8 external factors with date and week reference.

---

## 🤖 ML Models

| Model | Algorithm | Target | Key Features |
|-------|-----------|--------|--------------|
| DemandForecaster | XGBoost (300 trees) | `demand_units` | Price, cost, seasonality, macro |
| ElasticityEstimator | Log-Log OLS (per category) | `ln(demand)` | `ln(price)` + controls |
| MarginPredictor | Ridge Regression | `gross/ebit_margin_pct` | Full feature set |
| TimeSeriesForecaster | SARIMA(1,1,1)(1,0,1,52) | `revenue`, `ebit` | Weekly series per category |

---

## ⚙️ Optimiser

- **Algorithm**: SLSQP (Sequential Least Squares Programming) with Differential Evolution fallback
- **Objective**: Maximise EBIT
- **Decision Variables**: Unit price, Discount rate
- **Constraints**: Gross margin floor · Price band · Max discount cap · Minimum demand
- **Multi-objective**: Compare EBIT / Revenue / ROI / Gross Margin optimal points

---

## 🎲 Monte Carlo Simulation

- Samples uncertainty in: price (σ%), cost (σ%), demand (σ%), inflation shock
- 1,000–20,000 simulations (user-configurable)
- Outputs: P5/P50/P95 EBIT, probability of loss, CDF chart

---

## 🌍 External Factors Modelled

| Factor | Business Impact |
|--------|-----------------|
| Inflation (CPI) | Raises unit costs; partial margin pass-through |
| Fuel/Energy Index | Logistics & energy costs (Food & Bev most exposed) |
| Consumer Confidence | Demand multiplier for discretionary categories |
| Competitor Price Index | Cross-price elasticity effect on demand |
| EUR/USD Exchange Rate | Import cost sensitivity for sourced products |
| Interest Rate (ECB) | Dampens big-ticket demand (Electronics most sensitive) |
| Unemployment Rate | Baseline consumer spending capacity |
| GDP Growth | Long-run demand trend driver |

---

## ☁️ Deploy to Streamlit Cloud

1. Push to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entrypoint
4. Click **Deploy**

> Streamlit Cloud automatically installs `requirements.txt`.
> Models retrain on every cold start (~60 seconds).

---

## 📋 Industry Benchmarks Used

| Category | Gross Margin | EBIT Margin | ROI |
|----------|-------------|-------------|-----|
| Electronics | 25–40% | 3–10% | 8–20% |
| Apparel | 40–60% | 5–15% | 12–28% |
| Food & Beverage | 20–32% | 2–8% | 6–16% |
| Home & Garden | 32–48% | 4–12% | 10–22% |
| Sports | 35–52% | 4–14% | 10–24% |

---

## 👥 Team
- Insha Siddiqui (25248662)
- Kaariba Khan (25236617)
- Gaurav Aher (25235049)
- Siddhant Gadhe (25249512)
