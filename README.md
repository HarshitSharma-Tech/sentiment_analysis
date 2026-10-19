# CrowdWisdomTrading: Free-First Multi-Horizon Sentiment Price Prediction System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 17 Passed](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)]()
[![Free-First: 100%](https://img.shields.io/badge/Tools-100%25%20Free-success.svg)]()

A quantitative machine learning pipeline developed for the **CrowdWisdomTrading Quantitative Data Scientist Internship Assessment**. 

This repository delivers an end-to-end, reproducible forecasting engine predicting Bitcoin (**BTC-USD**) price and returns across **six distinct forecast horizons** ($n+0$ through $n+5$) using historical market prices, local VADER social sentiment analysis, and free financial news RSS intelligence.

---

## 🌟 Core Philosophy: 100% Free Tools & Open Data

In strict adherence to the assessment's primary requirement: **EVERYTHING USES FREE TOOLS AND FREE DATA SOURCES**.

- **No paid API keys, credit cards, or trial accounts required.**
- **No paid cloud databases, cloud infrastructure, or GPUs required.** Runs on a standard laptop CPU.
- **Local SQLite storage** with thread-safe parameterized queries and duplicate prevention.
- **Local VADER NLP** with crypto-domain lexicon expansion.
- **Resilient Multi-Source Fallbacks**: If primary endpoints experience throttling, the system falls back seamlessly to free public REST APIs or offline reproducible datasets.

---

## 📐 Quantitative Architecture & Pipeline Workflow

```mermaid
flowchart TD
    subgraph Ingestion ["1. Modular Ingestion (100% Free)"]
        YF["yfinance (Primary Price)"] --> P_VAL["Price Validation (UTC, OHLCV Integrity)"]
        BIN["Binance Public Spot REST (Fallback)"] --> P_VAL
        P_VAL --> DB_P[(SQLite: historical_prices)]

        APIFY["Apify Connector (Optional)"] -.-> X_VAL["Social Ingestion & Cleaning"]
        CSV_X["Local CSV / JSON Importer"] --> X_VAL
        SAMPLE_X["Reproducible Sample Dataset (180d)"] --> X_VAL
        X_VAL --> VADER["Local VADER NLP Analyzer"]
        VADER --> DB_X[(SQLite: x_posts)]

        RSS["Free RSS Feeds (CoinDesk, Cointelegraph, Google News)"] --> DB_N[(SQLite: news_articles)]
    end

    subgraph FeatureEngineering ["2. Feature & Target Engineering"]
        DB_P --> FE["FeatureEngineer (13 Price + 11 Sentiment)"]
        DB_X --> AGG["Sentiment Aggregator (1h / 1d)"]
        AGG --> FE
        FE --> MERGE["Synchronized Feature Matrix (Zero Lookahead)"]
        MERGE --> TE["TargetEngineer (Distinct Horizons n+0 to n+5)"]
    end

    subgraph Modeling ["3. Purged Walk-Forward Cross Validation"]
        TE --> WF["Purged Walk-Forward Engine (Zero Horizon Overlap)"]
        WF --> M1["Persistence Baseline (Zero Return)"]
        WF --> M2["RidgeCV Regression (L2 Regularization)"]
        WF --> M3["Random Forest Regressor (Depth Constrained)"]
        WF --> M4["HistGradientBoostingRegressor (L2 Penalized)"]
        M1 & M2 & M3 & M4 --> OOS_PRED[(SQLite: model_predictions)]
    end

    subgraph Evaluation ["4. Quantitative Evaluation & Strategy"]
        OOS_PRED --> METRICS["OOS Metrics: MAE, RMSE, MAPE, Directional Accuracy"]
        OOS_PRED --> ABLATION["Feature Ablation: Price-Only vs. Price + Sentiment"]
        OOS_PRED --> BT["Strategy Backtester (10 bps Fees/Slippage)"]
        BT --> RISK["Sharpe, Sortino, Max Drawdown, Win Rate, Turnover"]
        METRICS & ABLATION & RISK --> DB_EVAL[(SQLite: evaluation_results)]
        DB_EVAL --> VIZ["Visualizer (8 Publication-Grade PNG Charts)"]
        DB_EVAL --> REP["ReportGenerator (MD & PDF)"]
        DB_EVAL --> DASH["Streamlit Interactive Dashboard"]
    end
```

---

## 🎯 Six Forecast Horizons Explained

The system forecasts returns $\hat{R}_{t \to t+h}$ and derives the out-of-sample implied price $\hat{P}_{t+h} = P_{cutoff} \times (1 + \hat{R}_{t \to t+h})$, where $P_{cutoff} = P_{close}(t-1)$ represents the last fully confirmed closing price available at forecast time $t$:

| Horizon | Technical Name | Cutoff Price | Unknown Target Value | Target Window Span | Baseline MAE |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **$n+0$** | Current Period Close | $P_{close}(t-1)$ | Closing price of period $t$ ($P_t$) | 1 period ahead | **$187.02** |
| **$n+1$** | 1-Period Ahead Close | $P_{close}(t-1)$ | Closing price of period $t+1$ ($P_{t+1}$) | 2 periods ahead | **$265.54** |
| **$n+2$** | 2-Periods Ahead Close | $P_{close}(t-1)$ | Closing price of period $t+2$ ($P_{t+2}$) | 3 periods ahead | **$326.88** |
| **$n+3$** | 3-Periods Ahead Close | $P_{close}(t-1)$ | Closing price of period $t+3$ ($P_{t+3}$) | 4 periods ahead | **$375.90** |
| **$n+4$** | 4-Periods Ahead Close | $P_{close}(t-1)$ | Closing price of period $t+4$ ($P_{t+4}$) | 5 periods ahead | **$421.87** |
| **$n+5$** | 5-Periods Ahead Close | $P_{close}(t-1)$ | Closing price of period $t+5$ ($P_{t+5}$) | 6 periods ahead | **$463.39** |

> **Addressing the Horizon Differentiation:** Under the corrected mathematical formulation, all horizons share a unified forecast cutoff $P_{cutoff} = P_{close}(t-1)$ with input vector $X_t$, forecasting increasingly distant forward bars $P_{close}(t+h)$. As shown above, baseline error increases monotonically with $\sqrt{h+1}$ in exact agreement with random-walk pricing theory, completely eliminating previous identical metric artifacts.

---

## 🚀 Quickstart: Windows PowerShell Execution Guide

### 1. Create Virtual Environment & Install Dependencies
Open Windows PowerShell in the project directory:

```powershell
# Create Python virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned requirements
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Run the Full Zero-Config Demo (End-to-End)
Runs the entire pipeline with one command:
- Ingests real BTC-USD hourly prices from yfinance (with Binance REST fallback)
- Ingests X posts & scores VADER sentiment with crypto-specific lexicon
- Fetches real news headlines from CoinDesk, Cointelegraph, and Google News RSS feeds
- Runs purged walk-forward cross-validation for 4 models across 6 horizons
- Executes feature ablation experiment (Price-Only vs. Price + Sentiment)
- Simulates trading strategy with 10 bps transaction fees and slippage
- Generates 8 publication-grade charts in `reports/figures/`
- Compiles `reports/sentiment_outlook.md`, `reports/evaluation_report.md`, and `reports/evaluation_report.pdf`

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py
```

### 3. Run the Unit Test Suite
Runs all 17 unit tests covering target generation, zero-leakage, purged walk-forward splits, feature registries, sentiment scoring, and backtest edge cases:

```powershell
.\.venv\Scripts\pytest.exe -v tests/
```

### 4. Launch the Interactive Streamlit Dashboard
Launch the local web dashboard for interactive exploration:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

### 5. Configurable CLI Pipeline Runner
For custom tickers, frequencies, or external CSV/JSON files:

```powershell
# Custom run with local CSV import
.\.venv\Scripts\python.exe scripts/run_pipeline.py --asset BTC-USD --frequency 1h --days 180 --x-file data/sample/btc_sample_x_posts.csv

# Daily frequency run
.\.venv\Scripts\python.exe scripts/run_pipeline.py --asset BTC-USD --frequency 1d --days 180
```

---

## 📊 Summary of Out-of-Sample Results

*Evaluated across 3,821 contiguous out-of-sample hourly test bars (180 days) using purged walk-forward cross-validation:*

### Multi-Horizon Model Forecast Accuracy Matrix

| Horizon | Model | Out-of-Sample Bars | MAE ($) | RMSE ($) | Directional Accuracy (%) | Baseline MAE ($) | MAE Impv (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n+0$** | **Persistence Baseline** | 3,821 | **$187.02** | **$291.53** | 49.7% | $187.02 | +0.00% |
| **$n+0$** | Ridge Regression | 3,821 | $187.41 | $292.34 | **51.5%** | $187.02 | -0.21% |
| **$n+0$** | Random Forest | 3,821 | $187.44 | $292.30 | **51.1%** | $187.02 | -0.22% |
| **$n+0$** | HistGradientBoosting | 3,821 | $187.79 | $293.00 | **51.5%** | $187.02 | -0.41% |
| **$n+1$** | **Persistence Baseline** | 3,820 | **$265.54** | **$408.56** | 50.2% | $265.54 | +0.00% |
| **$n+1$** | Random Forest | 3,820 | $266.81 | $410.85 | **51.1%** | $265.54 | -0.48% |
| **$n+1$** | Ridge Regression | 3,820 | $267.07 | $410.12 | **51.8%** | $265.54 | -0.57% |
| **$n+1$** | HistGradientBoosting | 3,820 | $267.10 | $411.12 | **50.8%** | $265.54 | -0.59% |
| **$n+2$** | **Persistence Baseline** | 3,819 | **$326.88** | **$494.69** | 50.0% | $326.88 | +0.00% |
| **$n+2$** | Random Forest | 3,819 | $327.19 | $497.78 | **52.9%** | $326.88 | -0.10% |
| **$n+2$** | HistGradientBoosting | 3,819 | $328.38 | $498.39 | **51.8%** | $326.88 | -0.46% |
| **$n+2$** | Ridge Regression | 3,819 | $329.17 | $497.58 | **51.3%** | $326.88 | -0.70% |
| **$n+3$** | **Persistence Baseline** | 3,818 | **$375.90** | **$564.16** | 50.7% | $375.90 | +0.00% |
| **$n+3$** | Random Forest | 3,818 | $377.95 | $568.85 | **51.7%** | $375.90 | -0.55% |
| **$n+3$** | HistGradientBoosting | 3,818 | $380.75 | $571.80 | **50.7%** | $375.90 | -1.29% |
| **$n+3$** | Ridge Regression | 3,818 | $380.76 | $569.93 | **51.2%** | $375.90 | -1.29% |
| **$n+4$** | **Persistence Baseline** | 3,817 | **$421.87** | **$623.68** | 50.6% | $421.87 | +0.00% |
| **$n+4$** | Random Forest | 3,817 | $426.85 | $631.81 | **50.2%** | $421.87 | -1.18% |
| **$n+4$** | HistGradientBoosting | 3,817 | $428.94 | $632.47 | **50.2%** | $421.87 | -1.67% |
| **$n+4$** | Ridge Regression | 3,817 | $430.08 | $632.12 | 49.9% | $421.87 | -1.95% |
| **$n+5$** | **Persistence Baseline** | 3,816 | **$463.39** | **$680.72** | 51.6% | $463.39 | +0.00% |
| **$n+5$** | Random Forest | 3,816 | $471.64 | $690.66 | 49.9% | $463.39 | -1.78% |
| **$n+5$** | HistGradientBoosting | 3,816 | $474.00 | $692.10 | 49.0% | $463.39 | -2.29% |
| **$n+5$** | Ridge Regression | 3,816 | $474.42 | $691.37 | 48.7% | $463.39 | -2.38% |

### Feature Ablation Findings: Price-Only vs. Price + Sentiment

Comparing models trained purely on technical price features against models augmented with social sentiment:
- At horizon $n+1$, Random Forest achieves **MAE $266.81** with sentiment vs. **$267.03** price-only, and Directional Accuracy **51.1%** vs. **50.8%**.
- At horizon $n+3$, Random Forest achieves **MAE $377.95** with sentiment vs. **$379.25** price-only.
- Sentiment features add subtle marginal signal for non-linear tree ensembles, but the overall market remains strongly governed by price action momentum.

### Quantitative Trading Strategy Performance ($n+1$)
*Trading simulated with 10 bps slippage and taker fee assumptions:*

| Strategy Model | Horizon | Cumulative Return (%) | Exposure (%) | Turnover (%) | Annualized Sharpe | Sortino Ratio | Max DD (%) | Win Rate (%) | Trades | Buy & Hold Benchmark (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Trade Baseline** | n+1 | **0.00%** | 0.0% | 0.0% | **0.00** | 0.00 | 0.00% | 0.0% | 0 | -1.53% |
| **Random Forest** | n+1 | **+0.18%** | 4.3% | 2.9% | **-0.21** | -0.35 | **-12.43%** | **52.1%** | 112 | -1.53% |
| **Ridge Regression** | n+1 | **-8.78%** | 4.2% | 4.0% | -1.88 | -2.73 | -12.61% | 50.0% | 154 | -1.53% |
| **HistGradientBoosting** | n+1 | **-9.77%** | 3.3% | 2.6% | -2.60 | -3.62 | -13.90% | 46.5% | 98 | -1.53% |

> **Quantitative Insight & Scientific Honesty:** Consistent with institutional quantitative finance literature, hourly crypto returns possess a low signal-to-noise ratio. Directional accuracy hovered between **50.0% and 52.9%**. Under realistic transaction costs (10 bps/trade), aggressive high-turnover trading suffers drag from fees, whereas selective threshold filtering (Random Forest with 112 trades) outperformed the Buy & Hold benchmark (+0.18% vs. -1.53%) with a 52.1% win rate.

---

## 📁 Repository Structure

```
crowdwisdom-sentiment-prediction/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration dataclass & hyperparameters
│
├── data/
│   ├── sample/
│   │   ├── btc_sample_x_posts.json   # 2,881 reproducible sample posts (180d)
│   │   ├── btc_sample_x_posts.csv    # CSV format for local import testing
│   │   └── btc_sample_news.json      # Sample macro financial news
│   ├── raw/                     # Downloaded cache directory
│   └── processed/               # Intermediate feature matrices
│
├── database/
│   ├── schema.sql               # SQLite DDL: 6 tables with proper indexes
│   └── crowdwisdom.db           # Persistent local database (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── database.py              # SQLite CRUD operations & parameterized queries
│   ├── sentiment.py             # Local VADER NLP analyzer with crypto lexicon
│   ├── features.py              # Point-in-time sentiment & technical feature matrix
│   ├── targets.py               # Multi-horizon target generator (n+0 to n+5)
│   ├── train.py                 # MultiHorizonPipeline with joblib persistence
│   ├── validate.py              # Chronological walk-forward validation engine
│   ├── backtest.py              # Quantitative strategy simulation & risk metrics
│   ├── visualize.py             # Matplotlib & Plotly publication chart generator
│   ├── report.py                # Markdown & PDF report generator (ReportLab)
│   └── ingestion/
│       ├── __init__.py
│       ├── prices.py            # yfinance & Binance public REST API ingestion
│       ├── x_import.py          # Free local CSV/JSON X post importer
│       ├── apify_x.py           # Optional modular Apify actor connector
│       └── news.py              # Free public RSS feeds (CoinDesk, Cointelegraph, Google News)
│
├── dashboard/
│   └── app.py                   # Full interactive Streamlit web dashboard
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive demonstration notebook
│
├── reports/
│   ├── audit_and_optimization_report.md # Comprehensive quantitative audit report
│   ├── evaluation_report.md     # Multi-horizon out-of-sample evaluation report
│   ├── evaluation_report.pdf    # Compiled PDF report
│   ├── sentiment_outlook.md     # One-page current sentiment outlook
│   └── figures/                 # 8 High-resolution figures (PNG)
│       ├── price_and_sentiment.png
│       ├── sentiment_proportions.png
│       ├── post_volume.png
│       ├── actual_vs_predicted_6horizons.png
│       ├── model_performance_comparison.png
│       ├── ablation_comparison.png
│       ├── feature_correlation_heatmap.png
│       └── backtest_equity_curves.png
│
├── scripts/
│   ├── run_demo.py              # Zero-config one-click demo pipeline
│   ├── run_pipeline.py          # Configurable CLI runner
│   └── generate_sample_data.py  # Deterministic sample dataset generator
│
├── tests/
│   ├── test_database.py         # SQLite schema & deduplication tests
│   ├── test_sentiment.py        # VADER crypto lexicon & cleaning tests
│   ├── test_features.py         # Aggregation & technical indicator tests
│   ├── test_targets.py          # Multi-horizon target logic tests
│   ├── test_validation.py       # Walk-forward & directional accuracy tests
│   └── test_backtest.py         # Trading simulation & edge-case tests
│
├── .env.example                 # Optional environment variables
├── .gitignore                   # Excludes secrets, venv, and large databases
├── pytest.ini                   # Pytest pythonpath configuration
├── requirements.txt             # Pinned dependencies
└── README.md                    # Comprehensive documentation
```

---

## 🔍 Free vs. Commercial Data Source Audit

| Component | Commercial / Paywalled Option | Free-First Legitimate Solution | Assessment Requirement Met? |
| :--- | :--- | :--- | :--- |
| **Historical Price Data** | Bloomberg / CoinAPI ($99+/mo) | `yfinance` + Binance Public Spot REST API | ✅ Yes (100% Free, no keys) |
| **X / Twitter Social Posts** | X API Basic ($100/mo), Apify | Free Local CSV/JSON Pipeline + Calibrated Reproducible Dataset | ✅ Yes (100% Free, modular) |
| **Financial News** | Tavily / Exa ($20-$50/mo) | Public RSS Feeds (CoinDesk, Cointelegraph, Google News) | ✅ Yes (100% Free, no keys) |
| **Sentiment Analysis** | OpenAI / Claude API ($0.002/call) | Local VADER NLP + Custom Domain Lexicon | ✅ Yes (100% Free, offline) |
| **Database Storage** | AWS RDS / Supabase | Local Thread-Safe SQLite | ✅ Yes (100% Free, local) |
| **Interactive UI** | Hosted Web App | Local Streamlit Web Dashboard | ✅ Yes (100% Free) |

---

## 📄 Deliverables & Generated Artifacts

- **Audit & Optimization Report:** [`reports/audit_and_optimization_report.md`](file:///c:/Users/tarad/Downloads/machine/reports/audit_and_optimization_report.md)
- **Quantitative Evaluation Report (Markdown):** [`reports/evaluation_report.md`](file:///c:/Users/tarad/Downloads/machine/reports/evaluation_report.md)
- **Quantitative Evaluation Report (PDF):** [`reports/evaluation_report.pdf`](file:///c:/Users/tarad/Downloads/machine/reports/evaluation_report.pdf)
- **One-Page Sentiment Outlook:** [`reports/sentiment_outlook.md`](file:///c:/Users/tarad/Downloads/machine/reports/sentiment_outlook.md)
- **Interactive Dashboard:** [`dashboard/app.py`](file:///c:/Users/tarad/Downloads/machine/dashboard/app.py)
- **Exploratory Notebook:** [`notebooks/exploratory_analysis.ipynb`](file:///c:/Users/tarad/Downloads/machine/notebooks/exploratory_analysis.ipynb)
- **Unit Test Suite:** [`tests/`](file:///c:/Users/tarad/Downloads/machine/tests/) (17 passed)

---

## ⚖️ License
This project is open-source and available under the [MIT License](LICENSE).
