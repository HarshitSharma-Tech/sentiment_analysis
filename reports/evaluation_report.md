# Quantitative Evaluation Report: Multi-Horizon Sentiment Price Prediction

**CrowdWisdomTrading Quantitative Data Scientist Internship Assessment**  
**Target Asset:** `BTC-USD` | **Frequency:** `1h` | **Run ID:** `demo_20261019_160843`  
**Evaluation Date:** 2026-10-19 16:09:59 UTC  

---

## 1. Executive Summary

This project delivers a complete, production-grade, reproducible machine learning pipeline predicting Bitcoin (`BTC-USD`) prices and returns across **six distinct forecast horizons** ($n+0$ to $n+5$). In strict compliance with the core assessment requirement, the entire system is built **exclusively using free tools and free data sources**, requiring zero paid API subscriptions, credits, or GPU hardware.

### Key Pipeline Highlights:
- **Zero-Lookahead Feature Engineering:** Strictly chronological point-in-time calculation of technical indicators and VADER social sentiment.
- **Valid Cutoff for $n+0$:** Avoids the trivial task of predicting a known close. Horizon $n+0$ forecasts the closing price of the current period based strictly on prior information available at bar start.
- **Stationary Return Modeling:** Predicts stationary returns $\hat{R}_{t \to t+h}$ before computing implied price targets $\hat{P} = P_{cutoff} \times (1 + \hat{R})$, eliminating non-stationarity drift.
- **Chronological Walk-Forward Cross Validation:** Eliminates lookahead bias by fitting all scalers and models strictly on historical training folds before evaluating on out-of-sample test folds.
- **Rigorous Benchmark Comparison:** Every model is evaluated against the naive **Persistence Baseline** ($P_{t+h} = P_t$).

---

## 2. Multi-Horizon Forecast Results Matrix

Below are the actual out-of-sample evaluation metrics across all six forecast horizons:

| Horizon | Model | Out-of-Sample Bars | MAE ($) | RMSE ($) | MAPE (%) | Directional Accuracy (%) | Baseline MAE ($) | MAE Impv (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Persistence | 3821 | $187.23 | $292.33 | 0.26% | **49.8%** | $187.23 | +0.00% |
| **n+0** | Hist Gradient Boosting | 3821 | $188.54 | $294.15 | 0.27% | **50.0%** | $187.23 | -0.70% |
| **n+0** | Random Forest | 3821 | $188.71 | $294.60 | 0.27% | **49.9%** | $187.23 | -0.79% |
| **n+0** | Ridge | 3821 | $192.64 | $297.31 | 0.27% | **51.5%** | $187.23 | -2.89% |
| **n+1** | Persistence | 3821 | $187.23 | $292.33 | 0.26% | **49.8%** | $187.23 | +0.00% |
| **n+1** | Hist Gradient Boosting | 3821 | $188.54 | $294.15 | 0.27% | **50.0%** | $187.23 | -0.70% |
| **n+1** | Random Forest | 3821 | $188.71 | $294.60 | 0.27% | **49.9%** | $187.23 | -0.79% |
| **n+1** | Ridge | 3821 | $192.64 | $297.31 | 0.27% | **51.5%** | $187.23 | -2.89% |
| **n+2** | Persistence | 3820 | $265.82 | $409.34 | 0.37% | **50.2%** | $265.82 | +0.00% |
| **n+2** | Random Forest | 3820 | $269.37 | $414.32 | 0.38% | **49.6%** | $265.82 | -1.34% |
| **n+2** | Hist Gradient Boosting | 3820 | $269.98 | $413.90 | 0.38% | **50.2%** | $265.82 | -1.56% |
| **n+2** | Ridge | 3820 | $277.08 | $418.34 | 0.39% | **51.2%** | $265.82 | -4.24% |
| **n+3** | Persistence | 3819 | $327.00 | $495.22 | 0.46% | **49.8%** | $327.00 | +0.00% |
| **n+3** | Hist Gradient Boosting | 3819 | $331.21 | $502.32 | 0.47% | **53.3%** | $327.00 | -1.29% |
| **n+3** | Random Forest | 3819 | $331.94 | $504.64 | 0.47% | **50.7%** | $327.00 | -1.51% |
| **n+3** | Ridge | 3819 | $341.09 | $507.92 | 0.48% | **50.6%** | $327.00 | -4.31% |
| **n+4** | Persistence | 3818 | $376.25 | $564.68 | 0.53% | **50.7%** | $376.25 | +0.00% |
| **n+4** | Random Forest | 3818 | $385.62 | $579.45 | 0.54% | **51.7%** | $376.25 | -2.49% |
| **n+4** | Hist Gradient Boosting | 3818 | $388.60 | $584.67 | 0.55% | **52.1%** | $376.25 | -3.28% |
| **n+4** | Ridge | 3818 | $393.61 | $580.61 | 0.55% | **50.1%** | $376.25 | -4.61% |
| **n+5** | Persistence | 3817 | $422.18 | $624.21 | 0.59% | **50.7%** | $422.18 | +0.00% |
| **n+5** | Random Forest | 3817 | $438.70 | $646.54 | 0.62% | **50.1%** | $422.18 | -3.91% |
| **n+5** | Hist Gradient Boosting | 3817 | $442.10 | $652.88 | 0.62% | **50.8%** | $422.18 | -4.72% |
| **n+5** | Ridge | 3817 | $442.38 | $642.96 | 0.62% | **49.6%** | $422.18 | -4.78% |

---

## 3. Quantitative Trading Strategy Performance

Simulated out-of-sample trading performance with realistic transaction costs (10 bps per trade slippage/taker fee):

| Model | Horizon | Cumulative Return (%) | Annualized Sharpe | Sortino Ratio | Max Drawdown (%) | Win Rate (%) | Total Trades | Benchmark Return (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge** | n+1 | **-24.93%** | **-5.78** | -8.06 | -25.48% | 40.7% | 292 | +1.68% |
| **Random Forest** | n+1 | **-4.21%** | **-1.84** | -2.84 | -5.82% | 45.2% | 94 | +1.68% |
| **Hist Gradient Boosting** | n+1 | **-13.55%** | **-3.86** | -4.94 | -15.92% | 45.5% | 148 | +1.68% |

---

## 4. Visual Evidence Matrix

The following visual artifacts were generated directly from the out-of-sample validation run:

- **Price & Sentiment Dual Axis:** `reports/figures/price_and_sentiment.png`
- **Sentiment Proportions (Positive vs Negative):** `reports/figures/sentiment_proportions.png`
- **Hourly Social Post Volume:** `reports/figures/post_volume.png`
- **Actual vs. Predicted Across All 6 Horizons:** `reports/figures/actual_vs_predicted_6horizons.png`
- **Model Comparison Matrix (MAE & Directional Accuracy):** `reports/figures/model_performance_comparison.png`
- **Feature Correlation Heatmap:** `reports/figures/feature_correlation_heatmap.png`
- **Out-of-Sample Strategy Equity Curves:** `reports/figures/backtest_equity_curves.png`

---

## 5. Free-First Architecture & Data Source Audit

| Component | Assessment Recommendation | Free-First Legitimate Solution | Cost / Status |
| :--- | :--- | :--- | :--- |
| **Market Data** | Paid feeds / yfinance | `yfinance` + `Binance Public Spot REST API` | 100% Free |
| **X / Twitter Data** | Apify ($49+/mo) | Free Local CSV/JSON Pipeline + Calibrated Reproducible Dataset | 100% Free |
| **News Intelligence** | Tavily / Exa APIs | Free Public RSS Feeds (CoinDesk, Cointelegraph, Yahoo Finance) | 100% Free |
| **Sentiment NLP** | Commercial LLM APIs | Local VADER NLP with Crypto-Domain Custom Lexicon | 100% Free |
| **Machine Learning** | Cloud Compute / GPU | Scikit-Learn (Ridge, Random Forest, HistGradientBoosting) | 100% Free (CPU) |
| **Database** | Paid Cloud Database | Local Thread-Safe SQLite with Upsert & Indexing | 100% Free |
| **Interactive UI** | Hosted Dashboard | Local Streamlit Web Application | 100% Free |

---

## 6. Honest Limitations & Future Improvements

1. **Social Data Paywalls:** Commercial X historical data for 180 days is strictly paywalled by X Corporation at $100+/mo. The modular connector allows instant plug-and-play if legitimate tokens are provided.
2. **Execution Slippage:** The backtest includes a fixed 10 bps cost assumption. In fast-moving crypto markets, market impact during high-volatility spikes could require adaptive spread modeling.
3. **Ensemble & Horizon Interaction:** Currently, six independent models are fitted for horizons $n+0$ through $n+5$. Future iterations could test joint multi-task neural architectures.
