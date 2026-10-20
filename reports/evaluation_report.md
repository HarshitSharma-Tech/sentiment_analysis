# Quantitative Evaluation Report: Multi-Horizon Sentiment Price Prediction

**CrowdWisdomTrading Quantitative Data Scientist Internship Assessment**  
**Target Asset:** `BTC-USD` | **Frequency:** `1h` | **Run ID:** `demo_20261020_161001`  
**Evaluation Date:** 2026-10-20 16:11:03 UTC  

---

## 1. Executive Summary & Research Methodology

This report documents out-of-sample predictive performance across **six distinct forecast horizons** ($n+0$ to $n+5$) for Bitcoin (`BTC-USD`). The system is engineered to institutional quantitative standards using **100% free and open-source tools**.

### Core Methodology & Leakage Protections:
- **Strict Point-in-Time Cutoff:** All six forecast horizons share the identical reference price $P_{cutoff}(t) = P_{close}(t-1)$ and feature vector $X_t$ (shifted by 1 to use information strictly up to $close(t-1)$).
- **Target Horizons & Realization:**
  - $n+0$: Predicts $close(t)$ (1 hour ahead of cutoff; realizes at $t + 1\text{h}$).
  - $n+1$: Predicts $close(t+1)$ (2 hours ahead of cutoff; realizes at $t + 2\text{h}$).
  - $n+2 \dots n+5$: Predicts $close(t+h)$ (realizes at $t + (h+1)\text{h}$).
  - All targets are returns $R_t^{(h)} = (close(t+h) - close(t-1)) / close(t-1)$.
  - Predicted price is reconstructed as $\hat{P}_{t+h} = close(t-1) \times (1 + \hat{R}_t^{(h)})$.
- **Chronological Data Partitioning:**
  - **70% Development Period:** Used for model tuning, feature selection, and purged walk-forward cross-validation.
  - **30% Frozen Chronological Holdout:** Strictly quarantined and evaluated once without parameter retuning.
- **Purging + 5-Bar Embargo:** Eliminates overlapping label lookahead and auto-regressive rolling feature memory across fold boundaries.
- **Paired Statistical Tests:** Diebold-Mariano test (HLN 1997 adjusted) against Naive Persistence baseline with Benjamini-Hochberg False Discovery Rate (FDR) corrections.
- **Non-Overlapping Backtest:** Simulates hourly spot executions with 10 bps friction, eliminating multi-period compounding artifacts.

---

## 2. Multi-Horizon Forecast Results Matrix

### 2A. Development Period: Purged Walk-Forward Cross Validation (First 70% Window)
*Evaluated across rolling walk-forward test folds with $h$-step purging and 5-bar embargo:*

| Horizon | Model | Out-of-Sample Bars | MAE ($) | RMSE ($) | Dir Acc (%) | Maj Class Base (%) | Base MAE ($) | DM Stat | DM p-val (FDR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Persistence | 2513 | $184.83 | $292.42 | **49.1%** | 50.9% | $184.83 | +0.00 | 1.000 |
| **n+0** | Random Forest | 2513 | $184.85 | $292.87 | **52.3%** | 50.9% | $184.83 | -0.72 | 1.000 |
| **n+0** | Ridge | 2513 | $185.16 | $292.94 | **51.4%** | 50.9% | $184.83 | -0.70 | 1.000 |
| **n+0** | Hist Gradient Boosting | 2513 | $185.83 | $293.53 | **52.0%** | 50.9% | $184.83 | -1.34 | 1.000 |
| **n+1** | Persistence | 2513 | $262.56 | $406.61 | **49.1%** | 50.9% | $262.56 | +0.00 | 1.000 |
| **n+1** | Ridge | 2513 | $263.63 | $407.38 | **52.7%** | 50.9% | $262.56 | -0.48 | 1.000 |
| **n+1** | Random Forest | 2513 | $263.83 | $408.28 | **52.3%** | 50.9% | $262.56 | -1.21 | 1.000 |
| **n+1** | Hist Gradient Boosting | 2513 | $263.83 | $408.92 | **52.2%** | 50.9% | $262.56 | -1.73 | 1.000 |
| **n+2** | Persistence | 2513 | $321.57 | $490.77 | **49.1%** | 50.9% | $321.57 | +0.00 | 1.000 |
| **n+2** | Random Forest | 2513 | $322.36 | $493.37 | **52.3%** | 50.9% | $321.57 | -1.40 | 1.000 |
| **n+2** | Ridge | 2513 | $323.72 | $492.73 | **50.7%** | 50.9% | $321.57 | -0.77 | 1.000 |
| **n+2** | Hist Gradient Boosting | 2513 | $323.86 | $495.65 | **52.8%** | 50.9% | $321.57 | -2.44 | 1.000 |
| **n+3** | Persistence | 2513 | $369.21 | $556.72 | **50.0%** | 50.0% | $369.21 | +0.00 | 1.000 |
| **n+3** | Random Forest | 2513 | $373.61 | $562.33 | **49.9%** | 50.0% | $369.21 | -2.22 | 1.000 |
| **n+3** | Ridge | 2513 | $376.30 | $562.92 | **50.9%** | 50.0% | $369.21 | -1.84 | 1.000 |
| **n+3** | Hist Gradient Boosting | 2513 | $376.75 | $566.50 | **49.9%** | 50.0% | $369.21 | -3.25 | 1.000 |
| **n+4** | Persistence | 2513 | $413.68 | $614.36 | **49.9%** | 50.1% | $413.68 | +0.00 | 1.000 |
| **n+4** | Random Forest | 2513 | $421.61 | $623.53 | **49.1%** | 50.1% | $413.68 | -2.77 | 1.000 |
| **n+4** | Hist Gradient Boosting | 2513 | $422.88 | $624.61 | **49.5%** | 50.1% | $413.68 | -2.97 | 1.000 |
| **n+4** | Ridge | 2513 | $422.99 | $621.36 | **49.3%** | 50.1% | $413.68 | -1.82 | 1.000 |
| **n+5** | Persistence | 2513 | $455.21 | $670.18 | **51.3%** | 51.3% | $455.21 | +0.00 | 1.000 |
| **n+5** | Random Forest | 2513 | $466.57 | $681.48 | **46.8%** | 51.3% | $455.21 | -2.98 | 1.000 |
| **n+5** | Ridge | 2513 | $466.65 | $679.09 | **49.1%** | 51.3% | $455.21 | -2.00 | 1.000 |
| **n+5** | Hist Gradient Boosting | 2513 | $469.09 | $685.13 | **48.4%** | 51.3% | $455.21 | -3.74 | 1.000 |

### 2B. Frozen Chronological Holdout: Out-of-Sample Evaluation (Final 30% Window)
*Trained on Development Period with 5-bar embargo; tested strictly once on quarantined future data:*

| Horizon | Model | Holdout Bars | MAE ($) | RMSE ($) | Dir Acc (%) | Maj Class Base (%) | Base MAE ($) | DM Stat | DM p-val (FDR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Persistence | 1284 | $191.16 | $291.93 | **51.0%** | 51.0% | $191.16 | +0.00 | 0.997 |
| **n+0** | Random Forest | 1284 | $191.25 | $292.05 | **51.2%** | 51.0% | $191.16 | -0.18 | 0.997 |
| **n+0** | Hist Gradient Boosting | 1284 | $191.68 | $292.75 | **49.0%** | 51.0% | $191.16 | -1.27 | 0.997 |
| **n+0** | Ridge | 1284 | $192.73 | $293.56 | **50.8%** | 51.0% | $191.16 | -1.72 | 0.997 |
| **n+1** | Persistence | 1283 | $270.48 | $413.98 | **52.5%** | 52.5% | $270.48 | +0.00 | 0.997 |
| **n+1** | Random Forest | 1283 | $271.17 | $414.95 | **50.0%** | 52.5% | $270.48 | -0.83 | 0.997 |
| **n+1** | Hist Gradient Boosting | 1283 | $273.17 | $416.76 | **51.8%** | 52.5% | $270.48 | -2.29 | 0.997 |
| **n+1** | Ridge | 1283 | $274.58 | $417.47 | **50.4%** | 52.5% | $270.48 | -1.97 | 0.997 |
| **n+2** | Random Forest | 1282 | $334.49 | $503.14 | **51.2%** | 51.3% | $335.26 | -0.47 | 0.997 |
| **n+2** | Persistence | 1282 | $335.26 | $502.24 | **51.3%** | 51.3% | $335.26 | +0.00 | 0.997 |
| **n+2** | Hist Gradient Boosting | 1282 | $336.53 | $505.24 | **53.2%** | 51.3% | $335.26 | -1.75 | 0.997 |
| **n+2** | Ridge | 1282 | $339.91 | $508.53 | **50.6%** | 51.3% | $335.26 | -2.26 | 0.997 |
| **n+3** | Random Forest | 1281 | $386.62 | $579.12 | **53.6%** | 52.2% | $387.65 | -0.62 | 0.997 |
| **n+3** | Persistence | 1281 | $387.65 | $577.26 | **52.2%** | 52.2% | $387.65 | +0.00 | 0.997 |
| **n+3** | Hist Gradient Boosting | 1281 | $388.84 | $581.51 | **55.7%** | 52.2% | $387.65 | -1.83 | 0.997 |
| **n+3** | Ridge | 1281 | $391.77 | $584.27 | **52.5%** | 52.2% | $387.65 | -2.01 | 0.997 |
| **n+4** | Random Forest | 1280 | $436.47 | $643.48 | **53.4%** | 52.5% | $436.63 | -0.75 | 0.997 |
| **n+4** | Persistence | 1280 | $436.63 | $640.04 | **52.5%** | 52.5% | $436.63 | +0.00 | 0.997 |
| **n+4** | Hist Gradient Boosting | 1280 | $439.15 | $646.54 | **51.2%** | 52.5% | $436.63 | -1.70 | 0.997 |
| **n+4** | Ridge | 1280 | $444.65 | $652.73 | **50.9%** | 52.5% | $436.63 | -2.57 | 0.997 |
| **n+5** | Persistence | 1279 | $477.62 | $699.24 | **52.0%** | 52.0% | $477.62 | +0.00 | 0.997 |
| **n+5** | Random Forest | 1279 | $478.63 | $703.53 | **52.2%** | 52.0% | $477.62 | -0.77 | 0.997 |
| **n+5** | Hist Gradient Boosting | 1279 | $480.73 | $704.81 | **52.2%** | 52.0% | $477.62 | -1.23 | 0.997 |
| **n+5** | Ridge | 1279 | $488.89 | $715.25 | **50.5%** | 52.0% | $477.62 | -2.78 | 0.997 |

---

## 3. Feature Ablation Study: Price-Only vs. Price + Sentiment

> **[!IMPORTANT] Social Sentiment Data Provenance Disclosure:**
> The social sentiment ablation reported below was conducted using the calibrated offline reproducible sample dataset (2,881 synthetic posts), NOT live historical X Corporation firehose data. In an unpaywalled environment, these metrics demonstrate that the pipeline mechanics and ablation code execute correctly; they do NOT prove live social media alpha.

| Horizon | Model | Feature Set | MAE ($) | RMSE ($) | Directional Accuracy (%) | Baseline MAE ($) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **n+0** | Ridge | Price Only | $192.35 | $293.09 | **49.4%** | $191.16 |
| **n+0** | Ridge | Price + Sentiment | $192.73 | $293.56 | **50.8%** | $191.16 |
| **n+0** | Random Forest | Price Only | $191.60 | $292.46 | **49.6%** | $191.16 |
| **n+0** | Random Forest | Price + Sentiment | $191.25 | $292.05 | **51.2%** | $191.16 |
| **n+1** | Ridge | Price Only | $273.79 | $417.00 | **49.1%** | $270.48 |
| **n+1** | Ridge | Price + Sentiment | $274.58 | $417.47 | **50.4%** | $270.48 |
| **n+1** | Random Forest | Price Only | $272.04 | $415.53 | **52.8%** | $270.48 |
| **n+1** | Random Forest | Price + Sentiment | $271.17 | $414.95 | **50.0%** | $270.48 |
| **n+2** | Ridge | Price Only | $338.99 | $507.18 | **50.0%** | $335.26 |
| **n+2** | Ridge | Price + Sentiment | $339.91 | $508.53 | **50.6%** | $335.26 |
| **n+2** | Random Forest | Price Only | $334.95 | $504.16 | **53.3%** | $335.26 |
| **n+2** | Random Forest | Price + Sentiment | $334.49 | $503.14 | **51.2%** | $335.26 |
| **n+3** | Ridge | Price Only | $394.21 | $586.21 | **50.1%** | $387.65 |
| **n+3** | Ridge | Price + Sentiment | $391.77 | $584.27 | **52.5%** | $387.65 |
| **n+3** | Random Forest | Price Only | $386.75 | $580.37 | **54.1%** | $387.65 |
| **n+3** | Random Forest | Price + Sentiment | $386.62 | $579.12 | **53.6%** | $387.65 |
| **n+4** | Ridge | Price Only | $444.63 | $652.63 | **50.1%** | $436.63 |
| **n+4** | Ridge | Price + Sentiment | $444.65 | $652.73 | **50.9%** | $436.63 |
| **n+4** | Random Forest | Price Only | $436.08 | $644.65 | **55.4%** | $436.63 |
| **n+4** | Random Forest | Price + Sentiment | $436.47 | $643.48 | **53.4%** | $436.63 |
| **n+5** | Ridge | Price Only | $487.94 | $715.28 | **49.5%** | $477.62 |
| **n+5** | Ridge | Price + Sentiment | $488.89 | $715.25 | **50.5%** | $477.62 |
| **n+5** | Random Forest | Price Only | $479.51 | $706.70 | **55.1%** | $477.62 |
| **n+5** | Random Forest | Price + Sentiment | $478.63 | $703.53 | **52.2%** | $477.62 |

---

## 4. Quantitative Trading Strategy Performance

Simulated out-of-sample trading performance with realistic transaction costs (10 bps per trade slippage/taker fee) and non-overlapping spot return accounting:

| Model | Horizon | Split | Cum Ret (%) | Exposure (%) | Turnover (%) | Sharpe | Sortino | Autocorr (Lag 1) | Max DD (%) | Win Rate (%) | Trades | Buy & Hold Benchmark (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Trade Baseline** | n+1 | Dev Walk Forward | **+0.00%** | 0.0% | 0.0% | **0.00** | 0.00 | +0.00 | 0.00% | 0.0% | 0 | -1.39% |
| **Ridge** | n+1 | Dev Walk Forward | **-2.55%** | 4.4% | 5.1% | **-1.22** | -1.90 | -0.12 | -4.83% | 51.4% | 128 | -1.39% |
| **Random Forest** | n+1 | Dev Walk Forward | **-1.94%** | 3.5% | 3.2% | **-1.09** | -1.90 | +0.15 | -9.13% | 43.8% | 80 | -1.39% |
| **Hist Gradient Boosting** | n+1 | Dev Walk Forward | **-5.06%** | 3.3% | 3.3% | **-2.26** | -3.42 | +0.15 | -8.29% | 44.0% | 84 | -1.39% |
| **No Trade Baseline** | n+1 | Frozen Holdout | **+0.00%** | 0.0% | 0.0% | **0.00** | 0.00 | +0.00 | 0.00% | 0.0% | 0 | +4.08% |
| **Ridge** | n+1 | Frozen Holdout | **-8.37%** | 6.5% | 5.0% | **-6.07** | -7.32 | -0.03 | -8.49% | 43.4% | 64 | +4.08% |
| **Random Forest** | n+1 | Frozen Holdout | **-2.20%** | 5.4% | 3.3% | **-2.30** | -3.25 | -0.06 | -2.99% | 47.8% | 42 | +4.08% |
| **Hist Gradient Boosting** | n+1 | Frozen Holdout | **-1.90%** | 6.3% | 2.5% | **-2.00** | -2.82 | -0.12 | -2.85% | 49.4% | 32 | +4.08% |

---

## 5. Visual Artifacts Matrix

The following visual artifacts were generated directly from the out-of-sample validation run:

- **Price & Sentiment Dual Axis:** `reports/figures/price_and_sentiment.png`
- **Sentiment Proportions (Positive vs Negative):** `reports/figures/sentiment_proportions.png`
- **Hourly Social Post Volume:** `reports/figures/post_volume.png`
- **Actual vs. Predicted Across All 6 Horizons:** `reports/figures/actual_vs_predicted_6horizons.png`
- **Model Comparison Matrix (MAE & Directional Accuracy):** `reports/figures/model_performance_comparison.png`
- **Feature Ablation (Price vs. Sentiment):** `reports/figures/ablation_comparison.png`
- **Feature Correlation Heatmap:** `reports/figures/feature_correlation_heatmap.png`
- **Out-of-Sample Strategy Equity Curves:** `reports/figures/backtest_equity_curves.png`

---

## 6. Free-First Architecture & Data Source Audit

| Component | Assessment Recommendation | Free-First Legitimate Solution | Cost / Status |
| :--- | :--- | :--- | :--- |
| **Market Data** | Paid feeds / yfinance | `yfinance` + `Binance Public Spot REST API` | 100% Free |
| **X / Twitter Data** | Apify ($49+/mo) | Free Local CSV/JSON Pipeline + Calibrated Reproducible Dataset | 100% Free |
| **News Intelligence** | Tavily / Exa APIs | Free Public RSS Feeds (CoinDesk, Cointelegraph, Google News) | 100% Free |
| **Sentiment NLP** | Commercial LLM APIs | Local VADER NLP with Crypto-Domain Custom Lexicon | 100% Free |
| **Machine Learning** | Cloud Compute / GPU | Scikit-Learn (Ridge, Random Forest, HistGradientBoosting) | 100% Free (CPU) |
| **Database** | Paid Cloud Database | Local Thread-Safe SQLite with Upsert & Indexing | 100% Free |
| **Interactive UI** | Hosted Dashboard | Local Streamlit Web Application | 100% Free |

---

## 7. Honest Limitations & Future Improvements

1. **Social Data Paywalls:** Commercial X historical data for 180 days is strictly paywalled by X Corporation at $100+/mo. The modular connector allows instant plug-and-play if legitimate tokens are provided.
2. **Signal-to-Noise Ratio in Crypto Markets:** On hourly Bitcoin returns, directional accuracy hovers between 50% and 53%, consistent with semi-strong market efficiency. Models with aggressive trading turnover suffer from transaction fee drag.
3. **Execution Slippage:** The backtest applies a 10 bps fixed cost assumption. In volatile regime breaks, market impact and liquidity spreads may fluctuate dynamically.
