# Quantitative Evaluation Report: Multi-Horizon Sentiment Price Prediction

**CrowdWisdomTrading Quantitative Data Scientist Internship Assessment**  
**Target Asset:** `BTC-USD` | **Frequency:** `1h` | **Run ID:** `demo_20261019_170545`  
**Evaluation Date:** 2026-10-19 17:06:41 UTC  

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
| **n+0** | Persistence | 2530 | $185.23 | $292.74 | **49.2%** | 50.8% | $185.23 | +0.00 | 1.000 |
| **n+0** | Ridge | 2530 | $185.50 | $293.35 | **51.3%** | 50.8% | $185.23 | -0.83 | 1.000 |
| **n+0** | Random Forest | 2530 | $185.66 | $293.71 | **51.3%** | 50.8% | $185.23 | -1.35 | 1.000 |
| **n+0** | Hist Gradient Boosting | 2530 | $186.27 | $294.20 | **51.7%** | 50.8% | $185.23 | -1.70 | 1.000 |
| **n+1** | Persistence | 2530 | $263.65 | $407.30 | **49.1%** | 50.9% | $263.65 | +0.00 | 1.000 |
| **n+1** | Ridge | 2530 | $264.63 | $408.34 | **52.5%** | 50.9% | $263.65 | -0.65 | 1.000 |
| **n+1** | Random Forest | 2530 | $265.13 | $410.17 | **51.7%** | 50.9% | $263.65 | -1.96 | 1.000 |
| **n+1** | Hist Gradient Boosting | 2530 | $265.26 | $410.16 | **52.1%** | 50.9% | $263.65 | -2.15 | 1.000 |
| **n+2** | Persistence | 2530 | $323.00 | $492.09 | **49.2%** | 50.8% | $323.00 | +0.00 | 1.000 |
| **n+2** | Random Forest | 2530 | $323.80 | $495.37 | **52.8%** | 50.8% | $323.00 | -1.73 | 1.000 |
| **n+2** | Ridge | 2530 | $324.84 | $494.16 | **51.2%** | 50.8% | $323.00 | -0.81 | 1.000 |
| **n+2** | Hist Gradient Boosting | 2530 | $325.13 | $496.30 | **51.8%** | 50.8% | $323.00 | -2.16 | 1.000 |
| **n+3** | Persistence | 2530 | $370.47 | $558.40 | **50.0%** | 50.0% | $370.47 | +0.00 | 1.000 |
| **n+3** | Random Forest | 2530 | $374.47 | $564.34 | **50.1%** | 50.0% | $370.47 | -2.15 | 1.000 |
| **n+3** | Ridge | 2530 | $377.11 | $564.80 | **50.6%** | 50.0% | $370.47 | -1.87 | 1.000 |
| **n+3** | Hist Gradient Boosting | 2530 | $377.64 | $567.70 | **50.5%** | 50.0% | $370.47 | -3.01 | 1.000 |
| **n+4** | Persistence | 2530 | $414.78 | $616.37 | **49.9%** | 50.1% | $414.78 | +0.00 | 1.000 |
| **n+4** | Random Forest | 2530 | $422.31 | $625.93 | **50.8%** | 50.1% | $414.78 | -2.88 | 1.000 |
| **n+4** | Ridge | 2530 | $423.47 | $623.83 | **49.4%** | 50.1% | $414.78 | -1.91 | 1.000 |
| **n+4** | Hist Gradient Boosting | 2530 | $423.72 | $626.46 | **50.8%** | 50.1% | $414.78 | -2.94 | 1.000 |
| **n+5** | Persistence | 2530 | $456.28 | $672.26 | **51.4%** | 51.4% | $456.28 | +0.00 | 1.000 |
| **n+5** | Random Forest | 2530 | $466.54 | $684.25 | **47.9%** | 51.4% | $456.28 | -3.06 | 1.000 |
| **n+5** | Ridge | 2530 | $468.57 | $682.41 | **48.8%** | 51.4% | $456.28 | -2.21 | 1.000 |
| **n+5** | Hist Gradient Boosting | 2530 | $468.92 | $685.87 | **47.9%** | 51.4% | $456.28 | -3.36 | 1.000 |

### 2B. Frozen Chronological Holdout: Out-of-Sample Evaluation (Final 30% Window)
*Trained on Development Period with 5-bar embargo; tested strictly once on quarantined future data:*

| Horizon | Model | Holdout Bars | MAE ($) | RMSE ($) | Dir Acc (%) | Maj Class Base (%) | Base MAE ($) | DM Stat | DM p-val (FDR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Persistence | 1290 | $191.17 | $291.61 | **50.9%** | 50.9% | $191.17 | +0.00 | 0.998 |
| **n+0** | Random Forest | 1290 | $191.38 | $291.64 | **50.3%** | 50.9% | $191.17 | -0.04 | 0.998 |
| **n+0** | Hist Gradient Boosting | 1290 | $191.80 | $292.62 | **49.5%** | 50.9% | $191.17 | -1.46 | 0.998 |
| **n+0** | Ridge | 1290 | $192.68 | $293.17 | **50.9%** | 50.9% | $191.17 | -1.68 | 0.998 |
| **n+1** | Persistence | 1289 | $270.26 | $413.47 | **52.4%** | 52.4% | $270.26 | +0.00 | 0.998 |
| **n+1** | Random Forest | 1289 | $271.41 | $414.36 | **50.4%** | 52.4% | $270.26 | -0.66 | 0.998 |
| **n+1** | Hist Gradient Boosting | 1289 | $272.92 | $415.85 | **49.1%** | 52.4% | $270.26 | -2.00 | 0.998 |
| **n+1** | Ridge | 1289 | $274.22 | $416.85 | **50.0%** | 52.4% | $270.26 | -1.92 | 0.998 |
| **n+2** | Random Forest | 1288 | $334.02 | $502.54 | **52.9%** | 51.1% | $335.03 | -0.58 | 0.998 |
| **n+2** | Persistence | 1288 | $335.03 | $501.51 | **51.1%** | 51.1% | $335.03 | +0.00 | 0.998 |
| **n+2** | Hist Gradient Boosting | 1288 | $336.81 | $504.81 | **53.1%** | 51.1% | $335.03 | -1.91 | 0.998 |
| **n+2** | Ridge | 1288 | $339.64 | $507.66 | **51.2%** | 51.1% | $335.03 | -2.23 | 0.998 |
| **n+3** | Random Forest | 1287 | $386.11 | $578.18 | **55.3%** | 52.1% | $387.90 | -0.39 | 0.998 |
| **n+3** | Persistence | 1287 | $387.90 | $577.04 | **52.1%** | 52.1% | $387.90 | +0.00 | 0.998 |
| **n+3** | Hist Gradient Boosting | 1287 | $388.33 | $580.60 | **55.1%** | 52.1% | $387.90 | -1.61 | 0.998 |
| **n+3** | Ridge | 1287 | $392.01 | $583.87 | **51.7%** | 52.1% | $387.90 | -1.99 | 0.998 |
| **n+4** | Random Forest | 1286 | $436.63 | $643.53 | **53.3%** | 52.3% | $436.95 | -0.80 | 0.998 |
| **n+4** | Persistence | 1286 | $436.95 | $639.58 | **52.3%** | 52.3% | $436.95 | +0.00 | 0.998 |
| **n+4** | Hist Gradient Boosting | 1286 | $438.99 | $645.36 | **51.3%** | 52.3% | $436.95 | -1.51 | 0.998 |
| **n+4** | Ridge | 1286 | $445.10 | $652.14 | **50.5%** | 52.3% | $436.95 | -2.61 | 0.998 |
| **n+5** | Persistence | 1285 | $478.32 | $699.01 | **51.8%** | 51.8% | $478.32 | +0.00 | 0.998 |
| **n+5** | Random Forest | 1285 | $479.86 | $703.77 | **54.3%** | 51.8% | $478.32 | -0.80 | 0.998 |
| **n+5** | Hist Gradient Boosting | 1285 | $481.40 | $705.18 | **50.3%** | 51.8% | $478.32 | -1.30 | 0.998 |
| **n+5** | Ridge | 1285 | $489.53 | $714.82 | **49.6%** | 51.8% | $478.32 | -2.82 | 0.998 |

---

## 3. Feature Ablation Study: Price-Only vs. Price + Sentiment

> **[!IMPORTANT] Social Sentiment Data Provenance Disclosure:**
> The social sentiment ablation reported below was conducted using the calibrated offline reproducible sample dataset (2,881 synthetic posts), NOT live historical X Corporation firehose data. In an unpaywalled environment, these metrics demonstrate that the pipeline mechanics and ablation code execute correctly; they do NOT prove live social media alpha.

| Horizon | Model | Feature Set | MAE ($) | RMSE ($) | Directional Accuracy (%) | Baseline MAE ($) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **n+0** | Ridge | Price Only | $192.34 | $292.75 | **49.8%** | $191.17 |
| **n+0** | Ridge | Price + Sentiment | $192.68 | $293.17 | **50.9%** | $191.17 |
| **n+0** | Random Forest | Price Only | $191.76 | $292.28 | **48.7%** | $191.17 |
| **n+0** | Random Forest | Price + Sentiment | $191.38 | $291.64 | **50.3%** | $191.17 |
| **n+1** | Ridge | Price Only | $273.56 | $416.46 | **49.5%** | $270.26 |
| **n+1** | Ridge | Price + Sentiment | $274.22 | $416.85 | **50.0%** | $270.26 |
| **n+1** | Random Forest | Price Only | $271.89 | $415.16 | **49.8%** | $270.26 |
| **n+1** | Random Forest | Price + Sentiment | $271.41 | $414.36 | **50.4%** | $270.26 |
| **n+2** | Ridge | Price Only | $338.75 | $506.40 | **49.5%** | $335.03 |
| **n+2** | Ridge | Price + Sentiment | $339.64 | $507.66 | **51.2%** | $335.03 |
| **n+2** | Random Forest | Price Only | $334.55 | $503.72 | **55.2%** | $335.03 |
| **n+2** | Random Forest | Price + Sentiment | $334.02 | $502.54 | **52.9%** | $335.03 |
| **n+3** | Ridge | Price Only | $394.45 | $585.82 | **50.0%** | $387.90 |
| **n+3** | Ridge | Price + Sentiment | $392.01 | $583.87 | **51.7%** | $387.90 |
| **n+3** | Random Forest | Price Only | $387.82 | $580.94 | **55.2%** | $387.90 |
| **n+3** | Random Forest | Price + Sentiment | $386.11 | $578.18 | **55.3%** | $387.90 |
| **n+4** | Ridge | Price Only | $444.96 | $651.98 | **49.8%** | $436.95 |
| **n+4** | Ridge | Price + Sentiment | $445.10 | $652.14 | **50.5%** | $436.95 |
| **n+4** | Random Forest | Price Only | $440.73 | $648.35 | **53.9%** | $436.95 |
| **n+4** | Random Forest | Price + Sentiment | $436.63 | $643.53 | **53.3%** | $436.95 |
| **n+5** | Ridge | Price Only | $488.45 | $714.77 | **49.4%** | $478.32 |
| **n+5** | Ridge | Price + Sentiment | $489.53 | $714.82 | **49.6%** | $478.32 |
| **n+5** | Random Forest | Price Only | $482.95 | $709.44 | **54.7%** | $478.32 |
| **n+5** | Random Forest | Price + Sentiment | $479.86 | $703.77 | **54.3%** | $478.32 |

---

## 4. Quantitative Trading Strategy Performance

Simulated out-of-sample trading performance with realistic transaction costs (10 bps per trade slippage/taker fee) and non-overlapping spot return accounting:

| Model | Horizon | Split | Cum Ret (%) | Exposure (%) | Turnover (%) | Sharpe | Sortino | Autocorr (Lag 1) | Max DD (%) | Win Rate (%) | Trades | Buy & Hold Benchmark (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Trade Baseline** | n+1 | Dev Walk Forward | **+0.00%** | 0.0% | 0.0% | **0.00** | 0.00 | +0.00 | 0.00% | 0.0% | 0 | -2.15% |
| **Ridge** | n+1 | Dev Walk Forward | **-7.02%** | 4.3% | 4.3% | **-3.20** | -4.67 | +0.01 | -7.87% | 46.8% | 110 | -2.15% |
| **Random Forest** | n+1 | Dev Walk Forward | **-5.96%** | 4.0% | 3.2% | **-2.96** | -4.29 | +0.15 | -9.96% | 48.0% | 80 | -2.15% |
| **Hist Gradient Boosting** | n+1 | Dev Walk Forward | **-5.12%** | 3.3% | 2.8% | **-2.38** | -3.73 | +0.16 | -9.34% | 48.8% | 72 | -2.15% |
| **No Trade Baseline** | n+1 | Frozen Holdout | **+0.00%** | 0.0% | 0.0% | **0.00** | 0.00 | +0.00 | 0.00% | 0.0% | 0 | +3.27% |
| **Ridge** | n+1 | Frozen Holdout | **-8.30%** | 6.3% | 5.0% | **-6.01** | -7.24 | -0.03 | -8.42% | 42.0% | 64 | +3.27% |
| **Random Forest** | n+1 | Frozen Holdout | **-2.18%** | 5.7% | 2.9% | **-2.16** | -3.15 | -0.16 | -3.33% | 45.2% | 38 | +3.27% |
| **Hist Gradient Boosting** | n+1 | Frozen Holdout | **-1.86%** | 6.1% | 2.3% | **-1.97** | -2.78 | -0.11 | -2.62% | 50.6% | 30 | +3.27% |

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
