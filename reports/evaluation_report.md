# Quantitative Evaluation Report: Multi-Horizon Sentiment Price Prediction

**CrowdWisdomTrading Quantitative Data Scientist Internship Assessment**  
**Target Asset:** `BTC-USD` | **Frequency:** `1h` | **Run ID:** `demo_20261019_163052`  
**Evaluation Date:** 2026-10-19 16:31:55 UTC  

---

## 1. Executive Summary

This project delivers a production-grade, reproducible machine learning pipeline predicting Bitcoin (`BTC-USD`) prices and returns across **six distinct forecast horizons** ($n+0$ to $n+5$). In strict compliance with the core assessment requirement, the entire system is built **exclusively using free tools and free data sources**, requiring zero paid API subscriptions, credits, or GPU hardware.

### Key Rigor & Methodology Enhancements:
- **Zero-Lookahead Feature Engineering:** Strictly chronological point-in-time calculation of 24 curated technical and social sentiment features.
- **Distinct Target Definitions across $n+0$ to $n+5$:** At forecast cutoff $t$, all horizons share the identical known cutoff price $P_{cutoff} = P_{close}(t-1)$ and input vector $X_t$, forecasting future closes $P_{close}(t+h)$.
- **Purged Walk-Forward Cross Validation:** Eliminates multi-step target overlap by purging the $h$-step boundary before each test fold.
- **Regularization & Variance Reduction:** Models use adaptive shrinkage (RidgeCV) and constrained tree depth to prevent return noise overfitting.
- **Realistic Backtesting:** Incorporates 10 bps transaction costs, reporting exposure, turnover, win rate, Sharpe, Sortino, and comparing against Buy & Hold and No-Trade baselines.

---

## 2. Multi-Horizon Forecast Results Matrix

Below are the actual out-of-sample evaluation metrics across all six forecast horizons:

| Horizon | Model | Out-of-Sample Bars | MAE ($) | RMSE ($) | MAPE (%) | Directional Accuracy (%) | Baseline MAE ($) | MAE Impv (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Persistence | 3821 | $187.02 | $291.53 | 0.26% | **49.7%** | $187.02 | +0.00% |
| **n+0** | Ridge | 3821 | $187.41 | $292.34 | 0.26% | **51.5%** | $187.02 | -0.21% |
| **n+0** | Random Forest | 3821 | $187.44 | $292.30 | 0.26% | **51.1%** | $187.02 | -0.22% |
| **n+0** | Hist Gradient Boosting | 3821 | $187.79 | $293.00 | 0.26% | **51.5%** | $187.02 | -0.41% |
| **n+1** | Persistence | 3820 | $265.54 | $408.56 | 0.37% | **50.2%** | $265.54 | +0.00% |
| **n+1** | Random Forest | 3820 | $266.81 | $410.85 | 0.38% | **51.1%** | $265.54 | -0.48% |
| **n+1** | Ridge | 3820 | $267.07 | $410.12 | 0.38% | **51.8%** | $265.54 | -0.57% |
| **n+1** | Hist Gradient Boosting | 3820 | $267.10 | $411.12 | 0.38% | **50.8%** | $265.54 | -0.59% |
| **n+2** | Persistence | 3819 | $326.88 | $494.69 | 0.46% | **50.0%** | $326.88 | +0.00% |
| **n+2** | Random Forest | 3819 | $327.19 | $497.78 | 0.46% | **52.9%** | $326.88 | -0.10% |
| **n+2** | Hist Gradient Boosting | 3819 | $328.38 | $498.39 | 0.46% | **51.8%** | $326.88 | -0.46% |
| **n+2** | Ridge | 3819 | $329.17 | $497.58 | 0.46% | **51.3%** | $326.88 | -0.70% |
| **n+3** | Persistence | 3818 | $375.90 | $564.16 | 0.53% | **50.7%** | $375.90 | +0.00% |
| **n+3** | Random Forest | 3818 | $377.95 | $568.85 | 0.53% | **51.7%** | $375.90 | -0.55% |
| **n+3** | Hist Gradient Boosting | 3818 | $380.75 | $571.80 | 0.54% | **50.7%** | $375.90 | -1.29% |
| **n+3** | Ridge | 3818 | $380.76 | $569.93 | 0.54% | **51.2%** | $375.90 | -1.29% |
| **n+4** | Persistence | 3817 | $421.87 | $623.68 | 0.59% | **50.6%** | $421.87 | +0.00% |
| **n+4** | Random Forest | 3817 | $426.85 | $631.81 | 0.60% | **50.2%** | $421.87 | -1.18% |
| **n+4** | Hist Gradient Boosting | 3817 | $428.94 | $632.47 | 0.60% | **50.2%** | $421.87 | -1.67% |
| **n+4** | Ridge | 3817 | $430.08 | $632.12 | 0.61% | **49.9%** | $421.87 | -1.95% |
| **n+5** | Persistence | 3816 | $463.39 | $680.72 | 0.65% | **51.6%** | $463.39 | +0.00% |
| **n+5** | Random Forest | 3816 | $471.64 | $690.66 | 0.66% | **49.9%** | $463.39 | -1.78% |
| **n+5** | Hist Gradient Boosting | 3816 | $474.00 | $692.10 | 0.67% | **49.0%** | $463.39 | -2.29% |
| **n+5** | Ridge | 3816 | $474.42 | $691.37 | 0.67% | **48.7%** | $463.39 | -2.38% |

---

## 3. Feature Ablation Study: Price-Only vs. Price + Sentiment

Comparison of predictive power when adding X social sentiment to pure market price features:

| Horizon | Model | Feature Set | MAE ($) | RMSE ($) | Directional Accuracy (%) | Baseline MAE ($) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **n+0** | Ridge | Price Only | $187.09 | $292.09 | **51.4%** | $187.02 |
| **n+0** | Ridge | Price + Sentiment | $187.41 | $292.34 | **51.5%** | $187.02 |
| **n+0** | Random Forest | Price Only | $187.38 | $292.63 | **51.9%** | $187.02 |
| **n+0** | Random Forest | Price + Sentiment | $187.44 | $292.30 | **51.1%** | $187.02 |
| **n+1** | Ridge | Price Only | $266.65 | $410.03 | **51.6%** | $265.54 |
| **n+1** | Ridge | Price + Sentiment | $267.07 | $410.12 | **51.8%** | $265.54 |
| **n+1** | Random Forest | Price Only | $267.03 | $411.21 | **50.8%** | $265.54 |
| **n+1** | Random Forest | Price + Sentiment | $266.81 | $410.85 | **51.1%** | $265.54 |
| **n+2** | Ridge | Price Only | $329.45 | $498.21 | **52.0%** | $326.88 |
| **n+2** | Ridge | Price + Sentiment | $329.17 | $497.58 | **51.3%** | $326.88 |
| **n+2** | Random Forest | Price Only | $327.51 | $497.77 | **53.0%** | $326.88 |
| **n+2** | Random Forest | Price + Sentiment | $327.19 | $497.78 | **52.9%** | $326.88 |
| **n+3** | Ridge | Price Only | $380.57 | $569.74 | **51.3%** | $375.90 |
| **n+3** | Ridge | Price + Sentiment | $380.76 | $569.93 | **51.2%** | $375.90 |
| **n+3** | Random Forest | Price Only | $379.25 | $571.04 | **52.5%** | $375.90 |
| **n+3** | Random Forest | Price + Sentiment | $377.95 | $568.85 | **51.7%** | $375.90 |
| **n+4** | Ridge | Price Only | $428.89 | $631.70 | **50.9%** | $421.87 |
| **n+4** | Ridge | Price + Sentiment | $430.08 | $632.12 | **49.9%** | $421.87 |
| **n+4** | Random Forest | Price Only | $428.58 | $633.83 | **51.3%** | $421.87 |
| **n+4** | Random Forest | Price + Sentiment | $426.85 | $631.81 | **50.2%** | $421.87 |
| **n+5** | Ridge | Price Only | $472.85 | $691.14 | **49.9%** | $463.39 |
| **n+5** | Ridge | Price + Sentiment | $474.42 | $691.37 | **48.7%** | $463.39 |
| **n+5** | Random Forest | Price Only | $472.62 | $692.69 | **49.2%** | $463.39 |
| **n+5** | Random Forest | Price + Sentiment | $471.64 | $690.66 | **49.9%** | $463.39 |

---

## 4. Quantitative Trading Strategy Performance

Simulated out-of-sample trading performance with realistic transaction costs (10 bps per trade slippage/taker fee):

| Model | Horizon | Cumulative Return (%) | Exposure (%) | Turnover (%) | Sharpe | Sortino | Max DD (%) | Win Rate (%) | Trades | Buy & Hold Benchmark (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Trade Baseline** | n+1 | **+0.00%** | 0.0% | 0.0% | **0.00** | 0.00 | 0.00% | 0.0% | 0 | -1.53% |
| **Ridge** | n+1 | **-8.78%** | 4.2% | 4.0% | **-1.88** | -2.73 | -12.61% | 50.0% | 154 | -1.53% |
| **Random Forest** | n+1 | **+0.18%** | 4.3% | 2.9% | **-0.21** | -0.35 | -12.43% | 52.1% | 112 | -1.53% |
| **Hist Gradient Boosting** | n+1 | **-9.77%** | 3.3% | 2.6% | **-2.60** | -3.62 | -13.90% | 46.5% | 98 | -1.53% |

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
2. **Execution Slippage:** The backtest applies a 10 bps fixed cost assumption. In volatile regime breaks, market impact and liquidity spreads may fluctuate dynamically.
3. **Signal Strength in Efficient Markets:** Out-of-sample directional accuracy on hourly BTC bars hovers near 50-53%, reflecting efficient market pricing. Excess turnover quickly erodes gross alpha.
