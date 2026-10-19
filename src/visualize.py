"""
Visualization Suite for Sentiment and Price Predictions
Generates publication-quality charts using Matplotlib and Plotly:
1. Price and Sentiment dual-axis time-series
2. Positive vs Negative sentiment dynamics
3. Post volume distribution over time
4. Actual vs. Predicted prices across all 6 horizons (n+0 to n+5)
5. Model comparison matrix (MAE, RMSE, Directional Accuracy)
6. Residual error distribution and Q-Q style diagnostic
7. Feature correlation heatmap
8. Walk-forward strategy equity curves vs Buy & Hold

Outputs high-resolution PNGs and interactive HTML files to reports/figures/.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Configure clean aesthetic matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": True,
    "figure.dpi": 150,
})


class Visualizer:
    """Creates static and interactive financial charts for reporting and dashboard."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Price and Sentiment Dual Axis
    def plot_price_and_sentiment(
        self,
        prices_df: pd.DataFrame,
        agg_sentiment: pd.DataFrame,
        filename: str = "price_and_sentiment.png",
    ) -> Path:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color_price = "#1f77b4"
        color_sent = "#ff7f0e"

        ax1.set_title("BTC-USD Historical Price vs. Aggregated X/Twitter Sentiment", fontweight="bold", pad=15)
        ax1.plot(prices_df.index, prices_df["close"], color=color_price, linewidth=1.8, label="BTC Price (USD)")
        ax1.set_ylabel("Bitcoin Close Price ($)", color=color_price, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_price)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        if not agg_sentiment.empty and "mean_compound" in agg_sentiment.columns:
            rolling_sent = agg_sentiment["mean_compound"].rolling(12, min_periods=1).mean()
            ax2.plot(agg_sentiment.index, rolling_sent, color=color_sent, linewidth=1.5, alpha=0.85, label="Sentiment (12h Rolling Mean)")
            ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
            ax2.set_ylabel("VADER Compound Sentiment [-1, 1]", color=color_sent, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor=color_sent)
            ax2.set_ylim(-0.8, 0.8)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
        return save_path

    # 2. Positive vs Negative Sentiment Proportions
    def plot_pos_vs_neg(
        self,
        agg_sentiment: pd.DataFrame,
        filename: str = "sentiment_proportions.png",
    ) -> Path:
        if agg_sentiment.empty:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title("Positive vs. Negative X Sentiment Composition Over Time", fontweight="bold", pad=15)

        idx = agg_sentiment.index
        pos = agg_sentiment["pos_proportion"].rolling(6, min_periods=1).mean() * 100.0
        neg = agg_sentiment["neg_proportion"].rolling(6, min_periods=1).mean() * 100.0

        ax.fill_between(idx, pos, color="#2ca02c", alpha=0.5, label="Positive Sentiment %")
        ax.fill_between(idx, -neg, color="#d62728", alpha=0.5, label="Negative Sentiment % (Inverted)")
        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_ylabel("Sentiment Proportion (%)", fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(loc="upper left", frameon=True)
        fig.autofmt_xdate()

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 3. Post Volume Dynamics
    def plot_post_volume(
        self,
        agg_sentiment: pd.DataFrame,
        filename: str = "post_volume.png",
    ) -> Path:
        if agg_sentiment.empty:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title("X/Twitter Post Volume Dynamics Over Time", fontweight="bold", pad=15)
        ax.bar(agg_sentiment.index, agg_sentiment["post_count"], width=0.03, color="#4b8bbe", alpha=0.7, label="Hourly Post Count")
        rolling_vol = agg_sentiment["post_count"].rolling(24, min_periods=1).mean()
        ax.plot(agg_sentiment.index, rolling_vol, color="#111111", linewidth=1.5, label="24h Volume Moving Average")

        ax.set_ylabel("Number of Posts", fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(loc="upper right", frameon=True)
        fig.autofmt_xdate()

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 4. Actual vs. Predicted Prices across all six horizons (n+0 to n+5)
    def plot_actual_vs_predicted_horizons(
        self,
        predictions_df: pd.DataFrame,
        model_name: str = "ridge",
        filename: str = "actual_vs_predicted_6horizons.png",
    ) -> Path:
        sub = predictions_df[predictions_df["model_name"] == model_name].copy()
        if sub.empty:
            return self.output_dir / filename

        sub["forecast_timestamp"] = pd.to_datetime(sub["forecast_timestamp"])
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
        axes = axes.flatten()

        for h in range(6):
            ax = axes[h]
            h_data = sub[sub["horizon"] == h].sort_values("forecast_timestamp")
            if h_data.empty:
                ax.text(0.5, 0.5, f"No Data for n+{h}", ha="center", va="center")
                continue

            # Plot last 100 test points for clean visibility
            sample = h_data.tail(120)
            ax.plot(sample["forecast_timestamp"], sample["actual_price"], label="Actual Price", color="#1f77b4", linewidth=1.8)
            ax.plot(sample["forecast_timestamp"], sample["predicted_price"], label=f"Predicted ({model_name})", color="#e377c2", linestyle="--", linewidth=1.5)

            horizon_label = "n+0 (Current Period)" if h == 0 else f"n+{h} ({h} Periods Ahead)"
            ax.set_title(f"Forecast Horizon: {horizon_label}", fontweight="bold", fontsize=11)
            ax.set_ylabel("Price ($)")
            ax.legend(loc="upper left", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle(f"Walk-Forward Predictions vs. Actuals ({model_name.upper()}) Across All 6 Horizons", fontsize=14, fontweight="bold")
        fig.autofmt_xdate()

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 5. Model Performance Comparison Bar Chart
    def plot_model_comparison(
        self,
        metrics_df: pd.DataFrame,
        filename: str = "model_performance_comparison.png",
    ) -> Path:
        if metrics_df.empty:
            return self.output_dir / filename

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. MAE Comparison across horizons
        horizons = sorted(metrics_df["horizon"].unique())
        models = metrics_df["model_name"].unique()
        x = np.arange(len(horizons))
        width = 0.8 / len(models)

        for i, m in enumerate(models):
            sub = metrics_df[metrics_df["model_name"] == m].sort_values("horizon")
            maes = [sub[sub["horizon"] == h]["mae"].values[0] if len(sub[sub["horizon"] == h]) > 0 else 0 for h in horizons]
            ax1.bar(x + i * width, maes, width, label=m.replace("_", " ").title())

        ax1.set_title("Mean Absolute Error (MAE in $) Across Horizons", fontweight="bold")
        ax1.set_xticks(x + width * (len(models) - 1) / 2)
        ax1.set_xticklabels([f"n+{h}" for h in horizons])
        ax1.set_ylabel("MAE ($)")
        ax1.legend(frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # 2. Directional Accuracy Comparison
        for i, m in enumerate(models):
            sub = metrics_df[metrics_df["model_name"] == m].sort_values("horizon")
            accs = [sub[sub["horizon"] == h]["directional_accuracy"].values[0] if len(sub[sub["horizon"] == h]) > 0 else 0 for h in horizons]
            ax2.bar(x + i * width, accs, width, label=m.replace("_", " ").title())

        ax2.axhline(50.0, color="red", linestyle="--", label="Random Guess (50%)")
        ax2.set_title("Directional Accuracy (% Sign Hit Rate)", fontweight="bold")
        ax2.set_xticks(x + width * (len(models) - 1) / 2)
        ax2.set_xticklabels([f"n+{h}" for h in horizons])
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(40, 65)
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle=":", alpha=0.6)

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 6. Feature Correlation Heatmap
    def plot_correlation_heatmap(
        self,
        feature_df: pd.DataFrame,
        filename: str = "feature_correlation_heatmap.png",
    ) -> Path:
        # Select representative price and sentiment features
        core_cols = [
            c for c in [
                "return_1", "return_6", "return_24", "price_volatility_6", "rsi_14", "macd_diff",
                "mean_compound", "sentiment_dispersion", "engagement_weighted_sentiment",
                "post_volume_change_1", "sentiment_momentum", "target_return_h1"
            ] if c in feature_df.columns
        ]

        if len(core_cols) < 3:
            return self.output_dir / filename

        corr = feature_df[core_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson Correlation", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(core_cols)))
        ax.set_yticks(np.arange(len(core_cols)))
        ax.set_xticklabels(core_cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(core_cols, fontsize=9)
        ax.set_title("Feature Correlation Matrix (Price vs. Sentiment vs. Targets)", fontweight="bold", pad=15)

        # Annotate values
        for i in range(len(core_cols)):
            for j in range(len(core_cols)):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=8)

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 7. Out-of-Sample Equity Curves
    def plot_equity_curves(
        self,
        backtest_results_dict: Dict[str, Any],
        filename: str = "backtest_equity_curves.png",
    ) -> Path:
        if not backtest_results_dict:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Out-of-Sample Trading Strategy Equity Curves vs. Buy & Hold Benchmark", fontweight="bold", pad=15)

        # Plot Benchmark first
        first_res = list(backtest_results_dict.values())[0]
        bm_curve = first_res.benchmark_curve
        ax.plot(bm_curve.index, bm_curve.values, color="black", linestyle="--", linewidth=1.5, label="Buy & Hold BTC Benchmark")

        # Plot each strategy
        colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
        for i, (m_name, res) in enumerate(backtest_results_dict.items()):
            color = colors[i % len(colors)]
            c_ret = res.cumulative_return_pct
            sh = res.sharpe_ratio
            label = f"{m_name.replace('_', ' ').title()} Strategy (Ret: {c_ret:+.1f}%, Sharpe: {sh:.2f})"
            ax.plot(res.equity_curve.index, res.equity_curve.values, color=color, linewidth=2.0, label=label)

        ax.set_ylabel("Portfolio Value ($)", fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.autofmt_xdate()

        save_path = self.output_dir / filename
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
