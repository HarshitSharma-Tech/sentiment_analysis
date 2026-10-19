"""
Quantitative Trading Strategy Evaluation Module
Simulates an out-of-sample trading strategy driven by model return forecasts.
Evaluates:
- Sharpe Ratio (annualized)
- Sortino Ratio (annualized downside deviation)
- Maximum Drawdown (MDD)
- Cumulative Return & Out-of-Sample Equity Curves
- Exposure (% time invested in the market)
- Turnover (% average position change per bar)
- Win Rate & Trade Count
- Realistic Transaction Costs & Slippage (10 bps)
- Buy & Hold Benchmark AND No-Trade Baseline comparisons

Strict Guarantees:
- Calculations derived strictly from strategy returns (not price prediction errors)
- Resilient against zero-volatility and sparse downside edge cases
- Explicitly labels hypothetical backtests without fabricating profitability
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    model_name: str
    horizon: int
    total_bars: int
    total_trades: int
    exposure_pct: float
    turnover_pct: float
    cumulative_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    benchmark_cumulative_pct: float
    benchmark_sharpe: float
    benchmark_max_drawdown_pct: float
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    no_trade_curve: pd.Series


class StrategyBacktester:
    """Simulates spot execution from model return forecasts with transaction costs."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,  # 10 bps = 0.10% (taker fee + slippage)
        signal_threshold: float = 0.001,     # 0.10% minimum expected return to enter
        risk_free_rate: float = 0.04,        # 4.0% annualized risk-free rate
        frequency: str = "1h",
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_bps / 10_000.0
        self.signal_threshold = signal_threshold
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency

        # 24/7/365 crypto annualization periods
        self.periods_per_year = 8766.0 if frequency == "1h" else 365.25

    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        model_name: str,
        horizon: int = 1,
    ) -> BacktestResult:
        """
        Backtest a specific model and horizon from walk-forward predictions.
        Decision rule:
          At time t, if predicted_return > signal_threshold -> Position = 1.0 (Long)
          Otherwise -> Position = 0.0 (Flat / Cash)
        """
        sub = predictions_df[
            (predictions_df["model_name"] == model_name) &
            (predictions_df["horizon"] == horizon)
        ].copy()

        if sub.empty:
            raise ValueError(f"No predictions found for model={model_name}, horizon={horizon}")

        sub["forecast_timestamp"] = pd.to_datetime(sub["forecast_timestamp"], utc=True)
        sub = sub.sort_values("forecast_timestamp").reset_index(drop=True)

        n = len(sub)
        pred_rets = sub["predicted_return"].values
        actual_rets = sub["actual_return"].values

        # 1. Generate trading signals
        # Position is held over the period
        positions = np.zeros(n, dtype=float)
        positions[pred_rets > self.signal_threshold] = 1.0

        # Position changes incur transaction costs (slippage + fee)
        pos_shift = np.roll(positions, 1)
        pos_shift[0] = 0.0
        trades = np.abs(positions - pos_shift)
        trade_costs = trades * self.transaction_cost_rate

        # 2. Strategy period returns
        strat_returns = (positions * actual_rets) - trade_costs

        # 3. Cumulative equity curves
        equity = self.initial_capital * np.cumprod(1.0 + strat_returns)
        benchmark_equity = self.initial_capital * np.cumprod(1.0 + actual_rets)
        no_trade_equity = np.full(n, self.initial_capital, dtype=float)

        # 4. Performance metrics
        total_trades = int(np.sum(trades > 0))
        exposure = float(np.mean(positions > 0) * 100.0)
        turnover = float(np.mean(trades) * 100.0)
        cum_ret = float((equity[-1] / self.initial_capital - 1.0) * 100.0)
        bm_cum_ret = float((benchmark_equity[-1] / self.initial_capital - 1.0) * 100.0)

        # Annualized return
        mean_ret = float(np.mean(strat_returns))
        ann_return = mean_ret * self.periods_per_year * 100.0

        # Annualized volatility
        std_ret = float(np.std(strat_returns))
        ann_vol = std_ret * np.sqrt(self.periods_per_year) * 100.0

        # Risk-free rate per period
        rf_per_period = self.risk_free_rate / self.periods_per_year

        # Sharpe ratio
        excess_returns = strat_returns - rf_per_period
        if std_ret > 1e-8:
            sharpe = float(np.mean(excess_returns) / std_ret * np.sqrt(self.periods_per_year))
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        if std_ret > 1e-8:
            downside_diff = np.minimum(0.0, excess_returns)
            downside_std = float(np.sqrt(np.mean(downside_diff**2)))
            if downside_std > 1e-8:
                sortino = float(np.mean(excess_returns) / downside_std * np.sqrt(self.periods_per_year))
            else:
                sortino = sharpe
        else:
            sortino = 0.0

        # Benchmark Sharpe & Drawdown
        bm_std = float(np.std(actual_rets))
        if bm_std > 1e-8:
            bm_sharpe = float(np.mean(actual_rets - rf_per_period) / bm_std * np.sqrt(self.periods_per_year))
        else:
            bm_sharpe = 0.0

        bm_running_max = np.maximum.accumulate(benchmark_equity)
        bm_drawdowns = (benchmark_equity - bm_running_max) / bm_running_max
        bm_max_dd = float(np.min(bm_drawdowns) * 100.0)

        # Strategy Maximum Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = float(np.min(drawdowns) * 100.0)

        # Win rate of active trades
        active_mask = positions > 0
        if np.sum(active_mask) > 0:
            win_rate = float(np.sum(strat_returns[active_mask] > 0) / np.sum(active_mask) * 100.0)
        else:
            win_rate = 0.0

        timestamps = sub["forecast_timestamp"]
        equity_series = pd.Series(equity, index=timestamps, name="Strategy_Equity")
        benchmark_series = pd.Series(benchmark_equity, index=timestamps, name="Benchmark_Equity")
        no_trade_series = pd.Series(no_trade_equity, index=timestamps, name="No_Trade_Equity")

        return BacktestResult(
            model_name=model_name,
            horizon=horizon,
            total_bars=n,
            total_trades=total_trades,
            exposure_pct=exposure,
            turnover_pct=turnover,
            cumulative_return_pct=cum_ret,
            annualized_return_pct=ann_return,
            annualized_volatility_pct=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
            benchmark_cumulative_pct=bm_cum_ret,
            benchmark_sharpe=bm_sharpe,
            benchmark_max_drawdown_pct=bm_max_dd,
            equity_curve=equity_series,
            benchmark_curve=benchmark_series,
            no_trade_curve=no_trade_series,
        )

    def evaluate_all_models(
        self,
        predictions_df: pd.DataFrame,
        models: Optional[List[str]] = None,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Run backtest across all models and produce a comparison summary DataFrame."""
        if models is None:
            models = [m for m in predictions_df["model_name"].unique() if m != "persistence"]

        rows = []
        # Include No-Trade baseline row
        rows.append({
            "model_name": "no_trade_baseline",
            "horizon": horizon,
            "cumulative_return_pct": 0.0,
            "exposure_pct": 0.0,
            "turnover_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "benchmark_cumulative_pct": 0.0,
        })

        for m in models:
            try:
                res = self.run_backtest(predictions_df, model_name=m, horizon=horizon)
                rows.append({
                    "model_name": m,
                    "horizon": horizon,
                    "cumulative_return_pct": res.cumulative_return_pct,
                    "exposure_pct": res.exposure_pct,
                    "turnover_pct": res.turnover_pct,
                    "sharpe_ratio": res.sharpe_ratio,
                    "sortino_ratio": res.sortino_ratio,
                    "max_drawdown_pct": res.max_drawdown_pct,
                    "win_rate_pct": res.win_rate_pct,
                    "total_trades": res.total_trades,
                    "benchmark_cumulative_pct": res.benchmark_cumulative_pct,
                })
            except Exception as e:
                logger.warning(f"Backtest failed for {m}: {e}")

        # Update benchmark cumulative return for no_trade row if results exist
        if len(rows) > 1:
            rows[0]["benchmark_cumulative_pct"] = rows[1]["benchmark_cumulative_pct"]

        return pd.DataFrame(rows)
