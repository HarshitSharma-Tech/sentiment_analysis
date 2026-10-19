"""
Zero-Configuration End-to-End Demo Script
CrowdWisdomTrading Quantitative Data Scientist Assessment
Runs the complete forecasting pipeline using 100% free tools and free data sources:
1. Initializes local SQLite database
2. Downloads real BTC-USD historical prices (via yfinance or Binance public fallback)
3. Ingests reproducible crypto X/Twitter posts and analyzes sentiment locally with VADER
4. Ingests free financial news via public RSS feeds
5. Computes leakage-free technical and sentiment feature matrices
6. Generates targets for all 6 horizons (n+0 through n+5)
7. Runs chronological walk-forward cross validation across 4 models
8. Simulates quantitative trading strategies with realistic transaction costs
9. Generates all visualization charts in reports/figures/
10. Compiles one-page sentiment outlook and complete evaluation report (Markdown + PDF)
"""

from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_config
from src.database import DatabaseManager
from src.ingestion.prices import PriceIngestion
from src.ingestion.x_import import XPostImporter
from src.ingestion.news import NewsIngestion
from src.sentiment import get_analyzer
from src.features import FeatureEngineer
from src.targets import TargetEngineer
from src.validate import WalkForwardValidator
from src.backtest import StrategyBacktester
from src.visualize import Visualizer
from src.report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_demo")


def run_demo() -> None:
    print("=" * 80)
    print(" CROWDWISDOM TRADING: QUANTITATIVE SENTIMENT PRICE PREDICTION SYSTEM")
    print(" 100% Free-First & Reproducible End-to-End Pipeline Demo")
    print("=" * 80)

    cfg = get_config()
    run_id = f"demo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Initialized Run ID: {run_id}")

    # 1. Database Initialization
    logger.info(f"1/8 Initializing SQLite database at {cfg.database_path}...")
    db = DatabaseManager(cfg.database_path)

    # 2. Historical Price Ingestion (Free yfinance or Binance fallback)
    logger.info("2/8 Ingesting real historical market prices...")
    price_ingestion = PriceIngestion(db_manager=db)
    prices_df = price_ingestion.get_or_download_prices(
        ticker=cfg.asset_ticker,
        frequency=cfg.data_frequency,
        days=cfg.history_days,
        force_download=False,
    )
    logger.info(f"Loaded {len(prices_df)} price bars ({prices_df.index.min()} to {prices_df.index.max()}).")

    # 3. Social Data Ingestion (Free local reproducible dataset)
    logger.info("3/8 Ingesting X/Twitter posts & performing local VADER sentiment analysis...")
    sample_json = cfg.sample_dir / "btc_sample_x_posts.json"
    if not sample_json.exists():
        logger.warning("Sample JSON not found. Generating fresh sample data...")
        from scripts.generate_sample_data import generate_sample_data
        generate_sample_data(days=cfg.history_days)

    importer = XPostImporter(db_manager=db)
    posts = importer.import_from_json(sample_json, source_label="sample_reproducible")
    posts_df = db.get_x_posts(ticker="BTC")
    logger.info(f"Total processed X posts available: {len(posts_df)}")

    # 4. News Ingestion (Free public RSS feeds)
    logger.info("4/8 Fetching recent financial news via free public RSS feeds...")
    news_ingestion = NewsIngestion(db_manager=db, tavily_api_key=cfg.tavily_api_key, exa_api_key=cfg.exa_api_key)
    try:
        news_articles = news_ingestion.fetch_rss_feeds(feeds=cfg.rss_feeds, max_per_feed=15)
    except Exception as e:
        logger.warning(f"RSS fetch encountered warning ({e}); importing sample news...")
        sample_news = cfg.sample_dir / "btc_sample_news.json"
        if sample_news.exists():
            news_articles = news_ingestion.import_from_json(sample_news)
    recent_news_df = db.get_news_articles(limit=20)
    logger.info(f"Total news articles available: {len(recent_news_df)}")

    # 5. Feature Engineering
    logger.info("5/8 Computing point-in-time sentiment & technical price features...")
    fe = FeatureEngineer(
        sentiment_windows=cfg.sentiment_rolling_windows,
        price_windows=cfg.price_rolling_windows,
        rsi_period=cfg.rsi_period,
    )
    agg_sentiment = fe.aggregate_posts_by_frequency(posts_df, frequency=cfg.data_frequency)
    db.save_aggregated_sentiment(agg_sentiment, frequency=cfg.data_frequency)

    feature_matrix = fe.merge_features(
        prices_df=prices_df,
        agg_sentiment=agg_sentiment,
        frequency=cfg.data_frequency,
    )
    logger.info(f"Constructed feature matrix with shape {feature_matrix.shape}.")

    # 6. Target Construction (n+0 to n+5)
    logger.info("6/8 Generating multi-horizon forecasting targets (n+0 to n+5)...")
    te = TargetEngineer()
    df_with_targets = te.create_targets(feature_matrix)

    # Exclude non-feature reference columns
    exclude_cols = [
        "open", "high", "low", "close", "volume",
        "cutoff_price_h0", "target_price_h0", "target_return_h0",
        "cutoff_price_h1", "target_price_h1", "target_return_h1",
        "cutoff_price_h2", "target_price_h2", "target_return_h2",
        "cutoff_price_h3", "target_price_h3", "target_return_h3",
        "cutoff_price_h4", "target_price_h4", "target_return_h4",
        "cutoff_price_h5", "target_price_h5", "target_return_h5",
    ]
    feature_cols = [c for c in df_with_targets.columns if c not in exclude_cols]
    logger.info(f"Selected {len(feature_cols)} clean predictive features.")

    # 7. Walk-Forward Validation
    logger.info("7/8 Running chronological walk-forward cross validation...")
    validator = WalkForwardValidator(
        min_train_size=cfg.wf_min_train_size,
        step_size=cfg.wf_step_size,
        test_size=cfg.wf_test_size,
        expanding=True,
        db_manager=db,
    )

    models_to_test = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
    metrics_df, preds_df = validator.run_full_validation(
        df_with_targets=df_with_targets,
        feature_cols=feature_cols,
        models=models_to_test,
        horizons=cfg.horizons,
        run_id=run_id,
    )

    # 8. Strategy Backtesting
    logger.info("8/8 Evaluating out-of-sample quantitative trading strategies...")
    backtester = StrategyBacktester(
        initial_capital=cfg.initial_capital,
        transaction_cost_bps=cfg.transaction_cost_bps,
        signal_threshold=cfg.signal_threshold,
        risk_free_rate=cfg.risk_free_rate,
        frequency=cfg.data_frequency,
    )

    backtest_results_dict = {}
    backtest_summary_rows = []

    for m in models_to_test:
        if m == "persistence":
            continue
        try:
            res = backtester.run_backtest(preds_df, model_name=m, horizon=1)
            backtest_results_dict[m] = res
            backtest_summary_rows.append({
                "model_name": m,
                "horizon": 1,
                "cumulative_return_pct": res.cumulative_return_pct,
                "sharpe_ratio": res.sharpe_ratio,
                "sortino_ratio": res.sortino_ratio,
                "max_drawdown_pct": res.max_drawdown_pct,
                "win_rate_pct": res.win_rate_pct,
                "total_trades": res.total_trades,
                "benchmark_cumulative_pct": res.benchmark_cumulative_pct,
            })
        except Exception as e:
            logger.warning(f"Backtest error for {m}: {e}")

    backtest_summary_df = pd.DataFrame(backtest_summary_rows)

    # Save evaluation results to SQLite
    eval_records = []
    for _, m_row in metrics_df.iterrows():
        m_name = m_row["model_name"]
        h = int(m_row["horizon"])
        bt_match = backtest_summary_df[
            (backtest_summary_df["model_name"] == m_name) & (backtest_summary_df["horizon"] == h)
        ]
        sh = bt_match["sharpe_ratio"].values[0] if not bt_match.empty else 0.0
        so = bt_match["sortino_ratio"].values[0] if not bt_match.empty else 0.0
        mdd = bt_match["max_drawdown_pct"].values[0] if not bt_match.empty else 0.0
        cum = bt_match["cumulative_return_pct"].values[0] if not bt_match.empty else 0.0
        win = bt_match["win_rate_pct"].values[0] if not bt_match.empty else 0.0
        trades = int(bt_match["total_trades"].values[0]) if not bt_match.empty else 0

        eval_records.append({
            "run_id": run_id,
            "model_name": m_name,
            "horizon": h,
            "frequency": cfg.data_frequency,
            "train_bars": cfg.wf_min_train_size,
            "test_bars": int(m_row["test_bars"]),
            "mae": float(m_row["mae"]),
            "rmse": float(m_row["rmse"]),
            "mape": float(m_row["mape"]),
            "directional_accuracy": float(m_row["directional_accuracy"]),
            "r2_score": float(m_row["r2_score"]),
            "baseline_mae": float(m_row["baseline_mae"]),
            "baseline_rmse": float(m_row["baseline_rmse"]),
            "sharpe_ratio": sh,
            "sortino_ratio": so,
            "max_drawdown": mdd,
            "cumulative_return": cum,
            "win_rate": win,
            "total_trades": trades,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
    db.save_evaluation_results(eval_records)

    # 9. Visualizations
    logger.info("Generating publication-grade charts...")
    vis = Visualizer(output_dir=cfg.figures_dir)
    vis.plot_price_and_sentiment(prices_df, agg_sentiment)
    vis.plot_pos_vs_neg(agg_sentiment)
    vis.plot_post_volume(agg_sentiment)
    vis.plot_actual_vs_predicted_horizons(preds_df, model_name="ridge")
    vis.plot_model_comparison(metrics_df)
    vis.plot_correlation_heatmap(df_with_targets)
    vis.plot_equity_curves(backtest_results_dict)

    # 10. Reports
    logger.info("Compiling reports...")
    rg = ReportGenerator(reports_dir=cfg.reports_dir)
    outlook_path = rg.generate_sentiment_outlook(
        asset_ticker=cfg.asset_ticker,
        prices_df=prices_df,
        agg_sentiment=agg_sentiment,
        recent_posts_df=posts_df,
        recent_news_df=recent_news_df,
        data_source_label="Reproducible Sample Social Data + Real Market Prices",
        is_live_x=False,
    )
    report_path = rg.generate_evaluation_report(
        metrics_df=metrics_df,
        backtest_results_df=backtest_summary_df,
        asset_ticker=cfg.asset_ticker,
        frequency=cfg.data_frequency,
        run_id=run_id,
    )

    print("\n" + "=" * 80)
    print(" DEMO EXECUTION COMPLETE!")
    print("=" * 80)
    print(f" SQLite Database:      {cfg.database_path}")
    print(f" Sentiment Outlook:    {outlook_path}")
    print(f" Evaluation Report:    {report_path}")
    print(f" PDF Report:           {report_path.with_suffix('.pdf')}")
    print(f" Figures Directory:    {cfg.figures_dir}")
    print("=" * 80)

    # Print summary table
    print("\n--- OUT-OF-SAMPLE MODEL PERFORMANCE SUMMARY ---")
    summary_cols = ["horizon", "model_name", "mae", "rmse", "directional_accuracy", "baseline_mae", "mae_improvement_pct"]
    print(metrics_df[summary_cols].to_string(index=False))

    print("\n--- OUT-OF-SAMPLE TRADING STRATEGY (HORIZON n+1) ---")
    if not backtest_summary_df.empty:
        print(backtest_summary_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()
