"""
Optimized End-to-End Demo Script
CrowdWisdomTrading Quantitative Data Scientist Assessment
Runs the complete forecasting pipeline using 100% free tools and free data sources:
1. Initializes local SQLite database
2. Downloads real BTC-USD historical prices (via yfinance or Binance public fallback)
3. Ingests reproducible crypto X/Twitter posts & analyzes sentiment locally with VADER
4. Ingests free financial news via public RSS feeds (CoinDesk, Cointelegraph, Google News)
5. Computes leakage-free technical and sentiment feature matrices (Price vs Price+Sentiment)
6. Generates targets for all 6 horizons (n+0 through n+5) with distinct target spans
7. Runs purged chronological walk-forward cross validation across 4 models
8. Executes feature ablation: Price-Only vs. Price+Sentiment
9. Simulates quantitative trading strategies with realistic 10 bps transaction costs
10. Generates all 8 visualization charts in reports/figures/
11. Compiles one-page sentiment outlook and complete evaluation report (Markdown + PDF)
"""

from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_config
from src.database import DatabaseManager
from src.ingestion.prices import PriceIngestion
from src.ingestion.x_import import XPostImporter
from src.ingestion.news import NewsIngestion
from src.sentiment import get_analyzer
from src.features import FeatureEngineer, PRICE_FEATURE_COLS, ALL_FEATURE_COLS
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
    print(" 100% Free-First, Purged Walk-Forward & Multi-Horizon Pipeline Demo")
    print("=" * 80)

    cfg = get_config()
    run_id = f"demo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Initialized Run ID: {run_id}")

    # 1. Database Initialization
    logger.info(f"1/9 Initializing SQLite database at {cfg.database_path}...")
    db = DatabaseManager(cfg.database_path)

    # 2. Historical Price Ingestion (Free yfinance or Binance fallback)
    logger.info("2/9 Ingesting real historical market prices...")
    price_ingestion = PriceIngestion(db_manager=db)
    prices_df = price_ingestion.get_or_download_prices(
        ticker=cfg.asset_ticker,
        frequency=cfg.data_frequency,
        days=cfg.history_days,
        force_download=False,
    )
    logger.info(f"Loaded {len(prices_df)} price bars ({prices_df.index.min()} to {prices_df.index.max()}).")

    # 3. Social Data Ingestion (Free local reproducible dataset)
    logger.info("3/9 Ingesting X/Twitter posts & performing local VADER sentiment analysis...")
    sample_json = cfg.sample_dir / "btc_sample_x_posts.json"
    if not sample_json.exists():
        logger.warning("Sample JSON not found. Generating fresh sample data...")
        from scripts.generate_sample_data import generate_sample_data
        generate_sample_data(days=cfg.history_days)

    importer = XPostImporter(db_manager=db)
    posts = importer.import_from_json(sample_json, source_label="sample_reproducible")
    posts_df = db.get_x_posts(ticker="BTC")
    logger.info(f"Total processed X posts available: {len(posts_df)}")

    # 4. News Ingestion (Free public RSS feeds with 5s timeout)
    logger.info("4/9 Fetching recent financial news via free public RSS feeds...")
    news_ingestion = NewsIngestion(db_manager=db, tavily_api_key=cfg.tavily_api_key, exa_api_key=cfg.exa_api_key)
    try:
        news_articles = news_ingestion.fetch_rss_feeds(feeds=cfg.rss_feeds, max_per_feed=15)
    except Exception as e:
        logger.warning(f"RSS fetch warning ({e}); using local news articles...")
        sample_news = cfg.sample_dir / "btc_sample_news.json"
        if sample_news.exists():
            news_articles = news_ingestion.import_from_json(sample_news)
    recent_news_df = db.get_news_articles(limit=20)
    logger.info(f"Total news articles available: {len(recent_news_df)}")

    # 5. Feature Engineering
    logger.info("5/9 Computing point-in-time curated sentiment & technical price features...")
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

    # 6. Target Construction (n+0 through n+5)
    logger.info("6/9 Generating multi-horizon forecasting targets (n+0 to n+5)...")
    te = TargetEngineer()
    df_with_targets = te.create_targets(feature_matrix)

    available_price_feats = [c for c in PRICE_FEATURE_COLS if c in df_with_targets.columns]
    available_all_feats = [c for c in ALL_FEATURE_COLS if c in df_with_targets.columns]
    logger.info(f"Feature set: {len(available_price_feats)} price features, {len(available_all_feats)} total (price+sentiment) features.")

    # 7. Walk-Forward Validation
    logger.info("7/9 Running purged chronological walk-forward cross validation...")
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
        feature_cols=available_all_feats,
        models=models_to_test,
        horizons=cfg.horizons,
        run_id=run_id,
    )

    # 8. Feature Ablation: Price-Only vs. Price + Sentiment
    logger.info("8/9 Running feature ablation study (Price-Only vs. Price+Sentiment)...")
    price_metrics_df, _ = validator.run_full_validation(
        df_with_targets=df_with_targets,
        feature_cols=available_price_feats,
        models=["ridge", "random_forest"],
        horizons=cfg.horizons,
        run_id=f"{run_id}_price_only",
    )

    ablation_rows = []
    for h in cfg.horizons:
        for m in ["ridge", "random_forest"]:
            p_row = price_metrics_df[(price_metrics_df["model_name"] == m) & (price_metrics_df["horizon"] == h)]
            s_row = metrics_df[(metrics_df["model_name"] == m) & (metrics_df["horizon"] == h)]
            if not p_row.empty:
                ablation_rows.append({
                    "horizon": h,
                    "model_name": m,
                    "feature_set": "Price Only",
                    "mae": float(p_row["mae"].values[0]),
                    "rmse": float(p_row["rmse"].values[0]),
                    "directional_accuracy": float(p_row["directional_accuracy"].values[0]),
                    "baseline_mae": float(p_row["baseline_mae"].values[0]),
                })
            if not s_row.empty:
                ablation_rows.append({
                    "horizon": h,
                    "model_name": m,
                    "feature_set": "Price + Sentiment",
                    "mae": float(s_row["mae"].values[0]),
                    "rmse": float(s_row["rmse"].values[0]),
                    "directional_accuracy": float(s_row["directional_accuracy"].values[0]),
                    "baseline_mae": float(s_row["baseline_mae"].values[0]),
                })
    ablation_df = pd.DataFrame(ablation_rows)

    # 9. Strategy Backtesting
    logger.info("9/9 Evaluating out-of-sample quantitative trading strategies...")
    backtester = StrategyBacktester(
        initial_capital=cfg.initial_capital,
        transaction_cost_bps=cfg.transaction_cost_bps,
        signal_threshold=cfg.signal_threshold,
        risk_free_rate=cfg.risk_free_rate,
        frequency=cfg.data_frequency,
    )

    backtest_results_dict = {}
    backtest_summary_df = backtester.evaluate_all_models(preds_df, horizon=1)

    for m in models_to_test:
        if m == "persistence":
            continue
        try:
            res = backtester.run_backtest(preds_df, model_name=m, horizon=1)
            backtest_results_dict[m] = res
        except Exception as e:
            logger.warning(f"Backtest error for {m}: {e}")

    # Save evaluation results to SQLite
    eval_records = []
    for _, m_row in metrics_df.iterrows():
        m_name = m_row["model_name"]
        h = int(m_row["horizon"])
        bt_match = backtest_summary_df[
            (backtest_summary_df["model_name"] == m_name) & (backtest_summary_df["horizon"] == h)
        ]
        sh = float(bt_match["sharpe_ratio"].values[0]) if not bt_match.empty else 0.0
        so = float(bt_match["sortino_ratio"].values[0]) if not bt_match.empty else 0.0
        mdd = float(bt_match["max_drawdown_pct"].values[0]) if not bt_match.empty else 0.0
        cum = float(bt_match["cumulative_return_pct"].values[0]) if not bt_match.empty else 0.0
        win = float(bt_match["win_rate_pct"].values[0]) if not bt_match.empty else 0.0
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

    # 10. Visualizations
    logger.info("Generating publication-grade charts...")
    vis = Visualizer(output_dir=cfg.figures_dir)
    vis.plot_price_and_sentiment(prices_df, agg_sentiment)
    vis.plot_pos_vs_neg(agg_sentiment)
    vis.plot_post_volume(agg_sentiment)
    vis.plot_actual_vs_predicted_horizons(preds_df, model_name="ridge")
    vis.plot_model_comparison(metrics_df)
    vis.plot_correlation_heatmap(df_with_targets)
    vis.plot_equity_curves(backtest_results_dict)
    vis.plot_ablation_comparison(price_metrics_df, metrics_df, model_name="ridge")

    # 11. Reports
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
        ablation_df=ablation_df,
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
