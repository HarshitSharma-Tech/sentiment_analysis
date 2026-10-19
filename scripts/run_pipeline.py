"""
Configurable Command-Line Pipeline Runner
Allows custom assets, frequencies, custom X/Twitter CSV/JSON imports,
optional Apify execution, and model subset selections.
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_config
from src.database import DatabaseManager
from src.ingestion.prices import PriceIngestion
from src.ingestion.x_import import XPostImporter
from src.ingestion.apify_x import ApifyXConnector
from src.ingestion.news import NewsIngestion
from src.features import FeatureEngineer
from src.targets import TargetEngineer
from src.validate import WalkForwardValidator
from src.backtest import StrategyBacktester
from src.visualize import Visualizer
from src.report import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CrowdWisdom Sentiment Price Prediction Pipeline.")
    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset ticker (e.g. BTC-USD, ETH-USD)")
    parser.add_argument("--frequency", type=str, default="1h", choices=["1h", "1d"], help="Observation frequency")
    parser.add_argument("--days", type=int, default=180, help="Historical lookback window in days")
    parser.add_argument("--x-file", type=str, default="", help="Path to local X/Twitter CSV or JSON file")
    parser.add_argument("--use-apify", action="store_true", help="Attempt Apify collection if API token is present")
    parser.add_argument("--force-download", action="store_true", help="Force fresh download of prices")
    parser.add_argument("--models", nargs="+", default=["persistence", "ridge", "random_forest", "hist_gradient_boosting"], help="Models to train")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    cfg.asset_ticker = args.asset
    cfg.data_frequency = args.frequency
    cfg.history_days = args.days

    run_id = f"cli_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Running pipeline for {cfg.asset_ticker} ({cfg.data_frequency}) | Run ID: {run_id}")

    db = DatabaseManager(cfg.database_path)

    # 1. Prices
    price_ing = PriceIngestion(db_manager=db)
    prices_df = price_ing.get_or_download_prices(
        ticker=cfg.asset_ticker,
        frequency=cfg.data_frequency,
        days=cfg.history_days,
        force_download=args.force_download,
    )

    # 2. X Posts
    is_live_x = False
    if args.use_apify and cfg.apify_api_token:
        apify = ApifyXConnector(api_token=cfg.apify_api_token, db_manager=db)
        posts = apify.fetch_posts(query=f"{cfg.asset_ticker.split('-')[0]} crypto", max_posts=200)
        is_live_x = len(posts) > 0

    if args.x_file:
        importer = XPostImporter(db_manager=db)
        p = Path(args.x_file)
        if p.suffix.lower() == ".csv":
            importer.import_from_csv(p)
        else:
            importer.import_from_json(p)
    elif db.get_x_posts().empty:
        # Default sample import
        importer = XPostImporter(db_manager=db)
        importer.import_from_json(cfg.sample_dir / "btc_sample_x_posts.json", source_label="sample_reproducible")

    posts_df = db.get_x_posts(ticker=cfg.asset_ticker.split("-")[0])

    # 3. News
    news_ing = NewsIngestion(db_manager=db, tavily_api_key=cfg.tavily_api_key, exa_api_key=cfg.exa_api_key)
    try:
        news_ing.fetch_rss_feeds(feeds=cfg.rss_feeds, max_per_feed=15)
    except Exception as e:
        logger.warning(f"RSS fetch warning: {e}")
    recent_news_df = db.get_news_articles(limit=20)

    # 4. Features & Targets
    fe = FeatureEngineer()
    agg_sentiment = fe.aggregate_posts_by_frequency(posts_df, frequency=cfg.data_frequency)
    db.save_aggregated_sentiment(agg_sentiment, frequency=cfg.data_frequency)

    feature_matrix = fe.merge_features(prices_df=prices_df, agg_sentiment=agg_sentiment, frequency=cfg.data_frequency)
    te = TargetEngineer()
    df_with_targets = te.create_targets(feature_matrix)

    exclude_cols = [c for c in df_with_targets.columns if "target_" in c or "cutoff_" in c] + ["open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df_with_targets.columns if c not in exclude_cols]

    # 5. Walk-Forward Validation
    validator = WalkForwardValidator(
        min_train_size=cfg.wf_min_train_size,
        step_size=cfg.wf_step_size,
        test_size=cfg.wf_test_size,
        db_manager=db,
    )
    metrics_df, preds_df = validator.run_full_validation(
        df_with_targets=df_with_targets,
        feature_cols=feature_cols,
        models=args.models,
        horizons=cfg.horizons,
        run_id=run_id,
    )

    # 6. Backtest
    backtester = StrategyBacktester(frequency=cfg.data_frequency)
    bt_summary = backtester.evaluate_all_models(preds_df, horizon=1)

    # 7. Visualizations & Reports
    vis = Visualizer(output_dir=cfg.figures_dir)
    vis.plot_price_and_sentiment(prices_df, agg_sentiment)
    vis.plot_actual_vs_predicted_horizons(preds_df, model_name=args.models[1] if len(args.models) > 1 else args.models[0])
    vis.plot_model_comparison(metrics_df)

    rg = ReportGenerator(reports_dir=cfg.reports_dir)
    rg.generate_sentiment_outlook(
        asset_ticker=cfg.asset_ticker,
        prices_df=prices_df,
        agg_sentiment=agg_sentiment,
        recent_posts_df=posts_df,
        recent_news_df=recent_news_df,
        data_source_label="CLI Configured Ingestion",
        is_live_x=is_live_x,
    )
    rg.generate_evaluation_report(
        metrics_df=metrics_df,
        backtest_results_df=bt_summary,
        asset_ticker=cfg.asset_ticker,
        frequency=cfg.data_frequency,
        run_id=run_id,
    )

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
