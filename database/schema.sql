-- CrowdWisdomTrading Database Schema (SQLite)
-- Supports modular, restartable ingestion with duplicate prevention

-- 1. Raw and Parsed X/Twitter Posts
CREATE TABLE IF NOT EXISTS x_posts (
    post_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    text TEXT NOT NULL,
    likes INTEGER DEFAULT 0,
    reposts INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    author_followers INTEGER DEFAULT 0,
    query_ticker TEXT DEFAULT 'BTC',
    sentiment_compound REAL,
    sentiment_pos REAL,
    sentiment_neu REAL,
    sentiment_neg REAL,
    source TEXT NOT NULL, -- 'apify', 'local_import', 'sample_reproducible'
    collected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_x_posts_timestamp ON x_posts(timestamp ASC);
CREATE INDEX IF NOT EXISTS idx_x_posts_ticker ON x_posts(query_ticker);

-- 2. Historical Market Prices
CREATE TABLE IF NOT EXISTS historical_prices (
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    frequency TEXT NOT NULL, -- '1h' or '1d'
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    source TEXT NOT NULL, -- 'yfinance', 'binance_public', 'csv_import'
    collected_at TEXT NOT NULL,
    PRIMARY KEY (ticker, timestamp, frequency)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_time ON historical_prices(ticker, timestamp ASC);

-- 3. News Articles
CREATE TABLE IF NOT EXISTS news_articles (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT,
    url TEXT,
    summary TEXT,
    sentiment_compound REAL,
    collected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at ASC);

-- 4. Aggregated Sentiment Features (per time bucket)
CREATE TABLE IF NOT EXISTS aggregated_sentiment (
    timestamp TEXT NOT NULL,
    frequency TEXT NOT NULL,
    post_count INTEGER NOT NULL,
    mean_compound REAL NOT NULL,
    pos_proportion REAL NOT NULL,
    neu_proportion REAL NOT NULL,
    neg_proportion REAL NOT NULL,
    sentiment_dispersion REAL NOT NULL,
    total_likes INTEGER DEFAULT 0,
    total_reposts INTEGER DEFAULT 0,
    total_views INTEGER DEFAULT 0,
    engagement_weighted_sentiment REAL NOT NULL,
    PRIMARY KEY (timestamp, frequency)
);

CREATE INDEX IF NOT EXISTS idx_agg_sentiment_time ON aggregated_sentiment(timestamp ASC);

-- 5. Out-of-Sample Model Predictions
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    forecast_timestamp TEXT NOT NULL,
    target_timestamp TEXT NOT NULL,
    actual_price REAL,
    predicted_price REAL,
    actual_return REAL,
    predicted_return REAL,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (model_name, horizon, forecast_timestamp, run_id)
);

CREATE INDEX IF NOT EXISTS idx_preds_lookup ON model_predictions(run_id, model_name, horizon);

-- 6. Evaluation and Backtest Performance Metrics
CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    frequency TEXT NOT NULL,
    train_bars INTEGER NOT NULL,
    test_bars INTEGER NOT NULL,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    mape REAL NOT NULL,
    directional_accuracy REAL NOT NULL,
    r2_score REAL,
    baseline_mae REAL,
    baseline_rmse REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    cumulative_return REAL,
    win_rate REAL,
    total_trades INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results(run_id, model_name);
