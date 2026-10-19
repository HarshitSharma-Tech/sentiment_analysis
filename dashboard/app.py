"""
Interactive Streamlit Dashboard
CrowdWisdomTrading Quantitative Price Prediction System
Presents multi-horizon predictions, sentiment analysis, strategy backtests, and news intelligence.
Runs 100% locally on standard hardware without external server costs.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_config
from src.database import DatabaseManager

# Page Configuration
st.set_page_config(
    page_title="CrowdWisdom BTC Sentiment & Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_db() -> DatabaseManager:
    cfg = get_config()
    return DatabaseManager(cfg.database_path)


def main():
    cfg = get_config()
    db = load_db()

    st.markdown('<div class="main-title">📈 CrowdWisdomTrading: BTC Sentiment & Multi-Horizon Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative Machine Learning Pipeline across Horizons n+0 to n+5 | 100% Free-First Architecture</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Filters")
    frequency = st.sidebar.selectbox("Observation Frequency", ["1h", "1d"], index=0)
    asset = st.sidebar.text_input("Asset Ticker", value=cfg.asset_ticker)

    # Load Data from SQLite
    try:
        prices_df = db.get_prices(ticker=asset, frequency=frequency)
        posts_df = db.get_x_posts()
        agg_sentiment = db.get_aggregated_sentiment(frequency=frequency)
        preds_df = db.get_predictions()
        news_df = db.get_news_articles(limit=30)
    except Exception as e:
        st.error(f"Error connecting to local SQLite database: {e}")
        st.stop()

    if prices_df.empty or preds_df.empty:
        st.warning(
            "⚠️ Database is currently empty or has not been populated yet.\n\n"
            "Please run the demo script in your terminal to initialize and run the pipeline:\n"
            "```powershell\n"
            ".\\.venv\\Scripts\\python.exe scripts/run_demo.py\n"
            "```"
        )
        st.stop()

    # KPI Top Row
    latest_close = prices_df["close"].iloc[-1]
    prev_close = prices_df["close"].iloc[-2] if len(prices_df) > 1 else latest_close
    price_change_pct = ((latest_close - prev_close) / prev_close) * 100.0

    mean_sentiment = agg_sentiment["mean_compound"].tail(24).mean() if not agg_sentiment.empty else 0.0
    total_posts = len(posts_df)
    total_news = len(news_df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(f"{asset} Price", f"${latest_close:,.2f}", f"{price_change_pct:+.2f}%")
    with col2:
        sent_label = "Bullish" if mean_sentiment > 0.05 else ("Bearish" if mean_sentiment < -0.05 else "Neutral")
        st.metric("24h Sentiment", f"{sent_label}", f"{mean_sentiment:+.3f}")
    with col3:
        st.metric("X Posts Analyzed", f"{total_posts:,}", "VADER NLP")
    with col4:
        st.metric("Forecast Horizons", "6 (n+0 to n+5)", "Point-in-Time")
    with col5:
        st.metric("News Articles Ingested", f"{total_news}", "Free RSS Feeds")

    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Multi-Horizon Forecasts",
        "📊 Price & Sentiment Dynamics",
        "💼 Strategy Backtesting",
        "📰 News Intelligence",
        "🔍 Methodology & Audit",
    ])

    # =========================================================================
    # TAB 1: Multi-Horizon Predictions
    # =========================================================================
    with tab1:
        st.subheader("Multi-Horizon Price Predictions (Out-of-Sample)")
        st.caption("Walk-forward cross-validation predictions evaluated strictly on future unseen bars.")

        c_left, c_right = st.columns([1, 3])
        with c_left:
            horizon_sel = st.selectbox(
                "Select Forecast Horizon",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: f"n+{x} (Current Bar Close)" if x == 0 else f"n+{x} ({x} Steps Ahead)",
                index=1,
            )
            available_models = list(preds_df["model_name"].unique())
            model_sel = st.selectbox("Select Model Architecture", available_models, index=0)

        # Filter predictions
        h_preds = preds_df[
            (preds_df["horizon"] == horizon_sel) &
            (preds_df["model_name"] == model_sel)
        ].copy()

        if not h_preds.empty:
            h_preds["forecast_timestamp"] = pd.to_datetime(h_preds["forecast_timestamp"])
            h_preds = h_preds.sort_values("forecast_timestamp")

            # Metrics
            act = h_preds["actual_price"].values
            prd = h_preds["predicted_price"].values
            mae = float(np.mean(np.abs(act - prd)))
            rmse = float(np.sqrt(np.mean((act - prd) ** 2)))
            mape = float(np.mean(np.abs((act - prd) / act)) * 100.0)
            dir_acc = float(np.mean((h_preds["predicted_return"] >= 0) == (h_preds["actual_return"] >= 0)) * 100.0)

            with c_left:
                st.markdown("#### Out-of-Sample Performance")
                st.write(f"**MAE:** `${mae:,.2f}`")
                st.write(f"**RMSE:** `${rmse:,.2f}`")
                st.write(f"**MAPE:** `{mape:.2f}%`")
                st.write(f"**Directional Accuracy:** `{dir_acc:.1f}%`")

            with c_right:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=h_preds["forecast_timestamp"], y=h_preds["actual_price"],
                    mode="lines", name="Actual Price", line=dict(color="#1f77b4", width=2)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=h_preds["forecast_timestamp"], y=h_preds["predicted_price"],
                    mode="lines", name=f"Predicted ({model_sel.title()})", line=dict(color="#e377c2", width=1.8, dash="dot")
                ))
                fig_pred.update_layout(
                    title=f"Actual vs. Predicted Price for Horizon n+{horizon_sel} ({model_sel.upper()})",
                    xaxis_title="Forecast Timestamp (UTC)",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    height=450,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            # Error distribution
            with st.expander("🔬 View Residual Error Analysis"):
                errors = act - prd
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    fig_hist = px.histogram(errors, nbins=35, title="Prediction Error (Residuals) Distribution", labels={"value": "Error ($)"})
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_e2:
                    st.dataframe(h_preds[["forecast_timestamp", "actual_price", "predicted_price", "actual_return", "predicted_return"]].tail(15))
        else:
            st.info("No predictions found for this combination.")

    # =========================================================================
    # TAB 2: Price & Sentiment Dynamics
    # =========================================================================
    with tab2:
        st.subheader("Price Action vs. Social Sentiment Dynamics")
        if not agg_sentiment.empty:
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(
                go.Scatter(x=prices_df.index, y=prices_df["close"], name="BTC-USD Close", line=dict(color="#1e3a8a", width=2)),
                secondary_y=False,
            )
            rolling_sent = agg_sentiment["mean_compound"].rolling(12, min_periods=1).mean()
            fig_dual.add_trace(
                go.Scatter(x=agg_sentiment.index, y=rolling_sent, name="Sentiment (12h Rolling)", line=dict(color="#f97316", width=1.5)),
                secondary_y=True,
            )
            fig_dual.update_layout(
                title="Historical Price (USD) vs. VADER Social Polarity Index",
                xaxis_title="Timestamp (UTC)",
                hovermode="x unified",
                height=450,
            )
            fig_dual.update_yaxes(title_text="Bitcoin Price ($)", secondary_y=False)
            fig_dual.update_yaxes(title_text="VADER Compound Sentiment [-1, 1]", secondary_y=True, range=[-0.8, 0.8])
            st.plotly_chart(fig_dual, use_container_width=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                # Positive vs Negative sentiment proportions
                fig_prop = go.Figure()
                fig_prop.add_trace(go.Scatter(
                    x=agg_sentiment.index, y=agg_sentiment["pos_proportion"] * 100,
                    mode="lines", fill="tozeroy", name="Positive Sentiment %", line=dict(color="#10b981")
                ))
                fig_prop.add_trace(go.Scatter(
                    x=agg_sentiment.index, y=-agg_sentiment["neg_proportion"] * 100,
                    mode="lines", fill="tozeroy", name="Negative Sentiment % (Inverted)", line=dict(color="#ef4444")
                ))
                fig_prop.update_layout(title="Positive vs. Negative Sentiment Proportion (%)", height=320)
                st.plotly_chart(fig_prop, use_container_width=True)

            with col_s2:
                # Volume bar chart
                fig_vol = px.bar(agg_sentiment, x=agg_sentiment.index, y="post_count", title="Social Post Volume Distribution", labels={"post_count": "Post Count"})
                fig_vol.update_layout(height=320)
                st.plotly_chart(fig_vol, use_container_width=True)

    # =========================================================================
    # TAB 3: Strategy Backtesting
    # =========================================================================
    with tab3:
        st.subheader("Quantitative Strategy Evaluation (Out-of-Sample)")
        st.markdown(
            "Simulates execution of model return forecasts with **10 bps transaction costs & slippage**. "
            "Evaluated against a Buy & Hold benchmark."
        )

        h1_preds = preds_df[preds_df["horizon"] == 1].copy()
        if not h1_preds.empty:
            models = [m for m in h1_preds["model_name"].unique() if m != "persistence"]
            fig_eq = go.Figure()

            # Benchmark
            first_m = models[0]
            m_sub = h1_preds[h1_preds["model_name"] == first_m].sort_values("forecast_timestamp")
            m_sub["forecast_timestamp"] = pd.to_datetime(m_sub["forecast_timestamp"])
            bm_equity = 100_000.0 * np.cumprod(1.0 + m_sub["actual_return"].values)
            fig_eq.add_trace(go.Scatter(
                x=m_sub["forecast_timestamp"], y=bm_equity,
                mode="lines", name="Buy & Hold BTC Benchmark", line=dict(color="#64748b", dash="dash", width=1.5)
            ))

            for m in models:
                sub = h1_preds[h1_preds["model_name"] == m].sort_values("forecast_timestamp")
                sub["forecast_timestamp"] = pd.to_datetime(sub["forecast_timestamp"])
                pred_r = sub["predicted_return"].values
                act_r = sub["actual_return"].values

                pos = (pred_r > 0.001).astype(float)
                pos_prev = np.roll(pos, 1)
                pos_prev[0] = 0.0
                trades = np.abs(pos - pos_prev)
                costs = trades * (10.0 / 10_000.0)

                strat_r = (pos * act_r) - costs
                eq = 100_000.0 * np.cumprod(1.0 + strat_r)

                cum_ret = (eq[-1] / 100_000.0 - 1.0) * 100.0
                fig_eq.add_trace(go.Scatter(
                    x=sub["forecast_timestamp"], y=eq,
                    mode="lines", name=f"{m.title()} Strategy ({cum_ret:+.1f}%)", line=dict(width=2)
                ))

            fig_eq.update_layout(
                title="Out-of-Sample Portfolio Equity Curves (Initial Capital: $100,000)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

    # =========================================================================
    # TAB 4: News Intelligence
    # =========================================================================
    with tab4:
        st.subheader("Financial News Intelligence (100% Free Public RSS)")
        st.caption("Collected from CoinDesk, Cointelegraph, and Yahoo Finance RSS feeds.")

        if not news_df.empty:
            for _, item in news_df.iterrows():
                score = float(item.get("sentiment_compound", 0.0))
                badge = "🟢 POSITIVE" if score >= 0.05 else ("🔴 NEGATIVE" if score <= -0.05 else "⚪ NEUTRAL")
                with st.container():
                    st.markdown(f"**[{item['title']}]({item.get('url', '#')})**")
                    st.caption(f"Source: {item['source']} | Published: {item.get('published_at', 'N/A')} | Sentiment: {badge} (`{score:+.2f}`)")
                    if item.get("summary"):
                        st.write(item["summary"][:250] + "...")
                    st.markdown("---")

    # =========================================================================
    # TAB 5: Methodology & Audit
    # =========================================================================
    with tab5:
        st.subheader("Quantitative Methodology & Free-First Compliance")
        st.markdown("""
        ### Core Principles
        1. **100% Free Tools & Open Data:** Zero paid API subscriptions, credits, or GPU requirements.
        2. **Multi-Horizon Design:** Models forecast price and returns across $n+0$ (current bar close) and $n+1$ through $n+5$.
        3. **Valid Cutoff for $n+0$:** Uses strictly prior information available at bar start to forecast the unknown bar close.
        4. **Chronological Walk-Forward Validation:** Preprocessing (StandardScaler) and models are fitted exclusively on past training data, eliminating lookahead bias.
        5. **Persistence Benchmark:** Naive zero-change benchmark evaluated on identical out-of-sample points.
        """)


if __name__ == "__main__":
    main()
