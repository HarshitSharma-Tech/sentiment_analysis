"""Unit tests for sentiment analyzer."""
from src.sentiment import SentimentAnalyzer


def test_sentiment_clean_text():
    analyzer = SentimentAnalyzer()
    raw = "Check out this link https://example.com and follow @elonmusk for $BTC updates! 🚀"
    cleaned = analyzer.clean_text(raw)
    assert "https" not in cleaned
    assert "@elonmusk" not in cleaned
    assert "$BTC" in cleaned


def test_sentiment_crypto_lexicon():
    analyzer = SentimentAnalyzer()
    bullish_res = analyzer.analyze("Bitcoin is super bullish, breaking out to all-time highs! 🚀")
    assert bullish_res["compound"] > 0.3
    assert bullish_res["label"] == "positive"

    bearish_res = analyzer.analyze("Total rugpull and crash, long positions got liquidated and rekt.")
    assert bearish_res["compound"] < -0.3
    assert bearish_res["label"] == "negative"


def test_sentiment_neutral():
    analyzer = SentimentAnalyzer()
    res = analyzer.analyze("The CPI inflation report will be released on Wednesday at 8:30 AM.")
    assert -0.2 < res["compound"] < 0.2
