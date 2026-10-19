"""
Free & Local Sentiment Analysis Module
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner).
Completely free, fast, offline, and calibrated for social media text / microblogs.
No API keys, credits, or GPU required.
"""

import re
from typing import Any, Dict, List, Union
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Local sentiment analyzer for X/Twitter posts and financial headlines."""

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        # Enrich VADER with key crypto/trading domain lexicon
        crypto_lexicon = {
            "bullish": 2.5,
            "bearish": -2.5,
            "pump": 1.8,
            "dump": -2.2,
            "ath": 2.8,
            "all-time high": 2.8,
            "moon": 2.4,
            "breakout": 2.0,
            "rekt": -3.0,
            "fud": -2.0,
            "hodl": 1.5,
            "liquidation": -2.2,
            "liquidated": -2.4,
            "rugpull": -3.5,
            "scam": -3.2,
            "accumulate": 1.7,
            "halving": 1.5,
            "rally": 2.2,
            "crash": -3.0,
            "dip": -0.8,
            "buy the dip": 2.0,
        }
        self.analyzer.lexicon.update(crypto_lexicon)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean raw text while preserving financial cashtags and emojis."""
        if not isinstance(text, str):
            return ""

        # Remove raw URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove user handles (@username) but keep cashtags ($BTC)
        cleaned = re.sub(r"@\w+", "", cleaned)
        # Normalize excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def analyze(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of a single string.
        Returns:
            dict with compound (-1 to 1), pos, neu, neg, and sentiment_label.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neu": 1.0,
                "neg": 0.0,
                "label": "neutral",
            }

        scores = self.analyzer.polarity_scores(cleaned)
        compound = float(scores["compound"])

        if compound >= self.pos_threshold:
            label = "positive"
        elif compound <= self.neg_threshold:
            label = "negative"
        else:
            label = "neutral"

        return {
            "compound": compound,
            "pos": float(scores["pos"]),
            "neu": float(scores["neu"]),
            "neg": float(scores["neg"]),
            "label": label,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[float, str]]]:
        """Analyze a list of texts in batch."""
        return [self.analyze(t) for t in texts]


# Module-level convenience singleton
_default_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SentimentAnalyzer()
    return _default_analyzer
