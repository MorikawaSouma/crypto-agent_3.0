import pandas as pd
import numpy as np
import random
from datetime import datetime

class MockDataProvider:
    @staticmethod
    def get_onchain_data() -> dict:
        """
        Mock On-chain data
        """
        return {
            "net_flow": random.uniform(-1000, 1000),  # BTC/ETH flow
            "whale_holdings_change": random.uniform(-5, 5), # Percentage
            "active_addresses": int(random.uniform(500000, 1000000)),
            "mvrv_ratio": random.uniform(1.0, 3.5)
        }

    @staticmethod
    def get_sentiment_data() -> dict:
        """
        Mock Sentiment data
        """
        return {
            "twitter_sentiment": random.uniform(0, 100), # 0-100
            "reddit_sentiment": random.uniform(0, 100),
            "fear_greed_index": int(random.uniform(10, 90)),
            "google_trends": int(random.uniform(20, 100)),
            "news_sentiment_score": random.uniform(-1, 1)
        }

    @staticmethod
    def get_macro_data() -> dict:
        """
        Mock Macro data
        """
        return {
            "dxy": random.uniform(90, 110), # US Dollar Index
            "us_10y_yield": random.uniform(3.0, 5.0), # %
            "nasdaq_correlation": random.uniform(0.3, 0.8),
            "vix": random.uniform(10, 35) # Volatility Index
        }
