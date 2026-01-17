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

    @staticmethod
    def generate_price_history(symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Generate random walk price history for a symbol.
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Random starting price based on symbol hash (deterministic-ish)
        seed = sum(ord(c) for c in symbol)
        np.random.seed(seed)
        
        start_price = 100.0
        if "BTC" in symbol: start_price = 60000.0
        elif "ETH" in symbol: start_price = 3000.0
        elif "SOL" in symbol: start_price = 100.0
        elif "BNB" in symbol: start_price = 500.0
        
        # Generate returns
        volatility = 0.05
        returns = np.random.normal(0.0005, volatility, days)
        price_path = start_price * (1 + returns).cumprod()
        
        df = pd.DataFrame(index=dates)
        df['close'] = price_path
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.01, days))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.02, days)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.02, days)))
        df['volume'] = np.random.uniform(100000, 10000000, days)
        
        df.iloc[0] = df.iloc[1] # Fix first row NaNs
        return df

