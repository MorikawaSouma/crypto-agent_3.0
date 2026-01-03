import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET = os.getenv("BINANCE_SECRET")
    
    # Network
    PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:7897")
    PROXIES = {
        'http': PROXY_URL,
        'https': PROXY_URL
    }
    
    # Trading Settings
    SYMBOLS = [
        'BTC/USDT'
    ]
    # SYMBOLS = [
    #     'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    #     'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'LTC/USDT'
    # ]
    TIMEFRAMES = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d'
    }
    
    # DeepSeek Settings
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # Adjust if needed
    MODEL_NAME = "deepseek-chat" # or deepseek-chat
    
    # Backtest Settings
    INITIAL_CAPITAL = 100000.0
    
    # Analysis Time Range (Used for both Backtest and Main Analysis)
    # Set to None to use latest 'limit' candles for real-time analysis
    # Set to 'YYYY-MM-DD HH:MM:SS' to analyze specific historical period
    START_TIME = None
    END_TIME =None
    # Example for specific range:
    # START_TIME = '2023-02-01 00:00:00'
    # END_TIME = '2023-02-03 00:00:00'
    
    # Data Fetch Limit (Used when START_TIME is None)
    # Default to 1440 candles (60 days for 1h timeframe) to ensure sufficient history for indicators
    DEFAULT_LIMIT = 1440
