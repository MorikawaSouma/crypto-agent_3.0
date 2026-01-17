import os
from dotenv import load_dotenv

load_dotenv()
base_dir = os.path.dirname(__file__)
env_path = os.path.join(base_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

class Config:
    # API Keys
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET = os.getenv("BINANCE_SECRET")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
    GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    BLOCKCHAIR_API_KEY = os.getenv("BLOCKCHAIR_API_KEY")
    
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
     
    #SYMBOLS = [
    #    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    #     'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'LTC/USDT'
    # ]

    # DeepSeek Settings
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"  
   
   
   
    MODEL_NAME = "deepseek-chat" # or deepseek-reasoner
    AGENT_MODE = "debate"  # "voting" or "debate"
    DEBATE_ROUNDS = 3
    
    WARMUP_DAYS = 30          # extra days to fetch before START_TIME
    LATEST_WARMUP_EXTRA = 200 # extra candles to fetch for latest mode
    
    LLM_MOCK = False
    MEMORY_RETRIEVE_LIMIT = 10
    RESET_MEMORY_ON_BACKTEST = True # If True, clears ChromaDB memory before starting backtest
    WEIGHT_WINDOW_N = 64
    TRANSFORMER_LR = 0.001
    
    # Auxiliary Data (Macro/Sentiment/OnChain)
    USE_REAL_AUX_DATA = True  # If True, fetches real data from Yahoo/Blockchain/Alternative.me
                              # If False, uses fast mock data

    # ML Strategy Configuration
    ML_MODEL = "lightgbm"
    TOP_K = 1
    # Top 15 Liquid Cryptos by Market Cap (Expanded Universe)
    UNIVERSE = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        "TRX/USDT", "MATIC/USDT", "LTC/USDT", "SHIB/USDT", "BCH/USDT"
    ]
    # ML Parameters
    LOOKBACK_WINDOW = 60  # Extended lookback for Alpha158
    TRAIN_WINDOW = 365    # Days for training
    TEST_WINDOW = 30      # Days for testing/backtest (约一个月)
    FEATURE_WINDOW = 2000   # Max lookback for features (e.g. MA60)
    
    # Rolling Training
    ROLLING_TRAIN_INTERVAL = 30 # Retrain every 30 days

    # Backtest Settings
    INITIAL_CAPITAL = 100000.0
    WEIGHT_MODEL_PATH = "torch_weight_model.pt"
    TRADE_FEEDBACK_PATH = None

    
    # Analysis Time Range (Used for both Backtest and Main Analysis)
    # Set to None to use latest 'limit' candles for real-time analysis
    # Set to 'YYYY-MM-DD HH:MM:SS' to analyze specific historical period
    START_TIME ='2022-02-01 00:00:00'
    END_TIME ='2025-12-05 00:00:00'
    # Example for specific range:
    # START_TIME = '2023-02-01 00:00:00'
    # END_TIME = '2023-02-03 00:00:00'
    
    # Data Fetch Limit (Used when START_TIME is None)
    # Default to 1440 candles (60 days for 1h timeframe) to ensure sufficient history for indicators
    DEFAULT_LIMIT = 1440
