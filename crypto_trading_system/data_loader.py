import ccxt
import pandas as pd
import time
from typing import Dict, List, Optional
from .config import Config

class MarketDataManager:
    def __init__(self):
        # Configure proxies properly for ccxt
        # ccxt expects 'http' and 'https' keys with the full URL
        proxies = {
            'http': Config.PROXY_URL,
            'https': Config.PROXY_URL
        }
        
        config = {
            'proxies': proxies,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        }
        
        # Only add credentials if they are valid/present
        if Config.BINANCE_API_KEY and not Config.BINANCE_API_KEY.startswith('your_'):
            config['apiKey'] = Config.BINANCE_API_KEY
            config['secret'] = Config.BINANCE_SECRET
            
        self.exchange = ccxt.binance(config)
        
        # Explicitly set https_proxy environment variable as a fallback for some underlying libraries
        # import os
        # os.environ['https_proxy'] = Config.PROXY_URL
        # os.environ['http_proxy'] = Config.PROXY_URL
        # If proxies are not supported directly in the constructor for some ccxt versions/envs, 
        # we might need to set them via https_proxy env var, but ccxt usually supports 'proxies' or 'proxy'
        # Actually ccxt python uses 'proxies' dict for requests. 
        # However, sometimes it's better to set exchange.proxies
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, since: int = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance
        :param since: Timestamp in ms
        """
        try:
            # CCXT returns: [timestamp, open, high, low, close, volume]
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            if not ohlcv:
                print(f"Warning: No data returned for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_historical_data(self, symbol: str, start_time: str, end_time: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        Fetch historical data for a specific range
        start_time, end_time: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        """
        try:
            since = self.exchange.parse8601(start_time)
            if since is None:
                # Fallback to pandas parsing
                since = int(pd.Timestamp(start_time).timestamp() * 1000)
                
            end_ts = self.exchange.parse8601(end_time)
            if end_ts is None:
                end_ts = int(pd.Timestamp(end_time).timestamp() * 1000)
            
            if since is None or end_ts is None:
                print(f"Error parsing dates: {start_time} -> {since}, {end_time} -> {end_ts}")
                return pd.DataFrame()

            all_ohlcv = []
            while since < end_ts:
                # Calculate limit based on timeframe to optimize calls, but max is usually 1000 for Binance
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                
                start_of_batch = ohlcv[0][0]
                last_of_batch = ohlcv[-1][0]
                
                # Check if we moved forward
                if start_of_batch == since and len(ohlcv) == 1:
                     # Avoid infinite loop if only one candle is returned repeatedly
                     since += self.exchange.parse_timeframe(timeframe) * 1000
                     continue

                since = last_of_batch + 1 # Next batch starts after the last candle
                all_ohlcv.extend(ohlcv)
                
                # Brief sleep to avoid rate limits if needed, though ccxt handles it well usually
                # time.sleep(0.1) 
                
                if last_of_batch >= end_ts:
                    break
            
            if not all_ohlcv:
                # print(f"Warning: No OHLCV fetched for {symbol} in range.")
                return pd.DataFrame()

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter exact range
            mask = (df.index >= pd.to_datetime(start_time)) & (df.index <= pd.to_datetime(end_time))
            return df.loc[mask]
        except Exception as e:
            # print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_market_data(self, symbols: List[str] = None, timeframe: str = '1h', limit: int = None) -> Dict[str, pd.DataFrame]:
        if symbols is None:
            symbols = Config.SYMBOLS
        
        # Use Config limit if not specified
        if limit is None:
            limit = Config.DEFAULT_LIMIT
            
        data = {}
        for symbol in symbols:
            # Check if we should fetch specific range or latest candles
            if Config.START_TIME and Config.END_TIME:
                try:
                    start_dt = pd.to_datetime(Config.START_TIME)
                    fetch_start_dt = start_dt - pd.Timedelta(days=getattr(Config, "WARMUP_DAYS", 30))
                    fetch_start_str = fetch_start_dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    fetch_start_str = Config.START_TIME
                print(f"Fetching historical data for {symbol} ({fetch_start_str} to {Config.END_TIME})...")
                df = self.get_historical_data(symbol, fetch_start_str, Config.END_TIME, timeframe)
            else:
                extra = getattr(Config, "LATEST_WARMUP_EXTRA", 200)
                eff_limit = int(limit) + int(extra)
                df = self.fetch_ohlcv(symbol, timeframe, eff_limit)
                
            if not df.empty:
                data[symbol] = df
        return data

if __name__ == "__main__":
    # Test
    dm = MarketDataManager()
    print("Fetching BTC/USDT...")
    df = dm.fetch_ohlcv('BTC/USDT', limit=5)
    print(df)
