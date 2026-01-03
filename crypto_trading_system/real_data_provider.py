import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time

class RealDataProvider:
    def __init__(self):
        self.macro_data = pd.DataFrame()
        self.sentiment_data = pd.DataFrame()
        self.onchain_data = pd.DataFrame()
        self.is_initialized = False

    def fetch_all_data(self, start_date: str, end_date: str):
        """
        Fetch all auxiliary data for the given date range.
        Extend range slightly to ensure coverage.
        """
        if self.is_initialized:
            return

        print(f"Fetching REAL auxiliary data from {start_date} to {end_date}...")
        
        # Convert to datetime
        start_dt = pd.to_datetime(start_date) - timedelta(days=10)
        end_dt = pd.to_datetime(end_date) + timedelta(days=10)
        
        # 1. Fetch Macro Data (yfinance)
        self._fetch_macro_data(start_dt, end_dt)
        
        # 2. Fetch Sentiment Data (Alternative.me)
        self._fetch_sentiment_data()
        
        # 3. Fetch On-chain Data (Blockchain.com)
        self._fetch_onchain_data(start_dt, end_dt)
        
        self.is_initialized = True
        print("✓ Real auxiliary data fetched.")

    def _fetch_macro_data(self, start_dt, end_dt):
        print("  Fetching Macro data (S&P500, DXY, VIX, 10Y Yield)...")
        
        # Map our internal names to yfinance tickers
        ticker_map = {
            "sp500": "^GSPC",
            "dxy": "DX-Y.NYB", 
            "us10y": "^TNX",
            "vix": "^VIX"
        }
        
        combined_df = pd.DataFrame()

        # Create a session with custom headers to mimic browser and avoid some rate limits
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        # Helper for direct CSV download (Fallback)
        def download_yahoo_csv(ticker, start, end):
            try:
                period1 = int(start.timestamp())
                period2 = int(end.timestamp())
                url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
                
                # print(f"    DEBUG: Downloading CSV from {url}")
                response = session.get(url, timeout=10)
                response.raise_for_status()
                
                from io import StringIO
                csv_df = pd.read_csv(StringIO(response.text))
                csv_df['Date'] = pd.to_datetime(csv_df['Date'])
                csv_df.set_index('Date', inplace=True)
                return csv_df['Close']
            except Exception as e:
                # print(f"    DEBUG: CSV download failed: {e}")
                return None

        for name, ticker in ticker_map.items():
            try:
                # print(f"    Fetching {name} ({ticker})...")
                # Add a small delay to be nice to the API
                time.sleep(1.0)
                
                # 1. Try yfinance library
                try:
                    df = yf.download(ticker, start=start_dt, end=end_dt, progress=False, session=session)
                    
                    if not df.empty:
                        # Handle MultiIndex if present (yfinance changed recently)
                        if isinstance(df.columns, pd.MultiIndex):
                            try:
                                series = df.xs('Close', level=0, axis=1)[ticker]
                            except:
                                try:
                                    series = df.xs(ticker, level=1, axis=1)['Close']
                                except:
                                    series = df.iloc[:, 0] # Fallback
                        elif 'Close' in df.columns:
                            series = df['Close']
                        else:
                            series = df.iloc[:, 0]
                    else:
                        series = None
                except Exception as e:
                    # print(f"    ! yfinance lib failed for {ticker}: {e}")
                    series = None

                # 2. Fallback to direct CSV if library failed or returned empty
                if series is None or series.empty:
                    # print(f"    Using CSV fallback for {ticker}...")
                    series = download_yahoo_csv(ticker, start_dt, end_dt)

                if series is not None and not series.empty:
                    series.name = name
                    # Normalize index
                    series.index = series.index.tz_localize(None)
                    
                    if combined_df.empty:
                        combined_df = pd.DataFrame(series)
                    else:
                        combined_df = combined_df.join(series, how='outer')
                else:
                    print(f"    ! Warning: No data for {ticker} (Yahoo failed)")
                    
            except Exception as e:
                print(f"    ! Error fetching {ticker}: {e}")
        
        # 3. Fallback to Stooq if Yahoo failed for everything
        if combined_df.empty:
            print("    ! Yahoo Finance failed. Trying Stooq as fallback...")
            self._fetch_from_stooq(start_dt, end_dt)
            return

        if not combined_df.empty:
            # Process combined_df
            combined_df = combined_df.ffill().bfill()
            self.macro_data = combined_df
            print(f"    ✓ Macro data: {len(combined_df)} rows")
        else:
             print("    !! Failed to fetch any macro data. Using hardcoded fallback.")
             self.macro_data = pd.DataFrame()

    def _fetch_from_stooq(self, start_dt, end_dt):
        stooq_map = {
            "sp500": "^SPX",
            "dxy": "U.S. Dollar Index", # Stooq hard to get exact DXY ticker via CSV sometimes, trying generic or skipping
            "us10y": "10USY.B", 
            "vix": "^VIX" # Stooq might not have VIX easily
        }
        # Simplified Stooq map for most critical ones
        # ^SPX is standard. 
        # 10USY.B is 10 Year Yield.
        
        combined_df = pd.DataFrame()
        
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })
        
        for name, ticker in stooq_map.items():
            if name == "dxy" or name == "vix": continue # Skip difficult ones for now to ensure at least SPX works
            
            try:
                url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
                response = session.get(url, timeout=10)
                
                if response.status_code == 200:
                    from io import StringIO
                    content = response.content.decode('utf-8')
                    if "Exceeded the limit" in content:
                        print(f"    ! Stooq rate limit for {ticker}")
                        continue
                        
                    csv_df = pd.read_csv(StringIO(content))
                    
                    if 'Date' in csv_df.columns and 'Close' in csv_df.columns:
                        csv_df['Date'] = pd.to_datetime(csv_df['Date'])
                        csv_df.set_index('Date', inplace=True)
                        series = csv_df['Close']
                        series.name = name
                        
                        # Filter date
                        series = series[(series.index >= start_dt) & (series.index <= end_dt)]
                        
                        if combined_df.empty:
                            combined_df = pd.DataFrame(series)
                        else:
                            combined_df = combined_df.join(series, how='outer')
            except Exception as e:
                print(f"    ! Stooq failed for {ticker}: {e}")
                
        if not combined_df.empty:
            self.macro_data = combined_df.ffill().bfill()
            print(f"    ✓ Macro data (from Stooq): {len(combined_df)} rows")
        else:
            print("    !! Stooq also failed.")
            self.macro_data = pd.DataFrame()

    def _fetch_sentiment_data(self):
        print("  Fetching Crypto Fear & Greed Index...")
        try:
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['metadata']['error'] is None:
                records = data['data']
                df = pd.DataFrame(records)
                # Cast to numeric before converting to datetime to avoid FutureWarning
                df['timestamp'] = pd.to_numeric(df['timestamp'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['value'] = pd.to_numeric(df['value'])
                df = df.set_index('timestamp').sort_index()
                
                # Normalize index
                df.index = df.index.tz_localize(None)
                
                self.sentiment_data = df[['value', 'value_classification']]
                print(f"    ✓ Sentiment data: {len(df)} rows")
            else:
                print("    !! Error from Fear&Greed API")
                
        except Exception as e:
            print(f"    !! Error fetching sentiment data: {e}")
            self.sentiment_data = pd.DataFrame()

    def _fetch_onchain_data(self, start_dt, end_dt):
        print("  Fetching On-chain data (Blockchain.com)...")
        # Blockchain.com API is free for charts
        # We need Active Addresses and Transactions
        
        try:
            # Helper to fetch chart
            def fetch_chart(chart_name):
                url = f"https://api.blockchain.info/charts/{chart_name}?timespan=2years&format=json&sampled=true"
                # Note: timespan=2years should cover our backtest. 
                # If backtest is older, need 'all' or specific.
                r = requests.get(url, timeout=10)
                d = r.json()
                values = d['values']
                df = pd.DataFrame(values)
                # Cast to numeric before converting to datetime
                df['x'] = pd.to_numeric(df['x'])
                df['x'] = pd.to_datetime(df['x'], unit='s')
                df = df.set_index('x').sort_index()
                df.columns = [chart_name]
                return df

            active_addr = fetch_chart('n-unique-addresses')
            tx_count = fetch_chart('n-transactions')
            
            # Merge
            df = pd.concat([active_addr, tx_count], axis=1)
            
            # Fill
            df = df.ffill().bfill()
            
            # Normalize index
            df.index = df.index.tz_localize(None)
            
            # Filter range
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            self.onchain_data = df
            print(f"    ✓ On-chain data: {len(df)} rows")
            
        except Exception as e:
            print(f"    !! Error fetching on-chain data: {e}")
            self.onchain_data = pd.DataFrame()

    def get_macro_at(self, date: pd.Timestamp) -> dict:
        """Get macro data for a specific date (or nearest previous)"""
        if self.macro_data.empty:
            return {"dxy": 100, "us_10y_yield": 4.0, "vix": 20, "nasdaq_correlation": 0.5} # Fallback
            
        try:
            # As of method
            idx = self.macro_data.index.asof(date)
            if pd.isna(idx):
                row = self.macro_data.iloc[0] # Use first available if date is before start
            else:
                row = self.macro_data.loc[idx]
                
            return {
                "dxy": row.get('dxy', 100),
                "us_10y_yield": row.get('us10y', 4.0),
                "vix": row.get('vix', 20),
                "nasdaq_correlation": 0.8 # Static for now as calculation is complex
            }
        except:
            return {"dxy": 100, "us_10y_yield": 4.0, "vix": 20, "nasdaq_correlation": 0.5}

    def get_sentiment_at(self, date: pd.Timestamp) -> dict:
        if self.sentiment_data.empty:
            return {"fear_greed_index": 50, "news_sentiment_score": 0}
            
        try:
            # Fear & Greed is daily
            # Normalize date to midnight
            d = date.normalize()
            idx = self.sentiment_data.index.asof(d)
            
            if pd.isna(idx):
                val = 50
            else:
                val = self.sentiment_data.loc[idx, 'value']
                
            return {
                "fear_greed_index": int(val),
                "twitter_sentiment": val, # Proxy
                "reddit_sentiment": val, # Proxy
                "google_trends": val, # Proxy
                "news_sentiment_score": (val - 50) / 50.0 # -1 to 1
            }
        except:
            return {"fear_greed_index": 50, "news_sentiment_score": 0}

    def get_onchain_at(self, date: pd.Timestamp) -> dict:
        if self.onchain_data.empty:
            return {"active_addresses": 800000, "net_flow": 0}
            
        try:
            idx = self.onchain_data.index.asof(date)
            if pd.isna(idx):
                row = self.onchain_data.iloc[0]
            else:
                row = self.onchain_data.loc[idx]
                
            return {
                "active_addresses": int(row.get('n-unique-addresses', 800000)),
                "transaction_count": int(row.get('n-transactions', 300000)),
                "net_flow": 0, # Hard to get free
                "mvrv_ratio": 2.0 # Hard to get free
            }
        except:
             return {"active_addresses": 800000, "net_flow": 0}
