import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import xml.etree.ElementTree as ET
from .config import Config
from .mock_data import MockDataProvider

class RealDataProvider:
    def __init__(self):
        self.macro_data = pd.DataFrame()
        self.sentiment_data = pd.DataFrame()
        self.onchain_data = pd.DataFrame()
        self.blockchair_data = {} # {symbol: DataFrame}
        self.defillama_data = {} # {symbol: {'tvl': Series, 'fees': Series, 'stablecoins': Series}}
        self.news_data = {} # {symbol: DataFrame or list of dicts}
        self.glassnode_data = {} # {symbol: DataFrame}
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
        
        # 3.5 Fetch Blockchair Data (Non-BTC, Free Multi-chain)
        self._fetch_blockchair_data(start_dt, end_dt)
        
        # 3.6 Fetch Glassnode Data (Non-BTC, optional paid source)
        self._fetch_glassnode_data(start_dt, end_dt)
        
        # 4. Fetch DefiLlama Data (TVL, Stablecoins)
        self._fetch_defillama_data()

        # 5. Fetch News Data (CryptoPanic or RSS)
        self._fetch_news_data(start_dt, end_dt)
        
        self.is_initialized = True
        print("✓ Real auxiliary data fetched.")

    def _fetch_defillama_data(self):
        print("  Fetching DefiLlama data (TVL, Stablecoins, Fees)...")
        self.defillama_data = {}
        
        # DefiLlama API endpoints
        # /v2/historicalChainTvl/{chain}
        # Map our symbols to DefiLlama chain names
        chain_map = {
            "ETH": "Ethereum",
            "BNB": "BSC",
            "SOL": "Solana",
            "AVAX": "Avalanche",
            "MATIC": "Polygon",
            "TRX": "Tron",
            "DOT": "Polkadot", 
            "ADA": "Cardano",
            "BTC": "Bitcoin"
        }
        
        session = requests.Session()
        
        for symbol, chain in chain_map.items():
            symbol_data = {}
            
            # 1. TVL
            try:
                url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    df = pd.DataFrame(data)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        df.set_index('date', inplace=True)
                        df = df.sort_index()
                        df.index = df.index.tz_localize(None)
                        symbol_data['tvl'] = df['tvl']
            except Exception as e:
                # print(f"    ! Error fetching TVL for {chain}: {e}")
                pass

            # 2. Stablecoins Market Cap (Total on chain)
            try:
                url = f"https://stablecoins.llama.fi/stablecoincharts/{chain}"
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # list of {date: ts, totalCirculating: {usd: val}}
                    processed = []
                    for item in data:
                        processed.append({
                            'date': pd.to_datetime(int(item['date']), unit='s'),
                            'stable_mcap': item.get('totalCirculating', {}).get('usd', 0)
                        })
                    
                    df = pd.DataFrame(processed)
                    if not df.empty:
                        df.set_index('date', inplace=True)
                        df = df.sort_index()
                        df.index = df.index.tz_localize(None)
                        symbol_data['stable_mcap'] = df['stable_mcap']
            except Exception as e:
                # print(f"    ! Error fetching Stablecoins for {chain}: {e}")
                pass

            # 3. Fees/Revenue (Summary)
            # This endpoint might be heavy, use summary if possible or skip if fails
            try:
                # Use simple summary endpoint for daily fees
                # https://api.llama.fi/summary/fees/{chain}?dataType=dailyFees
                url = f"https://api.llama.fi/summary/fees/{chain}?dataType=dailyFees"
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # {totalDataChart: [[ts, val], ...]}
                    chart = data.get('totalDataChart', [])
                    if chart:
                        df = pd.DataFrame(chart, columns=['date', 'fees'])
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        df.set_index('date', inplace=True)
                        df = df.sort_index()
                        df.index = df.index.tz_localize(None)
                        symbol_data['fees'] = df['fees']
            except Exception as e:
                pass

            if symbol_data:
                self.defillama_data[symbol] = symbol_data
                
        print(f"    ✓ DefiLlama data: {len(self.defillama_data)} chains")

    def get_defillama_at(self, symbol: str, date: pd.Timestamp) -> dict:
        # Extract base symbol from pair (e.g. ETH/USDT -> ETH)
        base = symbol.split('/')[0]
        if base not in self.defillama_data:
            return {}
            
        metrics = self.defillama_data[base]
        result = {}
        
        try:
            for key, series in metrics.items():
                idx = series.index.asof(date)
                if not pd.isna(idx):
                    result[key] = series.loc[idx]
        except:
            pass
            
        return result

    def _fetch_news_data(self, start_dt, end_dt):
        print("  Fetching News data (CryptoPanic/RSS)...")
        self.news_data = {} 
        
        api_key = Config.CRYPTOPANIC_API_KEY
        
        # Major coins to track for news
        target_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "AVAX"]
        
        if api_key:
            self._fetch_cryptopanic(api_key, target_symbols, start_dt, end_dt)
        else:
            print("    (No CryptoPanic Key found, defaulting to RSS fallback)")
            self._fetch_rss_news(target_symbols)
            
        count = sum(len(df) for df in self.news_data.values()) if self.news_data else 0
        print(f"    ✓ News data: {count} items (across symbols)")

    def _fetch_cryptopanic(self, api_key, symbols, start_dt, end_dt):
        session = requests.Session()
        currencies_str = ",".join(symbols)
        
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={currencies_str}&kind=news"
        
        all_posts = []
        page = 1
        max_pages = 5 # Limit to avoid excessive calls
        
        print(f"    Fetching CryptoPanic (max {max_pages} pages)...")
        
        while url and page <= max_pages:
            try:
                resp = session.get(url, timeout=10)
                data = resp.json()
                
                if 'results' in data:
                    posts = data['results']
                    for post in posts:
                        try:
                            created_at = pd.to_datetime(post['created_at'])
                            if created_at.tzinfo is not None:
                                created_at = created_at.tz_localize(None)
                            
                            item = {
                                'date': created_at,
                                'title': post['title'],
                                'url': post['url'],
                                'currencies': [c['code'] for c in post.get('currencies', []) if 'code' in c],
                                'source': post.get('source', {}).get('title', 'Unknown')
                            }
                            all_posts.append(item)
                        except Exception as e:
                            continue
                    
                    url = data.get('next')
                    page += 1
                    time.sleep(0.5) 
                else:
                    break
            except Exception as e:
                print(f"    ! Error CryptoPanic page {page}: {e}")
                break
                
        if all_posts:
            df_all = pd.DataFrame(all_posts)
            df_all.set_index('date', inplace=True)
            df_all.sort_index(inplace=True)
            
            for symbol in symbols:
                mask = df_all['currencies'].apply(lambda x: symbol in x)
                symbol_news = df_all[mask]
                if not symbol_news.empty:
                    self.news_data[symbol] = symbol_news
            
            self.news_data['GLOBAL'] = df_all

    def _fetch_rss_news(self, symbols):
        # Additional RSS feeds for better coverage
        rss_urls = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cryptoslate.com/feed/",
            "https://decrypt.co/feed"
        ]
        
        # Enhanced keyword mapping for better matching
        # Symbol -> List of keywords (Case insensitive matching)
        keyword_map = {
            "BTC": ["Bitcoin", "BTC"],
            "ETH": ["Ethereum", "Ether", "ETH"],
            "SOL": ["Solana", "SOL"],
            "BNB": ["Binance Coin", "BNB", "BSC", "Binance Smart Chain"],
            "XRP": ["Ripple", "XRP"],
            "ADA": ["Cardano", "ADA"],
            "DOT": ["Polkadot", "DOT"],
            "AVAX": ["Avalanche", "AVAX"],
            "DOGE": ["Dogecoin", "DOGE"],
            "LTC": ["Litecoin", "LTC"],
            "TRX": ["Tron", "TRX"],
            "MATIC": ["Polygon", "MATIC"],
            "SHIB": ["Shiba Inu", "SHIB"]
        }
        
        all_items = []
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        for url in rss_urls:
            try:
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    try:
                        # Handle potential encoding issues
                        content = resp.content
                        root = ET.fromstring(content)
                        
                        # Handle different RSS formats (standard RSS vs Atom)
                        items = root.findall('./channel/item')
                        if not items:
                            # Try Atom format (e.g. Decrypt sometimes uses entry)
                            # But standard RSS usually has channel/item
                            pass
                            
                        for item in items:
                            title = item.find('title').text if item.find('title') is not None else ""
                            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ""
                            link = item.find('link').text if item.find('link') is not None else ""
                            
                            try:
                                dt = pd.to_datetime(pubDate)
                                if dt.tzinfo is not None:
                                    dt = dt.tz_localize(None)
                            except:
                                dt = datetime.now()
                                
                            all_items.append({
                                'date': dt,
                                'title': title,
                                'url': link,
                                'source': 'RSS'
                            })
                    except Exception as e:
                        print(f"    ! Error parsing XML from {url}: {e}")
            except Exception as e:
                print(f"    ! Error fetching RSS {url}: {e}")
        
        if all_items:
            df = pd.DataFrame(all_items)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Remove duplicates based on title
            df = df.drop_duplicates(subset=['title'])
            
            for symbol in symbols:
                # Get keywords for this symbol, default to just symbol itself if not in map
                keywords = keyword_map.get(symbol, [symbol])
                
                # Build regex pattern for any of the keywords
                # Use word boundaries \b to avoid partial matches (e.g. "SOL" matching "SOLD")
                # pattern = r'\b(' + '|'.join(keywords) + r')\b'
                # Note: Simple str.contains might be safer than strict regex with boundaries due to varying punctuation
                
                # Let's use a simpler approach: iterate and check
                mask = pd.Series([False] * len(df), index=df.index)
                
                for kw in keywords:
                    mask |= df['title'].str.contains(kw, case=False, regex=False)
                
                symbol_news = df[mask]
                if not symbol_news.empty:
                    self.news_data[symbol] = symbol_news
            
            self.news_data['GLOBAL'] = df

    def get_news_at(self, symbol: str, date: pd.Timestamp, window_days=3) -> list:
        """Get news items for a symbol in the window ending at date"""
        base = symbol.split('/')[0]
        
        news_df = self.news_data.get(base)
        if news_df is None or news_df.empty:
            news_df = self.news_data.get('GLOBAL')
            
        if news_df is None or news_df.empty:
            return []
            
        start_window = date - timedelta(days=window_days)
        
        try:
            mask = (news_df.index >= start_window) & (news_df.index <= date)
            relevant = news_df[mask]
            # Return most recent 5
            return relevant.tail(5)[['title', 'source']].to_dict('records')
        except:
            return []



    def _fetch_macro_data(self, start_dt, end_dt):
        print("  Fetching Macro data (S&P500, DXY, VIX, 10Y Yield)...")
        
        sources = []
        stooq_df = self._fetch_from_stooq(start_dt, end_dt)
        if stooq_df is not None and not stooq_df.empty:
            sources.append(stooq_df)
        yahoo_df = self._fetch_macro_from_yahoo(start_dt, end_dt)
        if yahoo_df is not None and not yahoo_df.empty:
            sources.append(yahoo_df)
        elif Config.ALPHA_VANTAGE_API_KEY:
            print("    ! Yahoo Finance failed or returned empty.")
        alpha_df = None
        if Config.ALPHA_VANTAGE_API_KEY:
            alpha_df = self._fetch_macro_from_alpha_vantage(start_dt, end_dt, Config.ALPHA_VANTAGE_API_KEY)
            if alpha_df is not None and not alpha_df.empty:
                sources.append(alpha_df)
        if not sources:
            print("    !! Failed to fetch any macro data from all sources.")
            self.macro_data = pd.DataFrame()
            return
        indicators = ["sp500", "dxy", "us10y", "vix"]
        combined_df = pd.DataFrame()
        for name in indicators:
            merged_series = None
            for df in sources:
                if name in df.columns:
                    series = df[name].dropna()
                    if series.empty:
                        continue
                    if merged_series is None:
                        merged_series = series
                    else:
                        merged_series = merged_series.combine_first(series)
            if merged_series is not None and not merged_series.empty:
                merged_series = merged_series.sort_index()
                if combined_df.empty:
                    combined_df = pd.DataFrame(merged_series)
                else:
                    combined_df = combined_df.join(merged_series, how="outer")
        if combined_df.empty:
            print("    !! Macro indicators missing after merging all sources.")
            self.macro_data = pd.DataFrame()
            return
        combined_df = combined_df.sort_index()
        combined_df = combined_df.ffill().bfill()
        self.macro_data = combined_df
        print(f"    ✓ Macro data (merged): {len(combined_df)} rows, columns: {list(combined_df.columns)}")

    def _fetch_macro_from_yahoo(self, start_dt, end_dt):
        from io import StringIO

        ticker_map = {
            "sp500": "^GSPC",
            "dxy": "DX-Y.NYB", 
            "us10y": "^TNX",
            "vix": "^VIX"
        }
        
        combined_df = pd.DataFrame()

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        try:
            session.proxies.update(Config.PROXIES)
        except Exception:
            pass
        
        def download_yahoo_csv(ticker, start, end):
            try:
                period1 = int(start.timestamp())
                period2 = int(end.timestamp())
                url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
                resp = session.get(url, timeout=10)
                resp.raise_for_status()
                csv_df = pd.read_csv(StringIO(resp.text))
                csv_df['Date'] = pd.to_datetime(csv_df['Date'])
                csv_df.set_index('Date', inplace=True)
                return csv_df['Close']
            except Exception:
                return None

        for name, ticker in ticker_map.items():
            try:
                time.sleep(1.0)
                try:
                    df = yf.download(ticker, start=start_dt, end=end_dt, progress=False, session=session)
                    if not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            try:
                                series = df.xs('Close', level=0, axis=1)[ticker]
                            except Exception:
                                try:
                                    series = df.xs(ticker, level=1, axis=1)['Close']
                                except Exception:
                                    series = df.iloc[:, 0]
                        elif 'Close' in df.columns:
                            series = df['Close']
                        else:
                            series = df.iloc[:, 0]
                    else:
                        series = None
                except Exception:
                    series = None

                if series is None or series.empty:
                    series = download_yahoo_csv(ticker, start_dt, end_dt)

                if series is not None and not series.empty:
                    series.name = name
                    series.index = series.index.tz_localize(None)
                    if combined_df.empty:
                        combined_df = pd.DataFrame(series)
                    else:
                        combined_df = combined_df.join(series, how='outer')
            except Exception as e:
                print(f"    ! Error fetching {ticker} from Yahoo: {e}")

        return combined_df

    def _fetch_from_stooq(self, start_dt, end_dt):
        stooq_map = {
            "sp500": "^SPX",     # S&P 500
            "dxy": "dx.f",       # U.S. Dollar Index futures on ICE (DX.F)
            "us10y": "10USY.B",  # 10 Year Yield
            "vix": "vi.f"        # S&P 500 VIX (VI.F)
        }
        
        combined_df = pd.DataFrame()
        
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })
        try:
            session.proxies.update(Config.PROXIES)
        except Exception:
            pass
        
        for name, ticker in stooq_map.items():
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
                        
                        series = series[(series.index >= start_dt) & (series.index <= end_dt)]
                        
                        if combined_df.empty:
                            combined_df = pd.DataFrame(series)
                        else:
                            combined_df = combined_df.join(series, how='outer')
            except Exception as e:
                print(f"    ! Stooq failed for {ticker}: {e}")
                
        if not combined_df.empty:
            combined_df = combined_df.ffill().bfill()
            print(f"    ✓ Macro data (from Stooq): {len(combined_df)} rows")
            return combined_df
        else:
            print("    !! Stooq also failed.")
            return pd.DataFrame()

    def _fetch_macro_from_alpha_vantage(self, start_dt, end_dt, api_key: str):
        base_url = "https://www.alphavantage.co/query"
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        try:
            session.proxies.update(Config.PROXIES)
        except Exception:
            pass

        combined_df = pd.DataFrame()

        def fetch_time_series_daily(symbol: str):
            try:
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": symbol,
                    "apikey": api_key,
                    "outputsize": "full"
                }
                resp = session.get(base_url, params=params, timeout=10)
                if resp.status_code != 200:
                    print(f"    ! Alpha Vantage HTTP {resp.status_code} for {symbol}")
                    return None
                data = resp.json()
                key = "Time Series (Daily)"
                if key not in data:
                    err = data.get("Information") or data.get("Note") or data.get("Error Message")
                    if err:
                        msg = str(err).replace("\\n", " ")
                        print(f"    ! Alpha Vantage {symbol} info: {msg[:120]}...")
                    else:
                        print(f"    ! Alpha Vantage {symbol} unexpected response.")
                    return None
                ts = data[key]
                df = pd.DataFrame.from_dict(ts, orient="index")
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                close = pd.to_numeric(df["4. close"], errors="coerce")
                close = close[(close.index >= start_dt) & (close.index <= end_dt)]
                return close
            except Exception as e:
                print(f"    ! Alpha Vantage daily failed for {symbol}: {e}")
                return None

        def fetch_treasury_yield():
            try:
                params = {
                    "function": "TREASURY_YIELD",
                    "interval": "daily",
                    "maturity": "10year",
                    "apikey": api_key
                }
                resp = session.get(base_url, params=params, timeout=10)
                if resp.status_code != 200:
                    print(f"    ! Alpha Vantage HTTP {resp.status_code} for TREASURY_YIELD")
                    return None
                data = resp.json()
                if "data" not in data:
                    err = data.get("Error Message") or data.get("Note")
                    if err:
                        print(f"    ! Alpha Vantage TREASURY_YIELD error: {err}")
                    else:
                        print("    ! Alpha Vantage TREASURY_YIELD unexpected response structure.")
                    return None
                df = pd.DataFrame(data["data"])
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                y = pd.to_numeric(df["value"], errors="coerce")
                y = y[(y.index >= start_dt) & (y.index <= end_dt)]
                return y
            except Exception as e:
                print(f"    ! Alpha Vantage treasury yield failed: {e}")
                return None

        # S&P500 proxy: SPY
        spy = fetch_time_series_daily("SPY")
        if spy is not None and not spy.empty:
            spy.name = "sp500"
            combined_df = pd.DataFrame(spy)

        # DXY: symbol "DXY" (Alpha Vantage supports via TIME_SERIES_DAILY)
        dxy = fetch_time_series_daily("DXY")
        if dxy is not None and not dxy.empty:
            dxy.name = "dxy"
            if combined_df.empty:
                combined_df = pd.DataFrame(dxy)
            else:
                combined_df = combined_df.join(dxy, how="outer")

        # VIX: use VIX index via symbol "^VIX" proxy "VIX"
        vix = fetch_time_series_daily("VIX")
        if vix is not None and not vix.empty:
            vix.name = "vix"
            if combined_df.empty:
                combined_df = pd.DataFrame(vix)
            else:
                combined_df = combined_df.join(vix, how="outer")

        # 10Y yield
        us10y = fetch_treasury_yield()
        if us10y is not None and not us10y.empty:
            us10y.name = "us10y"
            if combined_df.empty:
                combined_df = pd.DataFrame(us10y)
            else:
                combined_df = combined_df.join(us10y, how="outer")

        if combined_df.empty:
            print("    !! Alpha Vantage macro fetch returned empty.")
        else:
            print(f"    ✓ Macro data (from Alpha Vantage): {len(combined_df)} rows")

        return combined_df

    def _fetch_sentiment_data(self):
        print("  Fetching Crypto Fear & Greed Index...")
        try:
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            response = requests.get(url, timeout=10, proxies=Config.PROXIES)
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
                r = requests.get(url, timeout=10, proxies=Config.PROXIES)
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

    def _fetch_blockchair_data(self, start_dt, end_dt):
        print("  Fetching Blockchair data (Non-BTC On-chain)...")
        self.blockchair_data = {}
        
        api_key = Config.BLOCKCHAIR_API_KEY
        
        symbol_chain_map = {
            "ETH": "ethereum",
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "DOGE": "dogecoin",
            "DASH": "dash",
            "ZEC": "zcash"
        }
        
        session = requests.Session()
        
        for symbol, chain in symbol_chain_map.items():
            try:
                url = f"https://api.blockchair.com/{chain}/stats"
                if api_key:
                    url = f"{url}?key={api_key}"
                
                resp = session.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                
                data = resp.json().get("data", {})
                if not data:
                    continue
                
                tx_24h = data.get("transactions_24h")
                active_24h = data.get("addresses_24h") or data.get("active_addresses_24h") or data.get("new_addresses_24h")
                
                row = {}
                if tx_24h is not None:
                    row["transaction_count"] = tx_24h
                if active_24h is not None:
                    row["active_addresses"] = active_24h
                
                if not row:
                    continue
                
                idx = pd.to_datetime(end_dt).tz_localize(None)
                df = pd.DataFrame([row], index=[idx])
                self.blockchair_data[symbol] = df
            except Exception:
                continue
        
        print(f"    ✓ Blockchair data: {len(self.blockchair_data)} symbols")

    def _fetch_glassnode_data(self, start_dt, end_dt):
        print("  Fetching Glassnode data (Non-BTC On-chain)...")
        api_key = Config.GLASSNODE_API_KEY
        if not api_key:
            print("    (No Glassnode API Key found, skipping non-BTC on-chain data)")
            return

        # Symbols to try. Glassnode uses symbols like ETH, LTC, MATIC, etc.
        # Note: Some symbols might be behind paywall for recent data.
        symbols = ["ETH", "LTC", "MATIC", "AVAX", "SOL", "BNB", "AAVE", "UNI", "LINK"]
        
        session = requests.Session()
        
        # Convert start/end to unix timestamp
        s = int(start_dt.timestamp())
        u = int(end_dt.timestamp())

        for symbol in symbols:
            # print(f"    Fetching Glassnode for {symbol}...")
            symbol_data = pd.DataFrame()
            
            try:
                # 1. Active Addresses
                url_addr = "https://api.glassnode.com/v1/metrics/addresses/active_count"
                params = {'a': symbol, 'api_key': api_key, 's': s, 'u': u, 'i': '24h'}
                
                resp = session.get(url_addr, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json() # list of {t, v}
                    if data:
                        df_addr = pd.DataFrame(data)
                        df_addr['t'] = pd.to_datetime(df_addr['t'], unit='s')
                        df_addr.set_index('t', inplace=True)
                        df_addr.rename(columns={'v': 'active_addresses'}, inplace=True)
                        symbol_data = df_addr
            except Exception as e:
                # print(f"      ! Error fetching addresses for {symbol}: {e}")
                pass

            try:
                # 2. Transaction Count
                url_tx = "https://api.glassnode.com/v1/metrics/transactions/count"
                params = {'a': symbol, 'api_key': api_key, 's': s, 'u': u, 'i': '24h'}
                
                resp = session.get(url_tx, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df_tx = pd.DataFrame(data)
                        df_tx['t'] = pd.to_datetime(df_tx['t'], unit='s')
                        df_tx.set_index('t', inplace=True)
                        df_tx.rename(columns={'v': 'transaction_count'}, inplace=True)
                        
                        if symbol_data.empty:
                            symbol_data = df_tx
                        else:
                            symbol_data = symbol_data.join(df_tx, how='outer')
            except Exception as e:
                pass
                
            if not symbol_data.empty:
                symbol_data.index = symbol_data.index.tz_localize(None)
                self.glassnode_data[symbol] = symbol_data
                
        print(f"    ✓ Glassnode data: {len(self.glassnode_data)} symbols")

    def get_macro_at(self, date: pd.Timestamp) -> dict:
        """Get macro data for a specific date (or nearest previous)"""
        if self.macro_data.empty:
            return {}
            
        try:
            # As of method
            idx = self.macro_data.index.asof(date)
            if pd.isna(idx):
                return {} # No data available before this date
            else:
                row = self.macro_data.loc[idx]
                
            return {
                "dxy": row.get('dxy', None),
                "us_10y_yield": row.get('us10y', None),
                "vix": row.get('vix', None),
                "sp500": row.get('sp500', None)
            }
        except:
            return {}

    def get_sentiment_at(self, date: pd.Timestamp) -> dict:
        if self.sentiment_data.empty:
            return {}
            
        try:
            # Fear & Greed is daily
            # Normalize date to midnight
            d = date.normalize()
            idx = self.sentiment_data.index.asof(d)
            
            if pd.isna(idx):
                return {}
            else:
                val = self.sentiment_data.loc[idx, 'value']
                classification = self.sentiment_data.loc[idx, 'value_classification']
                
            return {
                "fear_greed_index": int(val),
                "fear_greed_classification": classification
            }
        except:
            return {}

    def get_onchain_at(self, date: pd.Timestamp, symbol: str = "BTC") -> dict:
        # 1. If BTC, try generic onchain_data (Blockchain.com) first
        if symbol == "BTC":
            if not self.onchain_data.empty:
                try:
                    idx = self.onchain_data.index.asof(date)
                    if not pd.isna(idx):
                        row = self.onchain_data.loc[idx]
                        return {
                            "active_addresses": int(row.get('n-unique-addresses', 0)) if not pd.isna(row.get('n-unique-addresses')) else None,
                            "transaction_count": int(row.get('n-transactions', 0)) if not pd.isna(row.get('n-transactions')) else None,
                        }
                except:
                    pass
        
        # 2. If not BTC or BTC failed, try Blockchair data first (free multi-chain)
        base_symbol = symbol.split('/')[0]
        
        if base_symbol in self.blockchair_data:
            df_bc = self.blockchair_data[base_symbol]
            try:
                idx = df_bc.index.asof(date)
                if not pd.isna(idx):
                    row = df_bc.loc[idx]
                    return {
                        "active_addresses": int(row.get("active_addresses", 0)) if not pd.isna(row.get("active_addresses")) else None,
                        "transaction_count": int(row.get("transaction_count", 0)) if not pd.isna(row.get("transaction_count")) else None,
                    }
            except:
                pass
        
        # 3. If Blockchair missing, fall back to Glassnode data when available
        # Handle symbol mapping if needed (Glassnode usually uses standard tickers)
        if base_symbol in self.glassnode_data:
            df = self.glassnode_data[base_symbol]
            try:
                idx = df.index.asof(date)
                if not pd.isna(idx):
                    row = df.loc[idx]
                    return {
                        "active_addresses": int(row.get('active_addresses', 0)) if not pd.isna(row.get('active_addresses')) else None,
                        "transaction_count": int(row.get('transaction_count', 0)) if not pd.isna(row.get('transaction_count')) else None,
                    }
            except:
                pass
                
        return {}
