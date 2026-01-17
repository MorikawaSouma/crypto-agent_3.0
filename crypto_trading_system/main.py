import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system.config import Config
from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.feature_engineering import FeatureEngineer
from crypto_trading_system.llm_client import DeepSeekClient
from crypto_trading_system.agents.master_agent import MasterAgent
from crypto_trading_system.rolling_model import RollingModelManager


def main():
    print("Initializing Multi-Agent Quantitative Trading System (Live Analysis)...")
    dm = MarketDataManager()
    fe = FeatureEngineer()
    llm = DeepSeekClient(mock=getattr(Config, "LLM_MOCK", False))
    master = MasterAgent(llm)
    end_time = Config.END_TIME if Config.END_TIME else pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    if Config.START_TIME:
        start_time = Config.START_TIME
    else:
        limit_hours = Config.DEFAULT_LIMIT if Config.DEFAULT_LIMIT else 1440
        start_time = (pd.Timestamp.now() - pd.Timedelta(hours=limit_hours + 24)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Fetching daily data for ML selection from universe ({len(Config.UNIVERSE)} symbols)...")
    analysis_end_dt = pd.to_datetime(end_time)
    daily_start_date = (analysis_end_dt - pd.Timedelta(days=Config.FEATURE_WINDOW)).strftime('%Y-%m-%d')
    daily_end_date = analysis_end_dt.strftime('%Y-%m-%d')
    universe_data = {}
    for symbol in Config.UNIVERSE:
        df_daily = dm.get_historical_data(symbol, daily_start_date, daily_end_date, timeframe='1d')
        if df_daily.empty:
            print(f"No daily data for {symbol}, skipping in ML selection.")
            continue
        df_daily.index = pd.to_datetime(df_daily.index)
        universe_data[symbol] = df_daily
    if not universe_data:
        print("No universe data for ML selection, falling back to Config.SYMBOLS.")
        selected_symbols = Config.SYMBOLS
    else:
        rolling_manager = RollingModelManager()
        now_ts = pd.Timestamp(analysis_end_dt)
        rolling_manager.check_and_retrain(now_ts, universe_data)
        current_data_map = {}
        lookback_start = analysis_end_dt - pd.Timedelta(days=Config.FEATURE_WINDOW + 10)
        for sym, df_u in universe_data.items():
            mask = df_u.index >= lookback_start
            sub_df = df_u[mask]
            if not sub_df.empty:
                current_data_map[sym] = sub_df
        top_k_symbols = rolling_manager.predict(current_data_map, k=Config.TOP_K)
        if not top_k_symbols:
            print("ML selection returned empty set, falling back to Config.SYMBOLS.")
            selected_symbols = Config.SYMBOLS
        else:
            selected_symbols = top_k_symbols
    print(f"Selected symbols for agent analysis: {selected_symbols}")
    print("Fetching real auxiliary data...")
    try:
        master.prepare_data(start_time, end_time)
    except Exception as e:
        print(f"Warning: Failed to fetch real auxiliary data: {e}")
    print(f"Fetching daily data for {len(selected_symbols)} selected symbols for agent analysis...")
    market_data = dm.get_market_data(symbols=selected_symbols, timeframe='1d')
    results = []
    for symbol, df in market_data.items():
        print(f"\nProcessing {symbol}...")
        df_processed = fe.process(df)
        if df_processed.empty:
            print(f"Not enough data for {symbol}")
            continue
        try:
            decision = master.analyze_symbol(symbol, df_processed)
            results.append(decision)
            print(f"Decision for {symbol}: {decision['action']} (Score: {decision['final_score']:.2f})")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    print("\n" + "=" * 50)
    print("FINAL TRADING REPORT")
    print("=" * 50)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.reset_index(drop=True)
        dalio_weights = []
        for _, row in df_results.iterrows():
            ad = row.get('agent_details', {})
            dalio = ad.get('dalio', {}) if isinstance(ad, dict) else {}
            w = dalio.get('weight_suggestion', 0)
            try:
                dalio_weights.append(float(w))
            except Exception:
                dalio_weights.append(0.0)
        df_results['dalio_weight'] = dalio_weights
        total_w = df_results['dalio_weight'].sum()
        if total_w > 0:
            df_results['allocation_pct'] = df_results['dalio_weight'] / total_w
        else:
            equal = 1.0 / len(df_results)
            df_results['allocation_pct'] = equal
        print("\nRecommended Portfolio Allocation (Dalio Risk-Parity):")
        print("-" * 60)
        print(f"{'Symbol':<10} | {'Score':<6} | {'Action':<10} | {'DalioWeight':<12} | {'Allocation':<10}")
        print("-" * 60)
        for _, row in df_results.iterrows():
            print(f"{row['symbol']:<10} | {row['final_score']:<6.1f} | {row['action']:<10} | {row['dalio_weight']:<12.1f} | {row['allocation_pct']:.2%}")
        print("-" * 60)
        df_results.to_csv('trading_report.csv', index=False)
        print("\nReport saved to trading_report.csv")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
