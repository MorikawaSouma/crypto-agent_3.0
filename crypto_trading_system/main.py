import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system.config import Config
from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.feature_engineering import FeatureEngineer
from crypto_trading_system.llm_client import DeepSeekClient
from crypto_trading_system.agents.master_agent import MasterAgent

def main():
    print("Initializing Multi-Agent Quantitative Trading System (Live Analysis)...")
    
    # 1. Initialize Components
    dm = MarketDataManager()
    fe = FeatureEngineer()
    llm = DeepSeekClient()
    master = MasterAgent(llm)
    
    # 2. Fetch Data
    print(f"Fetching data for {len(Config.SYMBOLS)} symbols...")
    # Now get_market_data respects Config.START_TIME and Config.END_TIME automatically
    market_data = dm.get_market_data() 
    
    # 2.1 Fetch Real Auxiliary Data (Macro, Sentiment, On-chain)
    print("Fetching real auxiliary data...")
    # Calculate time range based on Config
    end_time = Config.END_TIME if Config.END_TIME else pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if Config.START_TIME:
        start_time = Config.START_TIME
    else:
        # Use DEFAULT_LIMIT to determine start time (assuming 1h timeframe)
        limit_hours = Config.DEFAULT_LIMIT if Config.DEFAULT_LIMIT else 1440
        # Add a small buffer (e.g. 24h) to ensure we cover the beginning
        start_time = (pd.Timestamp.now() - pd.Timedelta(hours=limit_hours + 24)).strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        master.prepare_data(start_time, end_time)
    except Exception as e:
        print(f"Warning: Failed to fetch real auxiliary data: {e}")

    results = []
    
    # 3. Analyze each symbol
    for symbol, df in market_data.items():
        print(f"\nProcessing {symbol}...")
        
        # Feature Engineering
        df_processed = fe.process(df)
        
        if df_processed.empty:
            print(f"Not enough data for {symbol}")
            continue
            
        # Run Analysis
        try:
            decision = master.analyze_symbol(symbol, df_processed)
            results.append(decision)
            
            print(f"Decision for {symbol}: {decision['action']} (Score: {decision['final_score']:.2f})")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            
    # 4. Generate Report with Portfolio Allocation
    print("\n" + "="*50)
    print("FINAL TRADING REPORT")
    print("="*50)
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Calculate Rank-Squared Allocation
        # Sort by Score
        df_results = df_results.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        
        n = len(df_results)
        df_results['rank'] = df_results.index + 1 # 1-based rank
        
        # Calculate Rank Score: (N - Rank + 1)^2
        # Rank 1: (10 - 1 + 1)^2 = 100
        # Rank 2: (10 - 2 + 1)^2 = 81
        df_results['rank_weight_score'] = (n - df_results['rank'] + 1) ** 2
        
        total_weight_score = df_results['rank_weight_score'].sum()
        df_results['allocation_pct'] = df_results['rank_weight_score'] / total_weight_score
        
        # Format output
        print("\nRecommended Portfolio Allocation (Rank-Squared Method):")
        print("-" * 60)
        print(f"{'Symbol':<10} | {'Score':<6} | {'Action':<6} | {'Rank':<4} | {'Allocation':<10}")
        print("-" * 60)
        
        for _, row in df_results.iterrows():
            print(f"{row['symbol']:<10} | {row['final_score']:<6.1f} | {row['action']:<6} | {row['rank']:<4} | {row['allocation_pct']:.2%}")
            
        print("-" * 60)

        # Save to CSV
        df_results.to_csv('trading_report.csv', index=False)
        print("\nReport saved to trading_report.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
