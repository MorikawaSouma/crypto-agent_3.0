import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load env first
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system.config import Config
from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.agent_backtest import AgentBacktestEngine
from crypto_trading_system.agents.master_agent import MasterAgent
from crypto_trading_system.llm_client import DeepSeekClient

def test_llm_connection():
    print("Testing LLM connection before anything else...")
    client = DeepSeekClient()
    res = client.query("You are a bot.", "Say hi.")
    print(f"LLM Pre-check result: {res}")

def main():
    print("="*50)
    print("STARTING MULTI-ASSET AGENT BACKTEST SYSTEM")
    print("="*50)
    
    # test_llm_connection()

    # 1. Initialize Components
    print("\n[1/5] Initializing Components...")
    
    # NOTE: Enabling mock=True for backtest loop stability due to proxy conflicts.
    # The pre-check proved real LLM works, but loop+ccxt breaks it.
    llm_client = DeepSeekClient(mock=False) # Mock DISABLED to use real model (deepseek-reasoner)
    master_agent = MasterAgent(llm_client)
    dm = MarketDataManager()
    print("✓ Agents initialized successfully")
    print("✓ Data Manager initialized")

    # Configuration
    symbols = Config.SYMBOLS
    #start_time = '2024-01-01 00:00:00'
    #end_time = '2024-01-05 00:00:00' # Short period for demo speed
    
    start_time = Config.START_TIME
    end_time = Config.END_TIME
    
    if not start_time or not end_time:
        print("Error: START_TIME and END_TIME must be set in Config for backtesting.")
        return

    print(f"\n[2/5] Configuration:")
    print(f"Symbols: {symbols}")
    print(f"Time Range: {start_time} to {end_time}")
    print(f"Agents: {len(master_agent.agents)} (Buffett, Soros, Dalio, Simons, Sentiment)")
    
    # 3. Fetch Data for ALL Symbols
    print(f"\n[3/5] Fetching Historical Data...")
    
    # Calculate fetch start time (start_time - 30 days for warmup)
    try:
        fetch_start_dt = pd.to_datetime(start_time) - pd.Timedelta(days=30)
        fetch_start_str = fetch_start_dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error parsing start time: {e}")
        return

    print(f"Fetching from {fetch_start_str} to {end_time} (includes warmup)...")

    data_map = {}
    for symbol in symbols:
        try:
            print(f"Fetching {symbol}...")
            df = dm.get_historical_data(symbol, fetch_start_str, end_time)
            if not df.empty:
                data_map[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} candles")
            else:
                print(f"  ✗ {symbol}: No data")
        except Exception as e:
            print(f"  ✗ {symbol}: Error {e}")
            
    if not data_map:
        print("No data fetched. Aborting.")
        return

    # CLEANUP to avoid conflicts
    del dm
    import gc
    gc.collect()
    print("✓ Data Manager cleaned up")

    # 4. Run Backtest
    print(f"\n[4/5] Running Multi-Asset Agent Backtest (This may take a while)...")
    print("Step size: Every 24 hours")
    
    engine = AgentBacktestEngine(master_agent, initial_capital=Config.INITIAL_CAPITAL)
    
    try:
        results_df, trades = engine.run_multi_asset(data_map, start_time, end_time)
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        return
    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Visualize and Report
    print(f"\n[5/5] Generating Report...")
    
    if not results_df.empty:
        # Save results to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"backtest_results_multiset_{timestamp_str}.csv"
        results_df.to_csv(csv_filename)
        print(f"✓ Results saved to {csv_filename}")
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Subplot 1: Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(results_df.index, results_df['portfolio_value'], label='Portfolio Value', color='blue')
        plt.title(f'Multi-Asset Agent Backtest ({start_time} - {end_time})')
        plt.ylabel('Value (USDT)')
        plt.grid(True)
        plt.legend()
        
        # Subplot 2: Asset Allocation (Stacked Area)
        # We need to reconstruct allocation history
        # results_df['positions'] contains a dict string, but we can't plot that directly easily in matplotlib without parsing
        # But we can try to extract major assets
        
        # Since we have positions in df, let's extract
        # Note: positions is a dict in the dataframe cell? No, pandas might have issues storing dicts if loaded from CSV, 
        # but here it is in memory.
        
        # Extract positions into columns
        # position_df = pd.DataFrame(results_df['positions'].tolist(), index=results_df.index)
        # But prices change, so we want Value Allocation
        
        # Simpler: Just plot portfolio value for now to avoid complexity in this step
        
        plt.tight_layout()
        plot_filename = f"backtest_chart_multiset_{timestamp_str}.png"
        plt.savefig(plot_filename)
        print(f"✓ Chart saved to {plot_filename}")
        
        # Print Summary again just in case
        engine.calculate_metrics(results_df, trades)
        
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
