import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system.config import Config
from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.mock_data import MockDataProvider
from crypto_trading_system.rolling_model import RollingModelManager
from crypto_trading_system.agents.master_agent import MasterAgent

from crypto_trading_system.llm_client import DeepSeekClient
from crypto_trading_system.feature_engineering import FeatureEngineer
import os
import json

def run_live_signals():
    print("=== Crypto AI Agent - Live Signal Generation ===")
    
    # 1. Fetch Latest Data
    # Fetch enough history for rolling train and long feature window
    total_days = Config.FEATURE_WINDOW
    start_date = (datetime.now() - timedelta(days=total_days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching Live Data for Universe: {Config.UNIVERSE}")
    data_manager = MarketDataManager()
    universe_data = {}
    
    # Force Mock for Demo if Fetch fails or for speed
    USE_MOCK_FOR_DEMO = True
    
    for symbol in Config.UNIVERSE:
        if USE_MOCK_FOR_DEMO:
             print(f"Fetching {symbol}... (Mocking)", end="", flush=True)
             df = pd.DataFrame() # Force fail to trigger mock logic below
        else:
            print(f"Fetching {symbol}...", end="", flush=True)
            try:
                df = data_manager.get_historical_data(symbol, start_date, end_date, timeframe='1d')
            except Exception as e:
                print(f" Error: {e}")
                df = pd.DataFrame()
            
        if df.empty:
            print(f" Failed/Empty. Using Mock Data.")
            df = MockDataProvider.generate_price_history(symbol, days=total_days)
        else:
            print(f" Done ({len(df)} rows).")
        
        df.index = pd.to_datetime(df.index)
        universe_data[symbol] = df

    # 2. Train Rolling Model
    print("Training Rolling LightGBM Model on recent history...")
    rolling_manager = RollingModelManager()
    
    # Train using data up to yesterday (to predict today/tomorrow)
    # We pass 'now' as the current date, the manager slices internally
    now_ts = pd.Timestamp(datetime.now())
    rolling_manager.check_and_retrain(now_ts, universe_data)
    
    # 3. Predict Top K
    print("Predicting Top Candidates for Next 24H...")
    
    # Prepare current data map (last 60 days)
    current_data_map = {}
    lookback_start = now_ts - timedelta(days=Config.FEATURE_WINDOW + 10)
    
    for sym, df in universe_data.items():
        mask = df.index >= lookback_start
        sub_df = df[mask]
        if not sub_df.empty:
            current_data_map[sym] = sub_df
            
    top_k_symbols = rolling_manager.predict(current_data_map, k=Config.TOP_K)
    print(f"ML Model Selected Top {Config.TOP_K}: {top_k_symbols}")
    
    if not top_k_symbols:
        print("No signals generated.")
        return

    # 4. Agent Analysis (Debate)
    print("\n--- Starting Multi-Agent Debate Analysis ---")
    
    # Initialize LLM Client (Real, or Mock if Config says so)
    # User requested real data, so we prefer real LLM unless explicitly mocked in Config
    use_mock = getattr(Config, "LLM_MOCK", False)
    llm = DeepSeekClient(mock=use_mock)
    master_agent = MasterAgent(llm_client=llm, mode=getattr(Config, "AGENT_MODE", "debate"))
    feedback_path = getattr(Config, "TRADE_FEEDBACK_PATH", None)
    if feedback_path and isinstance(feedback_path, str) and os.path.isfile(feedback_path):
        try:
            with open(feedback_path, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
            if isinstance(feedback_list, list):
                for td in feedback_list:
                    if isinstance(td, dict):
                        master_agent.reflect_on_trade(td)
                print(f"Applied trade feedback from {feedback_path}")
        except Exception as e:
            print(f"Failed to apply trade feedback from {feedback_path}: {e}")
    
    # We construct a query that forces the Master Agent to focus on these specific coins
    # We pass the data implicitly via the MasterAgent's internal data preparation (it re-fetches or we need to patch it)
    
    # In a real system, we'd inject the specific data into the MasterAgent
    # Here we'll manually invoke the analysis for the top K
    
    results = []
    dalio_weights = {}
    
    for symbol in top_k_symbols:
        print(f"\n>>> Analyzing {symbol} with AI Agents...")
        
        # Get raw data
        df = universe_data.get(symbol)
        if df is None or df.empty:
            print(f"Skipping {symbol}: No data.")
            continue
            
        # Feature Engineering for Agent Consumption (Traditional Indicators)
        # Note: ML model used Alpha158, Agents use traditional indicators via FeatureEngineer
        try:
            df_processed = FeatureEngineer.process(df)
            
            # Run Analysis
            result = master_agent.analyze_symbol(symbol, df_processed)
            results.append(result)
            
            # Extract Dalio weight suggestion
            ad = result.get('agent_details', {})
            dalio = ad.get('dalio', {}) if isinstance(ad, dict) else {}
            w = dalio.get('weight_suggestion', 0)
            try:
                dalio_weights[symbol] = max(0.0, float(w))
            except Exception:
                dalio_weights[symbol] = 0.0
            
            # Output Result
            print(f"--- Result for {symbol} ---")
            print(f"Score: {result['final_score']:.2f}/100")
            print(f"Action: {result['action']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Dynamic Weights: {result['dynamic_weights']}")
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. Build Target Allocation using Dalio's weights (normalized)
    if dalio_weights:
        total_w = sum(dalio_weights.values())
        if total_w > 0:
            allocation = {sym: (w / total_w) for sym, w in dalio_weights.items()}
        else:
            # Fallback to equal weight among Top-K
            eq = 1.0 / len(top_k_symbols)
            allocation = {sym: eq for sym in top_k_symbols}
        
        print("\n--- Target Allocation (Dalio normalized) ---")
        for sym, pct in allocation.items():
            print(f"{sym}: {pct:.2%}")
    else:
        print("\nNo Dalio weights produced; cannot form allocation.")

if __name__ == "__main__":
    run_live_signals()
