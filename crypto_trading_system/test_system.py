import sys
import os
import pandas as pd
import time

# Add project root to path
# Assuming we run from d:\agent2\agent2 or the script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the parent of project_root to handle 'agent2' if needed, but likely not.
# We will import from crypto_trading_system directly
sys.path.insert(0, os.path.dirname(project_root))

from crypto_trading_system.agents.master_agent import MasterAgent
from crypto_trading_system.llm_client import DeepSeekClient

def test_system():
    print("=== Testing Enhanced System (Debate & ChromaDB) ===")
    
    # 1. Initialize with Mock LLM
    client = DeepSeekClient(mock=True)
    master = MasterAgent(client, mode="debate")
    
    # Mock External Data Provider to avoid network calls
    master.real_data_provider.get_macro_at = lambda ts: {
        "vix": 20, 
        "nasdaq_correlation": 0.5,
        "dxy": 102.5,
        "us_10y_yield": 4.2
    }
    master.real_data_provider.get_sentiment_at = lambda ts: {
        "fear_greed_index": 50, 
        "news_sentiment_score": 0.5,
        "twitter_sentiment": 0.6,
        "reddit_sentiment": 0.4,
        "google_trends": 75
    }
    master.real_data_provider.get_onchain_at = lambda ts, symbol='BTC': {
        "mvrv_ratio": 1.5, 
        "net_flow": 100,
        "active_addresses": 500000,
        "transaction_count": 200000
    }

    # 2. Create Mock Market Data (Need >200 days for SMA_200)
    periods = 250
    dates = pd.date_range(start="2024-01-01", periods=periods, freq="D")
    df = pd.DataFrame({
        "open": [40000 + i*10 for i in range(periods)],
        "high": [41000 + i*10 for i in range(periods)],
        "low": [39000 + i*10 for i in range(periods)],
        "close": [40500 + i*10 for i in range(periods)],
        "volume": [1000000 for _ in range(periods)]
    }, index=dates)
    
    # 3. Test 3-Round Debate
    print("\n[Test 1] Running 3-Round Debate...")
    start_time = time.time()
    decision = master.analyze_symbol("BTC-TEST", df)
    print(f"Debate completed in {time.time() - start_time:.2f}s")
    print(f"Final Action: {decision['action']}, Score: {decision['final_score']:.2f}")
    
    # 4. Test Reflection & Memory
    print("\n[Test 2] Testing Reflection (ChromaDB Integration)...")
    # Simulate a losing trade where agents were wrong (e.g. they said BUY, but it was a loss)
    # Mock agent details to ensure they said BUY
    agent_details_mock = {
        "buffett": {"action": "BUY", "score": 80, "reasoning": "Value is good"},
        "soros": {"action": "BUY", "score": 75, "reasoning": "Trend is up"}
    }
    
    trade_details = {
        "symbol": "BTC-TEST",
        "action": "BUY",
        "outcome": "loss",
        "pnl": -5.0,
        "agent_details": agent_details_mock
    }
    
    print("Triggering reflect_on_trade...")
    master.reflect_on_trade(trade_details)
    
    # Verify memory storage
    print("\n[Test 3] Verifying Memory Retrieval...")
    # Give ChromaDB a moment to persist/index if async
    time.sleep(1)
    
    # Retrieve from Buffett's memory
    memories = master.buffett.memory.retrieve_relevant("Value is good", limit=5)
    print(f"Retrieved {len(memories)} memories from Buffett Agent.")
    for i, mem in enumerate(memories):
        print(f"  {i+1}. Context: {mem.get('context', '')[:50]}... | Outcome: {mem.get('outcome')}")
        
    if len(memories) > 0:
        print("\nSUCCESS: Memory stored and retrieved via ChromaDB.")
    else:
        print("\nFAILURE: No memories retrieved.")

if __name__ == "__main__":
    test_system()
