import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List

from crypto_trading_system.config import Config
from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.ml_strategy import LightGBMStrategy
from crypto_trading_system.mock_data import MockDataProvider

def run_ml_backtest():
    print(f"Starting ML Backtest (Top-{Config.TOP_K} Strategy)...")
    print(f"Universe: {Config.UNIVERSE}")
    print(f"Model: {Config.ML_MODEL}")
    
    # 1. Initialize Components
    data_manager = MarketDataManager()
    strategy = LightGBMStrategy()
    
    # 2. Fetch Data
    # We need enough history for Training + Backtest
    # Let's say we want 1 year of training + 3 months of backtest
    total_days = Config.TRAIN_WINDOW + Config.TEST_WINDOW + Config.LOOKBACK_WINDOW
    start_date = (datetime.now() - timedelta(days=total_days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    universe_data: Dict[str, pd.DataFrame] = {}
    
    for symbol in Config.UNIVERSE:
        # Fetch data
        df = data_manager.get_historical_data(symbol, start_date, end_date, timeframe='1d')
        if not df.empty:
            universe_data[symbol] = df
        else:
            print(f"Warning: No data for {symbol}, using MOCK data.")
            universe_data[symbol] = MockDataProvider.generate_price_history(symbol, days=total_days)
            
    if not universe_data:
        print("No data available. Aborting.")
        return

    # 3. Time Splitting
    # Find common date range
    # Ideally, we align all data.
    # For simplicity, we just use the dates from BTC/USDT as the master clock
    master_symbol = "BTC/USDT"
    if master_symbol not in universe_data:
        master_symbol = list(universe_data.keys())[0]
        
    master_dates = universe_data[master_symbol].index
    
    # Split point
    # Train on first part, Test on last part
    split_idx = len(master_dates) - Config.TEST_WINDOW
    if split_idx < 100:
        print("Not enough data for split.")
        return
        
    split_date = master_dates[split_idx]
    print(f"Training data up to {split_date}. Backtesting from {split_date} to End.")
    
    # 4. Train Model
    print("Starting Training...")
    # Prepare training data: slice all DFs up to split_date
    train_data_map = {}
    for sym, df in universe_data.items():
        train_data_map[sym] = df[df.index < split_date].copy()
        
    try:
        strategy.train(train_data_map)
        print("Training Completed.")
    except Exception as e:
        print(f"Training Failed: {e}")
        return
    
    # 5. Backtest Loop
    # Iterate through the test period
    test_dates = master_dates[split_idx:]
    
    # Portfolio Value Tracking
    portfolio_value = 10000.0
    portfolio_history = []
    
    # Benchmarks
    # 1. Buy and Hold BTC
    btc_start_price = universe_data["BTC/USDT"].loc[test_dates[0]]['close']
    btc_holdings = 10000.0 / btc_start_price
    
    # 2. UCRP (Equal Weight Universe)
    ucrp_value = 10000.0
    
    print("\nRunning Simulation...")
    
    for i in range(len(test_dates) - 1):
        current_date = test_dates[i]
        next_date = test_dates[i+1]
        
        # 1. Get Data available up to current_date
        # In a real scenario, at Close of current_date, we have the full candle.
        current_data_map = {}
        for sym, df in universe_data.items():
            if current_date in df.index:
                # Slice up to current_date inclusive
                # We need enough history for features (LOOKBACK_WINDOW)
                # Optimization: pass full df, but strategy takes last row?
                # Actually strategy needs window.
                # Let's pass the last 60 days ending at current_date
                start_window = current_date - timedelta(days=60)
                mask = (df.index >= start_window) & (df.index <= current_date)
                slice_df = df[mask]
                if not slice_df.empty:
                    current_data_map[sym] = slice_df
        
        # 2. Strategy Selects Top K
        top_k_symbols = strategy.predict_top_k(current_data_map, k=Config.TOP_K)
        
        # 3. Calculate Returns for Next Day
        # Return = (Price_next / Price_current) - 1
        # Portfolio Return = Average of Top K Returns (Equal Weight)
        
        daily_rets = []
        for sym in top_k_symbols:
            if sym in universe_data:
                df = universe_data[sym]
                if next_date in df.index and current_date in df.index:
                    p_curr = df.loc[current_date]['close']
                    p_next = df.loc[next_date]['close']
                    ret = (p_next / p_curr) - 1
                    daily_rets.append(ret)
        
        if daily_rets:
            strategy_ret = sum(daily_rets) / len(daily_rets)
        else:
            strategy_ret = 0.0
            
        portfolio_value *= (1 + strategy_ret)
        
        # Benchmark Calculation
        # BTC Return
        btc_curr = universe_data["BTC/USDT"].loc[current_date]['close']
        btc_next = universe_data["BTC/USDT"].loc[next_date]['close']
        btc_ret = (btc_next / btc_curr) - 1
        
        # UCRP Return (Avg of all universe)
        univ_rets = []
        for sym in Config.UNIVERSE:
            if sym in universe_data:
                df = universe_data[sym]
                if next_date in df.index and current_date in df.index:
                    p_c = df.loc[current_date]['close']
                    p_n = df.loc[next_date]['close']
                    univ_rets.append((p_n / p_c) - 1)
        
        ucrp_ret = sum(univ_rets) / len(univ_rets) if univ_rets else 0.0
        ucrp_value *= (1 + ucrp_ret)
        
        portfolio_history.append({
            'date': next_date,
            'Strategy': portfolio_value,
            'BuyHold_BTC': btc_holdings * btc_next,
            'UCRP': ucrp_value
        })
        
        if i % 10 == 0:
            print(f"Date: {current_date.date()} | Port: {portfolio_value:.2f} | BTC: {btc_holdings * btc_next:.2f} | Selected: {top_k_symbols}")

    # 6. Final Results
    results_df = pd.DataFrame(portfolio_history).set_index('date')
    
    print("\n=== Final Backtest Results ===")
    print(results_df.tail(1))
    
    # Calculate Total Return
    strat_total_ret = (results_df['Strategy'].iloc[-1] / 10000.0) - 1
    btc_total_ret = (results_df['BuyHold_BTC'].iloc[-1] / 10000.0) - 1
    ucrp_total_ret = (results_df['UCRP'].iloc[-1] / 10000.0) - 1
    
    print(f"Strategy Total Return: {strat_total_ret*100:.2f}%")
    print(f"Buy&Hold BTC Return: {btc_total_ret*100:.2f}%")
    print(f"UCRP Total Return:   {ucrp_total_ret*100:.2f}%")
    
    # Win Rate? Sharpe?
    # Simple win rate of daily returns
    strat_rets = results_df['Strategy'].pct_change().dropna()
    sharpe = (strat_rets.mean() / strat_rets.std()) * (252**0.5) if strat_rets.std() != 0 else 0
    
    # Max Drawdown
    rolling_max = results_df['Strategy'].cummax()
    drawdown = (results_df['Strategy'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    print(f"Strategy Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Strategy'], label='ML Strategy')
    plt.plot(results_df.index, results_df['BuyHold_BTC'], label='Buy & Hold BTC', alpha=0.6)
    plt.plot(results_df.index, results_df['UCRP'], label='UCRP (Equal Weight)', alpha=0.6)
    
    plt.title(f'ML Strategy Backtest (Top-{Config.TOP_K} LightGBM)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.grid(True)
    
    output_file = 'ml_backtest_results_v2.png'
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    print("DONE PLOTTING")

if __name__ == "__main__":
    run_ml_backtest()
