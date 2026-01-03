import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.feature_engineering import FeatureEngineer
from crypto_trading_system.backtest_engine import BacktestEngine

def simple_strategy(df):
    """
    A simple MA Crossover strategy to simulate agent decisions
    """
    signals = pd.Series(0, index=df.index)
    
    # Buy when SMA_20 > SMA_50
    signals[df['SMA_20'] > df['SMA_50']] = 1
    
    # Sell when SMA_20 < SMA_50
    signals[df['SMA_20'] < df['SMA_50']] = -1
    
    # Introduce some noise to simulate "AI uncertainty" or varied opinions
    # signals = signals.apply(lambda x: x if np.random.random() > 0.1 else 0)
    
    return signals

def main():
    print("Starting Backtest Simulation...")
    
    # Configuration
    symbol = 'BTC/USDT'
    # Use Config time range if set, otherwise fallback to defaults or latest
    start_time = Config.START_TIME if Config.START_TIME else '2023-01-01 00:00:00'
    end_time = Config.END_TIME if Config.END_TIME else '2023-06-01 00:00:00'
    
    dm = MarketDataManager()
    fe = FeatureEngineer()
    engine = BacktestEngine()
    
    print(f"Fetching historical data for {symbol} from {start_time} to {end_time}...")
    # Use the new get_historical_data method
    df = dm.get_historical_data(symbol, start_time, end_time)
    
    if df.empty:
        print("No data fetched. Check proxy, internet connection, or date range.")
        # Fallback for testing without API connection
        print("Generating dummy data for testing...")
        dates = pd.date_range(start=start_time, end=end_time, freq='1H')
        df = pd.DataFrame({
            'open': np.linspace(20000, 30000, len(dates)),
            'high': np.linspace(20100, 30100, len(dates)),
            'low': np.linspace(19900, 29900, len(dates)),
            'close': np.linspace(20000, 30000, len(dates)) + np.random.normal(0, 100, len(dates)),
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
    
    print(f"Data points: {len(df)}")

    print("Processing features...")
    df = fe.process(df)
    
    print("Generating signals (Simulating Agent Logic)...")
    signals = simple_strategy(df)
    
    print("Running Backtest Engine...")
    result_df, trades = engine.run(df, signals, start_time, end_time)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(result_df.index, result_df['portfolio_value'], label='Portfolio Value')
    plt.title(f'Backtest Result: {symbol} ({start_time} - {end_time})')
    plt.xlabel('Date')
    plt.ylabel('Value (USDT)')
    plt.legend()
    plt.grid(True)
    
    output_file = 'backtest_chart.png'
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    main()
