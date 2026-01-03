import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BacktestEngine:
    def __init__(self, initial_capital=100000.0, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run(self, df: pd.DataFrame, signals: pd.Series, start_time: str = None, end_time: str = None):
        """
        Run backtest based on signals.
        signals: Series with index matching df, values: 1 (Buy), -1 (Sell), 0 (Hold)
        start_time, end_time: Optional string dates to filter backtest range
        """
        # Filter by time range if provided
        if start_time:
            df = df[df.index >= pd.to_datetime(start_time)]
            signals = signals[signals.index >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df.index <= pd.to_datetime(end_time)]
            signals = signals[signals.index <= pd.to_datetime(end_time)]
            
        capital = self.initial_capital
        position = 0.0 # Amount of asset
        
        portfolio_values = []
        trades = []
        
        for i in range(len(df)):
            price = df.iloc[i]['close']
            signal = signals.iloc[i]
            timestamp = df.index[i]
            
            # Execute Signal
            if signal == 1 and position == 0: # Buy
                amount_to_buy = (capital * 0.99) / price # Use 99% of capital
                cost = amount_to_buy * price
                fee = cost * self.commission
                
                if capital >= cost + fee:
                    capital -= (cost + fee)
                    position += amount_to_buy
                    trades.append({'type': 'BUY', 'price': price, 'time': timestamp, 'amount': amount_to_buy})
            
            elif signal == -1 and position > 0: # Sell
                revenue = position * price
                fee = revenue * self.commission
                
                capital += (revenue - fee)
                position = 0
                trades.append({'type': 'SELL', 'price': price, 'time': timestamp, 'amount': 0})
                
            # Calculate Portfolio Value
            current_val = capital + (position * price)
            portfolio_values.append(current_val)
            
        # Results
        result_df = pd.DataFrame({
            'timestamp': df.index,
            'portfolio_value': portfolio_values,
            'price': df['close']
        }).set_index('timestamp')
        
        self.calculate_metrics(result_df, trades)
        return result_df, trades
        
    def calculate_metrics(self, result_df, trades):
        initial = result_df['portfolio_value'].iloc[0]
        final = result_df['portfolio_value'].iloc[-1]
        
        total_return = (final - initial) / initial
        
        # Daily Returns (approximate if data is hourly)
        result_df['returns'] = result_df['portfolio_value'].pct_change()
        
        # Sharpe Ratio (assuming hourly data, annualized)
        sharpe = result_df['returns'].mean() / result_df['returns'].std() * np.sqrt(24*365) if result_df['returns'].std() != 0 else 0
        
        # Max Drawdown
        rolling_max = result_df['portfolio_value'].cummax()
        drawdown = (result_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print("\n=== Backtest Performance ===")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Total Trades: {len(trades)}")
        
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        }

if __name__ == "__main__":
    # Test with dummy data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'close': np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    }, index=dates)
    
    # Random signals
    signals = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates)
    
    engine = BacktestEngine()
    engine.run(df, signals)
