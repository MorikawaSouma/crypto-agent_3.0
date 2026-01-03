import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .agents.master_agent import MasterAgent
from .feature_engineering import FeatureEngineer

class AgentBacktestEngine:
    def __init__(self, master_agent: MasterAgent, initial_capital=100000.0, commission=0.001):
        self.master_agent = master_agent
        self.initial_capital = initial_capital
        self.commission = commission
        self.fe = FeatureEngineer()
        
    def run_multi_asset(self, data_map: Dict[str, pd.DataFrame], start_time: str = None, end_time: str = None):
        """
        Run backtest for multiple assets with portfolio allocation
        """
        # 1. Pre-calculate features for ALL symbols
        print("Pre-calculating features for all symbols...")
        processed_data = {}
        common_indices = None
        
        for symbol, df in data_map.items():
            print(f"Processing {symbol}...")
            df_proc = self.fe.process(df.copy())
            if df_proc.empty:
                print(f"Warning: Empty features for {symbol}, skipping.")
                continue
            processed_data[symbol] = df_proc
            
            # Find common time intersection
            if common_indices is None:
                common_indices = df_proc.index
            else:
                common_indices = common_indices.intersection(df_proc.index)
        
        if not processed_data:
            print("No valid data to backtest.")
            return pd.DataFrame(), []

        # 2. Filter time range
        if start_time:
            common_indices = common_indices[common_indices >= pd.to_datetime(start_time)]
        if end_time:
            common_indices = common_indices[common_indices <= pd.to_datetime(end_time)]
            
        if len(common_indices) == 0:
            print(f"No common data found in range {start_time} to {end_time}")
            return pd.DataFrame(), []

        print(f"Starting Multi-Asset Backtest. Symbols: {list(processed_data.keys())}")
        print(f"Period: {start_time} to {end_time}")
        print(f"Total Steps to Process: {len(common_indices)}")
        
        # Prepare auxiliary data (Real Data)
        if hasattr(self.master_agent, 'prepare_data'):
            self.master_agent.prepare_data(start_time, end_time)

        # Portfolio State
        cash = self.initial_capital
        positions = {sym: 0.0 for sym in processed_data} # Amount held
        portfolio_history = []
        trades = []
        
        # Step size: Every 24 hours
        # We need to find the integer locations in the common_indices
        # Since common_indices is a DatetimeIndex, we iterate directly
        # But for performance we might want to skip.
        # Let's just iterate over the filtered common_indices with step
        
        # Convert to list for slicing
        all_timestamps = common_indices.sort_values()
        step_timestamps = all_timestamps[::24] # Daily rebalance
        
        print(f"Total rebalance steps: {len(step_timestamps)}")
        
        for idx_i, current_ts in enumerate(step_timestamps):
            print(f"\nProcessing step {idx_i+1}/{len(step_timestamps)} [{current_ts}]...")
            
            # 1. Get Scores for all assets
            scores = {}
            prices = {}
            
            for symbol, df_proc in processed_data.items():
                # Slice up to current_ts
                # We need to find the location of current_ts in df_proc
                # Since we intersected indices, it must exist
                
                # Get data up to this point
                current_slice = df_proc.loc[:current_ts]
                if current_slice.empty:
                    continue
                    
                latest_price = current_slice.iloc[-1]['close']
                prices[symbol] = latest_price
                
                # Run Agent Analysis
                # To save tokens/time, we might want to cache or optimize
                # But here we run full analysis
                decision = self.master_agent.analyze_symbol(symbol, current_slice)
                scores[symbol] = decision['final_score']
                print(f"  {symbol}: Score {decision['final_score']:.1f}")

            # 2. Calculate Target Allocation (Score Squared Ranking)
            # Rank scores: High to Low
            sorted_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            
            # Assign Rank Scores: N^2, (N-1)^2, ... 1^2
            # Example for 3 assets: 3^2=9, 2^2=4, 1^2=1. Total=14. Weights=9/14, 4/14, 1/14.
            n = len(sorted_symbols)
            rank_scores = {}
            total_rank_score = 0
            
            for rank, sym in enumerate(sorted_symbols):
                # Rank 0 (Top) gets score (N-0)^2
                # Rank 1 gets (N-1)^2
                r_score = (n - rank) ** 2
                rank_scores[sym] = r_score
                total_rank_score += r_score
                
            # Calculate Target Weights
            target_weights = {}
            for sym in sorted_symbols:
                target_weights[sym] = rank_scores[sym] / total_rank_score
                
            print(f"  Target Allocation: { {k: f'{v:.1%}' for k,v in target_weights.items()} }")
            
            # 3. Rebalance Portfolio
            # Calculate current total portfolio value
            current_portfolio_value = cash
            for sym, amt in positions.items():
                if sym in prices:
                    current_portfolio_value += amt * prices[sym]
            
            # Execute Trades
            # Sell first to free up cash
            for sym in positions:
                if sym not in target_weights:
                    # Sell all if not in target (e.g. data missing)
                    if positions[sym] > 0:
                        revenue = positions[sym] * prices[sym]
                        fee = revenue * self.commission
                        cash += (revenue - fee)
                        trades.append({
                            'time': current_ts, 'symbol': sym, 'type': 'SELL', 
                            'price': prices[sym], 'amount': positions[sym], 'reason': 'Exit'
                        })
                        positions[sym] = 0.0
                        
            # Adjust positions to target
            for sym, target_w in target_weights.items():
                if sym not in prices:
                    print(f"  Warning: No price for {sym}, cannot trade.")
                    continue
                    
                target_value = current_portfolio_value * target_w
                current_sym_value = positions[sym] * prices[sym]
                
                diff = target_value - current_sym_value
                
                print(f"  DEBUG: {sym} TargetVal=${target_value:.2f} CurrVal=${current_sym_value:.2f} Diff=${diff:.2f} Price=${prices[sym]}")

                # Threshold for trading (avoid dust)
                if abs(diff) > (current_portfolio_value * 0.01): # 1% threshold
                    if diff > 0: # Buy
                        cost = diff
                        fee = cost * self.commission
                        
                        # Cap cost if it exceeds available cash (including fee)
                        if (cost + fee) > cash:
                            # Use slightly less than full cash to avoid precision issues
                            max_cost = cash / (1 + self.commission)
                            cost = max_cost * 0.999 # Safety buffer
                            fee = cost * self.commission
                            
                        if cash >= (cost + fee) and cost > 0:
                            amount = cost / prices[sym]
                            cash -= (cost + fee)
                            positions[sym] += amount
                            print(f"  ACTION: BUY {sym} {amount:.6f} @ ${prices[sym]:.2f}")
                            trades.append({
                                'time': current_ts, 'symbol': sym, 'type': 'BUY', 
                                'price': prices[sym], 'amount': amount, 'reason': f'Rebalance (Rank {sorted_symbols.index(sym)+1})'
                            })
                        else:
                             print(f"  DEBUG: Not enough cash to buy {sym}. Cost+Fee=${cost+fee:.2f}, Cash=${cash:.2f}")

                    elif diff < 0: # Sell
                        revenue = abs(diff)
                        fee = revenue * self.commission
                        amount = revenue / prices[sym]
                        if positions[sym] >= amount * 0.999: # Tolerance for float errors
                            # Cap amount to held position
                            if amount > positions[sym]:
                                amount = positions[sym]
                                
                            cash += (amount * prices[sym]) * (1 - self.commission)
                            positions[sym] -= amount
                            print(f"  ACTION: SELL {sym} {amount:.6f} @ ${prices[sym]:.2f}")
                            trades.append({
                                'time': current_ts, 'symbol': sym, 'type': 'SELL', 
                                'price': prices[sym], 'amount': amount, 'reason': f'Rebalance (Rank {sorted_symbols.index(sym)+1})'
                            })
                        else:
                            print(f"  DEBUG: Not enough position to sell {sym}. Need {amount}, have {positions[sym]}")
                else:
                    print(f"  DEBUG: Diff {diff:.2f} below threshold.")

            # Record History
            total_val = cash + sum([positions[s] * prices.get(s, 0) for s in positions])
            portfolio_history.append({
                'timestamp': current_ts,
                'portfolio_value': total_val,
                'cash': cash,
                'positions': positions.copy()
            })
            print(f"  Portfolio Value: ${total_val:.2f}")

        return pd.DataFrame(portfolio_history).set_index('timestamp'), trades

    def calculate_metrics(self, df: pd.DataFrame, trades: List[Dict]):
        if df.empty:
            return
            
        initial_value = self.initial_capital
        final_value = df.iloc[-1]['portfolio_value']
        
        # Total Return
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized Return (assuming daily data steps in df, but here we have steps)
        # We can use timestamps
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0
            
        # Sharpe Ratio
        # Calculate daily returns
        # We need to resample to daily if steps are not daily, but here they are 24h
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)
        daily_returns = df['returns']
        
        risk_free_rate = 0.02 # Assumed 2%
        excess_returns = daily_returns - (risk_free_rate / 365)
        sharpe_ratio = np.sqrt(365) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0
        
        # Max Drawdown
        df['cummax'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        print("\n" + "="*30)
        print("MULTI-ASSET PORTFOLIO PERFORMANCE")
        print("="*30)
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Value:     ${final_value:,.2f}")
        print(f"Total Return:    {total_return:.2%}")
        print(f"Sharpe Ratio:    {sharpe_ratio:.2f}")
        print(f"Max Drawdown:    {max_drawdown:.2%}")
        print(f"Total Trades:    {len(trades)}")
        print("="*30 + "\n")
